"""Freeze Blender G34 geometry without importing the candidate environment.

Run with Blender's Python 3.11 module, not the simulator's Python environment::

    .venvs/occlusion-oracle-py3.11/bin/python -m scripts.testing.freeze_occlusion_ball_oracle \
        --source-root /path/to/object-permanence --output /path/to/new/fixture

The authored scene is used without the dataset driver's diversity transforms.
Depth and object identity come from Blender's evaluated triangle ray caster at
pixel centers. This is a geometry oracle, not an EEVEE RGB/antialiasing oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import runpy
import subprocess
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "824baddcdad087810d35638bf8f273b884f74975"
SOURCE_SCRIPT = (
    "object_permanence/generator/tasks/G34_occ_small_ball_behind_screen/"
    "pb_task_00000022_pb_occ_ball_behind_screen_0023_blender.py"
)
OBJECTS = (
    "large_floor_base",
    "rear_backdrop_panel",
    "straight_gray_track",
    "wide_opaque_screen",
    "small_orange_ball",
)
# Dense coverage at both visibility boundaries, plus visible/hidden endpoints.
GEOMETRY_FRAMES = (1, 30, 42, 43, 44, 45, 46, 47, 48, 55, 60, 65, 73, 74, 75, 76, 77, 78, 79, 90, 120)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(source_root: Path, output: Path, width=320, height=180):
    import bpy
    from mathutils import Vector

    if bpy.app.version[:2] != (4, 4):
        raise RuntimeError("The source geometry oracle requires Blender 4.4.x")
    commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"Expected source {SOURCE_COMMIT}, got {commit}")
    if subprocess.check_output(["git", "-C", str(source_root), "diff", "HEAD", "--"], text=True):
        raise ValueError("Reference source must be unmodified")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen oracle: {output}")
    source = source_root / SOURCE_SCRIPT
    ns = runpy.run_path(str(source), run_name="occlusion_ball_reference")
    ns["clear_scene"]()
    objects = ns["build_scene_by_kind"]()
    ns["animate_scene_by_kind"](objects)
    scene, ball = objects["scene"], objects["ball"]
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.frame_set(1)
    bpy.context.view_layer.update()
    camera = scene.camera
    matrix = camera.matrix_world.copy()
    corners = camera.data.view_frame(scene=scene)
    xmax = max(p.x / -p.z for p in corners)
    ymax = max(p.y / -p.z for p in corners)
    origin = matrix.translation.copy()
    rotation = matrix.to_3x3()
    directions = []
    for row in range(height):
        for col in range(width):
            ray = Vector(((2 * (col + 0.5) / width - 1) * xmax, (1 - 2 * (row + 0.5) / height) * ymax, -1))
            directions.append((rotation @ ray).normalized())
    forward = rotation @ Vector((0, 0, -1))
    labels = {name: i + 1 for i, name in enumerate(OBJECTS)}
    positions, quaternions = [], []
    source_frames = np.arange(1, 121, dtype=np.int64)
    for frame in source_frames:
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        world = ball.matrix_world
        positions.append(list(world.translation))
        quaternions.append(list(world.to_quaternion()))
    depths, ids = [], []
    for frame in GEOMETRY_FRAMES:
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depth = np.zeros((height, width), np.float32)
        segmentation = np.zeros((height, width), np.int16)
        for pixel, direction in enumerate(directions):
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=100)
            if hit:
                row, col = divmod(pixel, width)
                depth[row, col] = (location - origin).dot(forward)
                segmentation[row, col] = labels[obj.name]
        depths.append(depth)
        ids.append(segmentation)
        print(f"source frame {frame}: ball pixels {int((segmentation == 5).sum())}", flush=True)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    geometry = {}
    for name in OBJECTS:
        obj = bpy.data.objects[name]
        geometry[name] = dict(position=list(obj.matrix_world.translation), dimensions=list(obj.dimensions))
    output.mkdir(parents=True)
    archive = output / "blender_g34_nominal.npz"
    arrays = dict(
        source_frames=source_frames,
        positions=np.asarray(positions, np.float64),
        quaternions=np.asarray(quaternions, np.float64),
        geometry_frames=np.asarray(GEOMETRY_FRAMES, np.int64),
        depth=np.asarray(depths, np.float32),
        object_ids=np.asarray(ids, np.int16),
        camera_matrix=np.asarray(matrix, np.float64),
    )
    np.savez_compressed(archive, **arrays)
    manifest = dict(
        schema="stable-envs/occlusion-ball-oracle-v1",
        source_url="https://github.com/hokindeng/object-permanence",
        source_commit=commit,
        source_script=SOURCE_SCRIPT,
        source_script_sha256=sha256(source),
        source_license="CC-BY-NC-4.0",
        source_profile="G34 authored nominal scene; no dataset diversity wrapper",
        generator="scripts/testing/freeze_occlusion_ball_oracle.py",
        motion="source-authored location/rotation keyframes; no rigid-body dynamics",
        fps=int(scene.render.fps),
        width=width,
        height=height,
        camera=dict(
            position=list(origin),
            matrix_world=np.asarray(matrix).tolist(),
            tan_half_fov_x=xmax,
            tan_half_fov_y=ymax,
            lens_mm=camera.data.lens,
            sensor_width_mm=camera.data.sensor_width,
        ),
        depth_convention="positive camera-axis distance in source scene units; 0 is no hit",
        mask_convention="visible triangle surface at pixel center; 0 is no hit",
        object_ids=labels,
        geometry=geometry,
        versions=dict(blender=bpy.app.version_string, numpy=np.__version__, python=platform.python_version()),
        file=archive.name,
        sha256=sha256(archive),
        arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()},
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((source_root / "LICENSE").read_text())
    print(f"Frozen {archive}: {manifest['sha256']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.source_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
