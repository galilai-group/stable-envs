"""Freeze audited rigid scenes directly from pinned Blender, without candidate imports.

The depth encoding XORs IEEE float32 bits against frame one, losslessly removing
repeated background data. It does not change depth precision or acceptance gates.
Existing output directories are never overwritten, including the G34 fixture.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import runpy
import subprocess
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "824baddcdad087810d35638bf8f273b884f74975"
# Deliberately independent of the candidate exporter's entry registry/extraction.
# These calls and targets were checked against each source main()/selected builder.
REFERENCE_CASES = {
    "G27": ("build_scene", "animate_scene", ("target_orange_ball",)),
    "G37": ("build_scene_by_kind", "animate_scene", ("orange_ball",)),
    "G38": ("build_scene_by_kind", "animate_scene", ("red_pendulum_bob",)),
    "G64": ("build_scene_by_kind", "animate_scene_by_kind", ("static_object_a_ball", "static_object_b_cylinder")),
    "G55": ("build_scene_by_kind", "animate_scene", ("red_block", "green_block")),
    "G63": ("build_scene_by_kind", "animate_scene", ("red_block", "green_block", "blue_block")),
    "G66": ("build_scene_by_kind", "animate_scene", ("hidden_ball",)),
    "G80": ("build_scene_by_kind", "animate_scene", ("red_cube_target", "blue_sphere_target")),
    "G81": ("build_scene_by_kind", "animate_scene", ("red_box", "blue_cylinder")),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_depth_parts(output, bits):
    """Store all lossless depth bits in <=30-frame, <=1000-KiB NPZ parts.

    Bound reference file sizes without reducing frame count or precision. A part
    with high temporal change splits in half; this is independent of task ID.
    """
    parts: list[dict] = []

    def publish(start, stop):
        stream = io.BytesIO()
        np.savez_compressed(stream, depth_xor=bits[start:stop])
        content = stream.getvalue()
        if len(content) > 1000 * 1024:
            if stop - start == 1:
                raise ValueError("a single reference depth frame exceeds repository file limit")
            middle = (start + stop) // 2
            publish(start, middle)
            publish(middle, stop)
            return
        path = output / f"depth_{start:04d}_{stop:04d}.npz"
        path.write_bytes(content)
        parts.append(
            {
                "file": path.name,
                "sha256": sha256(path),
                "start": start,
                "stop": stop,
                "shape": list(bits[start:stop].shape),
                "dtype": str(bits.dtype),
            }
        )

    for start in range(0, len(bits), 30):
        publish(start, min(start + 30, len(bits)))
    return parts


def freeze(root, output, task, width=320, height=180):
    import bpy
    from mathutils import Vector

    if bpy.app.version[:2] != (4, 4):
        raise ValueError("rigid scene oracle requires Blender 4.4.x")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ):
        raise ValueError("reference source must be the clean pinned commit")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle: {output}")
    if task not in REFERENCE_CASES:
        raise ValueError("unaudited independent reference entry")
    matches = list((root / "object_permanence/generator/tasks").glob(task + "_*/task.json"))
    if len(matches) != 1:
        raise ValueError("source task must be unique")
    task_manifest = matches[0]
    metadata = json.loads(task_manifest.read_text())
    if metadata["id"] != task:
        raise ValueError("source task identity differs")
    script = task_manifest.parent / metadata["member_scripts"][0]
    source_script = str(script.relative_to(root))
    build, animate, targets = REFERENCE_CASES[task]
    namespace = runpy.run_path(str(script), run_name="rigid_independent_reference")
    namespace["clear_scene"]()
    built = namespace[build]()
    if task == "G27":
        namespace[animate](bpy.context.scene, built["support_plate"], built["ball"])
    else:
        namespace[animate](*(built if isinstance(built, tuple) else (built,)))
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.frame_set(scene.frame_start)
    bpy.context.view_layer.update()
    camera = scene.camera
    world = camera.matrix_world.copy()
    corners = camera.data.view_frame(scene=scene)
    xmax, ymax = max(p.x / -p.z for p in corners), max(p.y / -p.z for p in corners)
    origin, rotation = world.translation.copy(), world.to_3x3()
    forward = rotation @ Vector((0, 0, -1))
    directions = [
        (rotation @ Vector(((2 * (x + 0.5) / width - 1) * xmax, (1 - 2 * (y + 0.5) / height) * ymax, -1))).normalized()
        for y in range(height)
        for x in range(width)
    ]
    names = sorted(o.name for o in scene.objects if o.type == "MESH")
    labels = {name: index + 1 for index, name in enumerate(names)}
    if not set(targets) <= set(names):
        raise ValueError("audited reference targets missing")
    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=np.int64)
    depths = np.zeros((len(frames), height, width), np.float32)
    ids = np.zeros((len(frames), height, width), np.int16)
    poses = np.empty((len(frames), len(names), 7), np.float64)
    for row, frame in enumerate(frames):
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        if not np.array_equal(world, scene.camera.matrix_world):
            raise ValueError("reference camera changed")
        for index, name in enumerate(names):
            matrix = bpy.data.objects[name].evaluated_get(depsgraph).matrix_world
            poses[row, index] = [*matrix.translation, *matrix.to_quaternion()]
        for pixel, direction in enumerate(directions):
            hit, location, _, _, obj, _ = scene.ray_cast(depsgraph, origin, direction, distance=camera.data.clip_end)
            if hit:
                y, x = divmod(pixel, width)
                depths[row, y, x] = (location - origin).dot(forward)
                ids[row, y, x] = labels[obj.name]
        if row % 10 == 0 or row == len(frames) - 1:
            print(
                f"{task} source frame {frame}: target pixels "
                + str([int((ids[row] == labels[n]).sum()) for n in targets]),
                flush=True,
            )
    arrays = {
        "source_frames": frames,
        "poses": poses,
        "depth_base": depths[0].copy(),
        "depth_xor": depths.view(np.uint32) ^ depths[0].view(np.uint32),
        "object_ids": ids,
        "camera_matrix": np.asarray(world, np.float64),
    }
    reconstructed = (arrays["depth_xor"] ^ arrays["depth_base"].view(np.uint32)).view(np.float32)
    np.testing.assert_array_equal(reconstructed, depths)
    output.mkdir(parents=True)
    archive = output / f"blender_{task.lower()}_nominal.npz"
    # Chunk only lossless depth bits, never downsample frames or reduce precision.
    # The moving board changes many pixels; one monolithic NPZ exceeds repo limits.
    depth_bits = arrays.pop("depth_xor")
    np.savez_compressed(archive, **arrays)
    depth_parts = write_depth_parts(output, depth_bits)
    manifest = {
        "schema": "stable-envs/object-permanence-rigid-oracle-v1",
        "task": task,
        "source_url": "https://github.com/hokindeng/object-permanence",
        "source_commit": commit,
        "source_script": source_script,
        "source_task_manifest_sha256": sha256(task_manifest),
        "source_script_sha256": sha256(script),
        "source_profile": "nominal first member; no diversity wrapper",
        "source_license": "CC-BY-NC-4.0",
        "generator": "scripts/testing/freeze_object_permanence.py",
        "generator_sha256": sha256(Path(__file__)),
        "versions": {"blender": bpy.app.version_string, "numpy": np.__version__, "python": platform.python_version()},
        "file": archive.name,
        "sha256": sha256(archive),
        "width": width,
        "height": height,
        "fps": float(scene.render.fps / scene.render.fps_base),
        "object_ids": labels,
        "object_order": names,
        "targets": list(targets),
        "target_visible_pixels": {name: (ids == labels[name]).sum(axis=(1, 2)).tolist() for name in targets},
        "depth_encoding": "float32 bitwise XOR against depth_base; lossless chunks of at most 30 frames",
        "depth_parts": depth_parts,
        "depth_convention": "positive camera-axis distance in source scene units; zero for no hit",
        "mask_convention": "visible source mesh at the pixel center; zero for no hit",
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((root / "LICENSE").read_text())
    (output / "README.md").write_text(
        f"# Blender {task} nominal geometry oracle\n\n"
        "All 120 source frames at 320x180, generated independently from pinned Blender 4.4.\n"
        "No candidate exporter, recording or renderer is imported. Source, camera dimensions,\n"
        "IDs, units, array shapes, hashes and versions are in manifest.json.\n"
        "Depth XOR chunks preserve every float32 bit; shader RGB is not an oracle.\n\n"
        "Regenerate in a new directory with `python -m scripts.testing.freeze_object_permanence\n"
        f"--source-root <pinned-checkout> --output <new-directory> --task {task}` (one command).\n"
        "Existing fixtures are never overwritten. CC-BY-NC-4.0 source-derived data;\n"
        "see SOURCE_LICENSE.txt for full attribution and license terms.\n"
    )
    print(f"frozen {archive}: {archive.stat().st_size} bytes", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(REFERENCE_CASES), required=True)
    args = parser.parse_args()
    freeze(args.source_root.resolve(), args.output.resolve(), args.task)


if __name__ == "__main__":
    main()
