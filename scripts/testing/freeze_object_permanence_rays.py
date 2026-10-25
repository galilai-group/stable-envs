"""Freeze versioned geometric ray references alongside untouched raw BVH fixtures."""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from .freeze_object_permanence import (
    REFERENCE_CASES,
    SOURCE_COMMIT,
    prepare_reference,
    reference_execution_context,
    sha256,
    write_depth_parts,
)
from .object_permanence_ray_reference import CONTRACT, draw


def freeze(root, output, task, width=320, height=180):
    import bpy

    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle: {output}")
    if bpy.app.version[:2] != (4, 4):
        raise ValueError("source reference requires Blender 4.4.x")
    if subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip() != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ):
        raise ValueError("source reference requires the clean pinned checkout")
    if task not in REFERENCE_CASES:
        raise ValueError("unaudited independent source entry")
    matches = list((root / "object_permanence/generator/tasks").glob(task + "_*/task.json"))
    if len(matches) != 1:
        raise ValueError("source task must be unique")
    task_path = matches[0]
    metadata = json.loads(task_path.read_text())
    script = task_path.parent / metadata["member_scripts"][0]
    build, animate, targets = REFERENCE_CASES[task]
    prepare_reference(script, task)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    names = sorted(o.name for o in scene.objects if o.type == "MESH")
    labels = {name: i + 1 for i, name in enumerate(names)}
    if not set(targets) <= set(names):
        raise ValueError("source target missing")
    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=np.int64)
    depth_rows, id_rows, poses = [], [], []
    first_camera = None
    for frame in frames:
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        cam = scene.camera.evaluated_get(graph)
        if cam.data.type != "PERSP" or cam.data.dof.use_dof:
            raise ValueError("source contract requires fixed perspective camera without DOF")
        camera = {
            "world": np.array(cam.matrix_world).tolist(),
            "projection": np.array(
                cam.calc_matrix_camera(
                    graph, x=width, y=height, scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y
                )
            ).tolist(),
            "near": float(cam.data.clip_start),
            "far": float(cam.data.clip_end),
        }
        if first_camera is not None and camera != first_camera:
            raise ValueError("source camera changed")
        first_camera = camera
        meshes = {}
        frame_poses = []
        for name in names:
            evaluated = scene.objects[name].evaluated_get(graph)
            matrix = np.asarray(evaluated.matrix_world, np.float64)
            frame_poses.append([*evaluated.matrix_world.translation, *evaluated.matrix_world.to_quaternion()])
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                vertices = np.array([v.co[:] for v in mesh.vertices], np.float64)
                vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
                meshes[name] = vertices[np.array([t.vertices[:] for t in mesh.loop_triangles])]
            finally:
                evaluated.to_mesh_clear()
        depth, labels_image = draw(meshes, camera, height, width)
        depth_rows.append(depth)
        id_rows.append(labels_image)
        poses.append(frame_poses)
        if int(frame) % 10 == 0:
            print(f"{task} canonical source frame {frame}", flush=True)
    depths, ids = np.array(depth_rows), np.array(id_rows)
    assert first_camera is not None
    arrays = {
        "source_frames": frames,
        "poses": np.array(poses, np.float64),
        "depth_base": depths[0].copy(),
        "object_ids": ids,
        "camera_matrix": np.array(first_camera["world"], np.float64),
    }
    output.mkdir(parents=True)
    archive = output / f"geometric_{task.lower()}_nominal.npz"
    np.savez_compressed(archive, **arrays)
    parts = write_depth_parts(output, depths.view(np.uint32) ^ depths[0].view(np.uint32))
    manifest = {
        "schema": "stable-envs/object-permanence-rigid-oracle-v2",
        "visibility_contract": CONTRACT,
        "camera": first_camera,
        "task": task,
        "source_commit": SOURCE_COMMIT,
        "source_url": "https://github.com/hokindeng/object-permanence",
        "source_script": str(script.relative_to(root)),
        "source_script_sha256": sha256(script),
        "source_task_manifest_sha256": sha256(task_path),
        "source_license": "CC-BY-NC-4.0",
        "source_profile": "nominal first member; no diversity wrapper",
        "generator": "scripts/testing/freeze_object_permanence_rays.py",
        "generator_sha256": sha256(Path(__file__)),
        "reference_algorithm_sha256": sha256(Path(__file__).with_name("object_permanence_ray_reference.py")),
        "versions": {"blender": bpy.app.version_string, "numpy": np.__version__, "python": platform.python_version()},
        "width": width,
        "height": height,
        "fps": float(scene.render.fps / scene.render.fps_base),
        "object_order": names,
        "object_ids": labels,
        "targets": list(targets),
        "target_visible_pixels": {name: (ids == labels[name]).sum((1, 2)).tolist() for name in targets},
        "file": archive.name,
        "sha256": sha256(archive),
        "depth_parts": parts,
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
    }
    context = reference_execution_context(script, task)
    if context is not None:
        manifest["source_profile"] = "nominal first member; surface diversity globals; no driver hooks"
        manifest["source_execution_context"] = context
        manifest["source_preparation_sha256"] = sha256(Path(__file__).with_name("freeze_object_permanence.py"))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((root / "LICENSE").read_text())
    (output / "README.md").write_text(
        f"# {task} independent geometric oracle v2\n\nContract: `{CONTRACT}`. All source frames, 320x180.\nGenerated from evaluated Blender source geometry without candidate imports or BVH queries.\nDepth and IDs share pixel-center rays; exactly equal depths choose lowest semantic ID.\nReproduce in a new directory with `python -m scripts.testing.freeze_object_permanence_rays --source-root <source> --output <new> --task {task}`.\nLegacy BVH references are retained separately; no thresholds are changed.\nCC-BY-NC-4.0 source-derived data; see SOURCE_LICENSE.txt.\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(REFERENCE_CASES), required=True)
    args = parser.parse_args()
    freeze(args.source_root.resolve(), args.output.resolve(), args.task)


if __name__ == "__main__":
    main()
