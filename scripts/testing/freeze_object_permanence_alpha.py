"""Freeze independently evaluated constant-alpha source layers and provenance."""

import argparse
import json
import platform
import runpy
import subprocess
from pathlib import Path

import numpy as np

from .freeze_object_permanence import SOURCE_COMMIT, sha256, write_depth_parts
from .object_permanence_alpha_reference import CONTRACT, MATERIAL, draw

CASES = {
    "G90": (
        "build_scene_by_kind",
        "animate_scene",
        ("large_red_ball_wider_than_gap", "small_orange_ball_narrower_than_gap"),
    ),
    "G36": ("build_scene_by_kind", "animate_scene_by_kind", ("orange_ball_from_u_track_inside_glass_box",)),
    "G54": ("build_scene_by_kind", "animate_scene_by_kind", ("serpentine_ball",)),
    "G61": ("build_scene_by_kind", "animate_scene_by_kind", ("spiral_ball",)),
}


def prepare(script, task):
    source = runpy.run_path(str(script), run_name="independent_alpha_source")
    source["clear_scene"]()
    build, animate, _ = CASES[task]
    objects = source[build]()
    source[animate](*(objects if isinstance(objects, tuple) else (objects,)))


def material(obj):
    if len(obj.material_slots) != 1:
        raise ValueError("source requires one material")
    m = obj.active_material
    if m.use_nodes:
        shader = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        outputs = [n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL"]
        if (
            len(m.node_tree.nodes) != 2
            or len(outputs) != 1
            or len(outputs[0].inputs["Surface"].links) != 1
            or outputs[0].inputs["Surface"].links[0].from_node != shader
            or outputs[0].inputs["Volume"].is_linked
            or outputs[0].inputs["Displacement"].is_linked
        ):
            raise ValueError("source requires a single surface shader without volume/displacement")
        if shader.inputs["Transmission Weight"].default_value != 0 or any(
            shader.inputs[k].is_linked for k in ("Alpha", "Base Color", "Transmission Weight")
        ):
            raise ValueError("source needs a different optical contract")
        return float(shader.inputs["Alpha"].default_value)
    return float(m.diffuse_color[3])


def freeze(root, output, task, *, width=320, height=180, max_layers=16):
    import bpy

    if (
        type(max_layers) is not int
        or max_layers < 1
        or type(width) is not int
        or type(height) is not int
        or min(width, height) < 1
    ):
        raise ValueError("source dimensions and layer capacity must be positive integers")
    if output.exists():
        raise FileExistsError("refusing to overwrite alpha oracle")
    if bpy.app.version[:2] != (4, 4):
        raise ValueError("source requires Blender 4.4")
    if subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip() != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ):
        raise ValueError("source requires clean pinned checkout")
    taskpath = next((root / "object_permanence/generator/tasks").glob(task + "_*/task.json"))
    taskmeta = json.loads(taskpath.read_text())
    script = taskpath.parent / taskmeta["member_scripts"][0]
    prepare(script, task)
    scene = bpy.context.scene
    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=np.int64)
    names = sorted(o.name for o in scene.objects if o.type == "MESH")
    depths, ids, poses = [], [], []
    first_camera = first_alpha = None
    for frame in frames:
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        objects = {o.name: o.evaluated_get(graph) for o in scene.objects if o.type == "MESH"}
        if sorted(objects) != names:
            raise ValueError("source membership changed")
        cam = scene.camera.evaluated_get(graph)
        camera = {
            "world": np.array(cam.matrix_world).tolist(),
            "projection": np.array(
                cam.calc_matrix_camera(
                    graph, x=width, y=height, scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y
                )
            ).tolist(),
            "near": float(cam.data.clip_start),
            "far": float(cam.data.clip_end),
            "projection_type": "orthographic" if cam.data.type == "ORTHO" else "perspective",
        }
        if cam.data.type not in ("PERSP", "ORTHO") or cam.data.dof.use_dof:
            raise ValueError("unsupported source camera")
        alphas = {n: material(objects[n]) for n in names}
        if first_camera is not None and (first_camera != camera or first_alpha != alphas):
            raise ValueError("source camera/material changed")
        first_camera, first_alpha = camera, alphas
        meshes = {}
        for name, obj in objects.items():
            mesh = obj.to_mesh()
            try:
                mesh.calc_loop_triangles()
                matrix = np.asarray(obj.matrix_world, np.float64)
                vertices = np.array([v.co[:] for v in mesh.vertices], np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
                meshes[name] = vertices[np.array([t.vertices[:] for t in mesh.loop_triangles])]
            finally:
                obj.to_mesh_clear()
        result = draw(meshes, alphas, camera, height, width, max_layers=max_layers)
        depths.append(result["layer_depth"])
        ids.append(result["layer_ids"])
        poses.append([[*objects[n].matrix_world.translation, *objects[n].matrix_world.to_quaternion()] for n in names])
        if int(frame) % 10 == 0:
            print(task, "independent alpha source frame", frame, flush=True)
    d = np.stack(depths)
    i = np.stack(ids)
    assert first_camera is not None and first_alpha is not None
    output.mkdir(parents=True)
    arrays = {
        "source_frames": frames,
        "poses": np.array(poses, np.float64),
        "layer_depth_base": d[0],
        "layer_ids_base": i[0],
    }
    path = output / "geometry.npz"
    np.savez_compressed(path, **arrays)
    parts = {}
    for field, rows in [("layer_depth", d.view(np.uint32)), ("layer_ids", i.astype(np.uint32))]:
        directory = output / field
        directory.mkdir()
        parts[field] = write_depth_parts(directory, rows ^ rows[:1])
        for part in parts[field]:
            part["file"] = field + "/" + part["file"]
    metadata = {
        "schema": "stable-envs/object-permanence-alpha-oracle-v1",
        "visibility_contract": CONTRACT,
        "material_convention": MATERIAL,
        "task": task,
        "source_commit": SOURCE_COMMIT,
        "source_script": str(script.relative_to(root)),
        "source_script_sha256": sha256(script),
        "source_task_manifest_sha256": sha256(taskpath),
        "source_preparation_recipe": dict(zip(("build", "animate"), CASES[task][:2])),
        "source_license": "CC-BY-NC-4.0",
        "versions": {"python": platform.python_version(), "blender": bpy.app.version_string, "numpy": np.__version__},
        "camera": first_camera,
        "object_order": names,
        "object_ids": {n: k + 1 for k, n in enumerate(names)},
        "alphas": [first_alpha[n] for n in names],
        "targets": list(CASES[task][2]),
        "max_layers": max_layers,
        "observed_layers_max": int((i > 0).sum(-1).max()),
        "width": width,
        "height": height,
        "fps": scene.render.fps / scene.render.fps_base,
        "file": path.name,
        "sha256": sha256(path),
        "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape)} for k, v in arrays.items()},
        "parts": parts,
        "generator_sha256": sha256(Path(__file__)),
        "reference_algorithm_sha256": sha256(Path(__file__).with_name("object_permanence_alpha_reference.py")),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    (output / "SOURCE_LICENSE.txt").write_text((root / "LICENSE").read_text())
    (output / "README.md").write_text(
        f"# {task} independent alpha-surface oracle\n\nAll 120 source frames; {CONTRACT}. Straight rays and constant surface alpha; no refraction.\nGenerated from source evaluated world triangles with independent projected-edge tests.\nDepth/ID layers retain objects visible through alpha surfaces; repeated triangles at one object/depth count once.\nCC-BY-NC-4.0; see SOURCE_LICENSE.txt.\n"
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=tuple(CASES), required=True)
    parser.add_argument("--max-layers", type=int, default=16)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    freeze(
        args.source_root.resolve(),
        args.output.resolve(),
        args.task,
        width=args.width,
        height=args.height,
        max_layers=args.max_layers,
    )


if __name__ == "__main__":
    main()
