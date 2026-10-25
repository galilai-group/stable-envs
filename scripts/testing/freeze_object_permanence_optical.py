"""Freeze independently evaluated source optical geometry; never load replay."""

import argparse
import ast
import json
import platform
import runpy
import subprocess
from pathlib import Path

import numpy as np

from .freeze_object_permanence import SOURCE_COMMIT, sha256, write_depth_parts
from .object_permanence_optical_reference import CONTRACT, MATERIAL, POLICY, draw


def prepare(script, task="G01"):
    """Independent audited text boundary; omit only legacy directory creation."""
    if task == "G03":
        source = runpy.run_path(str(script), run_name="independent_optical_reference")
        source["clear_scene"]()
        o = source["build_scene"]()
        source["animate_scene"](scene=o["scene"], wall=o["wall"], ball=o["ball"], ball_z=o["ball_z"])
        return
    boundary = "\ndef render_png("
    text = script.read_text()
    if text.count(boundary) != 1:
        raise ValueError("source preparation boundary changed")
    tree = ast.parse(text.split(boundary)[0])
    tree.body = [n for n in tree.body if ast.unparse(n) != "os.makedirs(OUT_DIR, exist_ok=True)"]
    exec(compile(tree, str(script), "exec"), {"__file__": str(script), "__name__": "independent_optical_reference"})


def material(obj):
    if len(obj.material_slots) != 1 or not obj.active_material.use_nodes:
        raise ValueError("source optical material must be one Principled surface")
    m = obj.active_material
    shaders = [n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"]
    outputs = [n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL"]
    if len(m.node_tree.nodes) != 2 or len(shaders) != 1 or len(outputs) != 1:
        raise ValueError("unsupported source optical graph")
    shader = shaders[0]
    out = outputs[0]
    if (
        len(out.inputs["Surface"].links) != 1
        or out.inputs["Surface"].links[0].from_node != shader
        or any(out.inputs[k].is_linked for k in ("Volume", "Displacement"))
    ):
        raise ValueError("unsupported source surface connection")
    if any(
        shader.inputs[k].is_linked or shader.inputs[k].default_value != 0
        for k in ("Metallic", "Coat Weight", "Sheen Weight")
    ):
        raise ValueError("unsupported source metallic or coated dielectric")
    sockets = {"alpha": "Alpha", "transmission": "Transmission Weight", "ior": "IOR", "roughness": "Roughness"}
    if any(shader.inputs[n].is_linked for n in (*sockets.values(), "Base Color")):
        raise ValueError("source optical parameters must be constant")
    return {k: float(shader.inputs[v].default_value) for k, v in sockets.items()}


def freeze(root, output, *, task="G01", member=0, width=320, height=180):
    import bpy

    if output.exists():
        raise FileExistsError("refusing to overwrite optical oracle")
    if (
        bpy.app.version[:2] != (4, 4)
        or type(member) is not int
        or (task, member) not in (("G01", 0), ("G01", 1), ("G03", 1))
    ):
        raise ValueError("requires audited optical member and Blender 4.4")
    if type(width) is not int or type(height) is not int or min(width, height) <= 0:
        raise ValueError("oracle dimensions must be positive integers")
    if subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip() != SOURCE_COMMIT or subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ):
        raise ValueError("requires clean pinned source")
    (taskpath,) = (root / "object_permanence/generator/tasks").glob(task + "_*/task.json")
    script = taskpath.parent / json.loads(taskpath.read_text())["member_scripts"][member]
    prepare(script, task)
    scene = bpy.context.scene
    frames = np.arange(scene.frame_start, scene.frame_end + 1, dtype=np.int64)
    names = sorted(o.name for o in scene.objects if o.type == "MESH")
    target = "target_orange_ball" if task == "G01" else "target_red_ball_blocked_by_transparent_wall"
    first = None
    segments = None
    rows = []
    poses = []
    for frame in frames:
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        objects = {o.name: o.evaluated_get(graph) for o in scene.objects if o.type == "MESH"}
        if sorted(objects) != names:
            raise ValueError("source object set changed")
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
            "projection_type": "perspective",
        }
        if cam.data.type != "PERSP" or cam.data.dof.use_dof:
            raise ValueError("unsupported source optical camera")
        materials = {n: material(objects[n]) for n in names}
        meshes = {}
        for name, obj in objects.items():
            mesh = obj.to_mesh()
            try:
                mesh.calc_loop_triangles()
                matrix = np.array(obj.matrix_world)
                vertices = np.array([v.co[:] for v in mesh.vertices], np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
                meshes[name] = vertices[np.array([t.vertices[:] for t in mesh.loop_triangles])]
            finally:
                obj.to_mesh_clear()
        if first is None:
            first = (camera, materials, {n: m.copy() for n, m in meshes.items() if n != target})
            _, segments = draw(first[2], materials, camera, height, width, save=True)
        elif (
            camera != first[0]
            or materials != first[1]
            or any(not np.array_equal(meshes[n], v) for n, v in first[2].items())
        ):
            raise ValueError("source static optics changed")
        row, _ = draw({target: meshes[target]}, materials, camera, height, width, segments=segments)
        rows.append(row)
        poses.append([[*objects[n].matrix_world.translation, *objects[n].matrix_world.to_quaternion()] for n in names])
        if frame % 10 == 0:
            print(task, member, "independent optical source frame", frame, flush=True)
    assert first is not None
    output.mkdir(parents=True)
    d = np.stack([r["depth"] for r in rows])
    path = np.stack([r["path_depth"] for r in rows])
    ids = np.stack([r["ids"] for r in rows])
    masks = np.stack([r["masks"] for r in rows])
    archive = output / "geometry.npz"
    arrays = {
        "source_frames": frames,
        "poses": np.array(poses, np.float64),
        "depth_base": d[0],
        "path_depth_base": path[0],
        "ids_base": ids[0],
        "masks_packed_base": np.packbits(masks[0], axis=-1),
    }
    np.savez_compressed(archive, **arrays)
    parts = {}
    for field, value in [
        ("depth", d),
        ("path_depth", path),
        ("ids", ids),
        ("masks_packed", np.packbits(masks, axis=-1)),
    ]:
        directory = output / field
        directory.mkdir()
        bits = value.view(np.uint32) if value.dtype == np.float32 else value.astype(np.uint32)
        parts[field] = write_depth_parts(directory, bits ^ bits[:1])
        for part in parts[field]:
            part["file"] = field + "/" + part["file"]
    metadata = {
        "schema": "stable-envs/object-permanence-optical-oracle-v1",
        "visibility_contract": CONTRACT,
        "material_convention": MATERIAL,
        "optical_policy": POLICY,
        "task": task,
        "member": member,
        "clip": task.lower() + ("-nominal" if member == 0 else f"-member{member}"),
        "source_commit": SOURCE_COMMIT,
        "source_script": str(script.relative_to(root)),
        "source_script_sha256": sha256(script),
        "source_task_manifest_sha256": sha256(taskpath),
        "source_preparation_recipe": {"build": "$prefix:render_png", "animate": ""}
        if task == "G01"
        else {"build": "build_scene", "animate": "animate_scene"},
        "source_license": "CC-BY-NC-4.0",
        "source_render_engine": scene.render.engine,
        "source_use_raytracing": bool(scene.eevee.use_raytracing),
        "versions": {"python": platform.python_version(), "blender": bpy.app.version_string, "numpy": np.__version__},
        "object_order": names,
        "targets": [target],
        "object_ids": {n: i + 1 for i, n in enumerate(names)},
        "materials": materials,
        "camera": first[0],
        "width": width,
        "height": height,
        "frames": len(frames),
        "file": archive.name,
        "sha256": sha256(archive),
        "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()},
        "parts": parts,
        "generator": "scripts/testing/freeze_object_permanence_optical.py",
        "generator_sha256": sha256(Path(__file__)),
        "reference_sha256": sha256(Path(__file__).with_name("object_permanence_optical_reference.py")),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--member", type=int, choices=(0, 1), default=0)
    p.add_argument("--task", choices=("G01", "G03"), default="G01")
    p.add_argument("--height", type=int, default=180)
    a = p.parse_args()
    freeze(a.source_root, a.output, task=a.task, member=a.member, height=a.height, width=a.height * 16 // 9)


if __name__ == "__main__":
    main()
