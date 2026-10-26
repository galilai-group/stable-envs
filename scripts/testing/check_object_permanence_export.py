"""Compare exported world triangles with independently evaluated source geometry.

This is a live serialization check, not a new frozen depth/mask oracle. Only
source preparation is shared with the exporter: pose/mesh extraction and rigid
reconstruction below do not call its sample_scene(), _pose() or _mesh().
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.export.object_permanence import AUDITED_TASKS, prepare
from scripts.object_permanence_member import Member


def _canonical(faces, normals):
    starts = faces.argmin(axis=1)
    order = (starts[:, None] + np.arange(3)) % 3
    faces = np.take_along_axis(faces, order, axis=1)
    normals = np.take_along_axis(normals, order[..., None], axis=1)
    rows = np.lexsort(faces.T[::-1])
    return faces[rows], normals[rows]


def check(root, dataset, task, *, member=0, profile="nominal"):
    import bpy
    from mathutils import Quaternion
    from mathutils.kdtree import KDTree

    entry = Member(task, member, profile)
    scene, script, _ = prepare(root, task, member=member, profile=profile)
    directory = dataset / "clips" / entry.clip
    manifest = json.loads((directory / "manifest.json").read_text())
    entry.check_source(manifest["source"])
    if manifest["source"]["script_sha256"] != hashlib.sha256(script.read_bytes()).hexdigest():
        raise ValueError("source script differs from recording")
    poses = np.load(directory / "poses.npy", allow_pickle=False)
    frames = np.load(directory / "frames.npy", allow_pickle=False)
    scales = np.load(directory / "scales.npy", allow_pickle=False) if "scales" in manifest["arrays"] else None
    exact = np.load(directory / "transforms.npy", allow_pickle=False) if "transforms" in manifest["arrays"] else None
    assets = {}
    for key in manifest["assets"]:
        with np.load(
            dataset / manifest["assets"][key].get("file", f"assets/{key}.npz"), allow_pickle=False
        ) as archive:
            assets[key] = dict(archive)
    worst_position, worst_normal = 0.0, 0.0
    for row, frame in enumerate(frames):
        scene.frame_set(int(frame))
        bpy.context.view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        converted = manifest.get("geometry_conversion") is not None
        types = ("MESH", "CURVE", "FONT") if converted else ("MESH",)
        source_objects = {o.name: o for o in scene.objects if o.type in types and not o.hide_render}
        assert set(source_objects) == {o["name"] for o in manifest["objects"]}
        camera = scene.camera.evaluated_get(depsgraph)
        if manifest["camera"].get("projection_type") == "orthographic":
            assert camera.data.type == "ORTHO"
        np.testing.assert_array_equal(manifest["camera"]["world"], np.asarray(camera.matrix_world))
        projection = camera.calc_matrix_camera(
            depsgraph,
            x=manifest["camera"]["width"],
            y=manifest["camera"]["height"],
            scale_x=scene.render.pixel_aspect_x,
            scale_y=scene.render.pixel_aspect_y,
        )
        np.testing.assert_array_equal(manifest["camera"]["projection"], np.asarray(projection))
        for record in manifest["objects"]:
            evaluated = source_objects[record["name"]].evaluated_get(depsgraph)
            if converted and evaluated.original.type in ("CURVE", "FONT"):
                description = {"type": evaluated.original.type}
                if evaluated.original.type == "FONT":
                    description.update(text=evaluated.data.body, font=evaluated.data.font.name)
                assert record.get("source_geometry") == description
            mesh = (
                bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=depsgraph)
                if converted
                else evaluated.to_mesh()
            )
            try:
                mesh.calc_loop_triangles()
                local = np.asarray([v.co[:] for v in mesh.vertices])
                faces = np.asarray([t.vertices[:] for t in mesh.loop_triangles])
                matrix = np.asarray(evaluated.matrix_world, np.float64)
                source_vertices = local @ matrix[:3, :3].T + matrix[:3, 3]
                unique, inverse = np.unique(source_vertices, axis=0, return_inverse=True)
                tree = KDTree(len(unique))
                for index, vertex in enumerate(unique):
                    tree.insert(vertex, index)
                tree.balance()
                p = record["pose"] if record["node"] == -1 else poses[row, record["node"]]
                rotation = np.asarray(Quaternion(p[3:]).normalized().to_matrix(), np.float64)
                scale = np.ones(3) if scales is None or record["node"] == -1 else scales[row, record["node"]]
                asset = assets[record["asset"]]
                actual = (asset["vertices"] * scale) @ rotation.T + p[:3]
                if exact is not None:
                    a = np.asarray(record["transform"]) if record["node"] < 0 else exact[row, record["node"]]
                    actual = asset["vertices"] @ a[:3, :3].T + a[:3, 3]
                matches = np.asarray([tree.find(vertex)[1] for vertex in actual])
                expected = unique[matches]
                np.testing.assert_allclose(
                    actual, expected, atol=2e-6, rtol=0, err_msg=f"{task}/{frame}/{record['name']}"
                )
                worst_position = max(worst_position, float(np.max(np.abs(actual - expected))))
                loops = np.asarray([t.loops[:] for t in mesh.loop_triangles])
                normals = np.asarray([n.vector[:] for n in mesh.corner_normals])
                normal_matrix = np.linalg.inv(matrix[:3, :3]).T
                expected_n = normals[loops] @ normal_matrix.T
                expected_n /= np.linalg.norm(expected_n, axis=-1, keepdims=True)
                actual_n = (asset["normals"] / scale) @ rotation.T
                if exact is not None:
                    actual_n = asset["normals"] @ np.linalg.inv(a[:3, :3])
                actual_n /= np.linalg.norm(actual_n, axis=-1, keepdims=True)
                actual_n = actual_n[asset["triangles"]]
                source_faces, expected_n = _canonical(inverse[faces], expected_n)
                recorded_faces, actual_n = _canonical(matches[asset["triangles"]], actual_n)
                np.testing.assert_array_equal(source_faces, recorded_faces)
                np.testing.assert_allclose(actual_n, expected_n, atol=3e-6, rtol=0)
                worst_normal = max(worst_normal, float(np.max(np.abs(actual_n - expected_n))))
            finally:
                if converted:
                    bpy.data.meshes.remove(mesh)
                else:
                    evaluated.to_mesh_clear()
    return {
        "task": task,
        "frames": len(frames),
        "objects": len(manifest["objects"]),
        "dynamic_nodes": manifest["dynamic_nodes"],
        "assets": len(assets),
        "world_triangle_max_abs_error": worst_position,
        "world_normal_max_abs_error": worst_normal,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", choices=(*AUDITED_TASKS, "G01"), required=True)
    parser.add_argument("--member", type=int, default=0)
    parser.add_argument("--profile", default="nominal")
    args = parser.parse_args()
    print(
        json.dumps(
            check(
                args.source_root.resolve(), args.dataset.resolve(), args.task, member=args.member, profile=args.profile
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
