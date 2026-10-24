"""Probe source pixel visibility with exact arithmetic, without candidate geometry.

Blender supplies evaluated world triangles, its BVH result and camera matrices.
Fraction arithmetic checks the represented triangles/rays exactly, preserving all
nearest objects when coplanar surfaces tie. Reports are diagnostics, not replacement
oracles and never change the normal depth/mask gates.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def _vector(values):
    return tuple(v if isinstance(v, Fraction) else Fraction.from_float(float(v)) for v in values)


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def _cross(a, b):
    return a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]


def _box(origin, direction, lower, upper):
    low, high = Fraction(0), None
    for o, d, lo, hi in zip(origin, direction, _vector(lower), _vector(upper), strict=True):
        if not d:
            if not lo <= o <= hi:
                return False
        else:
            first, last = sorted(((lo - o) / d, (hi - o) / d))
            low = max(low, first)
            high = last if high is None else min(high, last)
            if low > high:
                return False
    return True


def exact_triangle(origin, direction, triangle):
    """Return exact positive (t,u,v), or None; closed triangle edges, no epsilon."""
    origin, direction = _vector(origin), _vector(direction)
    a, b, c = (_vector(v) for v in triangle)
    e1, e2 = _sub(b, a), _sub(c, a)
    p = _cross(direction, e2)
    determinant = _dot(e1, p)
    if not determinant:
        return None
    relative = _sub(origin, a)
    u = _dot(relative, p) / determinant
    q = _cross(relative, e1)
    v, distance = _dot(direction, q) / determinant, _dot(e2, q) / determinant
    if 0 <= u <= 1 and 0 <= v and u + v <= 1 and distance > 0:
        return distance, u, v
    return None


def exact_nearest(origin, direction, meshes, *, max_distance=None):
    """Return all exactly nearest mesh names, using exact per-object AABB culling.

    ``meshes`` maps names to finite float64 [F,3,3] world triangles. Equality means
    exact equality for these represented numbers; arbitrarily close surfaces are
    never merged into a tie. This diagnoses numerical representations, not the
    unknowable exact real geometry before source/export floating-point rounding.
    """
    origin, direction = np.asarray(origin, np.float64), np.asarray(direction, np.float64)
    if (
        origin.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite([origin, direction]).all()
        or not direction.any()
    ):
        raise ValueError("ray requires finite origin and nonzero direction of shape (3,)")
    if max_distance is not None and (not np.isfinite(max_distance) or max_distance <= 0):
        raise ValueError("max_distance must be finite and positive")
    o, d = _vector(origin), _vector(direction)
    limit2 = None if max_distance is None else Fraction.from_float(float(max_distance)) ** 2
    norm2 = _dot(d, d)
    nearest: Fraction | None = None
    hits: list[dict] = []
    for name, triangles in meshes.items():
        triangles = np.asarray(triangles, np.float64)
        if (
            triangles.ndim != 3
            or triangles.shape[1:] != (3, 3)
            or not len(triangles)
            or not np.isfinite(triangles).all()
        ):
            raise ValueError("meshes require finite nonempty triangles [F,3,3]")
        if not _box(o, d, triangles.min(axis=(0, 1)), triangles.max(axis=(0, 1))):
            continue
        for index, triangle in enumerate(triangles):
            hit = exact_triangle(o, d, triangle)
            if hit is None:
                continue
            distance, u, v = hit
            if limit2 is not None and distance * distance * norm2 > limit2:
                continue
            if nearest is None or distance < nearest:
                nearest, hits = distance, []
            if distance == nearest:
                hits.append({"object": name, "triangle": index, "edge_margin": float(min(u, v, 1 - u - v))})
    return {
        "objects": sorted({h["object"] for h in hits}),
        "ray_parameter": None if nearest is None else float(nearest),
        "distance": None if nearest is None else float(nearest) * float(np.linalg.norm(direction)),
        "hits": hits,
    }


def canonical_object(nearest, object_ids):
    """Diagnostic lowest-semantic-ID policy for a future versioned contract.

    Only resolve an already proven exact tie set. This never combines close
    surfaces, modifies a frozen mask or supplies a legacy certification result.
    """
    values = list(object_ids.values())
    if any(type(i) is not int or i <= 0 for i in values) or len(set(values)) != len(values):
        raise ValueError("object IDs must be distinct positive integers")
    if not set(nearest["objects"]) <= set(object_ids):
        raise ValueError("nearest object missing from declared IDs")
    return min(nearest["objects"], key=object_ids.__getitem__) if nearest["objects"] else None


def probe(root, task, pixels, *, width=320, height=180):
    import bpy
    from mathutils import Vector

    from scripts.export.object_permanence import SOURCE_COMMIT, prepare, sha256

    scene, script, _ = prepare(root, task)
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    reports = []
    for frame, x, y in pixels:
        if not scene.frame_start <= frame <= scene.frame_end or not (0 <= x < width and 0 <= y < height):
            raise ValueError("query outside source frame/image bounds")
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        camera = scene.camera.evaluated_get(graph)
        if camera.data.type != "PERSP" or camera.data.shift_x != 0 or camera.data.shift_y != 0:
            raise ValueError("the existing oracle ray convention requires an unshifted perspective camera")
        world = np.asarray(camera.matrix_world, np.float64)
        corners = camera.data.view_frame(scene=scene)
        xmax, ymax = max(p.x / -p.z for p in corners), max(p.y / -p.z for p in corners)
        local = np.array([(2 * (x + 0.5) / width - 1) * xmax, (1 - 2 * (y + 0.5) / height) * ymax, -1.0])
        origin = camera.matrix_world.translation.copy()
        ray32 = (camera.matrix_world.to_3x3() @ Vector(local)).normalized()
        ray64 = world[:3, :3] @ local
        ray64 /= np.linalg.norm(ray64)
        projection = np.asarray(
            camera.calc_matrix_camera(
                graph, x=width, y=height, scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y
            ),
            np.float64,
        )
        camera_point = np.linalg.inv(projection) @ [2 * (x + 0.5) / width - 1, 1 - 2 * (y + 0.5) / height, -1, 1]
        ray_projection = world[:3, :3] @ camera_point[:3]
        ray_projection /= np.linalg.norm(ray_projection)
        hit, location, _, _, obj, _ = scene.ray_cast(graph, origin, ray32, distance=camera.data.clip_end)
        meshes = {}
        for source in scene.objects:
            if source.type != "MESH":
                continue
            evaluated = source.evaluated_get(graph)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                vertices = np.asarray([v.co[:] for v in mesh.vertices], np.float64)
                matrix = np.asarray(evaluated.matrix_world, np.float64)
                vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
                meshes[source.name] = vertices[np.asarray([t.vertices[:] for t in mesh.loop_triangles])]
            finally:
                evaluated.to_mesh_clear()
        exact = {
            name: exact_nearest(origin, ray, meshes, max_distance=camera.data.clip_end)
            for name, ray in (
                ("source_ray32", ray32),
                ("view_frame_ray64", ray64),
                ("projection_ray64", ray_projection),
            )
        }
        oracle_projection = projection.copy()
        oracle_projection[0, 0], oracle_projection[1, 1] = 1 / xmax, 1 / ymax
        oracle_projection[0, 2] = oracle_projection[1, 2] = 0
        labels = {name: i + 1 for i, name in enumerate(sorted(meshes))}
        for nearest in exact.values():
            nearest["canonical_object"] = canonical_object(nearest, labels)
        bvh = obj.name if hit else None
        reports.append(
            {
                "frame": frame,
                "xy": [x, y],
                "bvh_object": bvh,
                "bvh_camera_depth": float((location - origin).dot(camera.matrix_world.to_3x3() @ Vector((0, 0, -1))))
                if hit
                else 0.0,
                "camera_world": world.tolist(),
                "recorded_projection": projection.tolist(),
                "oracle_ray_projection": oracle_projection.tolist(),
                "origin": list(origin),
                "rays": {
                    "source_ray32": list(ray32),
                    "view_frame_ray64": ray64.tolist(),
                    "projection_ray64": ray_projection.tolist(),
                },
                "exact": exact,
                "bvh_agrees_with_exact_source_ray": bvh in exact["source_ray32"]["objects"]
                if hit
                else not exact["source_ray32"]["objects"],
            }
        )
    return {
        "source_commit": SOURCE_COMMIT,
        "source_script": str(script.relative_to(root)),
        "source_script_sha256": sha256(script),
        "task": task,
        "blender": bpy.app.version_string,
        "size": [height, width],
        "diagnostic_id_policy": "lowest-semantic-id-v1; alphabetical source names in this probe",
        "queries": reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--pixels", required=True, help="semicolon separated source-frame:x:y queries")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    pixels = [tuple(map(int, q.split(":"))) for q in args.pixels.split(";")]
    result = probe(args.source_root.resolve(), args.task, pixels)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
