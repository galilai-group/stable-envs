"""Explicit, sampled surface optics; independent of Blender's Principled BSDF.

This experimental policy is deliberately separate from straight alpha layers.
Samples share the pixel centre; the counter stream samples material events, not
coverage. Alpha-null events, diffuse absorption, Fresnel reflection and Snell
transmission form a probability partition. No nonzero path is silently cut off.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np


class OpticalPolicy(TypedDict):
    samples: int
    seed: int
    max_events: int
    ray_start: float


CONTRACT = "stable-envs/sampled-optical-rays-v1"
MATERIAL = "stable-envs/constant-dielectric-surfaces-v1"
POLICY: OpticalPolicy = {"samples": 8, "seed": 0, "max_events": 64, "ray_start": 1e-9}


def validate_material(value):
    """Reject linked, varying or unsupported optics before publishing a clip."""
    if not isinstance(value, dict) or set(value) != {"transmission", "ior", "roughness"}:
        raise ValueError("optics require transmission, ior and roughness")
    t, n, r = (value[k] for k in ("transmission", "ior", "roughness"))
    if not all(type(x) in (int, float) and np.isfinite(x) for x in (t, n, r)):
        raise ValueError("optical parameters must be finite scalars")
    if not 0 <= t <= 1 or not 1 <= n <= 3 or not 0 <= r <= 1:
        raise ValueError("invalid dielectric material")


def hierarchy(triangles, *, leaf_size=4):
    """Build a stackless local-space BVH, shared by all frames of one mesh.

    A BVH (包围盒层次结构) rejects groups of triangles before exact tests.
    Preorder escape pointers skip entire missed subtrees without a ray stack.
    Face numbers remain original; tree order never changes semantic tie rules.
    """
    triangles = np.asarray(triangles, np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("triangles must be finite [F,3,3]")
    if not len(triangles) or type(leaf_size) is not int or leaf_size < 1:
        raise ValueError("empty mesh or invalid leaf size")
    bounds: list[np.ndarray] = []
    records: list[list[int]] = []
    ordered: list[int] = []
    centres = triangles.mean(1)

    def visit(faces):
        index = len(records)
        bounds.append(np.concatenate((triangles[faces].min((0, 1)), triangles[faces].max((0, 1)))))
        records.append([0, 0, 0])
        if len(faces) <= leaf_size:
            records[index][1:] = [len(ordered), len(faces)]
            ordered.extend(faces)
        else:
            axis = np.ptp(centres[faces], axis=0).argmax()
            faces = faces[np.argsort(centres[faces, axis], kind="stable")]
            middle = len(faces) // 2
            visit(faces[:middle])
            visit(faces[middle:])
        records[index][0] = len(records)

    visit(np.arange(len(triangles)))
    return np.array(bounds), np.array(records, np.int32), np.array(ordered, np.int32)


def closed_volume(triangles):
    """Check watertight, consistently oriented triangle surfaces geometrically.

    Exported meshes duplicate vertices for corner normals. Topology therefore
    uses exact position keys, not vertex indices. Zero-area faces are rejected.
    """
    edges: dict[tuple, list[bool]] = {}
    for triangle in np.asarray(triangles):
        if np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) == 0:
            raise ValueError("dielectric volume contains a degenerate face")
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0), strict=True):
            a, b = tuple(a), tuple(b)
            key = tuple(sorted((a, b)))
            edges.setdefault(key, []).append(a < b)
    if any(len(v) != 2 or v[0] == v[1] for v in edges.values()):
        raise ValueError("dielectric meshes must be closed and consistently oriented")


def snell(direction, normal, incident, transmitted):
    """Return ideal reflected/refracted rays and unpolarized Fresnel reflectance.

    Unit normals face against incidence; indices are absolute refractive indices
    (折射率). Total internal reflection (全反射) returns reflectance one.
    This CPU helper specifies the equations and is useful without CUDA.
    """
    d, n = np.asarray(direction, np.float64), np.asarray(normal, np.float64)
    c = np.clip(-np.sum(d * n, axis=-1), 0, 1)
    eta = np.asarray(incident) / np.asarray(transmitted)
    k = 1 - eta * eta * (1 - c * c)
    ct = np.sqrt(np.maximum(k, 0))
    reflected = d + 2 * c[..., None] * n
    refracted = eta[..., None] * d + (eta * c - ct)[..., None] * n
    n1, n2 = np.asarray(incident), np.asarray(transmitted)
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = ((n1 * c - n2 * ct) / (n1 * c + n2 * ct)) ** 2
        rp = ((n1 * ct - n2 * c) / (n1 * ct + n2 * c)) ** 2
    f = np.where(n1 == n2, 0, np.where(k <= 0, 1, (rs + rp) / 2))
    return reflected, refracted, f
