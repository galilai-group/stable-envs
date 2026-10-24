"""Independent NumPy source reference for opaque-pixel-rays-v1.

No stable_envs, Torch, candidate recording or runtime hit code is imported.
Coverage uses projected edge coordinates, independently of CUDA ray/triangle
intersection. Planes are rationally normalized before binary64 evaluation.
"""

from fractions import Fraction

import numpy as np

CONTRACT = "stable-envs/opaque-pixel-rays-v1"


def plane(triangle):
    points = [[Fraction(float(x)) for x in row] for row in triangle]
    a = points[0]
    edges = [[p[i] - a[i] for i in range(3)] for p in points[1:]]
    e, f = edges
    n = [e[1] * f[2] - e[2] * f[1], e[2] * f[0] - e[0] * f[2], e[0] * f[1] - e[1] * f[0]]
    axis = max(range(3), key=lambda i: abs(n[i]))
    if n[axis] == 0:
        return np.zeros(4)
    d = -sum(n[i] * a[i] for i in range(3))
    return np.array([float(c / n[axis]) for c in [*n, d]], np.float64)


def draw(meshes, camera, height, width):
    """Render independent source world triangles, mapping names to semantic IDs."""
    p, world = np.asarray(camera["projection"]), np.asarray(camera["world"])
    view = np.linalg.inv(world)
    yy, xx = np.mgrid[:height, :width]
    rays = (
        np.stack(
            (
                (2 * (xx + 0.5) / width - 1 + p[0, 2]) / p[0, 0],
                (1 - 2 * (yy + 0.5) / height + p[1, 2]) / p[1, 1],
                -np.ones_like(xx),
            ),
            -1,
        )
        @ world[:3, :3].T
    )
    origin = world[:3, 3]
    depth = np.full((height, width), np.inf)
    ids = np.zeros((height, width), np.int16)
    labels = {name: i + 1 for i, name in enumerate(sorted(meshes))}
    for name, triangles in meshes.items():
        for tri in triangles:
            homogeneous = np.column_stack((tri, np.ones(3))) @ view.T @ p.T
            if np.any(homogeneous[:, 3] <= 0) or np.any(abs(homogeneous[:, 2]) >= homogeneous[:, 3]):
                raise ValueError("source contract does not support depth-plane crossing")
            xy = homogeneous[:, :2] / homogeneous[:, 3, None]
            xy = (xy + 1) * [width / 2, height / 2]
            xy[:, 1] = height - xy[:, 1]
            lo = np.maximum(0, np.floor(xy.min(0) - 1).astype(int))
            hi = np.minimum([width - 1, height - 1], np.ceil(xy.max(0) + 1).astype(int))
            if np.any(lo > hi):
                continue
            region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            x, y = xx[region] + 0.5, yy[region] + 0.5
            a, b, c = xy
            determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if determinant == 0:
                continue
            u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / determinant
            v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / determinant
            coefficients = plane(tri)
            n, offset = coefficients[:3], coefficients[3]
            d = rays[region]
            denominator = n[0] * d[..., 0] + n[1] * d[..., 1] + n[2] * d[..., 2]
            numerator = -(n[0] * origin[0] + n[1] * origin[1] + n[2] * origin[2] + offset)
            with np.errstate(divide="ignore", invalid="ignore"):
                distance = numerator / denominator
            hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (distance >= camera["near"]) & (distance <= camera["far"])
            current, current_id = depth[region], ids[region]
            label = labels[name]
            update = hit & (
                (distance < current) | ((distance == current) & ((current_id == 0) | (label < current_id)))
            )
            current[update], current_id[update] = distance[update], label
    return np.where(ids > 0, depth, 0).astype(np.float32), ids
