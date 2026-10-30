"""Independent projected-edge alpha events from source world triangles only.

No candidate recording, runtime visibility or Torch imports. All source events
use projected edge tests and NumPy sorting, independently of CUDA ray tests.
"""

import numpy as np

from .object_permanence_ray_reference import plane

CONTRACT = "stable-envs/transparent-pixel-rays-v1"
MATERIAL = "stable-envs/constant-alpha-surfaces-v1"


def draw(meshes, alphas, camera, height, width, *, max_layers=16):
    if type(max_layers) is not int or max_layers < 1 or set(meshes) != set(alphas):
        raise ValueError("source requires positive layer capacity and matching materials")
    names = sorted(meshes)
    labels = {name: i + 1 for i, name in enumerate(names)}
    alpha = np.array([0, *[alphas[n] for n in names]], np.float32)
    if not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1).any():
        raise ValueError("invalid source alpha")
    # Preserve binary64 opaque plane depths rather than rounding before sorting.
    depths = np.full((height, width, max_layers + 1), np.inf)
    winners = np.zeros(depths.shape, np.int16)
    p, world = np.asarray(camera["projection"]), np.asarray(camera["world"])
    view = np.linalg.inv(world)
    yy, xx = np.mgrid[:height, :width]
    orthographic = camera.get("projection_type") == "orthographic"
    local = np.stack(
        (
            (2 * (xx + 0.5) / width - 1 + (0 if orthographic else p[0, 2]) - (p[0, 3] if orthographic else 0))
            / p[0, 0],
            (1 - 2 * (yy + 0.5) / height + (0 if orthographic else p[1, 2]) - (p[1, 3] if orthographic else 0))
            / p[1, 1],
            np.zeros_like(xx) if orthographic else -np.ones_like(xx),
        ),
        -1,
    )
    rays = np.broadcast_to(-world[:3, 2], (height, width, 3)) if orthographic else local @ world[:3, :3].T
    origin = local @ world[:3, :3].T + world[:3, 3] if orthographic else world[:3, 3]
    # Insert all surfaces independently, with opaque termination applied after
    # sorting. Depths stay binary64 throughout. A shared-edge duplicate is one
    # (depth, object) event, not a second alpha application.
    for name in names:
        if alphas[name] == 0:
            continue
        label = labels[name]
        for tri in meshes[name]:
            homogeneous = np.column_stack((tri, np.ones(3))) @ view.T @ p.T
            if np.any(homogeneous[:, 3] <= 0) or (
                not orthographic and np.any(abs(homogeneous[:, 2]) >= homogeneous[:, 3])
            ):
                raise ValueError("source contract does not support depth-plane crossing")
            xy = (homogeneous[:, :2] / homogeneous[:, 3, None] + 1) * [width / 2, height / 2]
            xy[:, 1] = height - xy[:, 1]
            lo = np.maximum(0, np.floor(xy.min(0) - 1).astype(int))
            hi = np.minimum([width - 1, height - 1], np.ceil(xy.max(0) + 1).astype(int))
            if np.any(lo > hi):
                continue
            region = np.s_[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1]
            x, y = xx[region] + 0.5, yy[region] + 0.5
            a, b, c = xy
            det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if det == 0:
                continue
            u = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det
            v = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det
            n0, n1, n2, offset = plane(tri)
            r = rays[region]
            o = origin[region] if orthographic else origin
            numerator = -(n0 * o[..., 0] + n1 * o[..., 1] + n2 * o[..., 2] + offset)
            denominator = n0 * r[..., 0] + n1 * r[..., 1] + n2 * r[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                t = numerator / denominator
            hit = (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= camera["near"]) & (t <= camera["far"])
            old_d, old_i = depths[region], winners[region]
            hit &= ~((old_d == t[..., None]) & (old_i == label)).any(-1)
            if not hit.any():
                continue
            # Opaque faces only need nearest-winner replacement. Transparent
            # faces retain all ordered events until the nearest opaque layer.
            if alphas[name] == 1:
                opaque_mask = (old_i > 0) & (alpha[old_i] == 1)
                existing = np.where(opaque_mask, old_d, np.inf).min(-1)
                earlier = hit & (t < existing)
                equal = hit & (t == existing)
                existing_id = np.where(opaque_mask, old_i, 32767).min(-1)
                hit = earlier | (equal & (label < existing_id))
                if not hit.any():
                    continue
                remove = hit[..., None] & opaque_mask
                old_d[remove] = np.inf
                old_i[remove] = 0
            d = np.concatenate((old_d, np.where(hit, t, np.inf)[..., None]), -1)
            i = np.concatenate((old_i, np.where(hit, label, 0)[..., None]), -1)
            order = np.lexsort((i, d), axis=-1)[..., : max_layers + 1]
            old_d[:] = np.take_along_axis(d, order, -1)
            old_i[:] = np.take_along_axis(i, order, -1)
            stop = (old_i > 0) & (alpha[old_i] == 1)
            hidden = np.cumsum(stop, -1) - stop > 0
            old_d[hidden] = np.inf
            old_i[hidden] = 0
    if (winners[..., -1] > 0).any():
        raise ValueError("source alpha layer capacity exceeded")
    result = {
        "layer_depth": np.where(winners[..., :max_layers] > 0, depths[..., :max_layers], 0).astype(np.float32),
        "layer_ids": winners[..., :max_layers].copy(),
    }
    return result
