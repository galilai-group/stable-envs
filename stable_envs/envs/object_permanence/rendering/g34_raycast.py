"""Batched G34 rendering over cached native RGB/depth/semantic background plates.

Analytic sphere hits provide broad-phase candidates; exact latitude/longitude
facets retain the source silhouette. Only visible ball hits are shaded. Native
OpenGL is used for the initial static bake, never for individual dynamic rows.
"""

from __future__ import annotations

from threading import RLock

import mujoco
import numpy as np
import torch

from ....render._resources import BoundedCache
from ....render.primitives import INF, SPHERE, Primitives, dot3, mtv3, mv3
from ....render.raycast3d import cached_rays, cast_nearest, screen_span, shade
from ....render.uv_sphere import refine_uv_sphere
from .g34_native import bake_static, copy_output

_PLATES = BoundedCache(8, name="object_permanence.g34.plates")
_PLATE_LOCK = RLock()


def _plate(height, width):
    with _PLATE_LOCK:
        plate = _PLATES.get((height, width))
        if plate is not None:
            return plate
        plate = bake_static(height, width)
        for key in ("rgb", "depth", "ids", "color", "light_position"):
            plate[key] = torch.from_numpy(plate[key])
        origin, direction = cached_rays(plate["camera"], height, width, "cpu", torch.float64)
        forward = torch.tensor(plate["camera"].basis()[2], dtype=torch.float64)
        cosine = dot3(direction, forward)
        depth = plate["depth"].flatten().double()
        plate.update(
            origin=origin,
            direction=direction,
            cosine=cosine,
            radial=torch.where(depth > 0, depth / cosine, INF),
            prims=Primitives(
                kind=np.array([SPHERE]),
                size=np.array([[plate["radius"], 0, 0]]),
                mat_rgb=np.ones((1, 3)),
                tex_rgb=np.ones((1, 3)),
                spec=np.zeros(1),
            ),
        )
        return _PLATES.put((height, width), plate)


def _rotation(quaternions):
    # MuJoCo normalizes its mocap quaternions; preserve that rule after the
    # float32 render projection. Rotation work is tiny and independent of pixels.
    matrices = np.empty((len(quaternions), 9), np.float64)
    for i, quaternion in enumerate(quaternions.numpy()):
        mujoco.mju_quat2Mat(matrices[i], quaternion / np.linalg.norm(quaternion))
    return torch.from_numpy(matrices.reshape(-1, 3, 3))


def _window(plate, position, height, width):
    span = screen_span(plate["camera"], plate["prims"], position[:, None], height, width, "cpu")
    # A box crossing the camera plane has an unbounded projected footprint.
    forward = torch.tensor(plate["camera"].basis()[2], dtype=torch.float64)
    if span is None or bool((dot3(position - plate["origin"], forward) <= plate["radius"]).any()):
        return torch.arange(height * width).expand(len(position), -1)
    x, y, w, h = span
    w, h = min(w, width), min(h, height)
    x, y = x.clamp(0, width - w), y.clamp(0, height - h)
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    indices = (y[:, None] + yy.flatten()) * width + x[:, None] + xx.flatten()
    return indices


def _hits(plate, rows, height, width):
    position, rotation = rows[:, :3], _rotation(rows[:, 3:])
    indices = _window(plate, position, height, width)
    rays = plate["direction"][indices]
    origin = plate["origin"].expand_as(rays)
    size = torch.tensor([plate["radius"], 0, 0], dtype=torch.float64).expand(len(rows), 1, 3)
    t, _, _ = cast_nearest([SPHERE], size, position[:, None], rotation[:, None], origin, rays)
    b, p = (t < INF).nonzero(as_tuple=True)
    if not len(b):
        return b, p, t[b, p], rays[b, p]
    local_origin = mtv3(rotation[b], origin[b, p] - position[b])
    local_direction = mtv3(rotation[b], rays[b, p])
    distance, normal = refine_uv_sphere(
        local_origin,
        local_direction,
        t[b, p],
        radius=plate["radius"],
        slices=plate["slices"],
        stacks=plate["stacks"],
    )
    pixel = indices[b, p]
    # Compare radial distances first. The camera-axis conversion is only for
    # diagnostic depth output, never mixed with radial static occlusion depth.
    axis_depth = distance * plate["cosine"][pixel]
    visible = (distance < plate["radial"][pixel]) & (axis_depth >= plate["near"]) & (axis_depth <= plate["far"])
    return b[visible], pixel[visible], distance[visible], mv3(rotation[b[visible]], normal[visible])


def _colors(plate, point, normal, view):
    lamp = plate["light_position"] - point
    lamp /= torch.linalg.vector_norm(lamp, dim=-1, keepdim=True).clamp_min(1e-15)
    head = -torch.tensor(plate["camera"].basis()[2], dtype=torch.float64).expand_as(lamp)
    directions = torch.stack((head, lamp))
    ambient = torch.tensor([plate["head_ambient"], 0], dtype=torch.float64)
    diffuse = torch.tensor([plate["head_diffuse"], plate["light_diffuse"]], dtype=torch.float64)
    zero = torch.zeros(2, dtype=torch.float64)
    color = plate["color"].double().expand_as(normal)
    rgb = shade(normal, view, directions, ambient, diffuse, zero, 1.0, color, torch.ones_like(color), zero[0])
    return (rgb * 255).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def render_raycast(rows, *, height, width, mode="rgb", out=None):
    """Render CPU rows without mutating State, cached plates or other outputs."""
    plate = _plate(height, width)
    host = rows.detach().cpu().reshape(-1, 7).double()
    background = plate[{"rgb": "rgb", "depth": "depth", "ids": "ids"}[mode]]
    result = background.expand(len(host), *background.shape).clone()
    flat = result.reshape(len(host), height * width, *(() if mode != "rgb" else (3,)))
    # Bound even the full-frame fallback's temporary ray tensors.
    span = screen_span(plate["camera"], plate["prims"], host[:, None, :3], height, width, "cpu")
    forward = torch.tensor(plate["camera"].basis()[2], dtype=torch.float64)
    near_camera = bool((dot3(host[:, :3] - plate["origin"], forward) <= plate["radius"]).any())
    pixels = height * width if span is None or near_camera else min(span[2], width) * min(span[3], height)
    chunk_size = max(1, min(64, 65536 // max(pixels, 1)))
    for start in range(0, len(host), chunk_size):
        chunk = host[start : start + chunk_size]
        b, pixel, t, normal = _hits(plate, chunk, height, width)
        if not len(b):
            continue
        if mode == "ids":
            flat[start + b, pixel] = 5
        elif mode == "depth":
            flat[start + b, pixel] = (t * plate["cosine"][pixel]).float()
        else:
            ray = plate["direction"][pixel]
            point = plate["origin"] + t[:, None] * ray
            flat[start + b, pixel] = _colors(plate, point, normal, -ray)
    return copy_output(result, out)
