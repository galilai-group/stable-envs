"""Exact ray hits on a convex latitude/longitude sphere after analytic broad phase.

Each latitude/longitude quad is planar. Its supporting plane can be recovered
from angular cell indices, avoiding a search through every triangle for ordinary
camera rays. A complete convex-plane clip handles unresolved and interior rays.
"""

from __future__ import annotations

import math

import torch

from .primitives import INF, dot3


def _cell_normal(point, slices, stacks):
    longitude = torch.floor(torch.atan2(point[:, 1], point[:, 0]) / (2 * math.pi / slices))
    middle = (longitude + 0.5) * (2 * math.pi / slices)
    c, s = middle.cos(), middle.sin()
    # Ring edges are chords, not circles: account for the longitude-cell angle
    # when selecting a latitude, especially near poles and silhouette pixels.
    rho = (point[:, 0] * c + point[:, 1] * s) / math.cos(math.pi / slices)
    latitude = torch.floor(torch.atan2(rho, point[:, 2]) / (math.pi / stacks)).clamp(0, stacks - 1)
    middle = (latitude + 0.5) * (math.pi / stacks)
    return torch.stack((c * middle.sin(), s * middle.sin(), middle.cos() * math.cos(math.pi / slices)), -1)


def _clip(origin, direction, radius, slices, stacks):
    """Intersect all supporting planes; also handles rays starting inside the mesh."""
    longitude = (torch.arange(slices, device=origin.device, dtype=origin.dtype) + 0.5) * (2 * math.pi / slices)
    latitude = (torch.arange(stacks, device=origin.device, dtype=origin.dtype) + 0.5) * (math.pi / stacks)
    theta, phi = torch.meshgrid(longitude, latitude, indexing="ij")
    normals = torch.stack(
        (theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos() * math.cos(math.pi / slices)), -1
    ).reshape(-1, 3)
    height = radius * math.cos(math.pi / slices) * math.cos(math.pi / (2 * stacks))
    distance, result_normal = [], []
    # Bound fallback scratch independently of the caller's ray count.
    for start in range(0, len(origin), 128):
        o, d = origin[start : start + 128], direction[start : start + 128]
        denominator = dot3(d[:, None], normals[None])
        numerator = height - dot3(o[:, None], normals[None])
        parallel = denominator.abs() < 1e-15
        parameter = numerator / torch.where(parallel, 1.0, denominator)
        entering, enter_face = torch.where(denominator < -1e-15, parameter, -INF).max(-1)
        leaving, leave_face = torch.where(denominator > 1e-15, parameter, INF).min(-1)
        hit = (leaving >= entering.clamp_min(0)) & ~(parallel & (numerator < 0)).any(-1)
        outside = entering > 0
        distance.append(torch.where(hit, torch.where(outside, entering, leaving), INF))
        result_normal.append(normals[torch.where(outside, enter_face, leave_face)])
    return torch.cat(distance), torch.cat(result_normal)


@torch.no_grad()
def refine_uv_sphere(origin, direction, parameter, *, radius, slices, stacks, max_iterations=8):
    """Refine positive analytic-sphere hits to exact faceted-sphere hits and normals.

    Inputs are compact local-space rays ``[N,3]`` and positive analytic entry/exit
    parameters ``[N]``; misses from the analytic broad phase must be excluded.
    Output misses use INF. Returned normals are local unit face normals. Angular
    subdivisions describe the mesh, not an image-resolution approximation.
    """
    if len(parameter) == 0:
        return parameter.clone(), origin.clone()
    t = parameter.clone()
    normal = torch.zeros_like(origin)
    pending = dot3(origin, origin) > radius * radius
    fallback = ~pending
    height = radius * math.cos(math.pi / slices) * math.cos(math.pi / (2 * stacks))
    for _ in range(max_iterations):
        indices = pending.nonzero(as_tuple=True)[0]
        if not len(indices):
            break
        o, d, old = origin[indices], direction[indices], t[indices]
        n = _cell_normal(o + old[:, None] * d, slices, stacks)
        denominator = dot3(n, d)
        next_t = (height - dot3(n, o)) / torch.where(denominator.abs() > 1e-15, denominator, 1.0)
        hit = (denominator < 0) & (next_t > 0)
        converged = (next_t - old).abs() <= 1e-10
        t[indices] = torch.where(hit, torch.maximum(old, next_t), INF)
        normal[indices] = n
        pending[indices] = hit & ~converged
    fallback |= pending
    indices = fallback.nonzero(as_tuple=True)[0]
    if len(indices):
        t[indices], normal[indices] = _clip(origin[indices], direction[indices], radius, slices, stacks)
    normal /= torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1e-15)
    return t, normal
