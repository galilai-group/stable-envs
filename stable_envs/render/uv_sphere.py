"""Exact ray hits on a convex latitude/longitude sphere after analytic broad phase.

Each latitude/longitude quad is planar. Its supporting plane can be recovered
from angular cell indices, avoiding a search through every triangle for ordinary
camera rays. A complete convex-plane clip handles unresolved and interior rays.
"""

from __future__ import annotations

import math

import torch

from ._resources import BoundedCache
from .primitives import INF, dot3

try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError:
    triton = None

_PLANES = BoundedCache(16, name="uv_sphere.planes")


def supporting_planes(slices, stacks, device, dtype=torch.float64):
    """Cache unnormalized convex face normals by tessellation, dtype and device."""
    key = (slices, stacks, str(device), dtype)
    normals = _PLANES.get(key)
    if normals is None:
        longitude = (torch.arange(slices, dtype=dtype) + 0.5) * (2 * math.pi / slices)
        latitude = (torch.arange(stacks, dtype=dtype) + 0.5) * (math.pi / stacks)
        theta, phi = torch.meshgrid(longitude, latitude, indexing="ij")
        normals = (
            torch.stack((theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos() * math.cos(math.pi / slices)), -1)
            .reshape(-1, 3)
            .to(device)
        )
        _PLANES.put(key, normals)
    return normals


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
    normals = supporting_planes(slices, stacks, origin.device, origin.dtype)
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
    if origin.is_cuda and origin.dtype == torch.float64 and triton is not None:
        result, normal = torch.empty_like(parameter), torch.empty_like(origin)
        with torch.cuda.device(origin.device):
            _refine_kernel[(triton.cdiv(len(parameter), 128),)](
                origin.contiguous(),
                direction.contiguous(),
                parameter.contiguous(),
                supporting_planes(slices, stacks, origin.device),
                result,
                normal,
                len(parameter),
                radius,
                slices,
                stacks,
                radius * math.cos(math.pi / slices) * math.cos(math.pi / (2 * stacks)),
                math.cos(math.pi / slices),
                max_iterations,
                128,
                enable_fp_fusion=False,
            )
        return result, normal
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


if triton is not None:

    @triton.jit
    def _refine_cuda(
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        t,
        hit,
        planes,
        RADIUS: tl.constexpr,
        SLICES: tl.constexpr,
        STACKS: tl.constexpr,
        HEIGHT: tl.constexpr,
        LONG_COS: tl.constexpr,
        ITERATIONS: tl.constexpr = 8,
        INF: tl.constexpr = INF,
    ):
        """Double-precision counterpart of the angular iteration and complete clip.

        Called inside a fused renderer; no compaction or host readback is needed.
        ``planes`` contains the same unnormalized supporting normals as _clip.
        """
        outside = ox * ox + oy * oy + oz * oz > RADIUS * RADIUS
        pending = hit & outside
        nx, ny, nz = tl.full(t.shape, 0, tl.float64), tl.full(t.shape, 0, tl.float64), tl.full(t.shape, 0, tl.float64)
        iteration = 0
        while (iteration < ITERATIONS) & (tl.sum(pending.to(tl.int32), 0) > 0):
            px, py, pz = ox + t * dx, oy + t * dy, oz + t * dz
            longitude = tl.floor(libdevice.atan2(py, px) / (2 * math.pi / SLICES))
            middle = (longitude + 0.5) * (2 * math.pi / SLICES)
            c, s = libdevice.cos(middle), libdevice.sin(middle)
            rho = (px * c + py * s) / LONG_COS
            latitude = tl.minimum(tl.maximum(tl.floor(libdevice.atan2(rho, pz) / (math.pi / STACKS)), 0), STACKS - 1)
            middle = (latitude + 0.5) * (math.pi / STACKS)
            sx, sy, sz = c * libdevice.sin(middle), s * libdevice.sin(middle), libdevice.cos(middle) * LONG_COS
            denominator = sx * dx + sy * dy + sz * dz
            next_t = (HEIGHT - (sx * ox + sy * oy + sz * oz)) / tl.where(tl.abs(denominator) > 1e-15, denominator, 1.0)
            valid = (denominator < 0) & (next_t > 0)
            converged = tl.abs(next_t - t) <= 1e-10
            nx, ny, nz = tl.where(pending, sx, nx), tl.where(pending, sy, ny), tl.where(pending, sz, nz)
            t = tl.where(pending, tl.where(valid, tl.maximum(t, next_t), INF), t)
            pending = pending & valid & ~converged
            iteration += 1
        fallback = hit & (~outside | pending)
        if tl.sum(fallback.to(tl.int32), 0):
            entering, leaving = tl.full(t.shape, -INF, tl.float64), tl.full(t.shape, INF, tl.float64)
            ex, ey, ez, lx, ly, lz = nx, ny, nz, nx, ny, nz
            valid = fallback
            for face in range(SLICES * STACKS):
                sx, sy, sz = tl.load(planes + face * 3), tl.load(planes + face * 3 + 1), tl.load(planes + face * 3 + 2)
                denominator = sx * dx + sy * dy + sz * dz
                numerator = HEIGHT - (sx * ox + sy * oy + sz * oz)
                parameter = numerator / tl.where(tl.abs(denominator) < 1e-15, 1.0, denominator)
                enter = (denominator < -1e-15) & (parameter > entering)
                leave = (denominator > 1e-15) & (parameter < leaving)
                entering, leaving = tl.where(enter, parameter, entering), tl.where(leave, parameter, leaving)
                ex, ey, ez = tl.where(enter, sx, ex), tl.where(enter, sy, ey), tl.where(enter, sz, ez)
                lx, ly, lz = tl.where(leave, sx, lx), tl.where(leave, sy, ly), tl.where(leave, sz, lz)
                valid = valid & ~((tl.abs(denominator) < 1e-15) & (numerator < 0))
            valid = valid & (leaving >= tl.maximum(entering, 0))
            entry = entering > 0
            t = tl.where(fallback, tl.where(valid, tl.where(entry, entering, leaving), INF), t)
            nx = tl.where(fallback, tl.where(entry, ex, lx), nx)
            ny = tl.where(fallback, tl.where(entry, ey, ly), ny)
            nz = tl.where(fallback, tl.where(entry, ez, lz), nz)
        norm = libdevice.sqrt(nx * nx + ny * ny + nz * nz)
        norm = tl.maximum(norm, 1e-15)
        return t, nx / norm, ny / norm, nz / norm

    @triton.jit
    def _refine_kernel(
        origin,
        direction,
        parameter,
        planes,
        output,
        normal,
        COUNT: tl.constexpr,
        RADIUS: tl.constexpr,
        SLICES: tl.constexpr,
        STACKS: tl.constexpr,
        HEIGHT: tl.constexpr,
        LONG_COS: tl.constexpr,
        ITERATIONS: tl.constexpr,
        BLOCK: tl.constexpr,
        INF: tl.constexpr = INF,
    ):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = i < COUNT
        ox, oy, oz = (
            tl.load(origin + i * 3, valid, 0),
            tl.load(origin + i * 3 + 1, valid, 0),
            tl.load(origin + i * 3 + 2, valid, 0),
        )
        dx, dy, dz = (
            tl.load(direction + i * 3, valid, 0),
            tl.load(direction + i * 3 + 1, valid, 0),
            tl.load(direction + i * 3 + 2, valid, 0),
        )
        t = tl.load(parameter + i, valid, INF)
        t, nx, ny, nz = _refine_cuda(
            ox, oy, oz, dx, dy, dz, t, valid, planes, RADIUS, SLICES, STACKS, HEIGHT, LONG_COS, ITERATIONS
        )
        tl.store(output + i, t, valid)
        tl.store(normal + i * 3, nx, valid)
        tl.store(normal + i * 3 + 1, ny, valid)
        tl.store(normal + i * 3 + 2, nz, valid)
