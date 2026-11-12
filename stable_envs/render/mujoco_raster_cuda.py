"""CUDA backend of ``render.mujoco_raster``: the same packed scene, per-frame inputs and shading as the CPU backend.

The scene is a ``render.mujoco_scene.Packed`` extraction; ``draw`` takes the same per-frame geom poses, colours,
cameras, light rows and sky colours as ``mujoco_raster.draw``. Each frame has the 4 samples per pixel of an
NVIDIA multisampled buffer, at NVIDIA's sample positions; a sample is shaded at the pixel centre when the centre
lies in the triangle, else at the sample. A Triton kernel lights every corner with the CPU backend's formula.

nvdiffrast draws the opaque triangles in one pass per sample position; a geom that is translucent in a frame is
left out of that frame's opaque passes. A static plane with a 2-D texture at the world origin (the Cube/Scene
floor) is drawn as background instead: its corners are lit with the CPU backend's ``shade_points`` and textured
with nvdiffrast's trilinear filter, once per camera, light rows, plane colour and sky colour, together with the
skybox. The shadow light adds its light where neither the static shadow map of the CPU backend nor the window of
the moving casters drawn by Triton holds a nearer caster. Triton kernels list the translucent fragments per
sample and blend them in MuJoCo's order. Geoms with a cube texture are drawn in its mean colour.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import torch

from . import mujoco_raster as core
from . import mujoco_scene as ms
from ._resources import BoundedCache

_LOCK = threading.RLock()
_CONTEXTS: dict = {}
#: Translucent fragments kept per sample.
LAYERS = 16
#: NVIDIA's 4x multisample positions in a pixel, x right and y up from its lower left corner.
SAMPLES = ((0.125, 0.625), (0.625, 0.875), (0.375, 0.125), (0.875, 0.375))
#: Backgrounds kept per scene.
BACKGROUNDS = 64
#: Largest moving-caster shadow window per chunk, in bytes.
WINDOW_BYTES = 2 << 30


def _context(device):
    import nvdiffrast.torch as dr

    key = str(device)
    with _LOCK:
        hit = _CONTEXTS.get(key)
        if hit is None:
            hit = _CONTEXTS[key] = dr.RasterizeCudaContext(device=torch.device(device))
        return hit


class DeviceScene:
    """Device arrays of one packed scene."""

    def __init__(self, p, device):
        self.p = p
        self.device = device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("the CUDA backend of the MuJoCo rasterizer draws on CUDA devices")
        G = len(p.objid)
        self.G = G
        t = lambda a, dt=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)  # noqa: E731
        corner_geom = np.repeat(np.arange(G), np.diff(p.cstarts))
        self.tri_geom = np.repeat(np.arange(G), np.diff(p.starts))
        self.tri_local = np.arange(len(p.tris)) - p.starts[self.tri_geom]
        self.vpos, self.vnor = t(p.cpos), t(p.cnor)
        self.vgeom = t(corner_geom, torch.int32)
        self.material = t(p.material)
        self.lut = t(p.powlut)
        texmod = np.ones((G, 3))
        self.plane = _plane(p)
        for g in range(G):
            if p.texmode[g] == ms.TEX_CUBE:
                texmod[g] = np.mean([face[0].reshape(-1, 3).mean(0) for face in p.textures[p.texslot[g]]], 0) / 255.0
            elif p.texmode[g] == ms.TEX_2D and (self.plane is None or g != self.plane["geom"]):
                raise NotImplementedError("the CUDA backend draws a 2-D texture on a background plane only")
        self.texmod = t(texmod)
        keep = np.ones(len(p.tris), bool) if self.plane is None else self.tri_geom != self.plane["geom"]
        drawn = p.tris[keep]
        used, compact = np.unique(drawn, return_inverse=True)
        self.tri_opaque = t(drawn, torch.int32)
        self.opaque_vertices = t(used, torch.int64)
        self.tri_opaque_compact = t(compact.reshape(-1, 3), torch.int32)
        self.opaque_geom = t(self.tri_geom[keep], torch.int32)
        self.keep = keep
        self.shadow = p.shadow_light >= 0
        if self.shadow:
            self.static_map = t(core.shadow_map(p))
            self.slook = t(p.slook.reshape(-1))
            self.smat = t(p.smat.reshape(-1))
            cpos, tris, cstarts, starts, rbound = p.cast
            rows = [np.arange(starts[g], starts[g + 1]) for g in p.casters]
            rows = np.concatenate(rows) if rows else np.zeros(0, np.int64)
            self.cast_pos = t(cpos)
            self.cast_tri = t(tris[rows], torch.int32)
            self.cast_geom = t(np.repeat(p.casters, [starts[g + 1] - starts[g] for g in p.casters]), torch.int32)
            self.cast_rbound = np.asarray(rbound, np.float64)
        self.sky = None
        if p.has_sky:
            self.sky = t(np.stack([face[0] for face in p.sky]).astype(np.float32) / 255.0)
        if self.plane is not None:
            level = p.textures[p.texslot[self.plane["geom"]]][0][0]
            self.plane["texture"] = t(level.astype(np.float32) / 255.0)
        self.backgrounds = BoundedCache(BACKGROUNDS, name="mujoco_raster_cuda.background")
        self.glass_sets: dict = {}

    def translucent(self, geoms):
        """Candidate translucent triangles of ``geoms``: triangles, their geoms and their order in the geom."""
        key = tuple(int(g) for g in geoms)
        hit = self.glass_sets.get(key)
        if hit is None:
            rows = np.flatnonzero(np.isin(self.tri_geom, geoms) & self.keep)
            t = lambda a, dt: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=self.device)  # noqa: E731
            hit = (
                t(self.p.tris[rows], torch.int32),
                t(self.tri_geom[rows], torch.int32),
                t(self.tri_local[rows], torch.int32),
            )
            if len(self.glass_sets) > 64:
                self.glass_sets.clear()
            self.glass_sets[key] = hit
        return hit


def _plane(p):
    """The background plane: a static plane geom with a 2-D texture at the world origin, or None."""
    planes = [g for g in range(len(p.objid)) if p.type[g] == ms.PLANE and p.texmode[g] == ms.TEX_2D]
    if not planes:
        return None
    if len(planes) > 1:
        raise NotImplementedError("the CUDA backend draws one textured plane")
    g = planes[0]
    if not p.static[g] or np.any(p.static_pos[g] != 0) or not np.allclose(p.static_mat[g], np.eye(3).reshape(-1)):
        raise NotImplementedError("the CUDA backend draws a textured plane at the world origin only")
    corners = p.cpos[p.cstarts[g] : p.cstarts[g + 1]]
    tex = p.ctex[p.cstarts[g] : p.cstarts[g + 1]]
    lines = [np.unique(corners[:, k]) for k in range(2)]
    for line in lines:
        step = np.diff(line)
        if len(step) == 0 or not np.allclose(step, step[0], rtol=0, atol=1e-9):
            raise NotImplementedError("the CUDA backend expects an even plane grid")
    # The plane's texture coordinates are affine in x and in y.
    fit = [np.polyfit(corners[:, k], tex[:, k], 1) for k in range(2)]
    for k in range(2):
        if np.abs(np.polyval(fit[k], corners[:, k]) - tex[:, k]).max() > 1e-6:
            raise NotImplementedError("the CUDA backend expects affine plane texture coordinates")
    return {
        "geom": g,
        "half": np.array([lines[0][-1], lines[1][-1]]),
        "origin": np.array([lines[0][0], lines[1][0]]),
        "step": np.array([lines[0][1] - lines[0][0], lines[1][1] - lines[1][0]]),
        "cells": np.array([len(lines[0]) - 1, len(lines[1]) - 1]),
        "uv": np.array(fit),
    }


def _view_projection(p, cams):
    """(C, 16) float32 projection times view matrices of camera rows ``cams`` (C, 12), computed in float64."""
    f = 1.0 / p.tanh
    n, fa = p.near, p.far
    proj = np.zeros((4, 4))
    proj[0, 0] = proj[1, 1] = f
    proj[2, 2] = (fa + n) / (n - fa)
    proj[2, 3] = 2 * fa * n / (n - fa)
    proj[3, 2] = -1
    view = np.zeros((len(cams), 4, 4))
    rot = np.stack([cams[:, 3:6], cams[:, 6:9], -cams[:, 9:12]], 1)
    view[:, :3, :3] = rot
    view[:, :3, 3] = -np.einsum("cij,cj->ci", rot, cams[:, :3])
    view[:, 3, 3] = 1
    return np.einsum("ij,cjk->cik", proj, view).reshape(len(cams), 16)


def _offset(clip, res, sample):
    """Clip positions moved so that pixel centres sample position ``sample`` of each pixel."""
    sx, sy = SAMPLES[sample]
    shift = torch.tensor([(0.5 - sx) * 2 / res, (0.5 - sy) * 2 / res, 0.0, 0.0], device=clip.device)
    return (clip + shift * clip[..., 3:4]).contiguous()


def _rays(p, cams, S, px, py):
    """World ray directions (U, S, S, 3), not normalised, through window positions ``px``, ``py`` (S, S), rows up."""
    x = (px / S * 2 - 1) * p.tanh
    y = (py / S * 2 - 1) * p.tanh
    return (
        x[None, ..., None] * cams[:, None, None, 3:6]
        + y[None, ..., None] * cams[:, None, None, 6:9]
        + cams[:, None, None, 9:12]
    )


def _plane_hits(plane, campos, rays):
    """World points where ``rays`` meet the plane z = 0, and whether they meet it in front within its extent."""
    t = -campos[:, None, None, 2:3] / rays[..., 2:3]
    point = campos[:, None, None, :] + t * rays
    half = torch.as_tensor(plane["half"], device=rays.device, dtype=rays.dtype)
    inside = (t[..., 0] > 0) & (point[..., 0].abs() <= half[0]) & (point[..., 1].abs() <= half[1])
    return point, inside


def _background(ds, keys, inputs, S):
    """Background (plane, plane shadow inputs and sky) of each unique key, cached by key."""
    out, missing = [None] * len(keys), []
    for i, key in enumerate(keys):
        hit = ds.backgrounds.get((S, key))
        if hit is None:
            missing.append(i)
        else:
            out[i] = hit
    if missing:
        made = _make_background(ds, [inputs[i] for i in missing], S)
        for j, i in enumerate(missing):
            out[i] = ds.backgrounds.put((S, keys[i]), {name: value[j] for name, value in made.items()})
    return {name: torch.stack([entry[name] for entry in out]).contiguous() for name in out[0]}


def _make_background(ds, inputs, S):
    import nvdiffrast.torch as dr

    p, device = ds.p, ds.device
    U = len(inputs)
    cams = torch.as_tensor(np.stack([i[0] for i in inputs]), device=device)
    i = torch.arange(S, device=device, dtype=torch.float64)
    py, px = torch.meshgrid(i, i, indexing="ij")
    centre_ray = _rays(p, cams, S, px + 0.5, py + 0.5)
    out = {}
    if ds.sky is not None:
        ray = centre_ray / centre_ray.norm(dim=-1, keepdim=True)
        # MuJoCo turns a world direction 90 degrees about x into cube space.
        cube = torch.stack([ray[..., 0], ray[..., 2], -ray[..., 1]], -1).float().contiguous()
        sky = torch.floor(dr.texture(ds.sky[None], cube, filter_mode="linear", boundary_mode="cube") * 255 + 0.5)
    else:
        sky = torch.zeros((U, S, S, 3), device=device)
    for u, (_, _, _, flat) in enumerate(inputs):
        if flat[0] >= 0:
            sky[u] = torch.as_tensor(flat, device=device, dtype=sky.dtype)
    out["sky"] = sky.to(torch.uint8)
    plane = ds.plane
    if plane is None:
        out["cover"] = torch.zeros((U, 4, S, S), dtype=torch.uint8, device=device)
        out["depth"] = torch.ones((U, 4, S, S), device=device)
        out["xy"] = torch.zeros((U, S, S, 2), device=device)
        out["first"] = out["second"] = torch.zeros((U, S, S, 3), dtype=torch.uint8, device=device)
        return out
    campos = cams[:, :3]
    centre, _ = _plane_hits(plane, campos, centre_ray)
    pv = torch.as_tensor(_view_projection(p, cams.cpu().numpy()), device=device).reshape(U, 4, 4)
    cover, depth = [], []
    for sx, sy in SAMPLES:
        point, inside = _plane_hits(plane, campos, _rays(p, cams, S, px + sx, py + sy))
        m = pv[:, None, None]
        clip = m[..., :, 0] * point[..., None, 0] + m[..., :, 1] * point[..., None, 1] + m[..., :, 3]
        z = clip[..., 2] / clip[..., 3]
        inside = inside & (z >= -1) & (z <= 1)
        cover.append(inside)
        depth.append(torch.where(inside, z, torch.ones_like(z)))
    # Gouraud shading on the plane grid: the lit colours of the three corners of the sample's triangle.
    origin = torch.as_tensor(plane["origin"], device=device)
    step = torch.as_tensor(plane["step"], device=device)
    cells = torch.as_tensor(plane["cells"], device=device)
    local = (centre[..., :2] - origin) / step
    cell = torch.minimum(torch.maximum(torch.floor(local), torch.zeros_like(local)), (cells - 1).to(local.dtype))
    fx, fy = (local - cell).unbind(-1)
    lower = fx + fy <= 1
    x0, y0 = cell[..., 0], cell[..., 1]
    corner = torch.stack(
        [
            torch.where(lower[..., None], torch.stack([x0, y0], -1), torch.stack([x0 + 1, y0], -1)),
            torch.where(lower[..., None], torch.stack([x0 + 1, y0], -1), torch.stack([x0 + 1, y0 + 1], -1)),
            torch.stack([x0, y0 + 1], -1),
        ],
        -2,
    ).long()
    weight = torch.stack(
        [torch.where(lower, 1 - fx - fy, 1 - fy), torch.where(lower, fx, fx + fy - 1), torch.where(lower, fy, 1 - fx)],
        -1,
    )
    nx, ny = (int(n) + 1 for n in plane["cells"])
    gx, gy = np.meshgrid(
        plane["origin"][0] + np.arange(nx) * plane["step"][0],
        plane["origin"][1] + np.arange(ny) * plane["step"][1],
        indexing="ij",
    )
    points = np.stack([gx.reshape(-1), gy.reshape(-1), np.zeros(nx * ny)], 1)
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    lit = np.stack([core.shade_points(p, plane["geom"], points, normals, i[2], i[0], i[1]) for i in inputs])
    lit = torch.as_tensor(lit, device=device).reshape(U, nx, ny, 2, 3)
    ui = torch.arange(U, device=device)[:, None, None, None]
    corners = lit[ui, corner[..., 0], corner[..., 1]]
    c1 = (corners[..., 0, :] * weight[..., None]).sum(-2)
    c2 = (corners[..., 1, :] * weight[..., None]).sum(-2)
    # Texture at the pixel centre, with derivatives over each 2x2 pixel quad as the GPU takes them.
    fit = torch.as_tensor(plane["uv"], device=device)
    u = centre[..., 0] * fit[0, 0] + fit[0, 1]
    v = centre[..., 1] * fit[1, 0] + fit[1, 1]
    uv = torch.stack([u, v], -1)
    even = torch.cat([uv, uv[:, -1:]], 1) if S % 2 else uv
    even = torch.cat([even, even[:, :, -1:]], 2) if S % 2 else even
    quad = even[:, 0::2, 0::2]
    du_dx = (even[:, 0::2, 1::2] - quad).repeat_interleave(2, 1).repeat_interleave(2, 2)
    du_dy = (even[:, 1::2, 0::2] - quad).repeat_interleave(2, 1).repeat_interleave(2, 2)
    uv_da = torch.stack([du_dx[..., 0], du_dy[..., 0], du_dx[..., 1], du_dy[..., 1]], -1)[:, :S, :S]
    tex = dr.texture(
        plane["texture"][None],
        uv.float().contiguous(),
        uv_da.float().contiguous(),
        filter_mode="linear-mipmap-linear",
        boundary_mode="wrap",
    )
    first = torch.floor((c1 * tex).clamp(0, 1) * 255 + 0.5)
    second = torch.floor((first + (c2 * tex).clamp(0, 1) * 255).clamp(max=255) + 0.5)
    out["cover"] = torch.stack(cover, 1).to(torch.uint8)
    out["depth"] = torch.stack(depth, 1).float()
    out["xy"] = centre[..., :2].float()
    out["first"] = first.to(torch.uint8)
    out["second"] = second.to(torch.uint8)
    return out


def _window(ds, gpos):
    """Texel range [i0, i1) x [j0, j1) of the shadow map that holds the moving casters of every frame."""
    p = ds.p
    if not len(p.casters):
        return (0, 0), (0, 0)
    n = p.nshadow
    smat = p.smat
    sc = math.sqrt(smat[0, 0] ** 2 + smat[0, 1] ** 2 + smat[0, 2] ** 2)
    centre = gpos[:, p.casters] @ smat[:2, :3].T + smat[:2, 3]
    rad = ds.cast_rbound[p.casters] * sc + 1.0
    lo = np.floor((centre - rad[None, :, None]).min((0, 1))).astype(int)
    hi = np.floor((centre + rad[None, :, None]).max((0, 1))).astype(int)
    lo = np.maximum(lo, 1)
    hi = np.minimum(hi, n - 2)
    if np.any(lo > hi):
        return (0, 0), (0, 0)
    return (int(lo[0]), int(lo[1])), (int(hi[0]) + 1, int(hi[1]) + 1)


def _resolve(base, out, size, ss):
    pixel = torch.floor((base.float().sum(1) + 2) / 4).flip(1)
    if ss > 1:
        pixel = torch.round(pixel.reshape(len(base), size, ss, size, ss, 3).mean((2, 4)))
    out.copy_(pixel.to(torch.uint8))


def _chunk(ds, gpos, gmat, rgba, cams, lights, sky, ss, out):
    import nvdiffrast.torch as dr

    from . import mujoco_raster_triton as kernels

    p, device = ds.p, ds.device
    C, size = len(gpos), out.shape[1]
    S = size * ss
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a, np.float32), device=device)  # noqa: E731
    transform = f32(np.concatenate([gmat, gpos], -1))
    rgba_d = f32(rgba)
    clip, attr, trans = kernels.vertices(
        ds.vpos,
        ds.vnor,
        ds.vgeom,
        transform,
        rgba_d,
        ds.material,
        ds.lut,
        f32(cams),
        f32(_view_projection(p, cams)),
        f32(lights),
        p.ambient,
    )
    alpha = rgba_d[..., 3].contiguous()
    glass_corner = alpha[:, ds.vgeom.long()] < core.TRANSPARENT_ALPHA
    clip_opaque = torch.where(
        glass_corner[:, ds.opaque_vertices, None],
        torch.tensor([0.0, 0.0, 0.0, 1.0], device=device),
        clip[:, ds.opaque_vertices],
    )
    plane = ds.plane
    rows = [
        (cams[c], lights[c], rgba[c, plane["geom"]] if plane is not None else np.zeros(4), sky[c]) for c in range(C)
    ]
    keys = [b"".join(np.ascontiguousarray(a, np.float64).tobytes() for a in row) for row in rows]
    unique = list(dict.fromkeys(keys))
    first = {k: keys.index(k) for k in unique}
    index = torch.as_tensor([unique.index(k) for k in keys], device=device, dtype=torch.int32)
    background = _background(ds, unique, [rows[first[k]] for k in unique], S)
    shadow = None
    if ds.shadow:
        window = _window(ds, gpos)
        dyn = kernels.shadow(ds.cast_pos, ds.cast_tri, ds.cast_geom, transform, ds.smat, window)
        shadow = (ds.static_map, dyn, ds.slook, window)
    ctx = _context(device)
    base = torch.empty((C, 4, S, S, 3), dtype=torch.uint8, device=device)
    zbuf = torch.empty((C, 4, S, S), device=device)
    for k in range(4):
        if len(ds.tri_opaque):
            rast, rast_db = dr.rasterize(ctx, _offset(clip_opaque, S, k), ds.tri_opaque_compact, (S, S), grad_db=True)
        else:
            rast = rast_db = torch.zeros((C, S, S, 4), device=device)
        kernels.opaque(
            rast,
            rast_db,
            ds.tri_opaque,
            attr,
            ds.opaque_geom,
            ds.texmod,
            index,
            background,
            shadow,
            base,
            zbuf,
            k,
            SAMPLES[k],
        )
    glass = np.flatnonzero((rgba[..., 3] < core.TRANSPARENT_ALPHA).any(0))
    if plane is not None:
        glass = glass[glass != plane["geom"]]
    tri, tgeom, tlocal = ds.translucent(glass) if len(glass) else (None, None, None)
    if tri is None or not len(tri):
        _resolve(base, out, size, ss)
        return
    rank = ms.transparent_order(gpos, cams, p.sort_rbound, rgba[..., 3])
    pixel = kernels.blend(
        clip,
        tri,
        trans,
        torch.as_tensor(np.maximum(rank, 0), dtype=torch.int32, device=device),
        tgeom,
        tlocal,
        alpha,
        ds.texmod,
        base,
        zbuf,
        LAYERS,
        SAMPLES,
    )
    if ss > 1:
        pixel = torch.round(pixel.float().reshape(C, size, ss, size, ss, 3).mean((2, 4))).to(torch.uint8)
    out.copy_(pixel)


def draw(ds, gpos, gmat, rgba, cams, lights, sky, ss, out, chunk_size=None):
    """Draw frames of device scene ``ds`` into the CUDA uint8 tensor ``out`` (B, size, size, 3).

    The per-frame inputs are the host arrays of ``mujoco_raster.draw``.
    """
    B, size = len(gpos), out.shape[1]
    S = size * ss
    limit = chunk_size or max(1, (128 * 224 * 224) // (S * S))
    if ds.shadow:
        (i0, j0), (i1, j1) = _window(ds, gpos)
        limit = max(1, min(limit, WINDOW_BYTES // max(1, 4 * (i1 - i0) * (j1 - j0))))
    with _LOCK, torch.cuda.device(ds.device):
        for start in range(0, B, limit):
            stop = min(B, start + limit)
            part = slice(start, stop)
            _chunk(ds, gpos[part], gmat[part], rgba[part], cams[part], lights[part], sky[part], ss, out[part])
    return out
