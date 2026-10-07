"""Texture minification from projected pixel footprints, without frame resampling."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F


class Pyramid(NamedTuple):
    texels: torch.Tensor
    sizes: torch.Tensor
    offsets: torch.Tensor
    original_size: tuple[int, int]


def plane_texgen(size, repeat, uniform, version):
    """Plane UV scale/offset for MuJoCo's legacy texgen or 3.12+ explicit UVs."""
    size = np.asarray(size[:2], dtype=np.float64)
    repeat = np.asarray(repeat, dtype=np.float64)
    finite = size > 0
    if version >= (3, 12):
        repeat = np.where(repeat > 0, repeat, 1.0)
    scale = repeat / np.where(finite & ~np.asarray(uniform), size, 1.0)
    offset = np.full(2, -0.5)
    if version >= (3, 12):
        offset = np.where(finite, 0.5 * repeat * (size if uniform else 1.0), 0.0)
        if not finite.all():
            offset -= 0.5
    return scale, offset


def plane_grid(size: float, spacing: float, zfar: float, repeat: float, max_grid: int = 200) -> np.ndarray:
    """MuJoCo's plane tessellation coordinates along one local axis."""
    if size <= 0:
        step = 2 / repeat if repeat > 0 else 2.1 * zfar / (max_grid - 2)
        width = (2.1 * zfar + 2 * step) / max_grid
        return width * (np.arange(max_grid + 1) - max_grid // 2)
    spacing = max(spacing, size / (max_grid - 2))
    count = int(np.floor(size / max(1e-15, spacing)))
    count -= count % 2
    padding = size - count * spacing
    coordinates = [-size]
    for index in range(count + 2):
        left = -size if index == 0 else -size + padding + (index - 1) * spacing * 2
        right = left + (padding if index in (0, count + 1) else spacing * 2)
        coordinates.append(max(left, right))
    return np.unique(coordinates)


def mip_pyramid(texture: torch.Tensor) -> Pyramid:
    """Build an area-filtered mip chain for a floating-point HWC texture."""
    levels = [texture]
    while max(levels[-1].shape[:2]) > 1:
        h, w = levels[-1].shape[:2]
        levels.append(
            F.interpolate(
                levels[-1].permute(2, 0, 1)[None],
                size=(max(1, h // 2), max(1, w // 2)),
                mode="area",
            )[0]
            .permute(1, 2, 0)
            .contiguous()
        )
    sizes = torch.tensor([(level.shape[1], level.shape[0]) for level in levels], device=texture.device)
    counts = sizes[:, 0] * sizes[:, 1]
    offsets = torch.cat((torch.zeros(1, dtype=torch.long, device=texture.device), counts.cumsum(0)[:-1]))
    texels = torch.cat([level.reshape(-1, texture.shape[-1]) for level in levels])
    return Pyramid(texels, sizes, offsets, tuple(texture.shape[:2]))


def _trilinear(pyramid: Pyramid, u: torch.Tensor, v: torch.Tensor, lod: torch.Tensor, wrap: bool) -> torch.Tensor:
    """Gather the two neighboring mip levels together from contiguous texture storage."""
    count = len(pyramid.sizes)
    lod = lod.clamp(0, count - 1)
    low = lod.floor().long()
    fraction = (lod - low)[..., None]
    levels = torch.stack((low, (low + 1).clamp(max=count - 1)), dim=-1)
    width, height = pyramid.sizes[levels].unbind(-1)
    offset = pyramid.offsets[levels]
    x, y = u[..., None] * width - 0.5, v[..., None] * height - 0.5
    ix, iy = x.floor().long(), y.floor().long()
    fx, fy = (x - ix)[..., None], (y - iy)[..., None]
    if wrap:
        x0, x1, y0, y1 = ix % width, (ix + 1) % width, iy % height, (iy + 1) % height
    else:
        x0, x1 = ix.clamp_min(0).minimum(width - 1), (ix + 1).clamp_min(0).minimum(width - 1)
        y0, y1 = iy.clamp_min(0).minimum(height - 1), (iy + 1).clamp_min(0).minimum(height - 1)
    texels = pyramid.texels
    top = texels[offset + y0 * width + x0] * (1 - fx) + texels[offset + y0 * width + x1] * fx
    bottom = texels[offset + y1 * width + x0] * (1 - fx) + texels[offset + y1 * width + x1] * fx
    value = top * (1 - fy) + bottom * fy
    return value[..., 0, :] * (1 - fraction) + value[..., 1, :] * fraction


def sample_plane_filtered(
    pyramid: Pyramid,
    u: torch.Tensor,
    v: torch.Tensor,
    du_dx: torch.Tensor,
    dv_dx: torch.Tensor,
    du_dy: torch.Tensor,
    dv_dy: torch.Tensor,
) -> torch.Tensor:
    """Sample repeating UVs using their per-screen-pixel derivatives."""
    h, w = pyramid.original_size
    rho2 = torch.maximum((du_dx * w).square() + (dv_dx * h).square(), (du_dy * w).square() + (dv_dy * h).square())
    lod = 0.5 * rho2.clamp_min(1).log2()
    return _trilinear(pyramid, u, v, lod, wrap=True)


def _cube_coordinates(directions: torch.Tensor, face: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = directions.unbind(-1)
    denominator = torch.where(face < 2, x.abs(), torch.where(face < 4, y.abs(), z.abs()))
    s = torch.where(face == 0, -z, torch.where(face == 1, z, torch.where(face == 5, -x, x)))
    t = torch.where(face == 2, z, torch.where(face == 3, -z, -y))
    denominator = denominator.clamp_min(torch.finfo(directions.dtype).tiny)
    return (s / denominator + 1) * 0.5, (t / denominator + 1) * 0.5


def _neighbor(directions: torch.Tensor, axis: int) -> torch.Tensor:
    if directions.shape[axis] == 1:
        return directions
    # Use the backward difference at the last pixel; only derivative length matters.
    indices = torch.arange(directions.shape[axis], device=directions.device) + 1
    indices[-1] -= 2
    return directions.index_select(axis, indices)


def sample_skybox_filtered(sky: torch.Tensor, face_w: int, directions: torch.Tensor) -> torch.Tensor:
    """Filter a six-face vertical MuJoCo skybox for an H×W grid of world rays.

    Each mip is built within its own face. Neighbor rays are projected onto the
    current pixel's face to keep footprint estimates continuous across seams.
    """
    if sky.shape[:2] != (6 * face_w, face_w) or directions.ndim != 3 or directions.shape[-1] != 3:
        raise ValueError("Expected a six-face skybox and an H×W×3 direction grid")
    cube = torch.stack((directions[..., 0], directions[..., 2], -directions[..., 1]), dim=-1)
    axis = cube.abs().argmax(-1)
    component = cube.gather(-1, axis[..., None]).squeeze(-1)
    face = axis * 2 + (component <= 0).long()
    u, v = _cube_coordinates(cube, face)
    ux, vx = _cube_coordinates(_neighbor(cube, 1), face)
    uy, vy = _cube_coordinates(_neighbor(cube, 0), face)
    rho2 = torch.maximum((ux - u).square() + (vx - v).square(), (uy - u).square() + (vy - v).square()) * face_w**2
    lod = 0.5 * rho2.clamp_min(1).log2()
    result = torch.empty((*face.shape, sky.shape[-1]), device=sky.device, dtype=sky.dtype)
    for index in range(6):
        mask = face == index
        if mask.any():
            pyramid = mip_pyramid(sky[index * face_w : (index + 1) * face_w])
            result[mask] = _trilinear(pyramid, u[mask], v[mask], lod[mask], wrap=False)
    return result
