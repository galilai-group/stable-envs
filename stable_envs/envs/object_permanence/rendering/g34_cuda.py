"""Fused CUDA composition over the same cached G34 background and exact facets.

One program draws a pixel tile for one pose. Analytic misses skip all facet and
shading work. Float64 geometry and disabled FMA preserve the CPU hit convention.
Triton is optional: g34_raycast retains its Torch path when it is unavailable.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

from ....render.uv_sphere import _refine_cuda
from .g34_native import copy_output


@triton.jit
def _draw(
    rows,
    rays,
    cosine,
    radial,
    background,
    color,
    lamp,
    planes,
    output,
    PIXELS: tl.constexpr,
    MODE: tl.constexpr,
    CAMERA: tl.constexpr,
    RADIUS: tl.constexpr,
    SLICES: tl.constexpr,
    STACKS: tl.constexpr,
    HEIGHT: tl.constexpr,
    LONG_COS: tl.constexpr,
    NEAR: tl.constexpr,
    FAR: tl.constexpr,
    AMBIENT: tl.constexpr,
    HEAD: tl.constexpr,
    LAMP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    batch = tl.program_id(0)
    pixel = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = pixel < PIXELS
    px, py, pz = (
        tl.load(rows + batch * 7).to(tl.float64),
        tl.load(rows + batch * 7 + 1).to(tl.float64),
        tl.load(rows + batch * 7 + 2).to(tl.float64),
    )
    ox, oy, oz = CAMERA[0] - px, CAMERA[1] - py, CAMERA[2] - pz
    dx = tl.load(rays + pixel * 3, valid, 0)
    dy = tl.load(rays + pixel * 3 + 1, valid, 0)
    dz = tl.load(rays + pixel * 3 + 2, valid, 0)
    b = ox * dx + oy * dy + oz * dz
    disc = b * b - (ox * ox + oy * oy + oz * oz - RADIUS * RADIUS)
    sq = libdevice.sqrt(tl.maximum(disc, 0))
    t = -b - sq
    t = tl.where(t > 0, t, -b + sq)
    hit = valid & (disc > 0) & (t > 0)
    nx, ny, nz = tl.full((BLOCK,), 0, tl.float64), tl.full((BLOCK,), 0, tl.float64), tl.full((BLOCK,), 0, tl.float64)
    if tl.sum(hit.to(tl.int32), 0):
        qw, qx, qy, qz = (
            tl.load(rows + batch * 7 + 3).to(tl.float64),
            tl.load(rows + batch * 7 + 4).to(tl.float64),
            tl.load(rows + batch * 7 + 5).to(tl.float64),
            tl.load(rows + batch * 7 + 6).to(tl.float64),
        )
        norm = libdevice.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
        r00, r01, r02 = 1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)
        r10, r11, r12 = 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)
        r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
        local_ox, local_oy, local_oz = (
            r00 * ox + r10 * oy + r20 * oz,
            r01 * ox + r11 * oy + r21 * oz,
            r02 * ox + r12 * oy + r22 * oz,
        )
        local_dx, local_dy, local_dz = (
            r00 * dx + r10 * dy + r20 * dz,
            r01 * dx + r11 * dy + r21 * dz,
            r02 * dx + r12 * dy + r22 * dz,
        )
        t, lx, ly, lz = _refine_cuda(
            local_ox + tl.zeros((BLOCK,), tl.float64),
            local_oy + tl.zeros((BLOCK,), tl.float64),
            local_oz + tl.zeros((BLOCK,), tl.float64),
            local_dx,
            local_dy,
            local_dz,
            t,
            hit,
            planes,
            RADIUS,
            SLICES,
            STACKS,
            HEIGHT,
            LONG_COS,
        )
        nx, ny, nz = r00 * lx + r01 * ly + r02 * lz, r10 * lx + r11 * ly + r12 * lz, r20 * lx + r21 * ly + r22 * lz
    z = t * tl.load(cosine + pixel, valid, 0)
    visible = hit & (t < tl.load(radial + pixel, valid, 0)) & (z >= NEAR) & (z <= FAR)
    if MODE == 0:
        illumination = tl.full((BLOCK,), 0, tl.float64)
        if tl.sum(visible.to(tl.int32), 0):
            lx = tl.load(lamp) - (CAMERA[0] + t * dx)
            ly = tl.load(lamp + 1) - (CAMERA[1] + t * dy)
            lz = tl.load(lamp + 2) - (CAMERA[2] + t * dz)
            norm = tl.maximum(libdevice.sqrt(lx * lx + ly * ly + lz * lz), 1e-15)
            head = tl.maximum(-(nx * CAMERA[3] + ny * CAMERA[4] + nz * CAMERA[5]), 0) * HEAD
            illumination = tl.maximum(nx * (lx / norm) + ny * (ly / norm) + nz * (lz / norm), 0) * LAMP
        else:
            head = tl.full((BLOCK,), 0, tl.float64)
        for channel in tl.static_range(3):
            base = tl.load(color + channel).to(tl.float64)
            value = base * AMBIENT + base * head
            value = value + base * illumination
            value = libdevice.nearbyint(tl.minimum(tl.maximum(value, 0), 1) * 255).to(tl.uint8)
            static = tl.load(background + pixel * 3 + channel, valid, 0)
            tl.store(output + (batch * PIXELS + pixel) * 3 + channel, tl.where(visible, value, static), valid)
    else:
        static = tl.load(background + pixel, valid, 0)
        value = z.to(tl.float32) if MODE == 1 else 5
        tl.store(output + batch * PIXELS + pixel, tl.where(visible, value, static), valid)


def render_cuda(rows, plate, height, width, mode, out):
    """Draw CUDA RGB/depth/IDs, retaining owned outputs and the shared plate."""
    background = plate[mode]
    result = torch.empty((len(rows), *background.shape), dtype=background.dtype, device=rows.device)
    with torch.cuda.device(rows.device):
        _draw[(len(rows), triton.cdiv(height * width, 128))](
            rows.contiguous(),
            plate["direction"],
            plate["cosine"],
            plate["radial"],
            background,
            plate["color"],
            plate["light_position"],
            plate["planes"],
            result,
            height * width,
            {"rgb": 0, "depth": 1, "ids": 2}[mode],
            plate["cuda_camera"],
            plate["radius"],
            plate["slices"],
            plate["stacks"],
            plate["radius"] * math.cos(math.pi / plate["slices"]) * math.cos(math.pi / (2 * plate["stacks"])),
            math.cos(math.pi / plate["slices"]),
            plate["near"],
            plate["far"],
            plate["head_ambient"],
            plate["head_diffuse"],
            plate["light_diffuse"],
            128,
            enable_fp_fusion=False,
        )
    return copy_output(result, out)
