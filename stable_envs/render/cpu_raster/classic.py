"""Numba CPU twin of the classic-control Triton kernel: the same Ops records, one frame per thread."""

from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange

from ..gfxdraw.classic import (
    CM_A,
    CM_B,
    CM_B2,
    CM_C,
    CM_D,
    CM_G,
    CM_OP,
    CM_R,
    CM_X0,
    CM_X1,
    OP_BLIT,
    OP_DISC,
    OP_HLINE,
    OP_POLY,
    OP_VLINE,
    Ops,
)
from ._common import blend_px, fill_polygon, put, wu_edge


@njit(cache=True)
def _disc(img, disc, cx, cy, at, r, aa, H, W, fr, fg, fb):
    d = 2 * r + 1
    for dy in range(-r, r + 1):
        y = cy + dy
        if y < 0 or y >= H:
            continue
        hw = disc[at + abs(dy)]
        row = H - 1 - y
        for dx in range(-r, r + 1):
            x = cx + dx
            if x < 0 or x >= W:
                continue
            if aa:
                wo = at + r + 1 + (dy + r) * d + (dx + r)
                w0 = disc[wo]
                if w0 > 0:
                    blend_px(img, row, x, fr, fg, fb, w0)
                w1 = disc[wo + d * d]
                if w1 > 0:
                    blend_px(img, row, x, fr, fg, fb, w1)
            if abs(dx) <= hw:
                put(img, row, x, fr, fg, fb)


@njit(cache=True)
def _frame(b, meta, vert, edge, disc, bg, plate, plate_idx, sprite, aa, out, H, W):
    img = out[b]
    if plate.shape[0] > 0:
        u = plate_idx[b]
        for y in range(H):
            for x in range(W):
                img[H - 1 - y, x, 0] = plate[u, 0, y, x]
                img[H - 1 - y, x, 1] = plate[u, 1, y, x]
                img[H - 1 - y, x, 2] = plate[u, 2, y, x]
    else:
        for y in range(H):
            for x in range(W):
                img[y, x, 0] = bg[b, 0]
                img[y, x, 1] = bg[b, 1]
                img[y, x, 2] = bg[b, 2]
    for i in range(meta.shape[1]):
        m = meta[b, i]
        op = m[CM_OP]
        fr = np.int64(m[CM_R])
        fg = np.int64(m[CM_G])
        fb = np.int64(m[CM_B])
        pa = np.int64(m[CM_A])
        pb = np.int64(m[CM_B2])
        pc = np.int64(m[CM_C])
        pd = np.int64(m[CM_D])
        if op == OP_HLINE:
            if 0 <= pa < H:
                for x in range(m[CM_X0], m[CM_X1] + 1):
                    put(img, H - 1 - pa, x, fr, fg, fb)
        elif op == OP_VLINE:
            if 0 <= pa < W:
                for y in range(max(pb, 0), min(pc, H - 1) + 1):
                    put(img, H - 1 - y, pa, fr, fg, fb)
        elif op == OP_BLIT:
            if sprite.shape[0] > 1:
                off = (pd // 2) - (pb // 2)
                for ly in range(pb):
                    y = pc + off + ly
                    if y < 0 or y >= H:
                        continue
                    for lx in range(pb):
                        x = pc + off + lx
                        if x < 0 or x >= W:
                            continue
                        sa = np.int64(sprite[pa, pb, ly, lx, 3])
                        for c in range(3):
                            cur = np.int64(img[H - 1 - y, x, c])
                            img[H - 1 - y, x, c] = cur + (((np.int64(sprite[pa, pb, ly, lx, c]) - cur) * sa) >> 8)
        elif op == OP_DISC:
            _disc(img, disc, pa, pb, pc, pd, aa[b] != 0, H, W, fr, fg, fb)
        elif op == OP_POLY:
            nv = pa
            if aa[b] != 0:
                for e in range(nv):
                    if edge[b, pb, e, 0] != 0:
                        wu_edge(img, edge[b, pb, e], H, W, fr, fg, fb)
            fill_polygon(img, vert[b, pb], nv, H, W, fr, fg, fb)


@njit(parallel=True, cache=True)
def _frames(meta, vert, edge, disc, bg, plate, plate_idx, sprite, aa, out, H, W):
    for b in prange(meta.shape[0]):
        _frame(b, meta, vert, edge, disc, bg, plate, plate_idx, sprite, aa, out, H, W)


def render_classic_cpu(ops: Ops, out: torch.Tensor | None = None) -> torch.Tensor:
    """Render CPU ``ops`` to ``(B, H, W, 3)`` uint8 RGB, byte-identical to the Triton kernel.

    A supplied output buffer is overwritten and returned.
    """
    B, H, W = ops.B, ops.H, ops.W
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8)
    plate = ops.plate
    if plate is None:
        plate_np, idx = np.zeros((0, 3, 1, 1), np.uint8), np.zeros(1, np.int64)
    else:
        # Planar (3, U, H, W) as the CUDA kernel reads it; lanes map to images by lane index or ops.plate_idx.
        plate_np = plate.permute(1, 0, 2, 3).numpy()
        pidx = getattr(ops, "plate_idx", None)
        idx = pidx.numpy() if pidx is not None else np.arange(B, dtype=np.int64) % plate_np.shape[0]
    sprite = ops.sprite.numpy() if ops.sprite is not None else np.zeros((1, 1, 1, 1, 4), np.uint8)
    bg = ops.bg.numpy() if ops.bg is not None else np.zeros((B, 3), np.int32)
    _frames(
        ops.meta_t.numpy(),
        ops.vert_t.numpy(),
        ops.edge_t.numpy(),
        ops.discs.flat.numpy(),
        bg,
        plate_np,
        idx,
        sprite,
        ops.aa.numpy(),
        out.numpy(),
        H,
        W,
    )
    return out
