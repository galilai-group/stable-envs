"""Numba CPU twin of the navigation dot compositor: the Torch path's rounding points, one frame per thread."""

from __future__ import annotations

import math

import numpy as np
import torch
from numba import njit, prange

from .navdot import DOT_SIGMAS

_F0 = np.float32(0.0)
_F1 = np.float32(1.0)
_SIGMAS = np.float32(DOT_SIGMAS)


@njit(cache=True)
def _window(x, y, sigma, H, W):
    """Return (r0, r1, c0, c1) with the float32 arithmetic of ``navdot.dot_window``."""
    mx = np.float32(min(max(np.rint(x), _F0), np.float32(W - 1)))
    my = np.float32(min(max(np.rint(y), _F0), np.float32(H - 1)))
    ex = x - mx
    ey = y - my
    m2 = ex * ex + ey * ey
    s = sigma * _SIGMAS
    d = np.float32(math.sqrt(np.float64(s * s + m2))) + _F1
    # Clamping to integer bounds commutes with ceil and floor; clamping first keeps far dots in int range.
    c0 = math.ceil(min(max(x - d, _F0), np.float32(W)))
    c1 = math.floor(min(max(x + d, _F0), np.float32(W - 1))) + 1
    r0 = math.ceil(min(max(y - d, _F0), np.float32(H)))
    r1 = math.floor(min(max(y + d, _F0), np.float32(H - 1))) + 1
    return r0, r1, c0, c1


@njit(cache=True)
def _dot(img, x, y, sigma, cr, cg, cb):
    """Blend one normalized Gaussian dot into ``img`` inside its window, quantizing to uint8."""
    H, W = img.shape[0], img.shape[1]
    r0, r1, c0, c1 = _window(x, y, sigma, H, W)
    if r1 <= r0 or c1 <= c0:
        return
    std = max(np.float64(sigma), 1e-6)
    den = np.float32(2.0 * std * std)
    dot = np.empty((r1 - r0, c1 - c0), np.float32)
    m = np.float32(-np.inf)
    for r in range(r0, r1):
        dy = np.float32(r) - y
        dy2 = dy * dy
        for c in range(c0, c1):
            dx = np.float32(c) - x
            v = np.float32(np.exp(np.float64(-(dx * dx + dy2) / den)))
            dot[r - r0, c - c0] = v
            m = max(m, v)
    for r in range(r0, r1):
        for c in range(c0, c1):
            v = dot[r - r0, c - c0]
            a = min(max(v / m if m > _F0 else v, _F0), _F1)
            ia = _F1 - a
            img[r, c, 0] = np.uint8(np.float32(img[r, c, 0]) * ia + cr * a)
            img[r, c, 1] = np.uint8(np.float32(img[r, c, 1]) * ia + cg * a)
            img[r, c, 2] = np.uint8(np.float32(img[r, c, 2]) * ia + cb * a)


@njit(parallel=True, cache=True)
def _frames(plates, plate_idx, pos, sigma, colour, out):
    for b in prange(out.shape[0]):
        img = out[b]
        # A flat loop copies at memory speed; a 3-D slice assignment indexes every element.
        dst = img.reshape(-1)
        src = plates[plate_idx[b]].reshape(-1)
        for i in range(dst.shape[0]):
            dst[i] = src[i]
        for d in range(pos.shape[1]):
            _dot(
                img,
                pos[b, d, 0],
                pos[b, d, 1],
                sigma[b, d],
                np.float32(np.trunc(colour[b, d, 0])),
                np.float32(np.trunc(colour[b, d, 1])),
                np.float32(np.trunc(colour[b, d, 2])),
            )


def _f32(t: torch.Tensor, shape: tuple[int, ...]) -> np.ndarray:
    return t.detach().reshape(shape).to(torch.float32).contiguous().numpy()


def blend_dots(
    base: torch.Tensor, pos: torch.Tensor, sigma: torch.Tensor, colour: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """Composite dots in order onto a CPU uint8 base ``(B, H, W, 3)``, as ``navdot.blend_dots`` does on CUDA.

    Positions are (B, ND, 2), sigmas (B, ND) and colours (B, ND, 3). ``base`` may repeat one image
    over the batch with a zero stride; that image is read once. Write into ``out`` when supplied.
    """
    B, H, W, _ = base.shape
    ND = int(pos.shape[1]) if pos.dim() == 3 else 1
    if B > 1 and base.stride(0) == 0:
        plates, idx = base[:1].contiguous(), np.zeros(B, np.int64)
    else:
        plates, idx = base.contiguous(), np.arange(B, dtype=np.int64)
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8)
    _frames(
        plates.numpy(),
        idx,
        _f32(pos, (B, ND, 2)),
        _f32(sigma, (B, ND)),
        _f32(colour, (B, ND, 3)),
        out.numpy(),
    )
    return out
