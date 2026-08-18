"""Candidate closed form for Qt 5.9's non-smooth `QPainter::drawImage(QRectF, QImage)`.

This is the HYPOTHESIS; `probe03_fit_blit.py` is what decides whether it is true, by diffing it
against coordinate-coded frames produced by the shipped procgen binary.

The claim is that the raster engine takes the `qScaleFunctions` fast path (`qt_scale_image_*`),
which is a nearest-neighbour walk in 16.16 fixed point:

    tr    = targetRect.toRect() intersected with the device rect
    ix,iy = int(65536 * srcW/dstW), int(65536 * srcH/dstH)
    basex = uint32(srcLeft*65536) + (ceil((tx1 + 0.5 - dstLeft) * sx * 65536) - 1)
    srcy  = uint32(srcTop *65536) + (ceil((ty1 + 0.5 - dstTop ) * sy * 65536) - 1)
    sample(col j, row i) = src[(srcy + i*iy) >> 16][(basex + j*ix) >> 16]

with `QRectF::toRect()` = `QRect(qRound(x), qRound(y), qRound(x+w)-qRound(x), qRound(y+h)-qRound(y))`.

Geometry upstream of this is FLOAT32, because procgen computes every rect in `float` and only widens
to `qreal` at the `QRectF` constructor. `view_dim` for heist is 12.999999 rather than 13, which is
measurable in the state blob and shifts a rect edge by a whole pixel if it is computed in float64.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def qround(v):
    """Qt's `qRound(qreal)`: `int(v + 0.5)` for v >= 0, `int(v - 0.5)` for v < 0."""
    v = np.asarray(v, np.float64)
    return np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)


def to_rect(x, y, w, h):
    """`QRectF::toRect()` -> (left, top, width, height) in integer device pixels."""
    l = qround(x)
    t = qround(y)
    r = qround(np.asarray(x, np.float64) + np.asarray(w, np.float64)) - 1
    b = qround(np.asarray(y, np.float64) + np.asarray(h, np.float64)) - 1
    return l, t, r - l + 1, b - t + 1


def blit_index(x, y, w, h, sw, sh, W, H):
    """Which source pixel each device pixel samples, for one `drawImage(QRectF, QImage)`.

    Returns `(ys, xs, sy, sx)`: device rows/cols actually written, and the source row/col each
    samples. Empty arrays when the rect misses the device entirely.
    """
    l, t, tw, th = to_rect(x, y, w, h)
    l, t, tw, th = int(l), int(t), int(tw), int(th)
    # tr.intersected(clip) with clip = the device rect
    x0, y0 = max(l, 0), max(t, 0)
    x1, y1 = min(l + tw, W), min(t + th, H)
    if x1 <= x0 or y1 <= y0:
        return (np.empty(0, np.int64),) * 4
    sxf = float(sw) / float(w)
    syf = float(sh) / float(h)
    ix = int(0x00010000 * sxf)                       # C truncation toward zero
    iy = int(0x00010000 * syf)
    dstx = int(np.ceil((x0 + 0.5 - float(x)) * sxf * 65536)) - 1
    dsty = int(np.ceil((y0 + 0.5 - float(y)) * syf * 65536)) - 1
    basex = (0 + dstx) & 0xFFFFFFFF                  # sourceRect.left() == 0
    basey = (0 + dsty) & 0xFFFFFFFF
    cols = np.arange(x1 - x0, dtype=np.int64)
    rows = np.arange(y1 - y0, dtype=np.int64)
    sx = ((basex + cols * ix) & 0xFFFFFFFF) >> 16
    sy = ((basey + rows * iy) & 0xFFFFFFFF) >> 16
    return (np.arange(y0, y1), np.arange(x0, x1), sy, sx)


def premultiply(rgba: np.ndarray) -> np.ndarray:
    """`qPremultiply`, the exact integer form Qt uses when converting to ARGB32_Premultiplied."""
    a = rgba[..., 3].astype(np.uint32)
    out = np.empty_like(rgba)
    for c in range(3):
        t = rgba[..., c].astype(np.uint32) * a
        t = (t + (t >> 8) + 0x80) >> 8
        out[..., c] = (t & 0xFF).astype(np.uint8)
    out[..., 3] = rgba[..., 3]
    return out


def source_over(dst_rgb: np.ndarray, src_rgb: np.ndarray, src_a: np.ndarray) -> np.ndarray:
    """SDL-free: Qt's `PRIM_SOURCE_OVER` on premultiplied ARGB, `d = s + BYTE_MUL(d, 255-sa)`."""
    inv = (255 - src_a.astype(np.uint32))
    d = dst_rgb.astype(np.uint32)
    t = d * inv[..., None]
    t = (t + (t >> 8) + 0x80) >> 8
    return np.clip(src_rgb.astype(np.uint32) + (t & 0xFF), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ procgen's own geometry

def prepare(main_w, main_h, center_agent, center, min_visibility, rect_h=64):
    """`BasicAbstractGame::prepare_for_drawing`, in float32 exactly as the C++ does it."""
    if center_agent:
        cx, cy = F(center[0]), F(center[1])
        vis = F(min_visibility)                 # caller supplies the game's own `visibility`
    else:
        cx, cy = F(main_w) * F(0.5), F(main_h) * F(0.5)
        vis = F(max(main_w, main_h))
        if vis < F(min_visibility):
            vis = F(min_visibility)
    raw_unit = F(F(64) / vis)
    unit = F(np.float64(raw_unit) * (rect_h / 64.0))
    view_dim = F(64.0 / np.float64(raw_unit))
    x_off = F(unit * F(cx - F(view_dim / F(2))))
    y_off = F(unit * F(cy - F(view_dim / F(2))))
    return unit, view_dim, x_off, y_off, vis


def screen_rect(x, y, dx, dy, unit, view_dim, x_off, y_off, eps=0.0):
    """`get_screen_rect`. Every operand is float32; only the QRectF constructor widens."""
    e = F(eps)
    rx = F(F(F(x) - e) * unit - x_off)
    ry = F(F(F(view_dim) - F(y) - e) * unit + y_off)
    rw = F(F(F(dx) + F(2) * e) * unit)
    rh = F(F(F(dy) + F(2) * e) * unit)
    return float(rx), float(ry), float(rw), float(rh)


def abs_rect(x, y, dx, dy, unit):
    return float(F(F(x) * unit)), float(F(F(y) * unit)), float(F(F(dx) * unit)), float(F(F(dy) * unit))
