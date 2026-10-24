"""Shared integer drawing pieces for the Numba raster twins: blends, Wu edges, polygon spans, discs."""

from __future__ import annotations

import numpy as np
from numba import njit

M32 = (1 << 32) - 1
BIG = 1 << 40
K_VERT, K_HORIZ, K_YMAJ = 1, 2, 4
(EG_DRAW, EG_SX, EG_SY, EG_X0, EG_Y0, EG_DX, EG_DY, EG_XDIR, EG_KIND, EG_MAJOR, EG_E, EG_LO) = range(12)


@njit(cache=True, inline="always")
def blend(s, d, a, sh):
    """SDL_gfx packed-channel blend of one channel with alpha ``a`` and 32-bit wrapping."""
    diff = ((s - d) << sh) & M32
    v = ((diff * a) & M32) >> 8
    return (((d << sh) + v) & (255 << sh)) >> sh


@njit(cache=True, inline="always")
def put(img, y, x, r, g, b):
    img[y, x, 0] = r
    img[y, x, 1] = g
    img[y, x, 2] = b


@njit(cache=True, inline="always")
def blend_px(img, y, x, r, g, b, w):
    """Blend colour (r, g, b) into a pixel with fringe weight ``w`` below 256."""
    a = (255 * w) >> 8
    img[y, x, 0] = blend(r, np.int64(img[y, x, 0]), a, 16)
    img[y, x, 1] = blend(g, np.int64(img[y, x, 1]), a, 8)
    img[y, x, 2] = blend(b, np.int64(img[y, x, 2]), a, 0)


@njit(cache=True, inline="always")
def round16(v):
    return (v >> 16) + ((v & 32768) >> 15)


@njit(cache=True)
def wu_edge(img, ep, H, W, r, g, b):
    """Draw one clipped Wu edge from its table row; ``img`` rows are pre-flip y = H - 1 - row."""
    sx = ep[EG_SX]
    sy = ep[EG_SY]
    xx0 = ep[EG_X0]
    yy0 = ep[EG_Y0]
    dxe = ep[EG_DX]
    dye = ep[EG_DY]
    xdir = ep[EG_XDIR]
    kind = ep[EG_KIND]
    major = ep[EG_MAJOR]
    E = ep[EG_E]
    lo = ep[EG_LO]
    if 0 <= sx < W and 0 <= sy < H:
        put(img, H - 1 - sy, sx, r, g, b)
    if kind == K_VERT and dye > 0:
        for y in range(max(yy0, 0), min(yy0 + dye, H - 1) + 1):
            if 0 <= xx0 < W:
                put(img, H - 1 - y, xx0, r, g, b)
    elif kind == K_HORIZ:
        if 0 <= yy0 < H:
            for x in range(max(lo, 0), min(lo + dxe, W - 1) + 1):
                put(img, H - 1 - yy0, x, r, g, b)
    elif kind == K_YMAJ:
        for kk in range(1, major):
            acc = kk * E
            wgt = (acc & 0xFFFF) >> 8
            y = yy0 + kk
            if 0 <= y < H:
                xa = xx0 + xdir * (acc >> 16)
                if 0 <= xa < W:
                    blend_px(img, H - 1 - y, xa, r, g, b, 255 - wgt)
                xb = xa + xdir
                if 0 <= xb < W:
                    blend_px(img, H - 1 - y, xb, r, g, b, wgt)
    elif kind == 0:
        for kk in range(1, major):
            acc = kk * E
            wgt = (acc & 0xFFFF) >> 8
            x = xx0 + xdir * kk
            if 0 <= x < W:
                ya = yy0 + (acc >> 16)
                if 0 <= ya < H:
                    blend_px(img, H - 1 - ya, x, r, g, b, 255 - wgt)
                if 0 <= ya + 1 < H:
                    blend_px(img, H - 1 - (ya + 1), x, r, g, b, wgt)


@njit(cache=True)
def fill_polygon(img, v, nv, H, W, r, g, b):
    """Fill a convex integer polygon ``v`` (rows of x, y) with SDL_gfx scanline rules."""
    maxy = v[0, 1]
    miny = v[0, 1]
    for e in range(1, nv):
        maxy = max(maxy, v[e, 1])
        miny = min(miny, v[e, 1])
    for Y in range(max(miny, 0), min(maxy, H - 1) + 1):
        xlo = BIG
        xhi = -BIG
        nact = 0
        for e in range(nv):
            j = (e + nv - 1) % nv
            ax = v[j, 0]
            ay = v[j, 1]
            bx = v[e, 0]
            by = v[e, 1]
            if ay == by:
                continue
            ylo = min(ay, by)
            yhi = max(ay, by)
            if (Y >= ylo and Y < yhi) or (Y == maxy and Y > ylo and Y <= yhi):
                xl = bx if ay > by else ax
                xh = ax if ay > by else bx
                X = ((65536 * (Y - ylo)) // max(yhi - ylo, 1)) * (xh - xl) + 65536 * xl
                xlo = min(xlo, X)
                xhi = max(xhi, X)
                nact += 1
        if nact >= 2:
            xa = max(round16(xlo + 1), 0)
            xb = min(round16(xhi - 1), W - 1)
            for x in range(xa, xb + 1):
                put(img, H - 1 - Y, x, r, g, b)
