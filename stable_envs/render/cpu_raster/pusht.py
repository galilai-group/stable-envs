"""Numba CPU twin of the PushT rasterizer: the same primitive records as ``raster.draw``, one frame per thread."""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from ...envs.pusht.raster import (
    BX0,
    BX1,
    BY0,
    BY1,
    CANVAS,
    CIRCLE,
    COLOR,
    CR,
    CX,
    CY,
    KIND,
    LDX,
    LDY,
    LERR,
    LINE,
    LN,
    LSX,
    LSY,
    LX,
    LXINC,
    LY,
    PN,
    POLY,
    PX,
    PY,
    STATIC,
    circle_table,
    resize_coefficients,
)

_BIG = np.iinfo(np.int32).max


@njit(cache=True)
def _floordiv(a, b):
    q = a // b
    return q


@njit(cache=True)
def _truncdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


@njit(cache=True)
def _poly_row(rec, y, hits):
    """Fill ``hits`` with the sorted edge crossings of row ``y``; return (count, miny, maxy, minx, maxx)."""
    n = min(max(rec[PN] if rec[KIND] == POLY else 0, 0), 5)
    miny, maxy, minx, maxx = _BIG, -_BIG, _BIG, -_BIG
    for i in range(n):
        miny = min(miny, rec[PY + i])
        maxy = max(maxy, rec[PY + i])
        minx = min(minx, rec[PX + i])
        maxx = max(maxx, rec[PX + i])
    count = 0
    for i in range(n):
        p = (i - 1) % max(n, 1)
        xa, ya, xs, ys = rec[PX + p], rec[PY + p], rec[PX + i], rec[PY + i]
        if ya == ys:
            continue
        if ya < ys:
            y1, y2, x1, x2 = ya, ys, xa, xs
        else:
            y1, y2, x1, x2 = ys, ya, xs, xa
        if (y1 <= y < y2) or (y == maxy and y2 == maxy):
            den = y2 - y1
            hits[count] = _truncdiv((y - y1) * (x2 - x1) + x1 * den, den)
            count += 1
    for i in range(1, count):
        v = hits[i]
        j = i - 1
        while j >= 0 and hits[j] > v:
            hits[j + 1] = hits[j]
            j -= 1
        hits[j + 1] = v
    return count, miny, maxy, minx, maxx


@njit(cache=True)
def _fill(color, r, taps, lo, hi, c0, c1, value):
    """Set the sampled columns of row ``r`` whose canvas x lies in ``[lo, hi]`` and in ``[c0, c1)``."""
    a = max(np.searchsorted(taps, lo), c0)
    b = min(np.searchsorted(taps, hi, side="right"), c1)
    for c in range(a, b):
        color[r, c] = value


@njit(cache=True)
def _paint(color, records, p0, p1, taps, table, max_radius, hits):
    """Paint records ``p0 <= p < p1`` of one frame into the sampled canvas ``color`` in drawing order."""
    for p in range(p0, p1):
        rec = records[p]
        kind = rec[KIND]
        if kind != POLY and kind != CIRCLE and kind != LINE:
            continue
        value = rec[COLOR]
        # Taps are sorted, so the box covers one contiguous range of rows and of columns.
        r0, r1 = np.searchsorted(taps, rec[BY0]), np.searchsorted(taps, rec[BY1], side="right")
        c0, c1 = np.searchsorted(taps, rec[BX0]), np.searchsorted(taps, rec[BX1], side="right")
        if kind == LINE and rec[LXINC] == 0:
            # Column runs: each column holds a vertical run of four pixels.
            lx, ly, sx, sy, dx, dy, err, steps = (
                rec[LX],
                rec[LY],
                rec[LSX],
                rec[LSY],
                rec[LDX],
                rec[LDY],
                rec[LERR],
                rec[LN],
            )
            for c in range(c0, c1):
                k = (taps[c] - lx) * sx
                if k < 0 or k > steps:
                    continue
                yk = ly + sy * _floordiv(k * dy - err + dx - 1, max(dx, 1))
                a = max(np.searchsorted(taps, yk - 1), r0)
                b = min(np.searchsorted(taps, yk + 2, side="right"), r1)
                for r in range(a, b):
                    color[r, c] = value
            continue
        for r in range(r0, r1):
            y = taps[r]
            if kind == POLY:
                count, miny, maxy, minx, maxx = _poly_row(rec, y, hits)
                if miny == maxy:
                    if y == miny:
                        _fill(color, r, taps, minx, maxx, c0, c1, value)
                    continue
                if miny <= y <= maxy:
                    if count >= 2:
                        _fill(color, r, taps, hits[0], hits[1], c0, c1, value)
                    if count >= 4:
                        _fill(color, r, taps, hits[2], hits[3], c0, c1, value)
                n = min(max(rec[PN], 0), 5)
                for e in range(n):
                    q = (e - 1) % max(n, 1)
                    xa, ya, xs, ys = rec[PX + q], rec[PY + q], rec[PX + e], rec[PY + e]
                    if ya == ys and miny < ys < maxy and y == ys:
                        _fill(color, r, taps, min(xa, xs), max(xa, xs), c0, c1, value)
            elif kind == CIRCLE:
                dy = y - rec[CY]
                if -max_radius - 1 <= dy <= max_radius:
                    row = min(max(dy + max_radius + 1, 0), 2 * max_radius + 1)
                    rad = min(max(rec[CR], 0), max_radius)
                    _fill(color, r, taps, rec[CX] + table[rad, row, 0], rec[CX] + table[rad, row, 1], c0, c1, value)
            else:
                k = (y - rec[LY]) * rec[LSY]
                if 0 <= k <= rec[LN]:
                    dy = rec[LDY]
                    nk = _floordiv(k * rec[LDX] + rec[LERR] + dy - 1, max(dy, 1))
                    xk = rec[LX] + rec[LSX] * (nk if dy > 0 else 0)
                    _fill(color, r, taps, xk - 1, xk + 2, c0, c1, value)


@njit(parallel=True, cache=True)
def _plates(records, background, first, taps, table, max_radius, out):
    for u in prange(out.shape[0]):
        out[u, :, :] = background[u]
        _paint(out[u], records[u], 0, first[u], taps, table, max_radius, np.empty(5, np.int64))


@njit(parallel=True, cache=True)
def _frames(records, plates, plate, first, wx, wy, wb, wc, wn, taps, position, table, max_radius, i0, i1, w0, w1, out):
    n, P = records.shape[0], records.shape[1]
    S = out.shape[1]
    for f in prange(n):
        color = plates[plate[f]].copy()
        _paint(color, records[f], first[f], P, taps, table, max_radius, np.empty(5, np.int64))
        one = np.float32(1.0)
        for w in range(wn[f]):
            r, c = position[wy[f, w]], position[wx[f, w]]
            if r < 0 or c < 0:
                continue
            old, new, b = color[r, c], wc[f, w], np.float32(wb[f, w])
            v = 0
            for shift in (16, 8, 0):
                fg = np.float32((new >> shift) & 255)
                bg = np.float32((old >> shift) & 255)
                value = int(np.float32(b * fg) + np.float32((one - b) * bg))
                v |= min(max(value, 0), 255) << shift
            color[r, c] = v
        for oy in range(S):
            ra, rb, wa, wb_ = i0[oy], i1[oy], w0[oy], w1[oy]
            for ox in range(S):
                ca, cb, xa, xb = i0[ox], i1[ox], w0[ox], w1[ox]
                p00, p01, p10, p11 = color[ra, ca], color[ra, cb], color[rb, ca], color[rb, cb]
                for ch in range(3):
                    shift = 16 - 8 * ch
                    h0 = ((p00 >> shift) & 255) * xa + ((p01 >> shift) & 255) * xb
                    h1 = ((p10 >> shift) & 255) * xa + ((p11 >> shift) & 255) * xb
                    v = (((h0 >> 4) * wa) >> 16) + (((h1 >> 4) * wb_) >> 16)
                    out[f, oy, ox, ch] = min(max((v + 2) >> 2, 0), 255)


def draw(records, background, writes, size):
    """Return ``[N, size, size, 3]`` uint8 frames for host ``records`` and ``writes``, equal to ``raster.draw``.

    The leading run of static records (the goal, and walls drawn before any body) is drawn once per
    distinct background and static records, then copied into each frame.
    """
    s, a0, a1 = resize_coefficients(CANVAS, size)
    taps = np.unique(np.concatenate([s, s + 1])).astype(np.int64)
    position = np.full(CANVAS, -1, np.int64)
    position[taps] = np.arange(len(taps))
    max_radius = max(2, int(records[..., CR].max(initial=0)) if len(records) else 2)
    table = np.ascontiguousarray(circle_table(max_radius), dtype=np.int64)
    wx, wy, wb, wc, wn = writes
    out = np.empty((len(records), size, size, 3), np.uint8)
    if not len(records):
        return out
    records = np.ascontiguousarray(records, dtype=np.int64)
    background = np.ascontiguousarray(background, dtype=np.int64).reshape(-1)
    dynamic = (records[..., KIND] != 0) & (records[..., STATIC] == 0)
    first = np.where(dynamic.any(1), dynamic.argmax(1), records.shape[1]).astype(np.int64)
    keys, plate, reps = {}, np.empty(len(records), np.int64), []
    for f in range(len(records)):
        key = (int(background[f]), records[f, : first[f]].tobytes())
        if key not in keys:
            keys[key] = len(reps)
            reps.append(f)
        plate[f] = keys[key]
    reps = np.asarray(reps, np.int64)
    plates = np.empty((len(reps), len(taps), len(taps)), np.int32)
    _plates(records[reps], background[reps], first[reps], taps, table, max_radius, plates)
    _frames(
        records,
        plates,
        plate,
        first,
        np.ascontiguousarray(wx, dtype=np.int64),
        np.ascontiguousarray(wy, dtype=np.int64),
        np.ascontiguousarray(wb, dtype=np.float32),
        np.ascontiguousarray(wc, dtype=np.int64),
        np.ascontiguousarray(wn, dtype=np.int64),
        taps,
        position,
        table,
        max_radius,
        position[s],
        position[s + 1],
        a0.astype(np.int64),
        a1.astype(np.int64),
        out,
    )
    return out
