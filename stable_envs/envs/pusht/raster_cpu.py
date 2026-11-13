"""Numba CPU PushT rasterizer: the records of ``raster.primitives`` drawn as ``draw_cuda`` draws them, one frame per thread."""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange, types

from .raster import (
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
    RECORD,
    STATIC,
    circle_table,
    resize_coefficients,
)

_BIG = np.iinfo(np.int32).max
# CPython's math.cos and math.sin are libm's; Numba's own lowering misses them in the last bit for some angles.
_cos = types.ExternalFunction("cos", types.float64(types.float64))
_sin = types.ExternalFunction("sin", types.float64(types.float64))


@njit(cache=True)
def _sign(v, otherwise):
    return 1 if v > 0 else (-1 if v < 0 else otherwise)


@njit(cache=True)
def _inside(px, py):
    return 0 <= px < CANVAS and 0 <= py < CANVAS


@njit(cache=True)
def _offset(v):
    return int(v - 0.5) if v < 0 else int(v + 0.5)


@njit(cache=True)
def clipped_line(x1, y1, x2, y2):
    # pygame keeps the Bresenham error of the whole width-4 line, restarts it at the clipped start, and
    # continues past the clipped end while a span still touches the canvas; steps is -1 when it misses.
    xinc = abs(x1 - x2) <= abs(y1 - y2)
    dx, sx = abs(x2 - x1), (1 if x1 < x2 else -1)
    dy, sy = abs(y2 - y1), (1 if y1 < y2 else -1)
    err = dx if dx > dy else -dy
    err = err // 2 if err >= 0 else -((-err) // 2)
    end_x, end_y = x2, y2
    p1 = x1 - x2
    p2, p3 = -p1, y1 - y2
    p4 = -p3
    q1, q2, q3, q4 = x1, CANVAS - x1, y1, CANVAS - y1
    if (p1 == 0 and q1 < 0) or (p2 == 0 and q2 < 0) or (p3 == 0 and q3 < 0) or (p4 == 0 and q4 < 0):
        return x1, y1, -1
    nmax, pmin = 0.0, 1.0
    for p, q, pp, qq in ((p1, q1, p2, q2), (p3, q3, p4, q4)):
        if p:
            r1, r2 = q / p, qq / pp
            if p < 0:
                nmax, pmin = max(nmax, r1), min(pmin, r2)
            else:
                nmax, pmin = max(nmax, r2), min(pmin, r1)
    if nmax > pmin:
        return x1, y1, -1
    x1, y1, x2, y2 = x1 + _offset(p2 * nmax), y1 + _offset(p4 * nmax), x1 + _offset(p2 * pmin), y1 + _offset(p4 * pmin)
    x, y = x1, y1
    lt = (x1 if xinc else y1) - 1
    rb = (x1 if xinc else y1) + 2
    steps = 0
    while _sign(x1 - x2, sx) != sx or _sign(y1 - y2, sy) != sy:
        steps += 1
        e2 = err
        if e2 > -dx:
            err -= dy
            x1 += sx
            if xinc:
                lt, rb = lt + sx, rb + sx
        if e2 < dy:
            err += dx
            y1 += sy
            if not xinc:
                lt, rb = lt + sy, rb + sy
    if xinc:
        while y1 != end_y and (_inside(lt, y1) or _inside(rb, y1)):
            steps += 1
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
                lt, rb = lt + sx, rb + sx
            if e2 < dy:
                err += dx
                y1 += sy
    else:
        while x1 != end_x and (_inside(x1, lt) or _inside(x1, rb)):
            steps += 1
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
            if e2 < dy:
                err += dx
                y1 += sy
                lt, rb = lt + sy, rb + sy
    return x, y, steps


@njit(cache=True)
def _line(rec, color, x1, y1, x2, y2):
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    xinc = int(dx <= dy)
    sx, sy, steps = x1, y1, max(dx, dy)
    rec[KIND], rec[COLOR] = LINE, color
    if min(x1, x2) < 0 or max(x1, x2) >= CANVAS or min(y1, y2) < 0 or max(y1, y2) >= CANVAS:
        sx, sy, run = clipped_line(x1, y1, x2, y2)
        if run < 0:
            rec[KIND] = 0
        else:
            steps = run
        # Each step of a clipped run moves the minor axis at most one pixel.
        rec[BX0], rec[BX1], rec[BY0], rec[BY1] = sx - steps - 1, sx + steps + 2, sy - steps - 1, sy + steps + 2
    else:
        rec[BX0], rec[BX1] = min(x1, x2) - xinc, max(x1, x2) + 2 * xinc
        rec[BY0], rec[BY1] = min(y1, y2) - 1 + xinc, max(y1, y2) + 2 - 2 * xinc
    rec[LX], rec[LY], rec[LSX], rec[LSY] = sx, sy, 1 if x1 < x2 else -1, 1 if y1 < y2 else -1
    rec[LDX], rec[LDY], rec[LN], rec[LXINC] = dx, dy, steps, xinc
    rec[LERR] = dx // 2 if dx > dy else -(dy // 2)


@njit(cache=True)
def _poly(rec, color, n):
    rec[KIND], rec[COLOR], rec[PN] = POLY, color, n
    rec[BX0], rec[BX1] = min(rec[PX : PX + n]), max(rec[PX : PX + n])
    rec[BY0], rec[BY1] = min(rec[PY : PY + n]), max(rec[PY : PY + n])


@njit(cache=True)
def _circle(rec, color, x, y, r):
    rec[KIND], rec[COLOR] = CIRCLE if r >= 1 else 0, color
    rec[BX0], rec[BY0], rec[BX1], rec[BY1] = x - r, y - r, x + r, y + r
    rec[CX], rec[CY], rec[CR] = x, y, r


@njit(cache=True)
def fat_segment(rec, color, x1, y1, x2, y2):
    _line(rec[0], color, x1, y1, x2, y2)
    o0, o1 = abs(y2 - y1), abs(x2 - x1)
    drawn = o0 != 0 or o1 != 0
    # DrawOptions scales by 2 / pow(n, 0.5); rounded, that equals the sqrt result, as o * 2 / sqrt(n) never lies
    # within rounding error of a half.
    scale = 2.0 / math.sqrt(o0 * o0 + o1 * o1) if drawn else 2.0
    q0, q1 = int(np.rint(o0 * scale)), int(np.rint(o1 * scale))
    xs, ys = (x1 - q0, x1 + q0, x2 + q0, x2 - q0), (y1 - q1, y1 + q1, y2 + q1, y2 - q1)
    for k in range(4):
        rec[1, PX + k], rec[1, PY + k] = xs[k], ys[k]
    _poly(rec[1], color, 4)
    _circle(rec[2], color, x1, y1, 2)
    _circle(rec[3], color, x2, y2, 2)
    if not drawn:
        rec[1:, KIND] = 0


@njit(cache=True)
def _transform(pose):
    # Setting the position stores the center of gravity R cog + position; the transform subtracts R cog
    # again, which can differ from the position in the last bit.
    px, py, cogx, cogy = pose[0], pose[1], pose[2], pose[3]
    cos, sin = _cos(pose[4]), _sin(pose[4])
    gx = (cos * cogx + -sin * cogy) + px
    gy = (sin * cogx + cos * cogy) + py
    return cos, sin, gx - (cogx * cos - cogy * sin), gy - (cogx * sin + cogy * cos)


@njit(cache=True)
def _build_one(f, rec, pose, color, light, goal, shapes, owners, starts, discs, walls, wall_slots):
    rec[:] = 0
    for m in range(len(shapes)):
        b, r = owners[m], rec[starts[m]]
        cos, sin, tx, ty = _transform(pose[f, b])
        for k in range(5):
            x, y = shapes[m, k % 4, 0], shapes[m, k % 4, 1]
            r[PX + k], r[PY + k] = int(np.rint(cos * x + -sin * y + tx)), int(np.rint(sin * x + cos * y + ty))
        _poly(r, light[f, b] if b else color[f, 0], 5)
        if b == 0:
            r[KIND], r[STATIC] = POLY * goal[f], 1
            continue
        for e in range(4):
            s = starts[m] + 1 + 4 * e
            fat_segment(rec[s : s + 4], color[f, b], r[PX + e], r[PY + e], r[PX + e + 1], r[PY + e + 1])
    for d in range(len(discs)):
        b, s = int(discs[d, 0]), int(discs[d, 5])
        cos, sin, tx, ty = _transform(pose[f, b])
        x, y = discs[d, 1], discs[d, 2]
        cx, cy = int(np.rint(cos * x + -sin * y + tx)), int(np.rint(sin * x + cos * y + ty))
        _circle(rec[s], color[f, b], cx, cy, int(discs[d, 3]))
        _circle(rec[s + 1], light[f, b], cx, cy, int(discs[d, 4]))
    for w in range(len(wall_slots)):
        for k in range(RECORD):
            rec[wall_slots[w], k] = walls[w, k]


@njit(parallel=True, cache=True)
def build(lanes, pose, color, light, goal, shapes, owners, starts, discs, walls, wall_slots, out):
    for i in prange(len(lanes)):
        f = lanes[i]
        _build_one(f, out[f], pose, color, light, goal, shapes, owners, starts, discs, walls, wall_slots)


@njit(cache=True)
def _put(w, n, steep, x, y, brightness, color):
    px, py = (y, x) if steep else (x, y)
    if _inside(px, py):
        if n < len(w[0]):
            w[0][n], w[1][n], w[2][n], w[3][n] = px, py, brightness, color
        n += 1
    return n


@njit(cache=True)
def _aaline(w, n, segment):
    # pygame's draw_aaline: float32 throughout, every (int) cast truncates.
    one, lo, hi = np.float32(1), np.float32(-1), np.float32(CANVAS)
    fx, fy = np.float32(np.rint(segment[0])), np.float32(np.rint(segment[1]))
    tx, ty = np.float32(np.rint(segment[2])), np.float32(np.rint(segment[3]))
    color = (int(np.rint(segment[4])) << 16) | (int(np.rint(segment[5])) << 8) | int(np.rint(segment[6]))
    dx, dy = tx - fx, ty - fy
    steep = abs(dx) < abs(dy)
    if abs(dx) < 0.0001 and abs(dy) < 0.0001:
        return _put(w, n, steep, int(np.floor(fx + 0.5)), int(np.floor(fy + 0.5)), one, color)
    if steep:
        fx, fy, tx, ty, dx, dy = fy, fx, ty, tx, dy, dx
    if dx < 0:
        fx, fy, tx, ty, dx, dy = tx, ty, fx, fy, -dx, -dy
    if tx <= lo or fx >= hi:
        return n
    g = dy / dx
    if fx < lo:
        fx, fy = lo, fy + g * (lo - fx)
    if tx > hi:
        tx, ty = hi, ty + g * (hi - tx)
    if g > 0:
        if ty <= lo or fy >= hi:
            return n
        if fy < lo:
            fx, fy = fx + (lo - fy) / g, lo
        if ty > hi:
            tx, ty = tx + (hi - ty) / g, hi
    else:
        if fy <= lo or ty >= hi:
            return n
        if ty < lo:
            tx, ty = tx + (lo - ty) / g, lo
        if fy > hi:
            fx, fy = fx + (hi - fy) / g, hi
    fy, ty = fy + one, ty + one
    xs = int(fx)
    iy = ye = fy + g * (np.float32(xs) - fx)
    if tx > lo + one:
        gap = np.float32(1 + xs) - fx
        yi = int(ye)
        br = ye - np.float32(yi)
        if np.float32(yi) < ye:
            n = _put(w, n, steep, xs, yi, br * gap, color)
        n = _put(w, n, steep, xs, yi - 1, (one - br) * gap, color)
        iy, xs = iy + g, xs + 1
    xe = int(np.ceil(tx))
    if fx < hi - one:
        ye = ty + g * (np.float32(xe) - tx)
        gap = np.float32(1 - xe) + tx
        yi = int(ye)
        br = ye - np.float32(yi)
        if np.float32(yi) < ye:
            n = _put(w, n, steep, xe, yi, br * gap, color)
        n = _put(w, n, steep, xe, yi - 1, (one - br) * gap, color)
    for x in range(xs, xe):
        y = int(iy)
        br = (one - iy) + np.float32(y)
        n = _put(w, n, steep, x, y - 1, br, color)
        if np.float32(y) < iy:
            n = _put(w, n, steep, x, y, one - br, color)
        iy += g
    return n


@njit(parallel=True, cache=True)
def contacts(segments, counts, w, total):
    for f in prange(len(counts)):
        total[f], row = 0, (w[0][f], w[1][f], w[2][f], w[3][f])
        for k in range(min(counts[f], segments.shape[1])):
            total[f] = _aaline(row, total[f], segments[f, k])


@njit(cache=True)
def _truncdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


@njit(cache=True)
def _poly_row(rec, y, hits):
    """Fill ``hits`` with the sorted edge crossings of row ``y`` of a polygon record; return their count."""
    n, maxy = min(max(rec[PN], 0), 5), rec[BY1]
    count = 0
    for i in range(n):
        p = i - 1 if i else n - 1
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
    return count


@njit(cache=True)
def _tap(edge, x):
    return edge[min(max(x, 0), CANVAS)]


@njit(cache=True)
def _fill(color, r, edge, lo, hi, c0, c1, value):
    """Set the sampled columns of row ``r`` whose canvas x lies in ``[lo, hi]`` and in ``[c0, c1)``."""
    for c in range(max(_tap(edge, lo), c0), min(_tap(edge, hi + 1), c1)):
        color[r, c] = value


@njit(cache=True)
def _paint(color, records, rows, cols, taps, edge, table):
    """Paint the ``records`` of one frame into ``rows`` and ``cols`` of ``color`` in drawing order."""
    max_radius, hits = table.shape[0] - 1, np.empty(5, np.int64)
    for rec in records:
        kind = rec[KIND]
        if kind != POLY and kind != CIRCLE and kind != LINE:
            continue
        value = rec[COLOR]
        # Taps are sorted, so the box covers one contiguous range of rows and of columns.
        r0, r1 = max(_tap(edge, rec[BY0]), rows.start), min(_tap(edge, rec[BY1] + 1), rows.stop)
        c0, c1 = max(_tap(edge, rec[BX0]), cols.start), min(_tap(edge, rec[BX1] + 1), cols.stop)
        if r0 >= r1 or c0 >= c1:
            continue
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
                yk = ly + sy * ((k * dy - err + dx - 1) // max(dx, 1))
                for r in range(max(_tap(edge, yk - 1), r0), min(_tap(edge, yk + 3), r1)):
                    color[r, c] = value
            continue
        for r in range(r0, r1):
            y = taps[r]
            if kind == POLY:
                # A polygon's box is the bounds of its points.
                miny, maxy = rec[BY0], rec[BY1]
                if miny == maxy:
                    _fill(color, r, edge, rec[BX0], rec[BX1], c0, c1, value)
                    continue
                count = _poly_row(rec, y, hits)
                if count >= 2:
                    _fill(color, r, edge, hits[0], hits[1], c0, c1, value)
                if count >= 4:
                    _fill(color, r, edge, hits[2], hits[3], c0, c1, value)
                n = min(max(rec[PN], 0), 5)
                for e in range(n):
                    q = e - 1 if e else n - 1
                    xa, ya, xs, ys = rec[PX + q], rec[PY + q], rec[PX + e], rec[PY + e]
                    if ya == ys and miny < ys < maxy and y == ys:
                        _fill(color, r, edge, min(xa, xs), max(xa, xs), c0, c1, value)
            elif kind == CIRCLE:
                dy = y - rec[CY]
                if -max_radius - 1 <= dy <= max_radius:
                    row = min(max(dy + max_radius + 1, 0), 2 * max_radius + 1)
                    rad = min(max(rec[CR], 0), max_radius)
                    _fill(color, r, edge, rec[CX] + table[rad, row, 0], rec[CX] + table[rad, row, 1], c0, c1, value)
            else:
                k = (y - rec[LY]) * rec[LSY]
                if 0 <= k <= rec[LN]:
                    dy = rec[LDY]
                    nk = (k * rec[LDX] + rec[LERR] + dy - 1) // max(dy, 1)
                    xk = rec[LX] + rec[LSX] * (nk if dy > 0 else 0)
                    _fill(color, r, edge, xk - 1, xk + 2, c0, c1, value)


@njit(cache=True)
def _differ(a, b):
    for p in range(a.shape[0]):
        if a[p, STATIC] != b[p, STATIC]:
            return True
        if a[p, STATIC] != 0:
            for i in range(a.shape[1]):
                if a[p, i] != b[p, i]:
                    return True
    return False


@njit(cache=True)
def _group(records, background, plate):
    reps, k = np.empty(len(records), np.int64), 0
    for f in range(len(records)):
        j = 0
        while j < k and (background[reps[j]] != background[f] or _differ(records[reps[j]], records[f])):
            j += 1
        if j == k:
            reps[k], k = f, k + 1
        plate[f] = j
    return reps[:k]


@njit(cache=True)
def _resize(color, i0, i1, w0, w1, rows, cols, out):
    for oy in rows:
        ra, rb, wa, wb = color[i0[oy]], color[i1[oy]], w0[oy], w1[oy]
        for ox in cols:
            ca, cb, xa, xb = i0[ox], i1[ox], w0[ox], w1[ox]
            p00, p01, p10, p11 = ra[ca], ra[cb], rb[ca], rb[cb]
            # The weights of each axis sum to 2048, so equal taps resize to their own color.
            if p00 == p01 == p10 == p11:
                for ch in range(3):
                    out[oy, ox, ch] = p00 >> (16 - 8 * ch)
                continue
            for ch in range(3):
                shift = 16 - 8 * ch
                h0 = ((p00 >> shift) & 255) * xa + ((p01 >> shift) & 255) * xb
                h1 = ((p10 >> shift) & 255) * xa + ((p11 >> shift) & 255) * xb
                out[oy, ox, ch] = ((((h0 >> 4) * wa) >> 16) + (((h1 >> 4) * wb) >> 16) + 2) >> 2


@njit(cache=True)
def _outputs(edge, i0, i1, lo, hi):
    return np.searchsorted(i1, _tap(edge, lo)), np.searchsorted(i0, _tap(edge, hi + 1))


@njit(parallel=True, cache=True)
def _frames(records, background, statics, plate, wx, wy, wb, wc, wn, taps, edge, position, table, i0, i1, w0, w1, out):
    S = out.shape[1]
    for f in prange(len(records)):
        oy0, oy1, ox0, ox1 = 0, S, 0, S
        if plate[f] >= 0:
            src, dst = statics[plate[f]].reshape(-1), out[f].reshape(-1)
            for i in range(dst.size):
                dst[i] = src[i]
            x0, y0, x1, y1 = _BIG, _BIG, -_BIG, -_BIG
            for rec in records[f]:
                if rec[KIND] != 0 and rec[STATIC] == 0:
                    x0, y0, x1, y1 = min(x0, rec[BX0]), min(y0, rec[BY0]), max(x1, rec[BX1]), max(y1, rec[BY1])
            for w in range(wn[f]):
                x0, y0, x1, y1 = min(x0, wx[f, w]), min(y0, wy[f, w]), max(x1, wx[f, w]), max(y1, wy[f, w])
            oy0, oy1 = _outputs(edge, i0, i1, y0, y1)
            ox0, ox1 = _outputs(edge, i0, i1, x0, x1)
            if oy0 >= oy1 or ox0 >= ox1:
                continue
        ra, rb, ca, cb = i0[oy0], i1[oy1 - 1] + 1, i0[ox0], i1[ox1 - 1] + 1
        color = np.empty((len(taps), len(taps)), np.int32)
        for r in range(ra, rb):
            for c in range(ca, cb):
                color[r, c] = background[f]
        _paint(color, records[f], range(ra, rb), range(ca, cb), taps, edge, table)
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
        _resize(color, i0, i1, w0, w1, range(oy0, oy1), range(ox0, ox1), out[f])


def draw(records, background, writes, out):
    """Draw the frames of host ``records`` and ``writes`` into uint8 ``out [N, size, size, 3]``.

    The static records (goal and walls) are drawn and resized once per distinct background and static
    records; each frame copies that frame and redraws only the output pixels its moving records reach.
    """
    s, a0, a1 = resize_coefficients(CANVAS, out.shape[1])
    taps = np.unique(np.concatenate([s, s + 1]))
    position = np.full(CANVAS, -1, np.int64)
    position[taps] = np.arange(len(taps))
    edge = np.searchsorted(taps, np.arange(CANVAS + 1))
    table = circle_table(max(2, int(records[..., CR].max(initial=0, where=records[..., KIND] == CIRCLE))))
    plate = np.empty(len(records), np.int64)
    reps = _group(records, background, plate)
    statics = np.empty((len(reps), *out.shape[1:]), np.uint8)
    i0, i1 = position[s], position[s + 1]
    fixed = np.where(records[reps, :, STATIC, None] != 0, records[reps], 0)
    args, whole = (taps, edge, position, table, i0, i1, a0, a1), np.full(len(reps), -1)
    _frames(fixed, background[reps], statics, whole, *writes[:4], np.zeros_like(whole), *args, statics)
    _frames(records, background, statics, plate, *writes, *args, out)


@njit(cache=True)
def _area(x0, y0, x1, y1):
    w = min(x1, CANVAS - 1) - max(x0, 0) + 1
    h = min(y1, CANVAS - 1) - max(y0, 0) + 1
    return max(w, 0) * max(h, 0)


@njit(cache=True)
def _merge(boxes, count):
    """Move the nonempty boxes first and join two of them while their joint box covers at most their two areas.

    Return the number of boxes left.
    """
    m = 0
    for g in range(count):
        if boxes[g, 0] <= boxes[g, 2] and boxes[g, 1] <= boxes[g, 3]:
            for k in range(4):
                boxes[m, k] = boxes[g, k]
            m += 1
    joined = True
    while joined:
        joined = False
        for a in range(m):
            for b in range(a + 1, m):
                x0, y0 = min(boxes[a, 0], boxes[b, 0]), min(boxes[a, 1], boxes[b, 1])
                x1, y1 = max(boxes[a, 2], boxes[b, 2]), max(boxes[a, 3], boxes[b, 3])
                apart = _area(boxes[a, 0], boxes[a, 1], boxes[a, 2], boxes[a, 3])
                apart += _area(boxes[b, 0], boxes[b, 1], boxes[b, 2], boxes[b, 3])
                if _area(x0, y0, x1, y1) <= apart:
                    boxes[a, 0], boxes[a, 1], boxes[a, 2], boxes[a, 3] = x0, y0, x1, y1
                    for k in range(4):
                        boxes[b, k] = boxes[m - 1, k]
                    m -= 1
                    joined = True
                    break
            if joined:
                break
    return m


@njit(cache=True)
def _grow(boxes, g, x0, y0, x1, y1):
    boxes[g, 0], boxes[g, 1] = min(boxes[g, 0], x0), min(boxes[g, 1], y0)
    boxes[g, 2], boxes[g, 3] = max(boxes[g, 2], x1), max(boxes[g, 3], y1)


@njit(cache=True)
def _blend(color, r, c, new, brightness):
    # pygame's blend: float32 products, summed in float32 and truncated.
    one, b, old, v = np.float32(1.0), np.float32(brightness), color[r, c], 0
    for shift in (16, 8, 0):
        fg = np.float32((new >> shift) & 255)
        bg = np.float32((old >> shift) & 255)
        value = int(np.float32(b * fg) + np.float32((one - b) * bg))
        v |= min(max(value, 0), 255) << shift
    color[r, c] = v


@njit(cache=True)
def _region(canvas, rec, writes, count, background, oy0, oy1, ox0, ox1, tables, frame):
    """Draw output rows ``[oy0, oy1)`` and columns ``[ox0, ox1)`` of a frame: paint, blend ``count`` writes, resize."""
    taps, edge, position, table, i0, i1, w0, w1 = tables
    ra, rb, ca, cb = i0[oy0], i1[oy1 - 1] + 1, i0[ox0], i1[ox1 - 1] + 1
    for r in range(ra, rb):
        for c in range(ca, cb):
            canvas[r, c] = background
    _paint(canvas, rec, range(ra, rb), range(ca, cb), taps, edge, table)
    wx, wy, wb, wc = writes
    for w in range(count):
        r, c = position[wy[w]], position[wx[w]]
        if ra <= r < rb and ca <= c < cb:
            _blend(canvas, r, c, wc[w], wb[w])
    _resize(canvas, i0, i1, w0, w1, range(oy0, oy1), range(ox0, ox1), frame)


@njit(parallel=True, cache=True)
def frames(lanes, dest, scene, segments, segment_counts, plate, plates, layout, groups, tables, scratch, static, out):
    """Draw the frames of rows ``lanes`` into ``out[dest]``; one scratch row of ``scratch`` per thread.

    ``scene`` is ``(pose, color, light, goal, background)`` per row, ``layout`` the arrays of one template,
    ``groups`` the box group of each record slot (−1 for static slots), and ``tables`` the canvas taps and resize
    weights of the output size. A row with ``plate`` ≥ 0 copies ``plates[plate]``, the frame of its background,
    goal and walls alone, and redraws the output pixels of the boxes of its moving parts and contact lines. With
    ``static``, the frames hold the background, goal and walls only.
    """
    pose, color, light, goal, background = scene
    shapes, owners, starts, discs, walls, wall_slots = layout
    records, canvases, wx, wy, wb, wc = scratch
    edge, i0, i1 = tables[1], tables[4], tables[5]
    threads, n, size, other = len(records), len(lanes), out.shape[1], max(groups.max(), -1) + 1
    for t in prange(threads):
        rec, canvas = records[t], canvases[t]
        writes = (wx[t], wy[t], wb[t], wc[t])
        boxes = np.empty((other + 1, 4), np.int64)
        for i in range(t, n, threads):
            f = lanes[i]
            frame = out[dest[i]]
            _build_one(f, rec, pose, color, light, goal, shapes, owners, starts, discs, walls, wall_slots)
            count = 0
            if static:
                for p in range(len(rec)):
                    if rec[p, STATIC] == 0:
                        rec[p, KIND] = 0
            else:
                for k in range(min(segment_counts[f], segments.shape[1])):
                    count = _aaline(writes, count, segments[f, k])
            if plate[f] < 0:
                _region(canvas, rec, writes, count, background[f], 0, size, 0, size, tables, frame)
                continue
            src, dst = plates[plate[f]].reshape(-1), frame.reshape(-1)
            for k in range(dst.size):
                dst[k] = src[k]
            for g in range(other + 1):
                boxes[g, 0], boxes[g, 1], boxes[g, 2], boxes[g, 3] = _BIG, _BIG, -_BIG, -_BIG
            for p in range(len(rec)):
                if rec[p, KIND] != 0 and rec[p, STATIC] == 0:
                    g = groups[p] if p < len(groups) and groups[p] >= 0 else other
                    _grow(boxes, g, rec[p, BX0], rec[p, BY0], rec[p, BX1], rec[p, BY1])
            for w in range(count):
                _grow(boxes, other, wx[t][w], wy[t][w], wx[t][w], wy[t][w])
            for k in range(_merge(boxes, other + 1)):
                oy0, oy1 = _outputs(edge, i0, i1, boxes[k, 1], boxes[k, 3])
                ox0, ox1 = _outputs(edge, i0, i1, boxes[k, 0], boxes[k, 2])
                if oy0 < oy1 and ox0 < ox1:
                    _region(canvas, rec, writes, count, background[f], oy0, oy1, ox0, ox1, tables, frame)
