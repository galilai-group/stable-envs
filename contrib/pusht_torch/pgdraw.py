# This module reproduces the drawing algorithms of pygame 2.6.1 (src_c/draw.c: draw_fillpoly,
# draw_circle_filled, draw_line_width, clip_line, draw_aaline). pygame is distributed under the
# GNU Lesser General Public License, version 2.1 or later; this file is distributed under the same
# license. See https://github.com/pygame/pygame/blob/2.6.1/docs/LGPL.txt.
"""pygame.draw primitives as batched torch spans, pixel-identical to pygame 2.6.1.

Primitives become horizontal spans ``(frame, order, y, x0, x1)``. :meth:`SpanRaster.paint`
expands them to pixels and keeps, per pixel, the colour of the last primitive in draw order.
Antialiased lines read the canvas, so they are blended afterwards, in draw order.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import torch


def _cdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _sign(x, y):
    return 1 if x > 0 else (-1 if x < 0 else y)


def _hspan(spans, x1, y, x2, w, h):
    if y < 0 or y >= h:
        return
    if x2 < x1:
        x1, x2 = x2, x1
    x1, x2 = max(x1, 0), min(x2, w - 1)
    if x2 < 0 or x1 >= w:
        return
    spans.append((y, x1, x2))


def circle_filled(x0, y0, radius, w=1 << 30, h=1 << 30):
    """Spans of ``pygame.draw.circle(..., width=0)``."""
    spans = []
    f, ddf_x, ddf_y, x, y = 1 - radius, 0, -2 * radius, 0, radius
    while x < y:
        if f >= 0:
            y -= 1
            ddf_y += 2
            f += ddf_y
        x += 1
        ddf_x += 2
        f += ddf_x + 1
        if f >= 0:
            _hspan(spans, x0 - x, y0 + y - 1, x0 + x - 1, w, h)
            _hspan(spans, x0 - x, y0 - y, x0 + x - 1, w, h)
        _hspan(spans, x0 - y, y0 + x - 1, x0 + y - 1, w, h)
        _hspan(spans, x0 - y, y0 - x, x0 + y - 1, w, h)
    return spans


def _clip_line(x1, y1, x2, y2, w, h):
    p1, p3 = x1 - x2, y1 - y2
    p2, p4 = -p1, -p3
    q1, q2, q3, q4 = x1, w - x1, y1, h - y1
    nmax, pmin = 0.0, 1.0
    if (p1 == 0 and q1 < 0) or (p2 == 0 and q2 < 0) or (p3 == 0 and q3 < 0) or (p4 == 0 and q4 < 0):
        return None
    for p, q, pn, qn in ((p1, q1, p2, q2), (p3, q3, p4, q4)):
        if p:
            r1, r2 = q / p, qn / pn
            lo, hi = (r1, r2) if p < 0 else (r2, r1)
            nmax, pmin = max(nmax, lo) if lo > nmax else nmax, hi if hi < pmin else pmin
    if nmax > pmin:
        return None

    def rnd(v):
        return int(v - 0.5) if v < 0 else int(v + 0.5)

    return x1 + rnd(p2 * nmax), y1 + rnd(p4 * nmax), x1 + rnd(p2 * pmin), y1 + rnd(p4 * pmin)


def _thin_line(x1, y1, x2, y2, w, h):
    spans = []

    def point(x, y):
        if 0 <= x < w and 0 <= y < h:
            spans.append((y, x, x))

    if x1 == x2 and y1 == y2:
        point(x1, y1)
    elif y1 == y2:
        _hspan(spans, x1, y1, x2, w, h)
    elif x1 == x2:
        for y in range(min(y1, y2), max(y1, y2) + 1):
            point(x1, y)
    else:
        dx, sx = abs(x2 - x1), (1 if x1 < x2 else -1)
        dy, sy = abs(y2 - y1), (1 if y1 < y2 else -1)
        err = _cdiv(dx if dx > dy else -dy, 2)
        while x1 != x2 or y1 != y2:
            point(x1, y1)
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
            if e2 < dy:
                err += dx
                y1 += sy
        point(x2, y2)
    return spans


def line_width(x1, y1, x2, y2, width, w, h):
    """Spans of one ``pygame.draw.lines`` segment, including the clipping step."""
    spans = []
    end_x, end_y = x2, y2
    xinc = abs(x1 - x2) <= abs(y1 - y2)
    dx, sx = abs(x2 - x1), (1 if x1 < x2 else -1)
    dy, sy = abs(y2 - y1), (1 if y1 < y2 else -1)
    err = _cdiv(dx if dx > dy else -dy, 2)
    clipped = _clip_line(x1, y1, x2, y2, w, h)
    if clipped is None:
        return spans
    x1, y1, x2, y2 = clipped
    if width == 1:
        return _thin_line(x1, y1, x2, y2, w, h)

    def column(x, lo, hi):
        for y in range(lo, hi + 1):
            if 0 <= x < w and 0 <= y < h:
                spans.append((y, x, x))

    def inside(x, y):
        return 0 <= x < w and 0 <= y < h

    lt, rb = (x1, x1) if xinc else (y1, y1)
    lt, rb = lt - _cdiv(width - 1, 2), rb + _cdiv(width, 2)
    while _sign(x1 - x2, sx) != sx or _sign(y1 - y2, sy) != sy:
        if xinc:
            _hspan(spans, lt, y1, rb, w, h)
        else:
            column(x1, lt, rb)
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
        while y1 != end_y and (inside(lt, y1) or inside(rb, y1)):
            _hspan(spans, lt, y1, rb, w, h)
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
                lt, rb = lt + sx, rb + sx
            if e2 < dy:
                err += dx
                y1 += sy
        _hspan(spans, lt, y1, rb, w, h)
    else:
        while x1 != end_x and (inside(x1, lt) or inside(x1, rb)):
            column(x1, lt, rb)
            e2 = err
            if e2 > -dx:
                err -= dy
                x1 += sx
            if e2 < dy:
                err += dx
                y1 += sy
                lt, rb = lt + sy, rb + sy
        column(x1, lt, rb)
    return spans


def aaline_pixels(from_x, from_y, to_x, to_y, w, h):
    """``(x, y, brightness)`` written by ``pygame.draw.aalines`` (blend on), in order, float32 as in C."""
    f = np.float32
    from_x, from_y, to_x, to_y = f(from_x), f(from_y), f(to_x), f(to_y)
    out = []

    def put(x, y, b):
        if 0 <= x < w and 0 <= y < h:
            out.append((x, y, f(b)))

    dx, dy = f(to_x - from_x), f(to_y - from_y)
    if abs(float(dx)) < 0.0001 and abs(float(dy)) < 0.0001:
        put(int(math.floor(float(from_x) + 0.5)), int(math.floor(float(from_y) + 0.5)), f(1))
        return out
    clip_left = f(-1)
    clip_right = f(f(clip_left + f(w)) + f(1))
    clip_top = f(-1)
    clip_bottom = f(f(clip_top + f(h)) + f(1))
    steep = abs(float(dx)) < abs(float(dy))
    if steep:
        from_x, from_y, to_x, to_y, dx, dy = from_y, from_x, to_y, to_x, dy, dx
        clip_left, clip_top, clip_right, clip_bottom = clip_top, clip_left, clip_bottom, clip_right
    if dx < 0:
        from_x, to_x, from_y, to_y = to_x, from_x, to_y, from_y
        dx, dy = f(-dx), f(-dy)
    if to_x <= clip_left or from_x >= clip_right:
        return out
    gradient = f(dy / dx)
    if from_x < clip_left:
        from_y = f(from_y + f(gradient * f(clip_left - from_x)))
        from_x = clip_left
    if to_x > clip_right:
        to_y = f(to_y + f(gradient * f(clip_right - to_x)))
        to_x = clip_right
    if gradient > f(0):
        if to_y <= clip_top or from_y >= clip_bottom:
            return out
        if from_y < clip_top:
            from_x = f(from_x + f(f(clip_top - from_y) / gradient))
            from_y = clip_top
        if to_y > clip_bottom:
            to_x = f(to_x + f(f(clip_bottom - to_y) / gradient))
            to_y = clip_bottom
    else:
        if from_y <= clip_top or to_y >= clip_bottom:
            return out
        if to_y < clip_top:
            to_x = f(to_x + f(f(clip_top - to_y) / gradient))
            to_y = clip_top
        if from_y > clip_bottom:
            from_x = f(from_x + f(f(clip_bottom - from_y) / gradient))
            from_y = clip_bottom
    from_y, to_y = f(from_y + f(1)), f(to_y + f(1))

    def endpoint(xp, y_end, gap):
        b = f(y_end - f(int(y_end)))
        x, y = (int(y_end), xp) if steep else (xp, int(y_end))
        if f(int(y_end)) < y_end:
            put(x, y, f(b * gap))
        x, y = (x - 1, y) if steep else (x, y - 1)
        put(x, y, f(f(f(1) - b) * gap))

    x_start = int(from_x)
    y_end = intersect_y = f(from_y + f(gradient * f(f(x_start) - from_x)))
    if to_x > f(clip_left + f(1)):
        endpoint(x_start, y_end, f(f(f(1) + f(x_start)) - from_x))
        intersect_y = f(intersect_y + gradient)
        x_start += 1
    x_end = int(math.ceil(float(to_x)))
    if from_x < f(clip_right - f(1)):
        endpoint(x_end, f(to_y + f(gradient * f(f(x_end) - to_x))), f(f(f(1) - f(x_end)) + to_x))
    for x in range(x_start, x_end):
        y = int(intersect_y)
        b = f(f(f(1) - intersect_y) + f(y))
        put(*((y - 1, x) if steep else (x, y - 1)), b)
        if f(y) < intersect_y:
            put(*((y, x) if steep else (x, y)), f(f(1) - b))
        intersect_y = f(intersect_y + gradient)
    return out


@functools.cache
def _circle_rows(r):
    """Per row offset in [-r-1, r+1]: (left, right) offsets from the centre; (1, 0) when empty."""
    c = 1 << 20
    rows = {}
    for y, x0, x1 in circle_filled(c, c, r):
        lo, hi = rows.get(y - c, (x0 - c, x1 - c))
        rows[y - c] = (min(lo, x0 - c), max(hi, x1 - c))
    out = np.tile(np.array([[1, 0]], np.int64), (2 * r + 3, 1))
    for y, (lo, hi) in rows.items():
        out[y + r + 1] = (lo, hi)
    return out


class SpanRaster:
    """Batched span builders and painter for one canvas size."""

    def __init__(self, width, height, device):
        self.w, self.h = width, height
        self.device = torch.device(device)

    def _t(self, a, dtype=torch.int64):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.device)

    def circles(self, meta, params):
        """meta (N, 2) frame/order; params (N, 3) cx, cy, radius."""
        cx, cy, r = params[:, 0], params[:, 1], params[:, 2]
        rmax = int(r.max())
        n = 2 * rmax + 3
        table = np.tile(np.array([1, 0], np.int64), (len(params), n, 1))
        for ri in np.unique(r):
            if ri >= 1:
                table[r == ri, rmax - ri : rmax + ri + 3] = _circle_rows(int(ri))
        table = self._t(table)
        y = self._t(cy)[:, None] + (torch.arange(n, device=self.device) - (rmax + 1))[None]
        cx = self._t(cx)[:, None]
        return [(self._t(meta)[:, None, :].expand(-1, n, 2), y, table[..., 0] + cx, table[..., 1] + cx)]

    def polys(self, meta, px, py):
        """meta (N, 2); px, py (N, n) integer vertices in pygame order."""
        px, py = self._t(px), self._t(py)
        N, n = px.shape
        miny, maxy = py.min(1).values, py.max(1).values
        R = int((maxy - miny).max()) + 1
        rows = miny[:, None] + torch.arange(R, device=self.device)[None]
        y = rows[:, None, :].to(torch.float64)
        prev = torch.roll(torch.arange(n, device=self.device), 1)
        ya, yb, xa, xb = py[:, prev], py, px[:, prev], px
        swap = ya > yb
        y1 = torch.where(swap, yb, ya).to(torch.float64)[..., None]
        y2 = torch.where(swap, ya, yb).to(torch.float64)[..., None]
        x1 = torch.where(swap, xb, xa).to(torch.float64)[..., None]
        x2 = torch.where(swap, xa, xb).to(torch.float64)[..., None]
        top = maxy.view(N, 1, 1).to(torch.float64)
        edge = (ya != yb)[..., None]
        hit = edge & (((y >= y1) & (y < y2)) | ((y == top) & (y2 == top)))
        xi = torch.trunc((y - y1) * (x2 - x1) / torch.where(edge, y2 - y1, torch.ones_like(y2)) + x1).to(torch.int64)
        big = 1 << 40
        xi = torch.sort(torch.where(hit, xi, torch.full_like(xi, big)), dim=1).values
        flat = miny == maxy
        m = self._t(meta)
        out = []
        for i in range(0, n - 1, 2):
            a, b = xi[:, i], xi[:, i + 1]
            ok = (b < big) & ~flat[:, None]
            out.append((m[:, None, :].expand(-1, R, 2), rows, torch.where(ok, a, 1), torch.where(ok, b, 0)))
        for i in range(n):
            j = (i - 1) % n
            yy = py[:, i]
            on = (py[:, j] == yy) & (yy > miny) & (yy < maxy)
            lo, hi = torch.minimum(px[:, i], px[:, j]), torch.maximum(px[:, i], px[:, j])
            out.append((m[:, None, :], yy[:, None], torch.where(on, lo, 1)[:, None], torch.where(on, hi, 0)[:, None]))
        lo, hi = px.min(1).values, px.max(1).values
        out.append(
            (m[:, None, :], miny[:, None], torch.where(flat, lo, 1)[:, None], torch.where(flat, hi, 0)[:, None])
        )
        return out

    def lines(self, meta, params):
        """meta (N, 2); params (N, 5) x1, y1, x2, y2, width for segments inside the canvas."""
        x1, y1, x2, y2, w = (self._t(params[:, i]) for i in range(5))
        N = len(params)
        m = self._t(meta)
        dx, dy = (x2 - x1).abs(), (y2 - y1).abs()
        sx, sy = torch.where(x1 < x2, 1, -1), torch.where(y1 < y2, 1, -1)
        lo_off, hi_off = -((w - 1) // 2), w // 2
        steep = dx <= dy
        # Bresenham as used by pygame: step n of the major axis moves the minor axis by
        # floor((n * minor + major - 1 - major // 2) / major).
        R = int(dy.max()) + 1
        n = torch.arange(R, device=self.device)[None]
        ok = steep[:, None] & (n <= dy[:, None])
        xn = x1[:, None] + sx[:, None] * torch.div(
            n * dx[:, None] + dy[:, None] - 1 - dy[:, None] // 2, dy.clamp_min(1)[:, None], rounding_mode="floor"
        )
        xn = torch.where(dy[:, None] == 0, x1[:, None], xn)
        out = [
            (
                m[:, None, :].expand(-1, R, 2),
                y1[:, None] + sy[:, None] * n,
                torch.where(ok, xn + lo_off[:, None], 1),
                torch.where(ok, xn + hi_off[:, None], 0),
            )
        ]
        C = int(dx.max()) + 1
        k = torch.arange(C, device=self.device)[None]
        okc = (~steep)[:, None] & (k <= dx[:, None])
        yn = y1[:, None] + sy[:, None] * torch.div(
            k * dy[:, None] + dx[:, None] - 1 - dx[:, None] // 2, dx.clamp_min(1)[:, None], rounding_mode="floor"
        )
        cols = x1[:, None] + sx[:, None] * k
        # Each column is a vertical run; for a fixed row the covering columns are contiguous.
        wmax = int(w.max())
        t = torch.arange(wmax, device=self.device)
        rr = yn[..., None] + lo_off.view(N, 1, 1) + t.view(1, 1, -1)
        okr = okc[..., None] & (t.view(1, 1, -1) < w.view(N, 1, 1))
        base = torch.minimum(y1, y2) + lo_off
        span = int((dy + w).max()) + 1
        key = (
            torch.arange(N, device=self.device).view(N, 1, 1) * span + (rr - base.view(N, 1, 1)).clamp(0, span - 1)
        )[okr]
        cc = cols[..., None].expand(-1, -1, wmax)[okr]
        lo = torch.full((N * span,), 1 << 30, dtype=torch.int64, device=self.device)
        hi = torch.full((N * span,), -(1 << 30), dtype=torch.int64, device=self.device)
        lo.scatter_reduce_(0, key, cc, reduce="amin")
        hi.scatter_reduce_(0, key, cc, reduce="amax")
        lo, hi = lo.view(N, span), hi.view(N, span)
        valid = (~steep)[:, None] & (hi >= lo)
        out.append(
            (
                m[:, None, :].expand(-1, span, 2),
                base[:, None] + torch.arange(span, device=self.device)[None],
                torch.where(valid, lo, 1),
                torch.where(valid, hi, 0),
            )
        )
        return out

    def spans(self, meta, spans):
        """Precomputed (y, x0, x1) spans for one primitive."""
        arr = self._t(np.asarray(spans, np.int64).reshape(-1, 3))
        return [(self._t(meta).view(1, 2).expand(len(arr), 2), arr[:, 0], arr[:, 1], arr[:, 2])]

    def paint(self, colors, parts):
        """colors (B, n_ops, 3) uint8 with op 0 the background; parts from the span builders."""
        B, nops, _ = colors.shape
        W, H = self.w, self.h
        meta = torch.cat([p[0].reshape(-1, 2) for p in parts])
        y = torch.cat([p[1].reshape(-1) for p in parts])
        a = torch.cat([p[2].reshape(-1) for p in parts]).clamp_min(0)
        b = torch.cat([p[3].reshape(-1) for p in parts]).clamp_max(W - 1)
        keep = (b >= a) & (y >= 0) & (y < H)
        meta, y, a, b = meta[keep], y[keep], a[keep], b[keep]
        length = b - a + 1
        total = int(length.sum())
        start = (meta[:, 0] * H + y) * W + a
        idx = torch.repeat_interleave(start - torch.cumsum(length, 0) + length, length, output_size=total)
        idx += torch.arange(total, device=self.device)
        order = torch.repeat_interleave(meta[:, 1], length, output_size=total)
        winner = torch.zeros(B * H * W, dtype=torch.int64, device=self.device)
        winner.scatter_reduce_(0, idx, order, reduce="amax")
        table = self._t(colors, torch.uint8).view(B * nops, 3)
        base = torch.arange(B, device=self.device).view(B, 1) * nops
        return table[(winner.view(B, H * W) + base).view(-1)].view(B, H, W, 3)

    def blend_aalines(self, canvas, segments):
        """segments: (frame, order, (x1, y1), (x2, y2), rgb), blended per frame in draw order."""
        rows = []
        seg_of = {}
        for frame, _, p1, p2, rgb in sorted(segments, key=lambda s: (s[0], s[1])):
            seg = seg_of[frame] = seg_of.get(frame, -1) + 1
            seen = {}
            for x, y, b in aaline_pixels(*p1, *p2, self.w, self.h):
                k = seen[(x, y)] = seen.get((x, y), -1) + 1
                rows.append((seg, k, frame, y, x, float(b), *rgb))
        if not rows:
            return canvas
        arr = np.array(rows, dtype=np.float64)
        keys = arr[:, 0] * 4096 + arr[:, 1]
        flat = canvas.view(-1, 3)
        for key in np.unique(keys):
            sel = arr[keys == key]
            idx = self._t(((sel[:, 2] * self.h + sel[:, 3]) * self.w + sel[:, 4]).astype(np.int64))
            b = self._t(sel[:, 5].astype(np.float32), torch.float32)[:, None]
            c = self._t(sel[:, 6:9].astype(np.float32), torch.float32)
            flat[idx] = (b * c + (1 - b) * flat[idx].to(torch.float32)).to(torch.int64).to(torch.uint8)
        return canvas
