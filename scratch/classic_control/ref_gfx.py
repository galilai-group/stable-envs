"""Exact numpy reference for the four SDL_gfx primitives gymnasium's classic-control envs use.

This is the bridge between "read the render() source" and "write a GPU kernel": every rule here was
decided by pixel-diffing pygame (probe_*.py), and the whole module is checked frame-for-frame
against the real renderer in verify_ref.py. Porting it to batched torch is then a mechanical step
that cannot silently introduce a geometry bug, because the CPU reference is already known-exact.

The four rules:

``trunc_verts``   pygame's gfxdraw converts each float coordinate to Sint16 by C truncation toward
                  zero, so the polygon SDL_gfx actually rasterises has INTEGER vertices. Everything
                  downstream is integer arithmetic -- there is no sub-pixel geometry to approximate.
``filled_polygon`` fixed-point scanline fill; see the docstring on the function.
``filled_circle``  midpoint-circle span fill. Its per-row half-width is NOT
                  ``floor(sqrt(r^2-dy^2))`` (that disagrees for every r >= 2), so it is tabulated.
``aapolygon``      Wu line per edge with a 32-bit accumulator, closed-formed so it parallelises,
                  writing through a blend that operates on SDL's MASKED channel word -- which makes
                  the rounding a ceiling when the source is darker than the destination, and means
                  a weight of 255 still blends (at alpha 254) rather than writing solid.
"""
from __future__ import annotations

import numpy as np

M32 = (1 << 32) - 1
SOLID = 256           # sentinel weight: goes through pixelColor, not pixelColorWeight

# filled_circle row half-widths, hw[r][dy]; measured, see probe_exact_rules.py
_CIRCLE_HW = {
    0: [0], 1: [1, 0], 2: [2, 2, 0], 3: [3, 3, 2, 0], 4: [4, 4, 3, 2, 0], 5: [5, 5, 5, 4, 3, 0],
    6: [6, 6, 6, 5, 4, 3, 0], 7: [7, 7, 7, 6, 6, 5, 3, 0], 8: [8, 8, 8, 7, 7, 6, 5, 3, 0],
    9: [9, 9, 9, 8, 8, 7, 6, 5, 4, 0], 10: [10, 10, 10, 10, 9, 9, 8, 7, 6, 4, 0],
    11: [11, 11, 11, 11, 10, 10, 9, 8, 7, 6, 4, 0],
    12: [12, 12, 12, 12, 11, 11, 10, 10, 9, 8, 6, 4, 0],
}


def circle_hw(r: int):
    """Row half-widths of ``gfxdraw.filled_circle`` for integer radius ``r`` (index by |dy|)."""
    if r in _CIRCLE_HW:
        return list(_CIRCLE_HW[r])
    return _midpoint_hw(r)


def _midpoint_hw(r: int):
    """``filledCircleColor``'s span walk, including the ``ocx``/``ocy`` paint-once guards.

    Without the guards the last row widens from 0 to 1 and every disc grows a 5-pixel spur at the
    top and bottom -- which is invisible by eye and 64 pixels a frame in a diff.
    """
    hw = [-1] * (r + 1)

    def span(dy, half):
        if half > hw[dy]:
            hw[dy] = half

    cx, cy = 0, r
    ocx = ocy = -1
    df, d_e, d_se = 1 - r, 3, -2 * r + 5
    while True:
        if ocy != cy:
            span(cy, cx)
            ocy = cy
        if ocx != cx:
            if cx != cy:
                span(cx, cy)
            ocx = cx
        if df < 0:
            df, d_e, d_se = df + d_e, d_e + 2, d_se + 2
        else:
            df, d_e, d_se = df + d_se, d_e + 2, d_se + 4
            cy -= 1
        cx += 1
        if cx > cy:
            break
    return hw


def trunc_verts(v):
    """float (n,2) -> int (n,2), C truncation toward zero, exactly as pygame's pg_IntFromObj does."""
    return np.trunc(np.asarray(v, np.float64)).astype(np.int64)


def round16(v):
    """SDL_gfx's ``(v >> 16) + ((v & 32768) >> 15)`` -- round-half-up of a 16.16 fixed-point x."""
    return (v >> 16) + ((v & 32768) >> 15)


def filled_polygon_mask(vi, H, W):
    """``gfxdraw.filled_polygon`` as a boolean mask. ``vi`` is (n,2) int.

    For every scanline y in [miny, maxy] the crossing of each non-horizontal edge is evaluated in
    16.16 fixed point, ``X = (65536*(y-y1))//(y2-y1) * (x2-x1) + 65536*x1``; an edge counts when
    ``y1 <= y < y2``, with a special case that reopens the bottom edges on the last scanline. The
    sorted crossings are paired and the span ``[round16(Xlo+1), round16(Xhi-1)]`` is filled.
    """
    vx, vy = vi[:, 0], vi[:, 1]
    n = len(vx)
    maxy = int(vy.max())
    out = np.zeros((H, W), bool)
    for y in range(max(int(vy.min()), 0), min(maxy, H - 1) + 1):
        xs = []
        for i in range(n):
            i1, i2 = (n - 1, 0) if i == 0 else (i - 1, i)
            y1, y2, x1, x2 = int(vy[i1]), int(vy[i2]), int(vx[i1]), int(vx[i2])
            if y1 > y2:
                y1, y2, x1, x2 = y2, y1, x2, x1
            elif y1 == y2:
                continue
            if (y1 <= y < y2) or (y == maxy and y1 < y <= y2):
                xs.append(((65536 * (y - y1)) // (y2 - y1)) * (x2 - x1) + 65536 * x1)
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            xa = max(round16(xs[i] + 1), 0)
            xb = min(round16(xs[i + 1] - 1), W - 1)
            if xb >= xa:
                out[y, xa:xb + 1] = True
    return out


def filled_circle_mask(cx, cy, r, H, W):
    """``gfxdraw.filled_circle`` as a boolean mask."""
    hw = circle_hw(int(r))
    out = np.zeros((H, W), bool)
    for dy in range(-r, r + 1):
        y = cy + dy
        if not (0 <= y < H):
            continue
        h = hw[abs(dy)]
        if h < 0:
            continue
        xa, xb = max(cx - h, 0), min(cx + h, W - 1)
        if xb >= xa:
            out[y, xa:xb + 1] = True
    return out


def _clip_encode(x, y, left, top, right, bottom):
    code = 0
    if x < left:
        code |= 1
    elif x > right:
        code |= 2
    if y < top:
        code |= 8
    elif y > bottom:
        code |= 4
    return code


def clip_line(x1, y1, x2, y2, W, H):
    """SDL_gfx ``clipLine``: Cohen-Sutherland on Sint16 with a float32 slope. Returns
    ``(draw, x1, y1, x2, y2)``.

    ``aalineColorInt`` runs this BEFORE the Wu walk, so an edge that leaves the surface starts its
    error accumulator from the clipped endpoint and lays down a completely different fringe from the
    one an unclipped walk produces -- which is why MountainCar's car disagrees near the goal and
    CartPole's pole disagrees past |theta| ~ 2.4. Note the swap: when the first point is inside, the
    two endpoints are exchanged, and the exchange survives into the walk (it decides which endpoint
    gets the solid initial pixel).
    """
    import numpy as _np
    left, top, right, bottom = 0, 0, W - 1, H - 1
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    while True:
        c1 = _clip_encode(x1, y1, left, top, right, bottom)
        c2 = _clip_encode(x2, y2, left, top, right, bottom)
        if not (c1 | c2):
            return True, x1, y1, x2, y2
        if c1 & c2:
            return False, x1, y1, x2, y2
        if c1 == 0:
            x1, x2, y1, y2, c1 = x2, x1, y2, y1, c2
        m = _np.float32(y2 - y1) / _np.float32(x2 - x1) if x2 != x1 else _np.float32(1.0)
        if c1 & 1:
            y1 += int(_np.float32(left - x1) * m)
            x1 = left
        elif c1 & 2:
            y1 += int(_np.float32(right - x1) * m)
            x1 = right
        elif c1 & 4:
            if x2 != x1:
                x1 += int(_np.float32(bottom - y1) / m)
            y1 = bottom
        elif c1 & 8:
            if x2 != x1:
                x1 += int(_np.float32(top - y1) / m)
            y1 = top


def aaline_ops(x1, y1, x2, y2):
    """(x, y, weight) writes, in draw order, for ``_aalineColor(..., draw_endpoint=0)``."""
    xx0, yy0, xx1, yy1 = int(x1), int(y1), int(x2), int(y2)
    if yy0 > yy1:
        yy0, yy1, xx0, xx1 = yy1, yy0, xx1, xx0
    dx, dy = xx1 - xx0, yy1 - yy0
    xdir = 1 if dx >= 0 else -1
    dx = abs(dx)
    if dx == 0:
        return ([(int(x1), int(y1), SOLID)]
                + ([(xx0, y, SOLID) for y in range(yy0, yy0 + dy + 1)] if dy > 0 else []))
    if dy == 0:
        a, b = xx0, xx0 + xdir * dx
        return [(int(x1), int(y1), SOLID)] + [(x, yy0, SOLID)
                                              for x in range(min(a, b), max(a, b) + 1)]
    ops = [(int(x1), int(y1), SOLID)]
    if dy > dx:                                                     # y-major
        E = (dx << 16) // dy
        for k in range(1, dy):
            acc = k * E
            w, wgt = acc >> 16, (acc & 0xFFFF) >> 8
            ops.append((xx0 + xdir * w, yy0 + k, 255 - wgt))
            ops.append((xx0 + xdir * (w + 1), yy0 + k, wgt))
    else:                                                           # x-major
        E = (dy << 16) // dx
        for k in range(1, dx):
            acc = k * E
            w, wgt = acc >> 16, (acc & 0xFFFF) >> 8
            ops.append((xx0 + xdir * k, yy0 + w, 255 - wgt))
            ops.append((xx0 + xdir * k, yy0 + w + 1, wgt))
    return ops


_SHIFT = np.array([16, 8, 0], np.int64)      # plain pygame.Surface masks are (0xFF0000, 0xFF00, 0xFF)


def gfx_blend(src, dst, w):
    """``_putPixelAlpha`` on the masked channel word. ``src``/``dst`` are (...,3) int64."""
    if w == SOLID:
        return np.broadcast_to(src, dst.shape).copy()
    a = (255 * int(w)) >> 8
    S = np.left_shift(src, _SHIFT)
    D = np.left_shift(dst, _SHIFT)
    v = np.right_shift(np.bitwise_and(np.bitwise_and(S - D, M32) * a, M32), 8)
    return np.right_shift(np.bitwise_and(D + v, np.left_shift(255, _SHIFT)), _SHIFT)


def draw_aapolygon(canvas, vi, color, clip=True):
    """``gfxdraw.aapolygon``: one Wu line per edge, in vertex order, closing edge last."""
    H, W = canvas.shape[:2]
    color = np.asarray(color, np.int64)
    n = len(vi)
    for i in range(n):
        a, b = vi[i], vi[(i + 1) % n]
        ax, ay, bx, by = int(a[0]), int(a[1]), int(b[0]), int(b[1])
        if clip:
            draw, ax, ay, bx, by = clip_line(ax, ay, bx, by, W, H)
            if not draw:
                continue
        for (x, y, w) in aaline_ops(ax, ay, bx, by):
            if 0 <= x < W and 0 <= y < H:
                canvas[y, x] = gfx_blend(color, canvas[y, x], w)
    return canvas


def draw_polygon(canvas, verts_f, color):
    """The gymnasium idiom: ``aapolygon`` then ``filled_polygon``, same float vertices."""
    vi = trunc_verts(verts_f)
    draw_aapolygon(canvas, vi, color)
    canvas[filled_polygon_mask(vi, canvas.shape[0], canvas.shape[1])] = np.asarray(color, np.int64)
    return canvas


def aacircle_ops(cx, cy, r):
    """``gfxdraw.aacircle`` -> ``aaellipseColor(cx,cy,r,r)``: (x, y, weight) writes in draw order.

    SDL_gfx special-cases r < 2 to a plain pixel/filled-circle; otherwise it walks the ellipse
    analytically. Rather than transcribe that, the fringe is captured once per radius by diffing
    pygame (see ``bake_aacircle``); the radii classic control uses are a handful of small integers.
    """
    raise NotImplementedError("baked per radius instead; see bake_aacircle")


def bake_aacircle(r, color=(0, 0, 0), bgs=((255, 255, 255), (0, 0, 0), (100, 160, 200),
                                           (204, 77, 77), (17, 240, 90))):
    """The (dy, dx, weight, pass) fringe of ``aacircle`` at radius ``r``, recovered from pygame.

    ``aacircle`` writes only outside ``filled_circle`` (verified: the solid-colour pixel set is
    identical with and without it), so the fringe is a constant of the radius and the weight can be
    inverted out of the blend -- cheaper and safer than transcribing ``aaellipseColor``'s walker.

    The catch, and the reason for the multi-background fit: ``aaellipseColor`` writes some pixels
    TWICE (the four axis points, and wherever two octants meet). A single equivalent weight fitted on
    one background reproduces that pixel on that background and is wrong by 1/255 on any other, which
    is exactly the residual this used to leave wherever a wheel or a rod-end disc sat on something
    other than the page. Pixels that no single weight explains across every background are emitted as
    an ordered pair of writes instead.
    """
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    from pygame import gfxdraw
    S = 4 * r + 9
    c = S // 2
    src = np.asarray(color, np.int64)
    obs = []
    for bg in bgs:
        s = pygame.Surface((S, S))
        s.fill(bg)
        gfxdraw.aacircle(s, c, c, int(r), color)
        obs.append(np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).astype(np.int64))
    fill = filled_circle_mask(c, c, int(r), S, S)
    touched = np.zeros((S, S), bool)
    for bg, a in zip(bgs, obs):
        touched |= (a != np.asarray(bg)).any(-1)
    ys, xs = np.where(touched & ~fill)

    nb = len(bgs)
    ws = list(range(257))
    single = np.stack([np.stack([gfx_blend(src, np.asarray(bg, np.int64), w) for w in ws])
                       for bg in bgs])                                   # (nb, 257, 3)
    pair = np.stack([np.stack([gfx_blend(src, single[i], w2) for w2 in ws], 1)
                     for i in range(nb)])                                # (nb, 257, 257, 3)
    out = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        got = np.stack([a[y, x] for a in obs])                           # (nb,3)
        hit = np.where((single == got[:, None, :]).all(-1).all(0))[0]
        if hit.size:
            out.append((y - c, x - c, int(hit[0]), 0))
            continue
        h = np.argwhere((pair == got[:, None, None, :]).all(-1).all(0))
        if not len(h):
            raise RuntimeError(f"aacircle r={r} px ({y-c},{x-c}) needs more than two writes")
        out.append((y - c, x - c, int(h[0][0]), 0))
        out.append((y - c, x - c, int(h[0][1]), 1))
    return out
