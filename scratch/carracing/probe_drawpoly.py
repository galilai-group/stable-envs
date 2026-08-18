"""Recover ``pygame.draw.polygon``'s filled-polygon rule EXACTLY.

CarRacing draws the road and grass through ``gfxdraw`` (already reimplemented in
``stable_envs/render/gfxdraw.py``) but the CAR (4 hull fixtures, 4 wheels, up to 4 wheel-white
quads) and the INDICATOR panel go through ``pygame.draw.polygon``, which is a completely different
rasterizer.  Two things to pin down:

1. how a float vertex becomes an integer (truncate toward zero? floor? round?);
2. the scanline rule.

Candidate for (2) is pygame's ``draw_fillpoly``: per scanline, collect edge crossings with the
half-open rule ``(y1 <= y < y2)`` plus a special case on the last row, C-truncating integer
division for the crossing x, sort, fill pairs; then a fixup pass that draws any horizontal edge
strictly between miny and maxy.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
W, H = 160, 120


def cdiv(a, b):
    """Measured (probe_drawpoly5, 18392/18392 rows): the crossing column is FLOOR of the exact
    crossing, not C truncation toward zero.  Floor is anchor- and winding-independent because the
    vertices are integers, which is why it is stable under reordering the polygon."""
    return a // b if b > 0 else (-a) // (-b)


def fillpoly(px, py, W, H):
    """The candidate rule -> (H,W) bool."""
    n = len(px)
    m = np.zeros((H, W), bool)

    def hline(x1, x2, y):
        if y < 0 or y >= H:
            return
        a, b = (x1, x2) if x1 <= x2 else (x2, x1)
        a, b = max(a, 0), min(b, W - 1)
        if a <= b:
            m[y, a:b + 1] = True

    miny, maxy = min(py), max(py)
    if miny == maxy:
        hline(min(px), max(px), miny)
        return m
    for y in range(miny, maxy + 1):
        xs = []
        for i in range(n):
            ip = i - 1 if i else n - 1
            y1, y2 = py[ip], py[i]
            if y1 < y2:
                x1, x2 = px[ip], px[i]
            elif y1 > y2:
                y1, y2 = py[i], py[ip]
                x1, x2 = px[i], px[ip]
            else:
                continue
            if (y1 <= y < y2) or (y == maxy and y1 < y <= y2):
                xs.append(x1 + (y - y1) * (x2 - x1) / (y2 - y1))
        xs.sort()
        # measured (probe_drawpoly6): the span is the CLOSED interval the crossings touch --
        # floor on the left, ceil on the right, not floor/floor.
        for k in range(0, len(xs) - 1, 2):
            hline(int(np.floor(xs[k])), int(np.ceil(xs[k + 1])), y)
    for i in range(n):
        ip = i - 1 if i else n - 1
        y = py[i]
        if miny < y == py[ip] < maxy:
            hline(px[i], px[ip], y)
    return m


def ref(points):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), points)
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def int_conversion():
    """trunc / floor / round: put one vertex at a fractional coordinate and see which row moves."""
    print("=== float -> int vertex conversion ===")
    for frac in (0.2, 0.5, 0.7, -0.2, -0.5, -0.7):
        pts = [(20.0, 40.0 + frac), (100.0, 40.0 + frac), (100.0, 41.0 + frac), (20.0, 41.0 + frac)]
        r = ref(pts)
        rows = np.nonzero(r.any(1))[0]
        base = 40.0 + frac
        print(f"  y0={base:6.2f}  rows {rows.tolist()[:4]}   trunc {int(base)}  "
              f"floor {int(np.floor(base))}  round {int(round(base))}")


def main():
    print(f"pygame {pygame.version.ver}")
    int_conversion()

    rng = np.random.default_rng(0)
    HULL3 = [(25, 20), (50, -10), (50, -40), (20, -90), (-20, -90), (-50, -40), (-50, -10), (-25, 20)]
    cases = []
    # random convex quads, incl. degenerate/thin/offscreen
    for _ in range(400):
        c = rng.uniform(-40, W + 40, 2)
        r = rng.uniform(0.3, 90, 2)
        th = rng.uniform(0, 6.28)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        q = np.array([[-r[0], -r[1]], [r[0], -r[1]], [r[0], r[1]], [-r[0], r[1]]]) @ R.T + c
        cases.append([tuple(p) for p in q])
    # the car's 8-gon hull fixture at random poses/scales
    for _ in range(200):
        c = rng.uniform(0, W, 2)
        sc = rng.uniform(0.05, 1.2)
        th = rng.uniform(0, 6.28)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        q = (np.array(HULL3, float) * sc) @ R.T + c
        cases.append([tuple(p) for p in q])
    # axis-aligned indicator quads, including zero-height ones
    for _ in range(200):
        x0 = rng.uniform(0, W)
        w = rng.uniform(0, 30)
        y0 = rng.uniform(0, H)
        hh = rng.uniform(0, 3)
        cases.append([(x0, y0), (x0 + w, y0), (x0 + w, y0 + hh), (x0, y0 + hh)])

    for conv_name, conv in (("trunc", int), ("floor", lambda v: int(np.floor(v)))):
        tally = {"onscreen": [0, 0, 0], "offscreen": [0, 0, 0]}
        worst = {}
        for pts in cases:
            px = [conv(p[0]) for p in pts]
            py = [conv(p[1]) for p in pts]
            inside = (min(px) >= 0 and max(px) < W and min(py) >= 0 and max(py) < H)
            k = "onscreen" if inside else "offscreen"
            got = fillpoly(px, py, W, H)
            exp = ref(pts)
            n_bad = int((got != exp).sum())
            tally[k][0] += 1
            tally[k][1] += n_bad == 0
            tally[k][2] += n_bad
            if n_bad and k not in worst:
                worst[k] = (px, py, n_bad)
        print(f"\n=== vertex conversion {conv_name} ===")
        for k, (n, ok, npx) in tally.items():
            print(f"  {k:9s} {ok}/{n} polygons bit-identical, {npx} px wrong"
                  + (f"   first bad px={worst[k][0]} py={worst[k][1]} ({worst[k][2]} px)"
                     if k in worst else ""))


if __name__ == "__main__":
    main()
