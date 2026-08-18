"""`pygame.draw.polygon`'s fill rule, transcribed from pygame-ce 2.5.7 `src_c/draw.c::draw_fillpoly`
and then CHECKED against the shipped binary.

The rule that defeated the fitting sweep is the third line of the loop body:

    intersect = (y - y1) * (x2 - x1) / (float)(y2 - y1);       # float32 division
    if (n_intersections % 2 == 0) intersect = floor(intersect);
    else                          intersect = ceil(intersect);
    x_intersect[n_intersections++] = (int)intersect + x1;

`n_intersections` is a per-ROW counter over edges visited in VERTEX ORDER, and it is reset every row.
So whether a crossing is floored or ceiled is decided by how many earlier-indexed edges happen to be
active on that row — not by which side of the polygon it is on, and not by the row. As `y` passes a
vertex the active set changes, the same geometric edge changes parity, and its rounding flips
mid-edge. That is exactly the "flip row is the y of the right-hand vertex" observation, and no rule
written per edge or per side can reproduce it.

Then the crossings are SORTED and drawn in consecutive pairs, each as an inclusive horizontal span
(`drawhorzlineclipbounding` swaps its endpoints and clips to the surface). Horizontal edges strictly
between `miny` and `maxy` get one extra span in a trailing pass.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np                                                              # noqa: E402
import pygame                                                                   # noqa: E402

W = H = 200


def fill_mask(vx, vy, W=W, H=H):
    """`pygame.draw.polygon` for one integer polygon -> (H, W) bool.

    Vertex order is load-bearing, and THREE-vertex polygons take a different function
    (`draw_filltri`), which is a different rule — not an optimisation of the same one.
    """
    vx = np.asarray(vx, np.int64)
    vy = np.asarray(vy, np.int64)
    n = len(vx)
    out = np.zeros((H, W), bool)
    if n == 3:
        return _filltri(vx, vy, out)
    miny, maxy = int(vy.min()), int(vy.max())

    if miny == maxy:                                    # the 1-pixel-high special case
        _span(out, int(vx.min()), miny, int(vx.max()))
        return out

    for y in range(miny, maxy + 1):
        xs = []
        for i in range(n):
            p = i - 1 if i else n - 1
            y1, y2 = int(vy[p]), int(vy[i])
            if y1 < y2:
                x1, x2 = int(vx[p]), int(vx[i])
            elif y1 > y2:
                y1, y2 = y2, y1
                x1, x2 = int(vx[i]), int(vx[p])
            else:
                continue                                # horizontal edges are the trailing pass
            if (y1 <= y < y2) or (y == maxy and y2 == maxy):
                v = np.float32((y - y1) * (x2 - x1)) / np.float32(y2 - y1)
                v = np.floor(v) if len(xs) % 2 == 0 else np.ceil(v)
                xs.append(int(v) + x1)
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            _span(out, xs[k], y, xs[k + 1])

    for i in range(n):                                  # horizontal border lines
        p = i - 1 if i else n - 1
        y = int(vy[i])
        if miny < y < maxy and int(vy[p]) == y:
            _span(out, int(vx[i]), y, int(vx[p]))
    return out


def _filltri(vx, vy, out):
    """`draw_filltri`: sort by y, three float32 slopes with a 1e-17 guard, truncating `(int)`."""
    p = [[int(vx[i]), int(vy[i])] for i in range(3)]
    if p[1][1] < p[0][1]:
        p[1], p[0] = p[0], p[1]
    if p[2][1] < p[1][1]:
        p[1], p[2] = p[2], p[1]
        if p[1][1] < p[0][1]:
            p[1], p[0] = p[0], p[1]
    (p0x, p0y), (p1x, p1y), (p2x, p2y) = p
    if p0y == p1y == p2y and p0x == p1x and p1x != p2x:
        p1x, p1y, p2x, p2y = p2x, p2y, p1x, p1y
    d1 = np.float32((p2x - p0x) / ((p2y - p0y) + 1e-17))
    d2 = np.float32((p1x - p0x) / ((p1y - p0y) + 1e-17))
    d3 = np.float32((p2x - p1x) / ((p2y - p1y) + 1e-17))
    for y in range(p0y, p2y + 1):
        x1 = p0x + int(np.trunc(np.float32(y - p0y) * d1))
        if y < p1y:
            x2 = p0x + int(np.trunc(np.float32(y - p0y) * d2))
        else:
            x2 = p1x + int(np.trunc(np.float32(y - p1y) * d3))
        _span(out, x1, y, x2)
    return out


def _span(out, x1, y, x2):
    H, W = out.shape
    if not (0 <= y < H):
        return
    if x2 < x1:
        x1, x2 = x2, x1
    x1, x2 = max(x1, 0), min(x2, W - 1)
    if x2 < 0 or x1 >= W:
        return
    out[y, x1:x2 + 1] = True


def reference(vx, vy):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), list(zip(vx.tolist(), vy.tolist())))
    return pygame.surfarray.array3d(s).transpose(1, 0, 2)[..., 0] > 0


def main():
    pygame.init()
    rng = np.random.default_rng(0)
    cases = [(np.array([74, 56, 95, 113]), np.array([102, 53, 39, 88]))]        # the reported case
    # convex quads, which is what a Box2D fixture is, in both winding orders
    for k in range(400):
        c = rng.uniform(40, 160, 2)
        r = rng.uniform(5, 60, 2)
        a = np.sort(rng.uniform(0, 2 * np.pi, 4))
        if k % 2:
            a = a[::-1]
        cases.append((np.trunc(c[0] + r[0] * np.cos(a)).astype(np.int64),
                      np.trunc(c[1] + r[1] * np.sin(a)).astype(np.int64)))
    # arbitrary polygons with 3..8 vertices, including non-convex and off-surface ones
    for k in range(400):
        m = int(rng.integers(3, 9))
        cases.append((rng.integers(-30, W + 30, m).astype(np.int64),
                      rng.integers(-30, H + 30, m).astype(np.int64)))
    # degenerate: flat, collinear, repeated vertices
    cases += [(np.array([10, 60, 90]), np.array([40, 40, 40])),
              (np.array([10, 60, 90, 30]), np.array([40, 40, 90, 90])),
              (np.array([20, 20, 20]), np.array([10, 50, 90]))]

    bad = exact = 0
    per_n = {}
    worst = None
    for i, (vx, vy) in enumerate(cases):
        a, b = reference(vx, vy), fill_mask(vx, vy)
        d = int((a != b).sum())
        bad += d
        exact += d == 0
        k = per_n.setdefault(len(vx), [0, 0, 0])
        k[0] += 1
        k[1] += d == 0
        k[2] += d
        if d and worst is None:
            worst = (vx.tolist(), vy.tolist(), d)
        if i == 0:
            print(f"the reported polygon vx={vx.tolist()} vy={vy.tolist()}: differing px = {d}")
    print(f"{exact}/{len(cases)} polygons bit-identical against pygame.draw.polygon, "
          f"{bad} differing pixels total")
    for n in sorted(per_n):
        t, e, d = per_n[n]
        print(f"    {n} vertices: {e}/{t} exact, {d} differing px"
              f"{'   (draw_filltri path)' if n == 3 else ''}")
    if worst:
        print(f"first mismatch: vx={worst[0]} vy={worst[1]} ({worst[2]} px)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
