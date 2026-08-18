"""Reproduce SDL_gfx's filled_polygon / filled_circle rules EXACTLY, in numpy, and prove it.

If these two rules can be emulated bit-for-bit then a batched GPU rasterizer is not an
approximation of the gymnasium renderer, it *is* the gymnasium renderer -- everywhere except the
thin ``aapolygon`` fringe, which is drawn before the fill and survives only outside it.

filled_polygon (SDL2_gfx ``filledPolygonRGBAMT``) is a fixed-point scanline fill on the
TRUNCATED-to-int vertices:

    for each scanline y in [miny, maxy]:
        for each edge (with y1 < y2 after ordering):
            active iff  (y >= y1 and y < y2)  or  (y == maxy and y > y1 and y <= y2)
            X = (65536*(y-y1)) // (y2-y1) * (x2-x1) + 65536*x1
        sort the X's; for each consecutive pair:
            xa = round16(X_lo + 1),  xb = round16(X_hi - 1),  hline(xa, xb, y)
        where round16(v) = (v >> 16) + ((v & 0x8000) >> 15)

filled_circle is a midpoint-circle span fill, so its per-row half-width is tabulated rather than
guessed at.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from pygame import gfxdraw

W = 96


def _surf():
    s = pygame.Surface((W, W))
    s.fill((255, 255, 255))
    return s


def _mask(s, col=(0, 0, 0)):
    a = np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))
    return (a == np.array(col)).all(-1)


def round16(v):
    """SDL_gfx's ``(v >> 16) + ((v & 32768) >> 15)`` on python ints/arrays (arithmetic shift)."""
    return (v >> 16) + ((v & 32768) >> 15)


def filled_polygon_ref(verts_int, H=W, Wd=W):
    """Exact numpy transcription of filledPolygonRGBAMT. ``verts_int`` is (n,2) int."""
    vx = verts_int[:, 0].astype(np.int64)
    vy = verts_int[:, 1].astype(np.int64)
    n = len(vx)
    miny, maxy = vy.min(), vy.max()
    out = np.zeros((H, Wd), bool)
    for y in range(max(miny, 0), min(maxy, H - 1) + 1):
        xs = []
        for i in range(n):
            ind1, ind2 = (n - 1, 0) if i == 0 else (i - 1, i)
            y1, y2 = vy[ind1], vy[ind2]
            if y1 < y2:
                x1, x2 = vx[ind1], vx[ind2]
            elif y1 > y2:
                y1, y2 = vy[ind2], vy[ind1]
                x1, x2 = vx[ind2], vx[ind1]
            else:
                continue
            if ((y >= y1) and (y < y2)) or ((y == maxy) and (y > y1) and (y <= y2)):
                xs.append(((65536 * (y - y1)) // (y2 - y1)) * (x2 - x1) + 65536 * x1)
        xs.sort()
        for i in range(0, len(xs) - 1, 2):
            xa = round16(xs[i] + 1)
            xb = round16(xs[i + 1] - 1)
            xa = max(xa, 0)
            xb = min(xb, Wd - 1)
            if xb >= xa:
                out[y, xa:xb + 1] = True
    return out


def main():
    pygame.init()
    rng = np.random.default_rng(0)

    print("=" * 72)
    print("filled_polygon: exact-rule emulation vs pygame, 400 random rotated quads")
    print("=" * 72)
    bad = 0
    worst = 0
    for trial in range(400):
        cx, cy = rng.uniform(20, 76, 2)
        th = rng.uniform(-np.pi, np.pi)
        hw, hh = rng.uniform(2, 25), rng.uniform(1, 12)
        loc = np.array([(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)])
        c, s_ = np.cos(th), np.sin(th)
        R = np.array([[c, -s_], [s_, c]])
        wv = loc @ R.T + np.array([cx, cy])
        s = _surf()
        gfxdraw.filled_polygon(s, [tuple(map(float, p)) for p in wv], (0, 0, 0))
        m = _mask(s)
        vi = np.trunc(wv).astype(np.int64)
        ref = filled_polygon_ref(vi)
        x = np.logical_xor(m, ref).sum()
        worst = max(worst, x)
        bad += x > 0
    print(f"  mismatching quads: {bad}/400     worst per-quad pixel xor: {worst}")

    print("\n  triangles (mountain-car flag shape), 200 random:")
    bad = worst = 0
    for _ in range(200):
        p = rng.uniform(5, 90, (3, 2))
        s = _surf()
        gfxdraw.filled_polygon(s, [tuple(map(float, q)) for q in p], (0, 0, 0))
        m = _mask(s)
        ref = filled_polygon_ref(np.trunc(p).astype(np.int64))
        x = np.logical_xor(m, ref).sum()
        worst = max(worst, x)
        bad += x > 0
    print(f"  mismatching tris : {bad}/200     worst xor: {worst}")

    print("\n" + "=" * 72)
    print("filled_circle: per-row half-widths (hw[dy]) for the radii classic-control uses")
    print("=" * 72)
    tables = {}
    for r in range(0, 26):
        s = _surf()
        gfxdraw.filled_circle(s, 48, 48, r, (0, 0, 0))
        m = _mask(s)
        hw = []
        ok = True
        for dy in range(0, r + 1):
            row = m[48 + dy]
            xs = np.where(row)[0]
            if xs.size == 0:
                hw.append(-1)
                continue
            lo, hi = xs.min() - 48, xs.max() - 48
            if lo != -hi:
                ok = False
            if not row[48 + lo:48 + hi + 1].all():
                ok = False
            hw.append(int(hi))
            # symmetric in y?
            if not np.array_equal(m[48 + dy], m[48 - dy]):
                ok = False
        tables[r] = hw
        naive = [int(np.floor(np.sqrt(max(r * r - dy * dy, 0)))) for dy in range(r + 1)]
        tables[r] = hw
        if r in (1, 2, 3, 4, 5, 6, 10, 12, 20, 25):
            print(f"  r={r:2d} n={m.sum():4d} sym/contig={ok}  hw={hw}")
            print(f"        floor(sqrt(r^2-dy^2)) = {naive}   {'MATCH' if hw == naive else 'DIFFER'}")
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "circle_hw.npy"),
            np.array([tables[r] + [-1] * (26 - len(tables[r])) for r in range(26)], np.int64))
    print("  wrote circle_hw.npy  (row half-widths, -1 = empty)")

    print("\n" + "=" * 72)
    print("closed form check: is hw[dy] == floor(sqrt(r^2 - dy^2)) for all r<=25?")
    print("=" * 72)
    nm = [r for r in range(26)
          if tables[r] != [int(np.floor(np.sqrt(max(r * r - dy * dy, 0)))) for dy in range(r + 1)]]
    print(f"  radii where the closed form FAILS: {nm}")
    for r in nm[:6]:
        naive = [int(np.floor(np.sqrt(max(r * r - dy * dy, 0)))) for dy in range(r + 1)]
        print(f"    r={r}: gfx {tables[r]}  vs  {naive}")

    print("\n" + "=" * 72)
    print("aacircle fringe: which pixels, and is the fill still exactly filled_circle?")
    print("=" * 72)
    for r in (2, 5, 10):
        s = _surf()
        gfxdraw.aacircle(s, 48, 48, r, (0, 0, 0))
        gfxdraw.filled_circle(s, 48, 48, r, (0, 0, 0))
        a = np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))
        s2 = _surf()
        gfxdraw.filled_circle(s2, 48, 48, r, (0, 0, 0))
        b = np.transpose(np.array(pygame.surfarray.pixels3d(s2)), (1, 0, 2))
        black_both = (a == 0).all(-1)
        black_fill = (b == 0).all(-1)
        print(f"  r={r:2d}: pure-black pixels equal? {np.array_equal(black_both, black_fill)}  "
              f"extra blended px {(((a != 255).any(-1)) & ~black_both).sum()}")

    print("\n" + "=" * 72)
    print("pygame.draw.line width=1 (acrobot ground) -- aliased Bresenham?")
    print("=" * 72)
    s = _surf()
    pygame.draw.line(s, (0, 0, 0), (5.3, 20.8), (90.7, 20.2))
    a = np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))
    nz = (a != 255).any(-1)
    ys, xs = np.where(nz)
    print(f"  horizontal-ish: n={nz.sum()} rows={sorted(set(ys.tolist()))} x[{xs.min()},{xs.max()}] "
          f"blended={(nz & (a != 0).any(-1)).sum()}")

    pygame.quit()


if __name__ == "__main__":
    main()
