"""What does SDL_gfx do when a polygon leaves the surface?

This is not academic: MountainCar's car is partly off the right edge near the goal, and CartPole's
pole leaves the top of the surface once |theta| exceeds about 2.4 rad. ``filledPolygonRGBAMT``
clamps ``miny``/``maxy`` to the surface before the scanline loop, and the ``y == maxy`` special case
that reopens the bottom edges then fires on the CLAMPED row -- a different span from the one the
unclipped rule gives. Whether that is what happens is decided here by diffing.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from pygame import gfxdraw

import ref_gfx as G

H, W = 60, 80


def _surf():
    s = pygame.Surface((W, H))
    s.fill((255, 255, 255))
    return s


def _mask(s):
    a = np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))
    return (a == 0).all(-1)


def fill_ref(vi, clamp_maxy, clamp_miny):
    """The scanline rule with optional clamping of miny/maxy to the surface."""
    vx, vy = vi[:, 0], vi[:, 1]
    n = len(vx)
    miny, maxy = int(vy.min()), int(vy.max())
    if clamp_miny:
        miny = max(miny, 0)
    if clamp_maxy:
        maxy = min(maxy, H - 1)
    out = np.zeros((H, W), bool)
    for y in range(max(miny, 0), min(maxy, H - 1) + 1):
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
            xa = max(G.round16(xs[i] + 1), 0)
            xb = min(G.round16(xs[i + 1] - 1), W - 1)
            if xb >= xa:
                out[y, xa:xb + 1] = True
    return out


def main():
    pygame.init()
    rng = np.random.default_rng(0)
    cases = {"both in": 0, "off top": 0, "off bottom": 0, "off left/right": 0}
    score = {(cm, cn): 0 for cm in (False, True) for cn in (False, True)}
    tot = {k: 0 for k in cases}
    for trial in range(600):
        cx = rng.uniform(-20, W + 20)
        cy = rng.uniform(-20, H + 20)
        th = rng.uniform(-np.pi, np.pi)
        hw, hh = rng.uniform(4, 30), rng.uniform(2, 15)
        loc = np.array([(-hw, -hh), (-hw, hh), (hw, hh), (hw, -hh)])
        c, s_ = np.cos(th), np.sin(th)
        wv = loc @ np.array([[c, -s_], [s_, c]]).T + np.array([cx, cy])
        s = _surf()
        gfxdraw.filled_polygon(s, [tuple(map(float, p)) for p in wv], (0, 0, 0))
        m = _mask(s)
        vi = G.trunc_verts(wv)
        key = ("both in" if vi[:, 1].min() >= 0 and vi[:, 1].max() < H
               else ("off top" if vi[:, 1].min() < 0 and vi[:, 1].max() < H
                     else "off bottom"))
        tot[key] += 1
        for cm in (False, True):
            for cn in (False, True):
                score[(cm, cn)] += int(np.logical_xor(m, fill_ref(vi, cm, cn)).sum())
    print("  polygons by clipping situation:", {k: v for k, v in tot.items() if v})
    print("\n  total mismatching pixels over 600 polygons, by rule:")
    for (cm, cn), v in sorted(score.items()):
        print(f"    clamp_maxy={str(cm):5s} clamp_miny={str(cn):5s} -> {v}")

    print("\n  aapolygon with off-surface vertices: does the Wu walk start from a clipped endpoint?")
    bad = {False: 0, True: 0}
    tot_px = {False: 0, True: 0}
    ntest = 0
    for trial in range(200):
        cx, cy = rng.uniform(-15, W + 15), rng.uniform(-15, H + 15)
        th = rng.uniform(-np.pi, np.pi)
        loc = np.array([(-20.0, -6.0), (-20.0, 6.0), (20.0, 6.0), (20.0, -6.0)])
        c, s_ = np.cos(th), np.sin(th)
        wv = loc @ np.array([[c, -s_], [s_, c]]).T + np.array([cx, cy])
        vi = G.trunc_verts(wv)
        if vi.min() >= 0 and vi[:, 0].max() < W and vi[:, 1].max() < H:
            continue
        ntest += 1
        s = _surf()
        gfxdraw.aapolygon(s, [tuple(map(float, p)) for p in wv], (0, 0, 0))
        got = np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).astype(np.int64)
        for clip in (False, True):
            cv = np.full((H, W, 3), 255, np.int64)
            G.draw_aapolygon(cv, vi, (0, 0, 0), clip=clip)
            d = int((cv != got).any(-1).sum())
            tot_px[clip] += d
            bad[clip] += d > 0
    for clip in (False, True):
        print(f"    clip_line={str(clip):5s}: off-surface aapolygons with a mismatch: "
              f"{bad[clip]}   total px {tot_px[clip]}")
    pygame.quit()


if __name__ == "__main__":
    main()
