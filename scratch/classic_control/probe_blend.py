"""Invert pygame's gfxdraw AA blend instead of guessing at it.

A single ``gfxdraw.aaline`` on a uniform background touches each pixel exactly once, so every
painted pixel is one clean observation of ``out = blend(src, dst, weight)`` with ``weight`` known
from the closed-form Wu accumulator. Tabulating ``out`` against ``weight`` for a black line on
white gives the map directly, and the same table taken on other (src, dst) pairs says whether the
map is a function of the weight alone or of the endpoints too.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from pygame import gfxdraw

from probe_aaline import aaline_ops

W = 128


def _surf(bg):
    s = pygame.Surface((W, W))
    s.fill(bg)
    return s


def _arr(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).astype(np.int64)


def main():
    pygame.init()

    print("=" * 74)
    print("do the derived aapolygon positions match pygame's? (triangles, singly-touched pixels)")
    print("=" * 74)
    rng = np.random.default_rng(1)
    pos_ok = 0
    trials = 0
    table = {}          # weight -> set of observed out values (black on white)
    pairs = []          # (src, dst, w, out)
    for bg, col in (((255, 255, 255), (0, 0, 0)),
                    ((255, 255, 255), (204, 77, 77)),
                    ((0, 0, 0), (202, 152, 101)),
                    ((100, 160, 200), (10, 250, 30))):
        for _ in range(80):
            v = rng.integers(8, 120, (3, 2))
            if len(set(map(tuple, v.tolist()))) < 3:
                continue
            s = _surf(bg)
            gfxdraw.aapolygon(s, [tuple(int(q) for q in p) for p in v], col)
            got = _arr(s)
            touched = (got != np.array(bg)).any(-1)
            ops = []
            for i in range(3):
                ops += aaline_ops(v[i][0], v[i][1], v[(i + 1) % 3][0], v[(i + 1) % 3][1])
            cnt = np.zeros((W, W), np.int64)
            wmap = np.full((W, W), -1, np.int64)
            pred_pos = np.zeros((W, W), bool)
            for (x, y, w) in ops:
                if 0 <= x < W and 0 <= y < W:
                    cnt[y, x] += 1
                    wmap[y, x] = w
                    if w > 0:
                        pred_pos[y, x] = True
            trials += 1
            pos_ok += np.array_equal(touched, pred_pos)
            ys, xs = np.where(cnt == 1)                    # dst is provably the background here
            for y, x in zip(ys.tolist(), xs.tolist()):
                w = int(wmap[y, x])
                if w <= 0:
                    continue
                pairs.append((np.array(col), np.array(bg), w, got[y, x].copy()))
                if bg == (255, 255, 255) and col == (0, 0, 0):
                    table.setdefault(w, set()).add(int(got[y, x][0]))
    print(f"  position sets identical on {pos_ok}/{trials} random aapolygons")

    print("\n  black-on-white: weight -> observed grey (should be single-valued)")
    ws = sorted(table)
    multi = [w for w in ws if len(table[w]) > 1]
    print(f"  distinct weights seen {len(ws)}, weights with >1 output {len(multi)}")
    show = ws[:: max(1, len(ws) // 24)]
    print("   w  out   255-out   (255*w)>>8   round(255*w/255)")
    for w in show:
        o = sorted(table[w])[0]
        print(f"  {w:3d} {o:4d}   {255-o:5d}   {(255*w)>>8:8d}   {w:8d}")

    src = np.stack([p[0] for p in pairs]).astype(np.int64)
    dst = np.stack([p[1] for p in pairs]).astype(np.int64)
    wg = np.array([p[2] for p in pairs], np.int64)[:, None]
    out = np.stack([p[3] for p in pairs]).astype(np.int64)
    print(f"\n  {len(pairs)} clean observations across 4 (src,dst) pairs")

    def score(name, pred):
        ok = (pred == out).all(-1)
        print(f"  {name:52s} exact {100*ok.mean():6.2f} %  max|err| {np.abs(pred-out).max()}")
        return ok.mean()

    a1 = (255 * wg) >> 8
    score("a=(255w)>>8, (s*a+d*(255-a))//255", np.where(wg == 255, src, (src * a1 + dst * (255 - a1)) // 255))
    score("a=w,        (s*a+d*(255-a))//255", np.where(wg == 255, src, (src * wg + dst * (255 - wg)) // 255))
    score("a=w,        d+((s-d)*a)//255", np.where(wg == 255, src, dst + ((src - dst) * wg) // 255))
    score("a=w,        d+((s-d)*a)>>8", np.where(wg == 255, src, dst + ((src - dst) * wg) // 256))
    score("a=w,        (s*a+d*(256-a))>>8", np.where(wg == 255, src, (src * wg + dst * (256 - wg)) >> 8))
    score("a=w,        d+ (((s-d)*a+((s-d)*a>>8)+1)>>8)",
          np.where(wg == 255, src, dst + (((src - dst) * wg + (((src - dst) * wg) >> 8) + 1) >> 8)))
    # SDL_BLENDMODE_BLEND on 8-bit: out = s*a/255 + d*(1-a/255) with SDL's DIV255 macro
    def div255(v):
        t = v + 128
        return (t + (t >> 8)) >> 8
    score("a=w, DIV255(s*a) + DIV255(d*(255-a))", np.where(wg == 255, src, div255(src * wg) + div255(dst * (255 - wg))))
    score("a=(255w)>>8, DIV255 form", np.where(wg == 255, src, div255(src * a1) + div255(dst * (255 - a1))))
    score("a=w, DIV255(  (s-d)*a ) + d", np.where(wg == 255, src, div255((src - dst) * wg) + dst))

    # brute force: assume out = d + round_k((s-d)*g(w)) for an unknown 0..255 monotone g. Solve g
    # per weight from the black-on-white observations and see whether it is (255w)>>8 or w.
    print("\n  solving the effective alpha per weight from black-on-white (s=0,d=255):")
    bad = []
    for w in ws:
        o = sorted(table[w])[0]
        # out = (0*a + 255*(255-a))//255 = 255-a  under the //255 form
        bad.append((w, 255 - o))
    arr = np.array(bad)
    print(f"    alpha_eff == w              on {100*(arr[:,1]==arr[:,0]).mean():.1f} % of weights")
    print(f"    alpha_eff == (255*w)>>8     on {100*(arr[:,1]==((255*arr[:,0])>>8)).mean():.1f} %")
    print(f"    alpha_eff - w  values: {sorted(set((arr[:,1]-arr[:,0]).tolist()))}")

    pygame.quit()


if __name__ == "__main__":
    main()
