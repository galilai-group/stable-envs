"""Decisive tests for pygame.draw.polygon: float-vs-int vertices, and a per-rule error histogram."""
from __future__ import annotations

import os
from collections import Counter

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
W, H = 300, 200


def ref(points):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), [(float(a), float(b)) for a, b in points])
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def ctrunc(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def float_test():
    print("=== decisive float-vs-int vertex test: tri (0,0),(0,100),(D,100) ===")
    for D in (10.0, 10.4, 10.99, 11.0):
        m = ref([(0, 0), (0, 100), (D, 100)])
        cols = [int(np.nonzero(m[y])[0][-1]) for y in (35, 65, 95)]
        ex = [D * y / 100.0 for y in (35, 65, 95)]
        print(f"  D={D:6.2f}  ref last @y=35,65,95 -> {cols}   exact {[round(e,3) for e in ex]}"
              f"   trunc(D)=10 would give {[int(10*y/100) for y in (35,65,95)]}")


def hist_test():
    print("\n=== per-rule error histogram on a controlled left edge ===")
    names = ["exact_floor", "exact_ceil", "exact_round", "top_trunc", "prev_trunc"]
    hist = {n: Counter() for n in names}
    rng = np.random.default_rng(2)
    for _ in range(300):
        y0, y1 = int(rng.integers(5, 190)), int(rng.integers(5, 190))
        if y0 == y1:
            continue
        xa, xb = int(rng.integers(20, 200)), int(rng.integers(20, 200))
        quad = [(xa, y0), (xb, y1), (280, y1), (280, y0)]
        m = ref(quad)
        lo, hi = min(y0, y1), max(y0, y1)
        dy = y1 - y0
        for y in range(lo + 1, hi):
            col = np.nonzero(m[y])[0]
            if not col.size:
                continue
            got = int(col[0])
            ex = xa + (y - y0) * (xb - xa) / dy
            if y0 <= y1:
                tx, ty, ox = xa, y0, xb
            else:
                tx, ty, ox = xb, y1, xa
            cand = {"exact_floor": int(np.floor(ex)), "exact_ceil": int(np.ceil(ex)),
                    "exact_round": int(np.floor(ex + 0.5)),
                    "top_trunc": tx + ctrunc((y - ty) * (ox - tx), abs(dy)),
                    "prev_trunc": xa + ctrunc((y - y0) * (xb - xa), dy)}
            for n in names:
                hist[n][int(cand[n] - got)] += 1
    for n in names:
        tot = sum(hist[n].values())
        top = sorted(hist[n].items(), key=lambda kv: -kv[1])[:5]
        print(f"  {n:12s} exact {100*hist[n][0]/tot:7.3f}%   "
              f"errors {[(d, c) for d, c in top]}")


def hline_test():
    print("\n=== is the span inclusive of both crossings? single-row probes ===")
    for (xa, xb) in ((40, 60), (60, 40), (40, 40)):
        m = ref([(xa, 30), (xb, 30), (xb, 31), (xa, 31)])
        c = np.nonzero(m.any(0))[0]
        print(f"  quad x {xa}->{xb}: cols {c[0]}..{c[-1]}  rows {np.nonzero(m.any(1))[0].tolist()}")


def main():
    print(f"pygame {pygame.version.ver}")
    float_test()
    hist_test()
    hline_test()


if __name__ == "__main__":
    main()
