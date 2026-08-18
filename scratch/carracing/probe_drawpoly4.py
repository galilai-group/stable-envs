"""Sweep one controlled edge and fit pygame.draw.polygon's crossing rule to it.

Quad = [(xa,y0), (xb,y1), (RIGHT,y1), (RIGHT,y0)] so its LEFT boundary is exactly the edge
(xa,y0)->(xb,y1) and its right boundary is a vertical line.  Also run the mirrored quad so the same
edge is the RIGHT boundary.  For every row, compare the reference's boundary column against five
candidate crossing rules, over both windings and both starting indices.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
W, H = 300, 200
RIGHT, LEFT = 280, 12


def ref(points):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), [(float(a), float(b)) for a, b in points])
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def ctrunc(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def rules(xa, y0, xb, y1, y):
    """Candidate x for the crossing of edge (xa,y0)-(xb,y1) with row y."""
    dy = y1 - y0
    ex = xa + (y - y0) * (xb - xa) / dy                        # exact
    # top-anchored: the C code swaps so (x1,y1) is the endpoint with the SMALLER y
    if y0 <= y1:
        tx, ty, ox = xa, y0, xb
    else:
        tx, ty, ox = xb, y1, xa
    top = tx + ctrunc((y - ty) * (ox - tx), abs(dy))
    prev = xa + ctrunc((y - y0) * (xb - xa), dy)               # anchored at the FIRST endpoint
    return {"exact_floor": int(np.floor(ex)), "exact_ceil": int(np.ceil(ex)),
            "exact_round": int(np.floor(ex + 0.5)), "top_trunc": top, "prev_trunc": prev}


def main():
    print(f"pygame {pygame.version.ver}")
    names = ["exact_floor", "exact_ceil", "exact_round", "top_trunc", "prev_trunc"]
    score = {n: [0, 0] for n in names}
    fails = {n: [] for n in names}
    rng = np.random.default_rng(2)
    for trial in range(500):
        y0, y1 = int(rng.integers(5, 190)), int(rng.integers(5, 190))
        if y0 == y1:
            continue
        xa, xb = int(rng.integers(LEFT + 1, 200)), int(rng.integers(LEFT + 1, 200))
        for side in ("left", "right"):
            if side == "left":
                quad = [(xa, y0), (xb, y1), (RIGHT, y1), (RIGHT, y0)]
            else:
                quad = [(xa, y0), (xb, y1), (LEFT, y1), (LEFT, y0)]
            for rot in (0, 2):
                for wind in (1, -1):
                    q = (quad[::wind] if wind == -1 else quad)
                    q = q[rot:] + q[:rot]
                    # the controlled edge must survive the rotation as an adjacent pair
                    m = ref(q)
                    lo, hi = min(y0, y1), max(y0, y1)
                    for y in range(lo + 1, hi):          # interior rows only: ends are corners
                        col = np.nonzero(m[y])[0]
                        if col.size == 0:
                            continue
                        got = int(col[0]) if side == "left" else int(col[-1])
                        cand = rules(xa, y0, xb, y1, y)
                        for n in names:
                            score[n][1] += 1
                            if cand[n] == got:
                                score[n][0] += 1
                            elif len(fails[n]) < 3:
                                fails[n].append((side, wind, rot, (xa, y0, xb, y1), y, got, cand[n]))
    print("\nboundary-column agreement:")
    for n in names:
        ok, tot = score[n]
        print(f"  {n:12s} {100*ok/max(tot,1):8.4f}%  ({ok}/{tot})")
    best = max(names, key=lambda n: score[n][0])
    print(f"\nfailures of {best}:")
    for f in fails[best]:
        print(f"  {f}")


if __name__ == "__main__":
    main()
