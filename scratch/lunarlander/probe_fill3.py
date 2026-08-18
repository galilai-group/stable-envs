"""Row by row on one polygon pygame and the scanline candidate disagree about, plus two variants.

The candidate in probe_fill.py reproduces a 3-vertex polygon exactly and is 3.6% short on 7-vertex
ones, so the difference is not the intersection formula per se. Printing the crossings under integer
division and under float division, next to pygame's own span, says which one it is -- and where the
extra crossings come from.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
H, W = 140, 180
BG, FG = (255, 255, 255), (0, 0, 0)
arr = lambda s: np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))
blank = lambda: (lambda s: (s.fill(BG), s)[1])(pygame.Surface((W, H)))
lit = lambda s: (arr(s) != np.array(BG, np.uint8)).any(-1)


def crossings(v, y, maxy, mode):
    xs = []
    n = len(v)
    for i in range(n):
        j = (i + 1) % n
        y1, y2, x1, x2 = int(v[i, 1]), int(v[j, 1]), int(v[i, 0]), int(v[j, 0])
        if y1 == y2:
            continue
        if y1 > y2:
            y1, y2, x1, x2 = y2, y1, x2, x1
        if (y1 <= y < y2) or (y == maxy and y1 < y <= y2):
            num, den = (y - y1) * (x2 - x1), (y2 - y1)
            if mode == "int":
                q = abs(num) // abs(den)
                xs.append((-q if (num < 0) != (den < 0) else q) + x1)
            else:
                xs.append(int(num / den) + x1)          # C truncation of a double
    return sorted(xs)


PTS = [(132.34, 92.0), (116.37, 103.36), (98.34, 105.56), (96.87, 105.36), (70.6, 46.59),
       (102.12, 29.06), (135.35, 47.04)]
v = np.trunc(np.asarray(PTS, np.float64)).astype(np.int64)
print("vertices (truncated):", [tuple(map(int, p)) for p in v])
s = blank()
pygame.draw.polygon(s, FG, PTS)
pg = lit(s)
maxy = int(v[:, 1].max())
print(" y  | pygame span | int-div crossings | float-div crossings")
for y in range(int(v[:, 1].min()), maxy + 1):
    if not pg[y].any():
        span = None
    else:
        xs_ = np.where(pg[y])[0]
        # every run on this row, not just the extremes
        runs, start = [], xs_[0]
        for a, b in zip(xs_[:-1], xs_[1:]):
            if b != a + 1:
                runs.append((int(start), int(a)))
                start = b
        runs.append((int(start), int(xs_[-1])))
        span = runs
    print(f"{y:4d} | {span} | {crossings(v, y, maxy, 'int')} | {crossings(v, y, maxy, 'float')}")
