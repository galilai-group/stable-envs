"""Diagnose where the candidate ``draw_fillpoly`` rule and pygame disagree, row by row."""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
W, H = 160, 120


def ref(points):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), points)
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def spans(mask):
    out = {}
    for y in range(mask.shape[0]):
        x = np.nonzero(mask[y])[0]
        if x.size:
            out[y] = (int(x[0]), int(x[-1]), int(x.size))
    return out


def cdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def crossings(px, py, y, maxy, mode):
    n = len(px)
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
            if mode == "cdiv":
                xs.append(cdiv((y - y1) * (x2 - x1), (y2 - y1)) + x1)
            elif mode == "floor":
                xs.append(((y - y1) * (x2 - x1)) // (y2 - y1) + x1)
            elif mode == "float":
                xs.append(int((y - y1) * (x2 - x1) / (y2 - y1) + x1))
            elif mode == "round":
                xs.append(int(round((y - y1) * (x2 - x1) / (y2 - y1))) + x1)
    return sorted(xs)


def main():
    rng = np.random.default_rng(0)
    shown = 0
    stats = {m: [0, 0] for m in ("cdiv", "floor", "float", "round")}
    for trial in range(600):
        c = rng.uniform(-20, W + 20, 2)
        r = rng.uniform(0.5, 60, 2)
        th = rng.uniform(0, 6.28)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        pts = [tuple(p) for p in (np.array([[-r[0], -r[1]], [r[0], -r[1]],
                                            [r[0], r[1]], [-r[0], r[1]]]) @ R.T + c)]
        px = [int(p[0]) for p in pts]
        py = [int(p[1]) for p in pts]
        miny, maxy = min(py), max(py)
        if miny == maxy:
            continue
        sp = spans(ref(pts))
        for mode in stats:
            for y in range(miny, maxy + 1):
                xs = crossings(px, py, y, maxy, mode)
                if len(xs) < 2:
                    continue
                a, b = max(xs[0], 0), min(xs[-1], W - 1)
                got = (a, b) if a <= b else None
                have = sp.get(y)
                have = (have[0], have[1]) if have else None
                if 0 <= y < H:
                    stats[mode][1] += 1
                    stats[mode][0] += int(got == have)
        if shown < 4:
            bad = []
            for y in range(max(miny, 0), min(maxy, H - 1) + 1):
                xs = crossings(px, py, y, maxy, "cdiv")
                a, b = (max(xs[0], 0), min(xs[-1], W - 1)) if len(xs) >= 2 else (1, 0)
                have = sp.get(y)
                if (a <= b) != bool(have) or (have and (a, b) != (have[0], have[1])):
                    bad.append((y, (a, b), have, xs))
            if bad:
                shown += 1
                print(f"\npoly px={px} py={py} miny={miny} maxy={maxy}")
                for y, ours, theirs, xs in bad[:6]:
                    print(f"   y={y:4d} ours {ours}  ref {theirs}   raw crossings {xs}")
    print("\nper-row span agreement by crossing-x rule:")
    for m, (ok, n) in stats.items():
        print(f"  {m:6s} {100*ok/max(n,1):7.3f}%  ({ok}/{n} rows)")


if __name__ == "__main__":
    main()
