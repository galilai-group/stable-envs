"""Row-by-row dump of pygame.draw.polygon vs the floor-crossing scanline rule, on failing quads."""
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
    pygame.draw.polygon(s, (255, 255, 255), [(float(a), float(b)) for a, b in points])
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def fdiv(a, b):
    return a // b if b > 0 else (-a) // (-b)


def cand_rows(px, py):
    n = len(px)
    miny, maxy = min(py), max(py)
    rows = {}
    if miny == maxy:
        return {miny: (min(px), max(px))}
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
        if len(xs) >= 2:
            rows[y] = (int(np.floor(xs[0])), int(np.ceil(xs[-1])), [round(v, 3) for v in xs])
    return rows


def dump(px, py, tag):
    m = ref(list(zip(px, py)))
    ours = cand_rows(px, py)
    print(f"\n{tag}: px={px} py={py}")
    for y in range(max(min(py), 0), min(max(py), H - 1) + 1):
        c = np.nonzero(m[y])[0]
        r = (int(c[0]), int(c[-1]), int(c.size)) if c.size else None
        o = ours.get(y)
        flag = "" if (r is None) == (o is None) and (r is None or (r[0], r[1]) == (o[0], o[1])) else "   <<<"
        print(f"   y={y:4d}  ref {str(r):18s} ours {str(o):40s}{flag}")


def main():
    print(f"pygame {pygame.version.ver}")
    dump([74, 56, 95, 113], [102, 53, 39, 88], "onscreen failure E")


if __name__ == "__main__":
    main()
