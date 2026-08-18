"""Pin pygame.draw.polygon's boundary rule with controlled triangles.

Two questions the random search could not separate:
  (a) does the fill use the FLOAT vertex or its truncation?
  (b) for a boundary at true x = t, which pixel column is the last one painted?
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
W, H = 300, 140


def ref(points):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.polygon(s, (255, 255, 255), points)
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))[..., 0] > 0


def right_edge_rule():
    print("=== ascending right edge: tri (0,0),(0,100),(D,100); true x(y) = D*y/100 ===")
    for D in (14, 101, 37, 200):
        m = ref([(0.0, 0.0), (0.0, 100.0), (float(D), 100.0)])
        print(f"  D={D}")
        for y in (1, 3, 7, 33, 50, 73, 99, 100):
            x = np.nonzero(m[y])[0]
            t = D * y / 100.0
            print(f"    y={y:4d} true {t:8.3f}  ref last {int(x[-1]) if x.size else None:>5}"
                  f"  floor {int(np.floor(t))} ceil {int(np.ceil(t))} round {int(round(t))}")


def left_edge_rule():
    print("\n=== descending left edge: tri (D,0),(0,100),(299,100); true x(y)=D-D*y/100 ===")
    for D in (14, 101):
        m = ref([(float(D), 0.0), (0.0, 100.0), (299.0, 100.0)])
        print(f"  D={D}")
        for y in (1, 3, 7, 33, 50, 73, 99):
            x = np.nonzero(m[y])[0]
            t = D - D * y / 100.0
            print(f"    y={y:4d} true {t:8.3f}  ref first {int(x[0]) if x.size else None:>5}"
                  f"  floor {int(np.floor(t))} ceil {int(np.ceil(t))} round {int(round(t))}")


def float_or_int():
    print("\n=== does the fill see the fraction of a vertex? ===")
    for fx in (0.0, 0.25, 0.5, 0.75, 0.99):
        m = ref([(0.0, 0.0), (0.0, 100.0), (100.0 + fx, 100.0)])
        cols = [int(np.nonzero(m[y])[0][-1]) for y in (25, 50, 75, 99)]
        print(f"  apex x = 100+{fx:4.2f}: last col at y=25,50,75,99 -> {cols}")
    for fy in (0.0, 0.25, 0.5, 0.75, 0.99):
        m = ref([(0.0, 0.0), (0.0, 100.0 + fy), (100.0, 100.0 + fy)])
        rows = np.nonzero(m.any(1))[0]
        print(f"  base y = 100+{fy:4.2f}: rows {rows[0]}..{rows[-1]}")


def horizontal_and_degenerate():
    print("\n=== degenerate: zero-height quad (the small indicator bars) ===")
    for hh in (0.0, 0.4, 0.9, 1.0, 1.6):
        m = ref([(10.0, 50.0), (40.0, 50.0), (40.0, 50.0 + hh), (10.0, 50.0 + hh)])
        rows = np.nonzero(m.any(1))[0]
        cols = np.nonzero(m.any(0))[0]
        print(f"  h={hh:4.2f}: rows {rows.tolist()}  cols {cols[0] if cols.size else None}.."
              f"{cols[-1] if cols.size else None}")


def main():
    print(f"pygame {pygame.version.ver}")
    right_edge_rule()
    left_edge_rule()
    float_or_int()
    horizontal_and_degenerate()


if __name__ == "__main__":
    main()
