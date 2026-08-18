"""Where exactly does a width-2 pygame line sit relative to its endpoints?

Every symmetric candidate came out ~50% wrong, which is the signature of a band that is offset to one
side rather than centred: 2 px cannot straddle a pixel centre symmetrically. This prints the lit rows
and columns directly, so the offset is read off rather than searched for.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import numpy as np
import pygame

pygame.display.init()
W = H = 60


def lit(a, b, w):
    s = pygame.Surface((W, H))
    s.fill((0, 0, 0))
    pygame.draw.line(s, (255, 255, 255), a, b, w)
    return np.moveaxis(pygame.surfarray.array3d(s), 0, 1)[..., 0] > 0


for w in (1, 2, 3, 4):
    m = lit((10, 30), (45, 30), w)
    rows = sorted(set(np.nonzero(m)[0].tolist()))
    cols = sorted(set(np.nonzero(m)[1].tolist()))
    print(f"width={w} horizontal y=30 x=10..45 -> rows {rows}  cols {cols[0]}..{cols[-1]} "
          f"({len(cols)} wide)")

for w in (1, 2, 3, 4):
    m = lit((30, 10), (30, 45), w)
    cols = sorted(set(np.nonzero(m)[1].tolist()))
    rows = sorted(set(np.nonzero(m)[0].tolist()))
    print(f"width={w} vertical x=30 y=10..45 -> cols {cols}  rows {rows[0]}..{rows[-1]} "
          f"({len(rows)} tall)")

print("\nreversed endpoints (does the offset follow the drawing direction?)")
for w in (2, 4):
    m = lit((45, 30), (10, 30), w)
    print(f"  width={w} horizontal drawn right-to-left -> rows "
          f"{sorted(set(np.nonzero(m)[0].tolist()))}")
    m = lit((30, 45), (30, 10), w)
    print(f"  width={w} vertical drawn bottom-to-top  -> cols "
          f"{sorted(set(np.nonzero(m)[1].tolist()))}")

print("\ndiagonal width=2, per-column lit rows (x: rows)")
m = lit((10, 10), (40, 25), 2)
for x in range(10, 20):
    print(f"  x={x}: {sorted(np.nonzero(m[:, x])[0].tolist())}")
