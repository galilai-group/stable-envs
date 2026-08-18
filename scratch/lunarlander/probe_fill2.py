"""What ``pygame.draw.polygon`` adds on top of a scanline fill: read the spans off, row by row.

probe_fill.py's scanline candidate is 3.6% short on convex polygons and every missing pixel sits at a
row END, which is what an outline stroke looks like. So: compare the filled polygon against the
outline-only polygon (``width=1``) and against the scanline fill, and see whether the union closes it.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame

pygame.init()
H, W = 140, 180
BG, FG = (255, 255, 255), (0, 0, 0)
arr = lambda s: np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))


def blank():
    s = pygame.Surface((W, H))
    s.fill(BG)
    return s


def lit(s):
    return (arr(s) != np.array(BG, np.uint8)).any(-1)


def ctrunc(a, b):
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def fillpoly(pts):
    v = np.trunc(np.asarray(pts, np.float64)).astype(np.int64)
    vx, vy = v[:, 0], v[:, 1]
    n = len(vx)
    miny, maxy = vy.min(), vy.max()
    out = np.zeros((H, W), bool)
    for y in range(miny, maxy + 1):
        xs = []
        for i in range(n):
            j = (i + 1) % n
            y1, y2, x1, x2 = vy[i], vy[j], vx[i], vx[j]
            if y1 == y2:
                continue
            if y1 > y2:
                y1, y2, x1, x2 = y2, y1, x2, x1
            if (y1 <= y < y2) or (y == maxy and y1 < y <= y2):
                xs.append(ctrunc((y - y1) * (x2 - x1), (y2 - y1)) + x1)
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a, b = int(xs[k]), int(xs[k + 1])
            if 0 <= y < H:
                out[y, max(0, a):min(W, b + 1)] = True
    return out


def outline(pts):
    s = blank()
    pygame.draw.polygon(s, FG, pts, 1)
    return lit(s)


tri = [(100.0, 30.0), (140.0, 100.0), (60.0, 100.0)]
s = blank()
pygame.draw.polygon(s, FG, tri)
pg = lit(s)
mine = fillpoly(tri)
ol = outline(tri)
print("triangle: per-row spans (pygame | scanline | outline-only)")
for y in range(28, 40):
    f = lambda m: (int(np.where(m[y])[0].min()), int(np.where(m[y])[0].max())) if m[y].any() else None
    print(f"  y={y}: {f(pg)}  {f(mine)}  {f(ol)}")
print(f"  pygame == fill|outline ? {np.array_equal(pg, mine | ol)}  "
      f"(diff {int((pg != (mine|ol)).sum())})")

rng = np.random.default_rng(2)
tot = tot2 = lit_tot = 0
worst = (0, None)
for trial in range(300):
    n = int(rng.integers(3, 8))
    c = rng.uniform(45, 130, 2)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(8, 40)
    pts = [(float(c[0] + r * np.cos(a)), float(c[1] + r * np.sin(a))) for a in ang]
    s = blank()
    pygame.draw.polygon(s, FG, pts)
    pg = lit(s)
    u = fillpoly(pts) | outline(pts)
    d = int((pg != u).sum())
    tot += d
    lit_tot += int(pg.sum())
    if d > worst[0]:
        worst = (d, pts, pg, u)
print(f"\nconvex, fill|outline: {tot} differing px over {lit_tot} lit "
      f"({100*tot/max(lit_tot,1):.4f}%), worst {worst[0]}")
if worst[0]:
    d, pts, pg, u = worst
    ys, xs = np.where(pg != u)
    print(f"   {[(int(a), int(b), bool(pg[a,b]), bool(u[a,b])) for a, b in list(zip(ys,xs))[:10]]}")

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
tot = lit_tot = 0
for trial in range(300):
    a = rng.uniform(-np.pi, np.pi)
    ca, sa = np.cos(a), np.sin(a)
    cx, cy = rng.uniform(45, 135), rng.uniform(45, 95)
    pts = [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in LANDER_POLY]
    s = blank()
    pygame.draw.polygon(s, FG, pts)
    pg = lit(s)
    u = fillpoly(pts) | outline(pts)
    tot += int((pg != u).sum())
    lit_tot += int(pg.sum())
print(f"LANDER_POLY, fill|outline: {tot} differing px over {lit_tot} lit")

# what IS the outline? pygame.draw.line per edge at width 1 -> Bresenham. Check that too.
def bres(x0, y0, x1, y1):
    """Integer Bresenham; the candidate for what draw.polygon's border stroke is."""
    pts = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err = dx - dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    return pts


tot = lit_tot = 0
for trial in range(200):
    n = int(rng.integers(3, 8))
    c = rng.uniform(45, 130, 2)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(8, 40)
    pts = [(float(c[0] + r * np.cos(a)), float(c[1] + r * np.sin(a))) for a in ang]
    ol = outline(pts)
    v = np.trunc(np.asarray(pts, np.float64)).astype(np.int64)
    m = np.zeros((H, W), bool)
    for i in range(len(v)):
        j = (i + 1) % len(v)
        for x, y in bres(v[i, 0], v[i, 1], v[j, 0], v[j, 1]):
            if 0 <= y < H and 0 <= x < W:
                m[y, x] = True
    tot += int((ol != m).sum())
    lit_tot += int(ol.sum())
print(f"\noutline-only vs integer Bresenham: {tot} differing px over {lit_tot} lit "
      f"({100*tot/max(lit_tot,1):.4f}%)")
