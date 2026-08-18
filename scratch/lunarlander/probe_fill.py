"""Fit ``pygame.draw.polygon``'s fill rule, ``pygame.draw.circle``'s stencil and the flag geometry.

``pygame.draw`` is a different renderer from ``pygame.gfxdraw`` -- measured in probe_raster.py at
7.3% of lit pixels on random polygons -- so the fill rule has to be recovered rather than borrowed.
Candidate here, and the only one tested against inputs it was not fitted on:

    for y in [miny, maxy]:
        for each edge with y1 != y2, ordered so y1 < y2:
            active iff (y1 <= y < y2) or (y == maxy and y1 < y <= y2)
            x = trunc_toward_zero((y - y1) * (x2 - x1) / (y2 - y1)) + x1     # C int division
        sort the crossings; fill [x[2i], x[2i+1]] INCLUSIVE

Everything is on truncated integer vertices, which probe_raster.py already showed (shifting a quad by
0.9 px moves nothing; by 1.0 px moves it one column).
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
    """C integer division: truncation toward zero, not floor."""
    q = np.abs(a) // np.abs(b)
    return np.where((a < 0) != (b < 0), -q, q)


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
                xs.append(int(ctrunc((y - y1) * (x2 - x1), (y2 - y1))) + x1)
        xs.sort()
        for k in range(0, len(xs) - 1, 2):
            a, b = int(xs[k]), int(xs[k + 1])
            if 0 <= y < H:
                out[y, max(0, a):min(W, b + 1)] = True
    return out


rng = np.random.default_rng(1)
tot = lit_tot = 0
worst = (0, None)
for trial in range(400):
    n = int(rng.integers(3, 8))
    c = rng.uniform(45, 130, 2)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(8, 40)                                  # CONVEX: one radius, sorted angles
    pts = [(float(c[0] + r * np.cos(a)), float(c[1] + r * np.sin(a))) for a in ang]
    s = blank()
    pygame.draw.polygon(s, FG, pts)
    pg = lit(s)
    mine = fillpoly(pts)
    d = int((pg != mine).sum())
    tot += d
    lit_tot += int(pg.sum())
    if d > worst[0]:
        worst = (d, pts, pg, mine)
print(f"convex polygons, 400 trials: {tot} differing px over {lit_tot} lit "
      f"({100*tot/max(lit_tot,1):.4f}%)  worst {worst[0]}")
if worst[0]:
    d, pts, pg, mine = worst
    ys, xs = np.where(pg != mine)
    print(f"   worst pts {[(round(a,2), round(b,2)) for a, b in pts]}")
    print(f"   first disagreements: {[(int(a), int(b), bool(pg[a,b]), bool(mine[a,b])) for a, b in list(zip(ys,xs))[:10]]}")

# non-convex too, since the terrain quads are convex but the lander poly is a hexagon that can go
# non-convex under no transform -- still, check the rule generalises
tot = lit_tot = 0
for trial in range(200):
    n = int(rng.integers(3, 9))
    c = rng.uniform(45, 130, 2)
    pts = [(float(c[0] + rng.uniform(-40, 40)), float(c[1] + rng.uniform(-40, 40))) for _ in range(n)]
    s = blank()
    pygame.draw.polygon(s, FG, pts)
    pg = lit(s)
    mine = fillpoly(pts)
    tot += int((pg != mine).sum())
    lit_tot += int(pg.sum())
print(f"arbitrary (possibly self-intersecting) polygons, 200 trials: {tot} differing px over "
      f"{lit_tot} lit ({100*tot/max(lit_tot,1):.4f}%)")

# the actual shapes this env draws
from Box2D.b2 import transform as b2t  # noqa: F401
LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
tot = lit_tot = 0
for trial in range(300):
    a = rng.uniform(-np.pi, np.pi)
    ca, sa = np.cos(a), np.sin(a)
    cx, cy = rng.uniform(40, 140), rng.uniform(40, 100)
    pts = [(cx + ca * x - sa * y, cy + sa * x + ca * y) for x, y in LANDER_POLY]
    s = blank()
    pygame.draw.polygon(s, FG, pts)
    pg = lit(s)
    mine = fillpoly(pts)
    tot += int((pg != mine).sum())
    lit_tot += int(pg.sum())
print(f"LANDER_POLY at 300 random poses: {tot} differing px over {lit_tot} lit")

# ---------------------------------------------------------------- circle stencil
print("\ncircle stencil (particles):")
for r in (2.0,):
    s = blank()
    pygame.draw.circle(s, FG, (60.0, 60.0), r)
    m = lit(s)
    ys, xs = np.where(m)
    print(f"  radius {r}: offsets from (60,60) = {sorted(zip((ys-60).tolist(), (xs-60).tolist()))}")
for cx, cy in ((60.0, 60.0), (60.7, 60.7), (61.0, 60.0), (59.5, 60.5)):
    s = blank()
    pygame.draw.circle(s, FG, (cx, cy), 2.0)
    ys, xs = np.where(lit(s))
    print(f"  centre ({cx},{cy}) -> bbox rows {ys.min()}..{ys.max()} cols {xs.min()}..{xs.max()}")
import gymnasium as gym
e = gym.make("LunarLander-v3", render_mode="rgb_array")
e.reset(seed=0)
u = e.unwrapped
e.step(2)
u.render()
p = u.particles[0]
f = p.fixtures[0]
print(f"  particle fixture radius {f.shape.radius!r} * SCALE = {f.shape.radius*30.0!r} "
      f"-> int {int(f.shape.radius*30.0)}")
print(f"  particle pos {tuple(f.body.transform * f.shape.pos)}  (circle centre before *SCALE)")

# ---------------------------------------------------------------- flag geometry and pygame.draw.line
print("\nflag geometry:")
print(f"  helipad_x1*SCALE = {u.helipad_x1*30.0!r}  x2 {u.helipad_x2*30.0!r}  "
      f"flagy1 {u.helipad_y*30.0!r}  flagy2 {u.helipad_y*30.0+50!r}")
s = blank()
pygame.draw.line(s, FG, (40.0, 30.0), (40.0, 80.0), 1)
ys, xs = np.where(lit(s))
print(f"  vertical width-1 line (40,30)->(40,80): cols {xs.min()}..{xs.max()} rows {ys.min()}..{ys.max()}")
s = blank()
pygame.draw.line(s, FG, (40.6, 30.0), (40.6, 80.0), 1)
ys, xs = np.where(lit(s))
print(f"  same at x=40.6: cols {xs.min()}..{xs.max()}  (truncation vs rounding)")
