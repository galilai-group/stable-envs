"""LunarLander draws through THREE rasterizers in one frame. Characterise each one against pygame.

The classic-control family goes through ``pygame.gfxdraw`` alone, which is integer arithmetic and
therefore exactly reproducible (``stable_envs/render/gfxdraw.py``). LunarLander does not:

    pygame.draw.polygon(color1)   +  gfxdraw.aapolygon(color1)  +  pygame.draw.aalines(color2)
    pygame.draw.circle(color1) x2                                   (particles)
    pygame.draw.line(width=1) + pygame.draw.polygon + gfxdraw.aapolygon   (flags)

So this asks, per primitive: is ``pygame.draw``'s fill the same pixel set as ``gfxdraw``'s, and if not
what IS it; what does ``draw.circle`` paint at the radius the particles use; and what does
``draw.aalines`` add on top. Nothing here is read off the C source -- every rule is fitted to pygame's
own output and then re-checked on inputs it was not fitted on.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
from pygame import gfxdraw

pygame.init()
H, W = 120, 160
BG, FG = (255, 255, 255), (0, 0, 0)


def blank():
    s = pygame.Surface((W, H))
    s.fill(BG)
    return s


def arr(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))


def lit(s):
    return (arr(s) != np.array(BG, np.uint8)).any(-1)


# ---------------------------------------------------------------- (a) draw.polygon vs filled_polygon
def gfx_fill_mask(pts):
    """stable_envs.render.gfxdraw.fill_polygon_mask, on the host, for one polygon."""
    v = np.trunc(np.asarray(pts, np.float64)).astype(np.int64)
    vx, vy = v[:, 0], v[:, 1]
    ay, by = np.roll(vy, 1), vy
    ax, bx = np.roll(vx, 1), vx
    sw = ay > by
    ylo, yhi = np.where(sw, by, ay), np.where(sw, ay, by)
    xlo, xhi = np.where(sw, bx, ax), np.where(sw, ax, bx)
    live = ay != by
    maxy = vy.max()
    out = np.zeros((H, W), bool)
    for y in range(H):
        act = live & (((y >= ylo) & (y < yhi)) | ((y == maxy) & (y > ylo) & (y <= yhi)))
        if act.sum() < 2:
            continue
        X = (65536 * (y - ylo)) // np.maximum(yhi - ylo, 1) * (xhi - xlo) + 65536 * xlo
        r16 = lambda v: (v >> 16) + ((v & 0x8000) >> 15)
        xa, xb = r16(X[act].min() + 1), r16(X[act].max() - 1)
        out[y, max(0, xa):min(W, xb + 1)] = True
    return out


rng = np.random.default_rng(0)
tot_pg_gfx = tot_px = 0
worst = None
for trial in range(300):
    n = int(rng.integers(3, 7))
    c = rng.uniform(30, 110, 2)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(6, 35, n)
    pts = [(float(c[0] + rr * np.cos(a)), float(c[1] + rr * np.sin(a))) for a, rr in zip(ang, r)]
    s = blank()
    pygame.draw.polygon(s, FG, pts)
    pg = lit(s)
    gx = gfx_fill_mask(pts)
    d = int((pg != gx).sum())
    tot_pg_gfx += d
    tot_px += int(pg.sum())
    if worst is None or d > worst[0]:
        worst = (d, pts, pg, gx)
print(f"(a) pygame.draw.polygon vs gfxdraw.filled_polygon, 300 random polygons: "
      f"{tot_pg_gfx} differing px over {tot_px} lit ({100*tot_pg_gfx/max(tot_px,1):.3f}%)")
d, pts, pg, gx = worst
print(f"    worst polygon: {d} px; pygame lit {pg.sum()}, gfx lit {gx.sum()}")
ys, xs = np.where(pg != gx)
print(f"    example disagreements (y,x, pygame, gfx): "
      f"{[(int(a), int(b), bool(pg[a,b]), bool(gx[a,b])) for a, b in list(zip(ys, xs))[:8]]}")

# does pygame.draw.polygon truncate its float vertices?
for dx in (0.0, 0.4, 0.9, 1.0):
    s = blank()
    pygame.draw.polygon(s, FG, [(10 + dx, 10.0), (40 + dx, 10.0), (40 + dx, 40.0), (10 + dx, 40.0)])
    m = lit(s)
    xs_ = np.where(m.any(0))[0]
    ys_ = np.where(m.any(1))[0]
    print(f"    axis-aligned quad shifted by {dx}: cols {xs_.min()}..{xs_.max()} rows {ys_.min()}..{ys_.max()}")

# ---------------------------------------------------------------- (b) draw.circle
print("\n(b) pygame.draw.circle row half-widths, by radius (particles use radius 2.0):")
for r in (1, 2, 3, 4, 5):
    s = blank()
    pygame.draw.circle(s, FG, (60.0, 60.0), float(r))
    m = lit(s)
    ys_ = np.where(m.any(1))[0]
    hw = []
    for y in range(60, ys_.max() + 1):
        xs_ = np.where(m[y])[0]
        hw.append(int(xs_.max() - 60) if len(xs_) else -1)
    print(f"    r={r}: rows {ys_.min()}..{ys_.max()}, half-widths from centre {hw}, "
          f"lit {int(m.sum())}")
# float centre: truncated or rounded?
for cx in (60.0, 60.4, 60.5, 60.9):
    s = blank()
    pygame.draw.circle(s, FG, (cx, 60.0), 2.0)
    xs_ = np.where(lit(s).any(0))[0]
    print(f"    centre x={cx}: cols {xs_.min()}..{xs_.max()}")
for r in (1.9, 2.0, 2.4, 2.5, 2.9):
    s = blank()
    pygame.draw.circle(s, FG, (60.0, 60.0), r)
    print(f"    radius {r}: lit {int(lit(s).sum())}")

# ---------------------------------------------------------------- (c) draw.aalines on top
print("\n(c) what pygame.draw.aalines adds on top of a filled polygon")
poly = [(30.3, 20.7), (70.9, 25.2), (60.1, 70.4), (25.5, 62.2)]
C1, C2 = (128, 102, 230), (77, 77, 128)
s = blank()
pygame.draw.polygon(s, C1, poly)
a_fill = arr(s).copy()
gfxdraw.aapolygon(s, poly, C1)
a_aap = arr(s).copy()
pygame.draw.aalines(s, C2, True, poly)
a_all = arr(s).copy()
d1 = (a_fill != a_aap).any(-1)
d2 = (a_aap != a_all).any(-1)
print(f"    aapolygon changed {int(d1.sum())} px; aalines then changed {int(d2.sum())} px")
inside = a_fill.reshape(-1, 3)
print(f"    colours after fill: {len(np.unique(inside,axis=0))}; after aapolygon "
      f"{len(np.unique(a_aap.reshape(-1,3),axis=0))}; after aalines "
      f"{len(np.unique(a_all.reshape(-1,3),axis=0))}")
# how deep does the aaline reach into the fill? (distance from the fill boundary)
fillmask = (a_fill != np.array(BG, np.uint8)).any(-1)
import scipy.ndimage as ndi
er1 = ndi.binary_erosion(fillmask, np.ones((3, 3), bool))
er2 = ndi.binary_erosion(er1, np.ones((3, 3), bool))
print(f"    aalines pixels strictly inside erode-1: {int((d2 & er1).sum())}, "
      f"inside erode-2: {int((d2 & er2).sum())}   (interior_mask erodes by 1)")

# ---------------------------------------------------------------- (d) flag drawn once per fixture
print("\n(d) the flags are drawn INSIDE the per-fixture loop; repeating an aapolygon is not idempotent")
for reps in (1, 2, 3, 7):
    s = blank()
    for _ in range(reps):
        pygame.draw.line(s, (255, 255, 255), (40.0, 30.0), (40.0, 80.0), 1)
        tri = [(40.0, 80.0), (40.0, 70.0), (65.0, 75.0)]
        pygame.draw.polygon(s, (204, 204, 0), tri)
        gfxdraw.aapolygon(s, tri, (204, 204, 0))
    globals()[f"flag{reps}"] = arr(s).copy()
for reps in (2, 3, 7):
    d = np.abs(globals()[f"flag{reps}"].astype(np.int16) - flag1.astype(np.int16))
    print(f"    {reps} repetitions vs 1: max {int(d.max())}, differing px {int((d.max(-1)>0).sum())}")
