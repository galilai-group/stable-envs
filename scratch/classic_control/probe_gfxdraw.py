"""Empirically pin down the pygame.gfxdraw primitives gymnasium's classic-control renderers use.

Nothing here is read off the SDL_gfx source; every rule is decided by drawing a primitive and
looking at the pixels. The questions that decide whether a batched GPU rasterizer can be
bit-identical:

  * how a float vertex becomes the Sint16 that SDL_gfx actually rasterises (trunc / round / floor),
  * the scanline fill rule of ``filled_polygon`` (which pixel centres are inside),
  * what ``aapolygon`` contributes once ``filled_polygon`` paints over it,
  * the disc rule of ``filled_circle`` / what ``aacircle`` adds,
  * the sign convention of ``pygame.math.Vector2.rotate_rad``.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from pygame import gfxdraw

W = 64


def surf():
    s = pygame.Surface((W, W))
    s.fill((255, 255, 255))
    return s


def arr(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2))


def mask(s, col=(0, 0, 0)):
    a = arr(s)
    return (a == np.array(col)).all(-1)


def hr(t):
    print("\n" + "-" * 70 + f"\n{t}\n" + "-" * 70)


pygame.init()

# ---------------------------------------------------------------- Vector2.rotate_rad
hr("pygame.math.Vector2.rotate_rad sign convention")
v = pygame.math.Vector2((1.0, 0.0)).rotate_rad(np.pi / 2)
print(f"  (1,0).rotate_rad(pi/2) = ({v[0]:+.6f}, {v[1]:+.6f})")
print("  => R(t) = [[cos,-sin],[sin,cos]] (CCW in a y-DOWN raster == visually clockwise)"
      if v[1] > 0 else "  => R(t) = [[cos,sin],[-sin,cos]]")

# ---------------------------------------------------------------- float -> int
hr("float vertex -> Sint16 conversion in gfxdraw")
# an axis-aligned rect whose left edge sweeps through a fractional coordinate
for xf in (10.0, 10.2, 10.5, 10.7, 10.9, 11.0, -0.5, -0.7):
    s = surf()
    gfxdraw.filled_polygon(s, [(xf, 5.0), (xf, 15.0), (30.0, 15.0), (30.0, 5.0)], (0, 0, 0))
    m = mask(s)
    cols = np.where(m.any(0))[0]
    print(f"  left x={xf:+6.2f} -> first filled column {cols.min() if cols.size else None}"
          f"  last {cols.max() if cols.size else None}")
print("  (trunc-toward-zero of the float is the SDL_gfx Sint16; compare to int(x))")
for xf in (10.2, 10.5, 10.7, -0.5, -0.7):
    print(f"    int({xf}) = {int(xf)}   round = {round(xf)}   floor = {int(np.floor(xf))}")

# ---------------------------------------------------------------- filled_polygon rule
hr("filled_polygon: axis-aligned rect with INTEGER vertices -> which pixels")
for (x0, y0, x1, y1) in [(10, 5, 30, 15), (10, 5, 11, 6), (0, 0, 5, 5)]:
    s = surf()
    gfxdraw.filled_polygon(s, [(x0, y0), (x0, y1), (x1, y1), (x1, y0)], (0, 0, 0))
    m = mask(s)
    ys, xs = np.where(m)
    print(f"  verts x[{x0},{x1}] y[{y0},{y1}] -> x in [{xs.min()},{xs.max()}] "
          f"y in [{ys.min()},{ys.max()}]  n={m.sum()}  "
          f"(w={xs.max()-xs.min()+1} h={ys.max()-ys.min()+1})")

hr("filled_polygon: a rotated quad -- is every filled pixel centre strictly inside?")
th = 0.37
c, sn = np.cos(th), np.sin(th)
loc = np.array([(-12.0, -4.0), (-12.0, 4.0), (12.0, 4.0), (12.0, -4.0)])
R = np.array([[c, -sn], [sn, c]])
wv = loc @ R.T + np.array([32.0, 32.0])
s = surf()
gfxdraw.filled_polygon(s, [tuple(p) for p in wv], (0, 0, 0))
m = mask(s)
iv = np.floor(wv).astype(np.int64)  # what SDL_gfx sees, assuming trunc (coords > 0 here)
ys, xs = np.mgrid[0:W, 0:W]
inside = np.ones((W, W), bool)
for i in range(4):
    a, b = iv[i], iv[(i + 1) % 4]
    cr = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
    inside &= cr >= 0
print(f"  gfx filled n={m.sum()}   sign-test(pixel centre, int verts, >=0) n={inside.sum()}   "
      f"IoU-diff={np.logical_xor(m, inside).sum()}")
inside_s = np.ones((W, W), bool)
for i in range(4):
    a, b = iv[i], iv[(i + 1) % 4]
    cr = (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])
    inside_s &= cr > 0
print(f"  strict > 0 -> n={inside_s.sum()}  diff={np.logical_xor(m, inside_s).sum()}")

# ---------------------------------------------------------------- aapolygon contribution
hr("aapolygon THEN filled_polygon (the gymnasium idiom): what survives?")
s = surf()
gfxdraw.aapolygon(s, [tuple(p) for p in wv], (0, 0, 0))
aa_only = arr(s).copy()
gfxdraw.filled_polygon(s, [tuple(p) for p in wv], (0, 0, 0))
both = arr(s).copy()
s2 = surf()
gfxdraw.filled_polygon(s2, [tuple(p) for p in wv], (0, 0, 0))
fill_only = arr(s2).copy()
d = (both != fill_only).any(-1)
print(f"  aa-only non-white px      : {(aa_only != 255).any(-1).sum()}")
print(f"  fill-only non-white px    : {(fill_only != 255).any(-1).sum()}")
print(f"  aa+fill differs from fill : {d.sum()} px  (these are the AA fringe)")
vals = np.unique(both[d], axis=0)
print(f"  distinct fringe colours   : {len(vals)}  e.g. {vals[:6].tolist()}")
print("  => AA fringe is ALPHA-BLENDED WITH THE BACKGROUND AT DRAW TIME, then the hard fill "
      "overwrites the interior")

hr("aapolygon on an AXIS-ALIGNED rect: does SDL_gfx take the hline/vline fast path (no AA)?")
s = surf()
gfxdraw.aapolygon(s, [(10, 5), (10, 15), (30, 15), (30, 5)], (0, 0, 0))
a = arr(s)
non_white = (a != 255).any(-1)
non_bw = ((a != 255).any(-1) & (a != 0).any(-1))
print(f"  non-white px {non_white.sum()}, of which NOT pure black (i.e. blended) {non_bw.sum()}")
print("  => axis-aligned edges are drawn solid; no anti-aliasing at all"
      if non_bw.sum() == 0 else "  => blended pixels present")

# ---------------------------------------------------------------- filled_circle
hr("filled_circle rule (integer centre + radius)")
for r in (2, 5, 10):
    s = surf()
    gfxdraw.filled_circle(s, 32, 32, r, (0, 0, 0))
    m = mask(s)
    ys, xs = np.mgrid[0:W, 0:W]
    d2 = (xs - 32) ** 2 + (ys - 32) ** 2
    for rule, cand in (("d2<=r^2", d2 <= r * r),
                       ("d2<=r^2+r", d2 <= r * r + r),
                       ("d2<(r+.5)^2", d2 < (r + 0.5) ** 2)):
        print(f"  r={r:2d} gfx n={m.sum():4d}  {rule:12s} n={cand.sum():4d} "
              f"xor={np.logical_xor(m, cand).sum()}")
    ys_, xs_ = np.where(m)
    print(f"        bbox x[{xs_.min()-32},{xs_.max()-32}] y[{ys_.min()-32},{ys_.max()-32}]")

hr("aacircle + filled_circle: fringe size")
for r in (2, 5, 10):
    s = surf()
    gfxdraw.aacircle(s, 32, 32, r, (0, 0, 0))
    gfxdraw.filled_circle(s, 32, 32, r, (0, 0, 0))
    both = arr(s)
    s2 = surf()
    gfxdraw.filled_circle(s2, 32, 32, r, (0, 0, 0))
    d = (both != arr(s2)).any(-1)
    print(f"  r={r:2d}: fringe px {d.sum():4d}  (fill n={mask(s2).sum()})")

# ---------------------------------------------------------------- hline / vline
hr("gfxdraw.hline(surf, x1, x2, y, col) -- inclusive?")
s = surf()
gfxdraw.hline(s, 5, 12, 20, (0, 0, 0))
m = mask(s)
ys, xs = np.where(m)
print(f"  hline(5,12,20) -> x in [{xs.min()},{xs.max()}] y in [{ys.min()},{ys.max()}] n={m.sum()}")
s = surf()
gfxdraw.hline(s, 0, W, 20, (0, 0, 0))  # cartpole calls with x2 == screen_width (out of range)
m = mask(s)
xs = np.where(m.any(0))[0]
print(f"  hline(0,{W},20) -> x in [{xs.min()},{xs.max()}] n={m.sum()} (clipped)")
s = surf()
gfxdraw.vline(s, 20, 5, 12, (0, 0, 0))
m = mask(s)
ys, xs = np.where(m)
print(f"  vline(20,5,12) -> x in [{xs.min()},{xs.max()}] y in [{ys.min()},{ys.max()}] n={m.sum()}")

# ---------------------------------------------------------------- pygame.draw.aalines
hr("pygame.draw.aalines (mountain-car terrain) -- blended?")
s = surf()
pygame.draw.aalines(s, points=[(2.0, 10.3), (30.0, 40.7), (60.0, 12.1)], closed=False,
                    color=(0, 0, 0))
a = arr(s)
nz = (a != 255).any(-1)
blend = nz & (a != 0).any(-1)
print(f"  non-white {nz.sum()}, blended {blend.sum()} -> genuinely anti-aliased, sub-pixel endpoints")

pygame.quit()
print("\nDONE")
