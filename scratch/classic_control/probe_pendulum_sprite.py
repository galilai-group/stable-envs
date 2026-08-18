"""The Pendulum torque arrow: how many distinct bitmaps can it actually be?

``render()`` blits ``assets/clockwise.png`` scaled to ``scale*|u|/2`` and flipped by ``sign(u)``.
If pygame truncates that size to an int, the sprite is one of a SMALL FINITE SET, which can be baked
once and indexed by state instead of being resized per frame -- keeping the frame a pure function of
(theta, last_u) without a resampler in the hot path.
"""
import os, sys
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
import numpy as np, pygame
from os import path
import gymnasium.envs.classic_control as cc
pygame.init()
f = path.join(path.dirname(cc.__file__), "assets/clockwise.png")
img = pygame.image.load(f)
scale = 500/4.4
print(f"asset {img.get_size()}  scale={scale:.4f}")
sizes = set()
for u in np.linspace(-2, 2, 20001):
    s = float(scale*abs(u)/2)
    sizes.add(pygame.transform.smoothscale(img,(s,s)).get_size())
print(f"distinct smoothscale output sizes over u in [-2,2]: {len(sizes)}  "
      f"min={min(sizes)} max={max(sizes)}")
print(f"  -> truncation of {scale:.4f}*|u|/2, i.e. int in [0, {int(scale*2/2)}]")
# is the sprite an exact function of that int size (+ flip)?
a = pygame.transform.smoothscale(img,(37.9,37.9)); b = pygame.transform.smoothscale(img,(37.0,37.0))
aa=pygame.surfarray.pixels3d(a).copy(); bb=pygame.surfarray.pixels3d(b).copy()
print(f"  smoothscale(37.9) == smoothscale(37.0): {np.array_equal(aa,bb)}")
tot = sum(w*h*4 for (w,h) in sizes)*2
print(f"  baking every (size, flip) variant costs {tot/1e6:.1f} MB of RGBA")
