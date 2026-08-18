import os
os.environ.setdefault("SDL_VIDEODRIVER","dummy")
import numpy as np, pygame
from os import path
import gymnasium.envs.classic_control as cc
pygame.init()
f = path.join(path.dirname(cc.__file__),"assets/clockwise.png")
img = pygame.image.load(f)
print("clockwise.png size", img.get_size(), "flags", img.get_flags(), "bitsize", img.get_bitsize(),
      "has alpha", img.get_alpha(), "masks", img.get_masks())
a = pygame.surfarray.pixels3d(img)
print("rgb unique colors", len(np.unique(a.reshape(-1,3),axis=0)))
try:
    al = pygame.surfarray.pixels_alpha(img)
    print("alpha min/max", al.min(), al.max(), "frac alpha<255", (al<255).mean())
except Exception as e:
    print("no per-pixel alpha:", e)
