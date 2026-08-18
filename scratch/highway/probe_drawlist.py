"""Capture the exact sequence of pygame draw calls highway-env issues for one frame.

Reading the source gives the intent; this gives the arguments. Everything that matters for parity
(int truncation in pos2pix, the stripe window, sprite rect offsets, the blit origin) shows up here as
numbers instead of as a guess.
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import sys
import numpy as np
import pygame
import gymnasium
import highway_env  # noqa: F401
from highway_env.road.graphics import LaneGraphics

EID = sys.argv[1] if len(sys.argv) > 1 else "highway-v0"
NSTEP = int(sys.argv[2]) if len(sys.argv) > 2 else 0

calls = []
_line, _rect, _poly = pygame.draw.line, pygame.draw.rect, pygame.draw.polygon
_blit = pygame.Surface.blit
_rotate = pygame.transform.rotate


def line(surf, color, a, b, w=1):
    calls.append(("line", id(surf), tuple(color), tuple(a), tuple(b), w))
    return _line(surf, color, a, b, w)


def rect(surf, color, r, w=0, *a, **k):
    calls.append(("rect", id(surf), tuple(color), tuple(r), w))
    return _rect(surf, color, r, w, *a, **k)


def poly(surf, color, pts, w=0, *a, **k):
    calls.append(("poly", id(surf), tuple(color), len(pts), w))
    return _poly(surf, color, pts, w, *a, **k)


def rotate(surf, angle):
    out = _rotate(surf, angle)
    calls.append(("rotate", id(surf), surf.get_size(), float(angle), out.get_size(), id(out)))
    return out


pygame.draw.line, pygame.draw.rect, pygame.draw.polygon = line, rect, poly
pygame.transform.rotate = rotate

from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.road.graphics import RoadObjectGraphics

for _cls in (VehicleGraphics, RoadObjectGraphics):
    _orig = _cls.blit_rotate

    def _mk(orig, tag):
        def f(surf, image, pos, angle, origin_pos=None, show_rect=False):
            w, h = image.get_size()
            box = [pygame.math.Vector2(p) for p in [(0, 0), (w, 0), (w, -h), (0, -h)]]
            br = [p.rotate(angle) for p in box]
            min_box = (min(br, key=lambda p: p[0])[0], min(br, key=lambda p: p[1])[1])
            max_box = (max(br, key=lambda p: p[0])[0], max(br, key=lambda p: p[1])[1])
            op = origin_pos if origin_pos is not None else (w / 2, h / 2)
            pv = pygame.math.Vector2(op[0], -op[1])
            pm = pv.rotate(angle) - pv
            org = (pos[0] - op[0] + min_box[0] - pm[0], pos[1] - op[1] - max_box[1] + pm[1])
            calls.append((tag, id(surf), image.get_size(), tuple(pos), float(angle),
                          tuple(round(float(o), 5) for o in org)))
            return orig(surf, image, pos, angle, origin_pos, show_rect)
        return staticmethod(f)

    _cls.blit_rotate = _mk(_orig, "blit_rot:" + _cls.__name__)

env = gymnasium.make(EID, render_mode="rgb_array")
env.reset(seed=0)
for _ in range(NSTEP):
    env.step(env.action_space.sample())
u = env.unwrapped
u.render()                      # build the viewer
calls.clear()
u.render()
surf = u.viewer.sim_surface
print(f"{EID}: sim_surface id={id(surf)} size={surf.get_size()} scaling={surf.scaling} "
      f"origin={surf.origin} stripes_count="
      f"{int(2*(surf.get_height()+surf.get_width())/(LaneGraphics.STRIPE_SPACING*surf.scaling))}")
print(f"ego pos={u.vehicle.position} heading={u.vehicle.heading}")
print(f"{len(calls)} draw calls")
n_sim = sum(1 for c in calls if c[1] == id(surf))
print(f"  {n_sim} directly on sim_surface")
for c in calls[:40]:
    print("   ", c)
print("  ...")
for c in calls[-60:]:
    print("   ", c)
