"""Who owns which pixels of the 96x96 observation?

Four rasterizers meet in one CarRacing frame and they are not equally reproducible:
  gfxdraw.filled_polygon + aapolygon   -- reimplemented bit-exactly in stable_envs/render/gfxdraw.py
  pygame.draw.polygon (car, bars)      -- a DIFFERENT rasterizer, rule not recovered (probe_drawpoly*)
  freesansbold glyphs (score)          -- a finite atlas
  pygame.transform.smoothscale         -- rule recovered exactly (probe_smoothscale4)

This measures what each is worth by disabling it in the OFFICIAL renderer and diffing the 96x96
observation, so the size of any unmatched rule is a number rather than a guess.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import gymnasium
from gymnasium.envs.box2d import car_racing as CR
import pygame
from pygame import gfxdraw


def frames(n=24, seed=0):
    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=seed)
    u = env.unwrapped
    out = []
    rng = np.random.default_rng(0)
    for i in range(n):
        for _ in range(6):
            env.step(np.array([float(rng.uniform(-0.4, 0.4)), 0.4, 0.0], np.float32))
        out.append(u)
    return env, u


def diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return float((d > 0).mean()), float(d.mean()), int(d.max())


def main():
    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    base, no_aa, no_car, no_ind, no_txt = [], [], [], [], []

    orig_aa = gfxdraw.aapolygon
    orig_draw = CR.Car.draw if hasattr(CR, "Car") else None
    from gymnasium.envs.box2d.car_dynamics import Car
    orig_car_draw = Car.draw
    orig_ind = CR.CarRacing._render_indicators

    for i in range(24):
        for _ in range(8):
            env.step(np.array([float(rng.uniform(-0.5, 0.5)), 0.5, 0.0], np.float32))
        base.append(u.render().copy())

        gfxdraw.aapolygon = lambda *a, **k: None
        CR.gfxdraw.aapolygon = gfxdraw.aapolygon
        no_aa.append(u.render().copy())
        gfxdraw.aapolygon = orig_aa
        CR.gfxdraw.aapolygon = orig_aa

        Car.draw = lambda *a, **k: None
        no_car.append(u.render().copy())
        Car.draw = orig_car_draw

        CR.CarRacing._render_indicators = lambda *a, **k: None
        no_ind.append(u.render().copy())
        CR.CarRacing._render_indicators = orig_ind

        # pygame.font.Font is immutable, so the score's share is measured by moving the ONLY
        # thing it depends on: the accumulated reward.
        keep = u.reward
        u.reward = keep + 137.0
        no_txt.append(u.render().copy())
        u.reward = keep

    b = np.stack(base)
    print("=== share of the 96x96 observation owned by each rasterizer (24 frames) ===")
    for name, arr in (("gfxdraw aapolygon (AA rim)", no_aa), ("car (pygame.draw.polygon)", no_car),
                      ("indicator bars (pygame.draw)", no_ind), ("score text (reward +137)", no_txt)):
        f, m, mx = diff(b, np.stack(arr))
        print(f"  {name:30s} {100*f:6.2f}% of pixels move,  mean |d| {m:6.3f}/255,  max {mx}")

    # interior fraction of the reference, for context
    try:
        import scipy.ndimage as ndi
        g = b.astype(np.int16)
        flat = (np.abs(np.diff(g, axis=1)).sum(-1)[:, :, :-1]
                + np.abs(np.diff(g, axis=2)).sum(-1)[:, :-1]) == 0
        interior = np.stack([ndi.minimum_filter(f, size=3) for f in flat])
        print(f"  interior (flat 3x3) fraction of the reference frame: {interior.mean():.4f}")
    except ImportError:
        pass
    env.close()


if __name__ == "__main__":
    main()
