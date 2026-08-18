"""Can Pendulum's torque arrow be an exact LUT, and is ``u = 0`` the same frame as ``last_u = None``?

Three things decide whether the torque arrow belongs in the state vector or in a documented exclusion:

  * ``smoothscale`` truncates its target size, so the sprite is one of 114 sizes x 2 flips. If the
    zero-size case blits nothing, then ``u = 0`` and ``last_u = None`` are the SAME frame and the state
    can be a plain ``(theta, torque)`` with no "is there a torque" flag.
  * the blit is an alpha composite of a single-colour mask onto an opaque surface, so its integer
    rounding has to be recovered the same way the gfxdraw blend was.
  * whichever rule wins has to reproduce the official frame for EVERY size, not on average.
"""
from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
from os import path

import gymnasium as gym
import gymnasium.envs.classic_control as cc

SCALE = 500 / 4.4
OFF = 250


def arr(s):
    return np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).astype(np.int64)


def main():
    pygame.init()
    img = pygame.image.load(path.join(path.dirname(cc.__file__), "assets/clockwise.png"))
    print(f"asset {img.get_size()} bitsize {img.get_bitsize()} masks {img.get_masks()}")

    print("\n" + "=" * 74)
    print("1. does u = 0 render the same frame as last_u = None?")
    print("=" * 74)
    env = gym.make("Pendulum-v1", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    same = True
    for th in np.linspace(-3, 3, 9):
        env.state = np.array([th, 0.0])
        env.last_u = None
        a = env.render().copy()
        env.state = np.array([th, 0.0])
        env.last_u = np.float32(0.0)
        b = env.render().copy()
        if not np.array_equal(a, b):
            same = False
            d = np.abs(a.astype(int) - b.astype(int)).max()
            print(f"  theta={th:+.3f}: DIFFER, max {d}")
    print(f"  identical on all 9 thetas: {same}")
    print("  => the state can be (theta, torque) with u=0 meaning 'no arrow'" if same
          else "  => a separate 'has torque' flag would be needed")

    print("\n" + "=" * 74)
    print("2. the blit: recover the composite rule from a known background")
    print("=" * 74)
    # one scaled sprite, blitted onto a flat background, gives (src_rgb, alpha, dst) -> out directly
    n_ok = {}
    for bg in ((255, 255, 255), (204, 77, 77), (0, 0, 0), (100, 160, 200)):
        s = pygame.Surface((500, 500))
        s.fill(bg)
        sc = pygame.transform.smoothscale(img, (91.0, 91.0))
        sc = pygame.transform.flip(sc, True, True)
        s.blit(sc, (OFF - sc.get_rect().centerx, OFF - sc.get_rect().centery))
        got = arr(s)
        rgb = np.transpose(np.array(pygame.surfarray.pixels3d(sc)), (1, 0, 2)).astype(np.int64)
        al = np.transpose(np.array(pygame.surfarray.pixels_alpha(sc)), (1, 0)).astype(np.int64)
        h, w = al.shape
        y0, x0 = OFF - sc.get_rect().centery, OFF - sc.get_rect().centerx
        sub = got[y0:y0 + h, x0:x0 + w]
        d = np.asarray(bg, np.int64)
        a3 = al[..., None]
        cands = {
            "d + ((s-d)*a)//255": d + ((rgb - d) * a3) // 255,
            "d + ((s-d)*a)>>8": d + ((rgb - d) * a3) // 256,
            "(s*a + d*(255-a))//255": (rgb * a3 + d * (255 - a3)) // 255,
            "s*a/255 + d*(255-a)/255 (SDL DIV255)":
                (((rgb * a3 + 128) + ((rgb * a3 + 128) >> 8)) >> 8)
                + (((d * (255 - a3) + 128) + ((d * (255 - a3) + 128) >> 8)) >> 8),
            "d + ((s-d)*a + ((s-d)*a>>8) + 1 >>8)":
                d + (((rgb - d) * a3 + (((rgb - d) * a3) >> 8) + 1) >> 8),
        }
        for k, v in cands.items():
            n_ok[k] = n_ok.get(k, 0) + int((v == sub).all(-1).sum())
        tot = h * w
    for k, v in sorted(n_ok.items(), key=lambda kv: -kv[1]):
        print(f"  {k:42s} {100*v/(4*tot):7.3f} % of {tot} px x 4 backgrounds")

    print("\n" + "=" * 74)
    print("3. full-frame replay: bake the 114 sprites, composite, compare to the official frame")
    print("=" * 74)
    best = max(n_ok, key=n_ok.get)
    print(f"  using: {best}")

    def blend(rgb, a3, d):
        return {
            "d + ((s-d)*a)//255": lambda: d + ((rgb - d) * a3) // 255,
            "d + ((s-d)*a)>>8": lambda: d + ((rgb - d) * a3) // 256,
            "(s*a + d*(255-a))//255": lambda: (rgb * a3 + d * (255 - a3)) // 255,
            "s*a/255 + d*(255-a)/255 (SDL DIV255)": lambda: (
                ((rgb * a3 + 128) + ((rgb * a3 + 128) >> 8)) >> 8)
                + (((d * (255 - a3) + 128) + ((d * (255 - a3) + 128) >> 8)) >> 8),
            "d + ((s-d)*a + ((s-d)*a>>8) + 1 >>8)": lambda: d + (
                ((rgb - d) * a3 + (((rgb - d) * a3) >> 8) + 1) >> 8),
        }[best]()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import ref_envs as E

    rng = np.random.default_rng(0)
    worst = 0
    nbad = 0
    for trial in range(24):
        th = float(rng.uniform(-np.pi, np.pi))
        u = float(rng.uniform(-2, 2))
        env.state = np.array([th, 0.0])
        env.last_u = np.float32(u)
        official = env.render().copy()

        # ours: vector part (pre-flip), then sprite, then axle, then flip
        cv = np.full((500, 500, 3), 255, np.int64)
        sc_, off = SCALE, 250
        rl, rw = 1.0 * sc_, 0.2 * sc_
        rod = [(E._rot(c, th + np.pi / 2)[0] + off, E._rot(c, th + np.pi / 2)[1] + off)
               for c in [(0.0, -rw / 2), (0.0, rw / 2), (rl, rw / 2), (rl, -rw / 2)]]
        E._poly(cv, rod, (204, 77, 77), True)
        E._circle(cv, off, off, int(rw / 2), (204, 77, 77), True)
        re = E._rot((rl, 0.0), th + np.pi / 2)
        E._circle(cv, int(re[0] + off), int(re[1] + off), int(rw / 2), (204, 77, 77), True)
        # sprite
        n = int(float(sc_ * abs(np.float32(u)) / 2))
        if n > 0:
            spr = pygame.transform.smoothscale(img, (float(sc_ * abs(np.float32(u)) / 2),) * 2)
            spr = pygame.transform.flip(spr, bool(np.float32(u) > 0), True)
            rgb = np.transpose(np.array(pygame.surfarray.pixels3d(spr)), (1, 0, 2)).astype(np.int64)
            al = np.transpose(np.array(pygame.surfarray.pixels_alpha(spr)), (1, 0)).astype(np.int64)
            h, w = al.shape
            y0, x0 = off - spr.get_rect().centery, off - spr.get_rect().centerx
            sub = cv[y0:y0 + h, x0:x0 + w]
            cv[y0:y0 + h, x0:x0 + w] = blend(rgb, al[..., None], sub)
        E._circle(cv, off, off, int(0.05 * sc_), (0, 0, 0), True)
        ours = np.flipud(cv).astype(np.uint8)
        d = int(np.abs(ours.astype(int) - official.astype(int)).max())
        nb = int((np.abs(ours.astype(int) - official.astype(int)).max(-1) > 0).sum())
        worst = max(worst, d)
        nbad += nb
        if nb and trial < 3:
            print(f"  trial {trial}: th={th:+.3f} u={u:+.3f} size={n} -> {nb} px, worst {d}")
    print(f"  24 frames: {nbad} differing px total, worst {worst}/255")
    print(f"  => sprite is EXACTLY reproducible from a baked LUT" if nbad == 0
          else "  => sprite composite is NOT exactly reproduced by this rule")

    print("\n  distinct (size, flip) sprites to bake and their memory:")
    sizes = sorted({int(float(SCALE * abs(np.float32(u)) / 2)) for u in np.linspace(-2, 2, 40001)})
    tot = sum(n * n * 4 for n in sizes) * 2
    print(f"    {len(sizes)} sizes x 2 flips = {2*len(sizes)} bitmaps, {tot/1e6:.2f} MB RGBA")
    pygame.quit()


if __name__ == "__main__":
    main()
