"""Does the CPU reference reproduce gymnasium's frames, pixel for pixel? Measured, not asserted."""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym

import ref_envs as E
import ref_gfx as G


def score(name, got, ref):
    d = np.abs(got.astype(np.int64) - ref.astype(np.int64)).max(-1)
    n = d.size
    print(f"  {name:28s} exact {100*(d==0).mean():8.4f} %   "
          f"differing px {int((d>0).sum()):6d}/{n}   worst {int(d.max()):3d}/255")
    return d


def main():
    rng = np.random.default_rng(0)

    print("=" * 78)
    print("CartPole-v1  (600x400)  state = (x, theta)")
    print("=" * 78)
    env = gym.make("CartPole-v1", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    states, frames = [], []
    for i in range(40):
        env.state = np.array([rng.uniform(-2.2, 2.2), rng.uniform(-2, 2),
                              rng.uniform(-0.5, 0.5), rng.uniform(-2, 2)])
        frames.append(env.render().copy())
        states.append((env.state[0], env.state[2]))
    frames = np.stack(frames)
    for aa in (False, True):
        got = np.stack([E.render_cartpole(x, th, aa=aa) for x, th in states])
        score(f"aa={aa}", got, frames)
    env.close()

    print("\n" + "=" * 78)
    print("Pendulum-v1  (500x500)  state = (theta,)   [reset frames: last_u is None]")
    print("=" * 78)
    env = gym.make("Pendulum-v1", render_mode="rgb_array").unwrapped
    ths, frames = [], []
    for i in range(40):
        env.reset(seed=i)
        env.state = np.array([rng.uniform(-np.pi, np.pi), 0.0])
        env.last_u = None
        frames.append(env.render().copy())
        ths.append(float(env.state[0]))
    frames = np.stack(frames)
    for aa in (False, True):
        got = np.stack([E.render_pendulum(t, aa=aa) for t in ths])
        score(f"aa={aa}", got, frames)
    env.close()

    print("\n" + "=" * 78)
    print("Acrobot-v1   (500x500)  state = (theta1, theta2)")
    print("=" * 78)
    env = gym.make("Acrobot-v1", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    sts, frames = [], []
    for i in range(40):
        env.state = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), 0.0, 0.0])
        frames.append(env.render().copy())
        sts.append((float(env.state[0]), float(env.state[1])))
    frames = np.stack(frames)
    for aa in (False, True):
        got = np.stack([E.render_acrobot(a, b, aa=aa) for a, b in sts])
        score(f"aa={aa}", got, frames)
    env.close()

    print("\n" + "=" * 78)
    print("MountainCar-v0 (600x400)  state = (position,)   [static terrain+flag baked]")
    print("=" * 78)
    env = gym.make("MountainCar-v0", render_mode="rgb_array").unwrapped
    env.reset(seed=0)
    # bake the static layer: render at a position, then take the official frame and delete the car.
    # Cleaner: draw the terrain/flag with pygame ONCE, which is what a real implementation would do
    # at construction time (it is a constant of the env, not of the state).
    import pygame
    from pygame import gfxdraw
    P = E.MOUNTAINCAR
    surf = pygame.Surface((P["W"], P["H"]))
    surf.fill((255, 255, 255))
    xs = np.linspace(P["minp"], P["maxp"], 100)
    ys = E.mc_height(xs)
    xys = list(zip((xs - P["minp"]) * P["scale"], ys * P["scale"]))
    pygame.draw.aalines(surf, points=xys, closed=False, color=(0, 0, 0))
    flagx = int((0.5 - P["minp"]) * P["scale"])
    flagy1 = int(float(E.mc_height(0.5)) * P["scale"])
    flagy2 = flagy1 + 50
    base_pre = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2)).astype(np.int64)
    poststatic = [("vline", flagx, flagy1, flagy2),
                  ("flag", [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)])]

    sts, frames = [], []
    for i in range(40):
        env.state = np.array([rng.uniform(-1.2, 0.6), 0.0])
        frames.append(env.render().copy())
        sts.append(float(env.state[0]))
    frames = np.stack(frames)

    def render_mc(pos, aa):
        cv = base_pre.copy()
        E._poly  # noqa
        ang = np.cos(3 * pos)
        cxp = (pos - P["minp"]) * P["scale"]
        cyp = P["clearance"] + float(E.mc_height(pos)) * P["scale"]
        l, r, t, b = -20.0, 20.0, 20.0, 0.0
        car = [(E._rot(c, ang)[0] + cxp, E._rot(c, ang)[1] + cyp)
               for c in [(l, b), (l, t), (r, t), (r, b)]]
        E._poly(cv, car, P["car_c"], aa)
        for c in [(10.0, 0.0), (-10.0, 0.0)]:
            c = E._rot(c, ang)
            E._circle(cv, int(c[0] + cxp), int(c[1] + cyp), int(P["carheight"] / 2.5),
                      P["wheel_c"], aa)
        # flag is drawn AFTER the car in the official renderer
        cv[flagy1:flagy2 + 1, flagx] = np.array((0, 0, 0), np.int64)
        tri = [(flagx, flagy2), (flagx, flagy2 - 10), (flagx + 25, flagy2 - 5)]
        E._poly(cv, tri, (204, 204, 0), aa)
        return np.flipud(cv).astype(np.uint8)

    _ = poststatic
    for aa in (False, True):
        got = np.stack([render_mc(p, aa) for p in sts])
        score(f"aa={aa}", got, frames)
    env.close()
    _ = G


if __name__ == "__main__":
    main()
