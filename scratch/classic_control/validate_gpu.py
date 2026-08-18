"""Does the batched GPU rasterizer draw the same frames as gymnasium? Measured, on identical states.

Both renderers are pure functions of the state, so they are pointed at the same states -- half from
real rollouts (the distribution that matters) and half sampled across the full render range (the
poses a rollout almost never reaches, where a geometry bug would hide). "Do they look the same" is
then a pixel count with an exact answer.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gymnasium as gym

import gpu_render as R


def _states_cartpole(env, n, rng):
    out = []
    env.reset(seed=0)
    for i in range(n):
        if i % 2 == 0:                                             # real rollout
            o, r, t, tr, _ = env.step(int(rng.integers(2)))
            if t or tr:
                env.reset()
        else:                                                      # full render range
            env.state = np.array([rng.uniform(-2.4, 2.4), rng.uniform(-3, 3),
                                  rng.uniform(-np.pi, np.pi), rng.uniform(-3, 3)])
        out.append((float(env.state[0]), float(env.state[2])))
    return out


def _states_pendulum(env, n, rng):
    out = []
    env.reset(seed=0)
    for i in range(n):
        if i % 2 == 0:
            env.step(rng.uniform(-2, 2, size=1).astype(np.float32))
        else:
            env.state = np.array([rng.uniform(-2 * np.pi, 2 * np.pi), rng.uniform(-8, 8)])
        env.last_u = None                                          # sprite layer measured separately
        out.append((float(env.state[0]),))
    return out


def _states_acrobot(env, n, rng):
    out = []
    env.reset(seed=0)
    for i in range(n):
        if i % 2 == 0:
            o, r, t, tr, _ = env.step(int(rng.integers(3)))
            if t or tr:
                env.reset()
        else:
            env.state = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi), 0.0, 0.0])
        out.append((float(env.state[0]), float(env.state[1])))
    return out


def _states_mountaincar(env, n, rng):
    out = []
    env.reset(seed=0)
    for i in range(n):
        if i % 2 == 0:
            o, r, t, tr, _ = env.step(int(rng.integers(3)))
            if t or tr:
                env.reset()
        else:
            env.state = np.array([rng.uniform(-1.2, 0.6), 0.0])
        out.append((float(env.state[0]),))
    return out


CASES = [("CartPole-v1", R.CartPoleGPU, _states_cartpole),
         ("Pendulum-v1", R.PendulumGPU, _states_pendulum),
         ("Acrobot-v1", R.AcrobotGPU, _states_acrobot),
         ("MountainCar-v0", R.MountainCarGPU, _states_mountaincar)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/"
                                     "scratch/classic_control/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  torch={torch.__version__}  gymnasium={gym.__version__}")

    for name, cls, sampler in CASES:
        print("\n" + "=" * 80)
        print(f"{name}   {cls.__name__}   state_dim={cls.STATE_DIM}   {cls.H}x{cls.W}")
        print("=" * 80)
        env = gym.make(name, render_mode="rgb_array").unwrapped
        rng = np.random.default_rng(0)
        states = sampler(env, args.n, rng)
        cpu = []
        for s in states:
            if name == "CartPole-v1":
                env.state = np.array([s[0], 0.0, s[1], 0.0])
            elif name == "Pendulum-v1":
                env.state = np.array([s[0], 0.0])
                env.last_u = None
            elif name == "Acrobot-v1":
                env.state = np.array([s[0], s[1], 0.0, 0.0])
            else:
                env.state = np.array([s[0], 0.0])
            cpu.append(env.render().copy())
        cpu = np.stack(cpu)
        env.close()

        st = np.asarray(states, np.float64)
        for aa in (True, False):
            rend = cls(device=dev, aa=aa)
            gpu = rend.render(st).cpu().numpy()
            d = np.abs(gpu.astype(np.int64) - cpu.astype(np.int64)).max(-1)
            nbad = int((d > 0).sum())
            per_frame = (d > 0).reshape(len(st), -1).sum(1)
            print(f"  aa={str(aa):5s}  exact {100*(d==0).mean():9.5f} %   "
                  f"differing px {nbad:8d}/{d.size}   worst {int(d.max()):3d}/255   "
                  f"frames bit-identical {int((per_frame==0).sum())}/{len(st)}")
            if aa and nbad:
                ys = np.argsort(-per_frame)[:3]
                for i in ys[:3]:
                    if per_frame[i] == 0:
                        break
                    yy, xx = np.where(d[i] > 0)
                    print(f"     worst frame {i} state={st[i].round(4).tolist()} "
                          f"{per_frame[i]} px, rows {yy.min()}..{yy.max()} cols {xx.min()}..{xx.max()}")
            if aa:
                import imageio.v2 as imageio
                k = min(4, len(st))
                rows = []
                for i in range(k):
                    dif = np.clip(np.abs(gpu[i].astype(np.int16) - cpu[i].astype(np.int16)) * 8,
                                  0, 255).astype(np.uint8)
                    sep = np.zeros((cls.H, 3, 3), np.uint8)
                    rows.append(np.concatenate([cpu[i], sep, gpu[i], sep, dif], 1))
                imageio.imwrite(os.path.join(args.out, f"{name.replace('/','_')}_sheet.png"),
                                np.concatenate(rows, 0))

        # float32 geometry, to show why the pipeline is float64
        rend = cls(device=dev, aa=True)
        gpu64 = rend.render(st).cpu().numpy()
        v32 = rend.render(st.astype(np.float32).astype(np.float64)).cpu().numpy()
        print(f"  float32-rounded states change {int((np.abs(gpu64.astype(int)-v32.astype(int))>0).sum())} px "
              f"(geometry sensitivity to the state itself)")

    print("\nwrote side-by-side sheets to", args.out)


if __name__ == "__main__":
    main()
