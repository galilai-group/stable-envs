"""Parity + throughput for the landed ``stable_envs`` classic-control envs.

Runs the same measurement the conformance suite runs, but prints the numbers rather than asserting
them, and adds the throughput table -- reported in Gpixel/s as well as frames/s because a
classic-control frame is 25x a PushT frame and raw frames/s is not comparable across resolutions.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

NAMES = ["CartPole", "Pendulum", "Acrobot", "MountainCar", "MountainCarContinuous"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[64, 1024, 4096])
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")

    print("\n" + "=" * 92)
    print(f"PARITY vs the official gymnasium renderer, {args.n} states/env")
    print("=" * 92)
    print(f"{'env':24s} {'HxW':10s} {'dim':>4s} {'exact':>11s} {'interior':>10s} "
          f"{'diff px':>12s} {'worst':>6s}")
    for name in NAMES:
        env = se.make(name, batch=args.n, device="cuda")
        cls = type(env)
        ref = cls.reference_env(size=env.size)
        st = env.sample_states(args.n)
        official = cls.reference_render(ref, st)
        ours = env.render(st).cpu().numpy()
        assert ours.shape == official.shape, (ours.shape, official.shape)
        r = C.pixel_agreement(ours, official)
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        H, W = env.native
        print(f"{name:24s} {f'{H}x{W}':10s} {env.state_dim:4d} {100*r['exact_frac']:10.5f}% "
              f"{100*r['interior_exact_frac']:9.5f}% {int((d>0).sum()):7d}/{d.size:<9d} "
              f"{r['max_abs']:5d}")

    print("\n" + "=" * 92)
    print("CONFORMANCE checks (the ones that do not need pytest)")
    print("=" * 92)
    for name in NAMES:
        env = se.make(name, batch=16, device="cuda")
        rt = C.check_state_roundtrip(env)["max_abs_err"]
        dd = C.check_state_dims_matter(env)["dead_dims"]
        bi = C.check_batch_invariance(env)["max_abs_diff"]
        det = C.check_determinism(env)["identical"]
        iv = C.check_variations_do_something(env)["inert_variations"]
        print(f"  {name:24s} roundtrip {rt:.1f}  dead_dims {dd}  batch_inv {bi}  "
              f"det {det}  inert {iv}")

    print("\n" + "=" * 92)
    print("THROUGHPUT  (aa=1 is the bit-exact mode; aa=0 is the documented speed/fidelity switch)")
    print("=" * 92)
    print(f"{'env':24s} {'mode':6s} {'B':>6s} {'frames/s':>11s} {'Gpix/s':>9s} {'vs pygame':>10s}")
    import gymnasium as gym
    for name in NAMES:
        env = se.make(name, batch=1, device="cuda")
        H, W = env.native
        g = gym.make(env.reference_id, render_mode="rgb_array").unwrapped
        type(env).reference_set_state(g, env.sample_states(1)[0])
        for _ in range(5):
            g.render()
        t0 = time.perf_counter()
        for _ in range(100):
            g.render()
        cpu = 100 / (time.perf_counter() - t0)
        print(f"{name:24s} {'pygame':6s} {1:6d} {cpu:11.1f} {cpu*H*W/1e9:9.4f} {'1.0x':>10s}")
        for aa in (1, 0):
            for B in args.batches:
                e = se.make(name, batch=B, device="cuda")
                st = torch.as_tensor(e.sample_states(B), device="cuda")
                var = {"rendering.antialias": aa}
                for _ in range(3):
                    e.render(st, variation=var)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(args.reps):
                    e.render(st, variation=var)
                torch.cuda.synchronize()
                f = B * args.reps / (time.perf_counter() - t0)
                print(f"{'':24s} {'aa' if aa else 'noaa':6s} {B:6d} {f:11.1f} "
                      f"{f*H*W/1e9:9.4f} {f/cpu:9.0f}x")
                del e
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
