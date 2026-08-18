"""End-to-end throughput: ``qpos`` in, frames out, against dm_control's own renderer.

Both sides do the same job -- take a state vector and produce the picture -- so the comparison is
between "regenerate from state on the GPU" and "regenerate from state through MuJoCo", which is the
choice a dataset consumer actually faces. Our side includes forward kinematics, which is a Python loop
over ``mj_kinematics`` and is what caps these domains rather than the ray-cast.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--batches", default="1,64,256,1024")
    ap.add_argument("--out", default="")
    ap.add_argument("--envs", default="")
    args = ap.parse_args()

    import stable_envs as se
    from stable_envs.envs import dmcontrol as DM

    rows = []
    names = args.envs.split(",") if args.envs else sorted(DM.DOMAINS)
    for name in names:
        # the VISUAL cells cast one ray per pixel and refuse a supersample; asking for one is a
        # ValueError rather than a slower render, so the flag is per-family and not global
        ss = 1 if name in DM.TRACKCOM_DOMAINS else args.ss
        env = se.make(name, batch=8, size=args.size, supersample=ss, device="cuda")
        cls = type(env)
        for B in [int(x) for x in args.batches.split(",")]:
            st = env.sample_states(B, rng=np.random.default_rng(0))
            t = timeit(lambda: env.render(st), iters=5 if B >= 512 else 20)
            rows.append({"env": name, "batch": B, "size": args.size, "ss": ss,
                         "ms": 1e3 * t, "fps": B / t})
            print(f"{name:22s} B={B:5d} {1e3*t:8.2f} ms  {B/t:10.1f} f/s", flush=True)
        # the reference, one frame at a time -- there is no batched dm_control renderer
        ref = cls.reference_env(size=args.size)
        st1 = env.sample_states(8, rng=np.random.default_rng(1))
        t = timeit(lambda: cls.reference_render(ref, st1), iters=5)
        rows.append({"env": name, "batch": 8, "size": args.size, "backend": "dm_control",
                     "ms": 1e3 * t, "fps": 8 / t})
        print(f"{name:22s} dm_control            {1e3*t:8.2f} ms  {8/t:10.1f} f/s", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
