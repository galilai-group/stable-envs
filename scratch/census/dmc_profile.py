"""Where does a DM Control frame's time actually go?

The suspicion is that it is neither the ray-caster nor the resolution: ``DMCBase`` walks the batch in
Python and calls ``mujoco.mj_kinematics`` once per environment on the CPU, so the cost per frame is a
serial forward-kinematics solve plus interpreter overhead, with the GPU idle for most of it.

``scene()`` cannot be used to split the phases -- the 3-D envs override ``render()`` and the base
``scene()`` raises -- so this profiles the real call with cProfile and reports the functions that
actually hold the time. OGBench is the control: same ``raycast3d`` backend, 197k f/s at 224px.
"""
from __future__ import annotations

import cProfile
import pstats
import io
import time

import numpy as np
import torch

import stable_envs as se

CASES = [("DMCCartpoleBalance", "5 geoms, no tendon"),
         ("DMCBallInCupCatch", "28-capsule catenary"),
         ("PointmazeMediumNavigate", "OGBench control, same backend")]
B, SIZE = 256, 84


def timeit(fn, n=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


for name, why in CASES:
    if name not in se.REGISTRY:
        print(f"{name:26s} not registered")
        continue
    env = se.make(name, batch=B, size=SIZE, device="cuda")
    st = env.sample_states(B, rng=np.random.default_rng(0))
    dt = timeit(lambda: env.render(st))
    print(f"\n{'=' * 78}\n{name}  [{why}]  {SIZE}px b{B}: "
          f"{dt * 1e3:.2f} ms/render = {B / dt:,.0f} f/s", flush=True)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        env.render(st)
    torch.cuda.synchronize()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(14)
    for line in s.getvalue().splitlines():
        t = line.strip()
        if t and not t.startswith(("Ordered by", "ncalls", "Random listing")) and "function calls" not in t:
            print("   " + line.rstrip()[:150])
    del env
    torch.cuda.empty_cache()
