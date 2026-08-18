"""How much of a render is GPU kernels and how much is the interpreter issuing them?

``cpu%`` cannot answer this. PyTorch's default synchronisation SPINS, so a render that is entirely
GPU-bound also reports 100% of a core, and a render that is entirely Python-bound reports the same
number. The two need separating before anything is built on top of the answer, because they have
opposite fixes: kernel time is removed by fusing arithmetic, issue time is removed by issuing less.

``torch.profiler`` gives both from one run — the summed DEVICE duration of every kernel against the
wall clock of the region. The gap is the interpreter (plus any real stall). It also names the top
kernels, which says WHICH arithmetic to fuse rather than leaving it to a guess.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import stable_envs as se

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="DMCCartpoleBalance,DMCReacherEasy,DMCFingerTurnEasy,"
                                 "DMCBallInCupCatch,MujocoPusher,MujocoReacher")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--top", type=int, default=8)
a = ap.parse_args()

print(f"{torch.cuda.get_device_name(0)}  {a.size}px b{a.batch}\n")
for name in [n for n in a.envs.split(",") if n]:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    env = se.make(name, batch=a.batch, size=a.size, device="cuda")
    s = env.sample_states(a.batch, rng=np.random.default_rng(0))
    for _ in range(3):
        env.render(s)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(3):
        env.render(s)
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t0) / 3

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        for _ in range(3):
            env.render(s)
        torch.cuda.synchronize()
    # DEVICE-side entries only. ``key_averages`` lists both the aten op and the kernels it launched,
    # and the op's ``self_device_time_total`` is the sum of its children's -- so summing everything
    # counts every kernel twice and reports 150% of wall, which is how the first version of this
    # script "measured" a negative interpreter share.
    evs = [e for e in pr.key_averages() if e.device_type == DeviceType.CUDA]
    dev = sum(int(getattr(e, "self_device_time_total", 0) or 0) for e in evs) / 3e6
    nlaunch = sum(int(e.count) for e in evs) / 3

    print(f"{name:24s} wall {wall * 1e3:7.1f} ms   kernels {dev * 1e3:7.1f} ms "
          f"({100 * dev / wall:4.1f}%)   interpreter {100 * (1 - dev / wall):4.1f}%   "
          f"{nlaunch:,.0f} launches   ceiling-if-python-free {a.batch / dev:,.0f} f/s")
    top = sorted(evs, key=lambda e: -(getattr(e, "self_device_time_total", 0) or 0))[:a.top]
    for e in top:
        d = (getattr(e, "self_device_time_total", 0) or 0) / 3e3
        if d <= 0:
            continue
        print(f"      {e.key[:58]:58s} {d:7.2f} ms  {100 * d / (dev * 1e3):5.1f}%  "
              f"x{int(e.count) / 3:,.0f}")
    del env
    torch.cuda.empty_cache()
