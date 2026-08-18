"""``RAY_BUDGET`` against speed AND peak memory, because it trades one for the other.

The budget sets how many environments a chunk holds. Raising it removes Python: a chunk re-issues the
whole render as a few hundred torch calls and synchronises twice to choose its screen windows, and at
256 px the old budget paid that 32 times per 1024 frames. Lowering it bounds memory: an env whose
mover window declines runs its reflection pass over the full frame, and there a chunk holds a dozen
(B, H*W*ss^2, 3) float32 tensors at once.

So the number is a real trade and neither end of it is free. This measures both ends on the same
states, and prints cpu% so a row that is host-bound says so.
"""
from __future__ import annotations

import argparse
import resource
import statistics
import time

import numpy as np
import torch

import stable_envs as se

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="DMCCartpoleBalance,DMCFingerTurnEasy,DMCBallInCupCatch,"
                                 "MujocoPusher,MujocoReacher")
ap.add_argument("--budgets", default="23,24,25,26")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--reps", type=int, default=5)
a = ap.parse_args()


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


print(f"{torch.cuda.get_device_name(0)}  {a.size}px b{a.batch}\n")
print(f"{'env':30s} {'budget':>8} {'envs/chunk':>11} {'f/s':>10} {'cpu%':>6} {'peak GB':>9}")
for name in [n for n in a.envs.split(",") if n]:
    if name not in se.REGISTRY:
        print(f"{name:30s} not registered")
        continue
    for bexp in [int(b) for b in a.budgets.split(",")]:
        env = se.make(name, batch=a.batch, size=a.size, device="cuda")
        type(env).RAY_BUDGET = 1 << bexp
        s = torch.as_tensor(env.sample_states(a.batch, rng=np.random.default_rng(0)),
                            device="cuda", dtype=torch.float32)
        buf = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
        per = (1 << bexp) // (a.size * a.size * env.ss * env.ss)
        try:
            for _ in range(2):
                env.render(s, out=buf)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            ts = []
            c0, w0 = cpu_s(), time.perf_counter()
            for _ in range(a.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                env.render(s, out=buf)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            cpu = (cpu_s() - c0) / (time.perf_counter() - w0)
            dt = statistics.median(ts)
            print(f"{name:30s} {'1<<' + str(bexp):>8} {per:>11} {a.batch / dt:10,.0f} "
                  f"{100 * cpu:5.0f}% {torch.cuda.max_memory_allocated() / 1e9:9.2f}")
        except torch.OutOfMemoryError:
            print(f"{name:30s} {'1<<' + str(bexp):>8} {per:>11} {'OOM':>10}")
        del env, s, buf
        torch.cuda.empty_cache()
