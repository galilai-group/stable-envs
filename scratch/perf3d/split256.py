"""Where does a 3-D ray-cast frame's time go at the size and batch the STORAGE baseline is quoted at?

The existing diagnosis was taken at 84 px, batch 256. Both knobs change which term dominates: the
mover passes are windowed and scale with the window, the shadow/reflection passes scale with the
frame, and the CPU forward-kinematics loop scales with the BATCH and with nothing else. So a profile
at 84px/b256 cannot rank the terms at 256px/b1024, which is the operating point the JPEG baseline
(16,889 f/s on 8 workers) is quoted at.

Reports, per env: total render, the ``_fk`` call alone, and total-minus-fk, so the split between the
serial CPU kinematics and everything on the GPU is visible without cProfile's attribution noise.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import socket
import statistics
import time

import numpy as np
import torch

import stable_envs as se

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="")
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--json", default="")
a = ap.parse_args()

DEFAULT = [
    # DM Control parity tier (raycast3d)
    "DMCCartpoleBalance", "DMCPendulumSwingup", "DMCAcrobotSwingup", "DMCPointMassEasy",
    "DMCReacherEasy", "DMCFingerSpin", "DMCFingerTurnEasy", "DMCBallInCupCatch",
    # gym MuJoCo parity tier (raycast3d)
    "MujocoReacher", "MujocoPusher", "MujocoInvertedPendulum", "MujocoInvertedDoublePendulum",
    # OGBench (raycast3d composite)
    "PointmazeMediumNavigate", "AntmazeMediumNavigate", "HumanoidmazeMediumNavigate",
    # maze2d / swm_nav
    "Maze2dUmaze", "Maze2dMedium",
]


def _cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def sync(fn, n):
    """``(median wall seconds, process CPU seconds per wall second)`` over the timed loop.

    The cpu% is reported with every figure because these families are HOST-bound, not GPU-bound:
    the per-chunk render is a few hundred Python-issued torch calls, so a row at or above 100% is a
    measurement of this process's interpreter and a row well above it is a measurement of the NODE.
    Without it, the same env on the same node has been seen at 3,664 f/s and at 356 f/s.
    """
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    c0, w0 = _cpu_seconds(), time.perf_counter()
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    cpu = (_cpu_seconds() - c0) / max(1e-9, time.perf_counter() - w0)
    return statistics.median(ts), cpu


names = [n for n in (a.envs.split(",") if a.envs else DEFAULT) if n]
la = os.getloadavg()
out = {"size": a.size, "batch": a.batch, "gpu": torch.cuda.get_device_name(0),
       "node": socket.gethostname(), "loadavg": la, "cpu_count": os.cpu_count(), "envs": {}}
print(f"{torch.cuda.get_device_name(0)}  {a.size}px  b{a.batch}  node {out['node']}  "
      f"loadavg {la[0]:.1f}/{la[1]:.1f} of {os.cpu_count()} cores\n")
print(f"{'env':32s} {'f/s':>10} {'ms/render':>10} {'cpu%':>6} {'fk ms':>9} {'fk %':>6} "
      f"{'gpu ms':>9}  P")
for name in names:
    if name not in se.REGISTRY:
        print(f"{name:32s} {'not registered':>10}")
        continue
    try:
        env = se.make(name, batch=a.batch, size=a.size, device="cuda")
        s = torch.as_tensor(env.sample_states(a.batch, rng=np.random.default_rng(0)),
                            device="cuda", dtype=torch.float32)
        buf = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
        total, cpu = sync(lambda: env.render(s, out=buf), a.reps)
        fk = 0.0
        P = -1
        try:
            sc = env._scene(a.size) if "size" in env._scene.__code__.co_varnames else env._scene()
            fk, _ = sync(lambda: env._fk(s, sc), a.reps)
            r = env._fk(s, sc)
            P = int(r[0].shape[1])
        except Exception as e:                                   # env without the same private shape
            print(f"  ({name}: fk probe unavailable: {type(e).__name__}: {e})")
        rec = dict(fps=a.batch / total, ms=total * 1e3, cpu_frac=cpu, fk_ms=fk * 1e3,
                   fk_frac=fk / total, gpu_ms=(total - fk) * 1e3, prims=P,
                   loadavg=la, node=out["node"])
        out["envs"][name] = rec
        print(f"{name:32s} {a.batch / total:10,.0f} {total * 1e3:10.1f} {100 * cpu:5.0f}% "
              f"{fk * 1e3:9.1f} {100 * fk / total:5.1f}% {(total - fk) * 1e3:9.1f}  {P}")
        del env, s, buf
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name:32s} FAILED {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", a.json)
