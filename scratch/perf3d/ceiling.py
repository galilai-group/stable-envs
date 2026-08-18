"""What sets the ceiling for the ray-cast parity families, decomposed rather than estimated.

Answers four things per env, at the operating point the storage baseline is quoted at:

  1. how much of a render is GPU kernels and how much is the interpreter issuing them (device-side
     profiler time against the wall clock — NOT cpu%, which spins and reads 100% either way);
  2. what the rate would be if the interpreter cost went to zero;
  3. what each pass costs, so the binding term has a name;
  4. the FLOOR: a render with the movers removed, i.e. the plate composite and the box filter alone.
     That is the cost of producing a 256 px frame at all on this path, and no amount of work on the
     ray passes can go below it.

Reports loadavg and cpu_frac on every row, because these families are host-sensitive: the same env
has measured 3,664 f/s on an idle node and 271 f/s on a contended one, and a ceiling computed on a
busy node is a property of the cluster rather than of the renderer.
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
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

import stable_envs as se
from stable_envs.render import raycast3d as R3

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="DMCCartpoleBalance,DMCPendulumSwingup,DMCAcrobotSwingup,"
                                 "DMCPointMassEasy,DMCReacherEasy,DMCFingerSpin,DMCFingerTurnEasy,"
                                 "DMCBallInCupCatch,MujocoReacher,MujocoPusher,"
                                 "MujocoInvertedPendulum,MujocoInvertedDoublePendulum")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--size", type=int, default=256)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--json", default="")
a = ap.parse_args()


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def timed(fn, n):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    c0, w0 = cpu_s(), time.perf_counter()
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts), (cpu_s() - c0) / (time.perf_counter() - w0)


def kernel_seconds(fn, n=3):
    """Summed DEVICE duration of every kernel, per call. Device entries only — ``key_averages``
    also lists the aten op that launched them and its time is the sum of theirs."""
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    evs = [e for e in pr.key_averages() if e.device_type == DeviceType.CUDA]
    return sum(int(getattr(e, "self_device_time_total", 0) or 0) for e in evs) / (n * 1e6)


la = os.getloadavg()
out = {"size": a.size, "batch": a.batch, "gpu": torch.cuda.get_device_name(0),
       "node": socket.gethostname(), "loadavg": la, "cpu_count": os.cpu_count(), "envs": {}}
print(f"{torch.cuda.get_device_name(0)}  {a.size}px b{a.batch}  node {out['node']}  "
      f"loadavg {la[0]:.1f}/{la[1]:.1f} of {os.cpu_count()} cores\n")
print(f"{'env':30s} {'f/s':>8} {'wall ms':>8} {'kern ms':>8} {'kern%':>6} {'cpu%':>5} "
      f"{'no-python f/s':>13} {'floor ss2':>10} {'floor ss1':>10} {'floorkern2':>11} {'P':>3}")

for name in [n for n in a.envs.split(",") if n]:
    if name not in se.REGISTRY:
        print(f"{name:30s} not registered")
        continue
    try:
        env = se.make(name, batch=a.batch, size=a.size, device="cuda")
        s = torch.as_tensor(env.sample_states(a.batch, rng=np.random.default_rng(0)),
                            device="cuda", dtype=torch.float32)
        buf = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
        wall, cpu = timed(lambda: env.render(s, out=buf), a.reps)
        kern = kernel_seconds(lambda: env.render(s, out=buf))

        # THE FLOOR: the same render with no primitives to draw. Every pass that depends on a mover
        # drops out and what is left is the plate promoted to float, the box filter and the cast to
        # uint8 -- the irreducible cost of emitting a supersampled 256 px frame on this path.
        #
        # Measured at BOTH supersample settings, because the hypothesis the floor tests is that the
        # ceiling is set by the anti-aliasing the parity claim requires rather than by the ray casts:
        # ss=2 evaluates 512x512 = 262,144 samples for a 65,536-pixel output. If the floor scales with
        # the sample count then that is the mechanism and no amount of work on the casts can move it.
        # Emptying ``prims`` does not work on every cell -- ball_in_cup's tendon and the gym MuJoCo
        # cells size their FK output from the original primitive count -- so the movers are removed
        # by making the CASTS return "nothing hit" instead. Every mover pass then contributes an
        # all-False coverage mask, the compactions select zero rays, and what is left is exactly the
        # plate promoted to float, the box filter and the cast to uint8.
        cls = type(env)
        keep_cast, keep_occ, keep_ss = cls._cast, cls._occluded, env.ss

        def _miss(sc, o, d, xpos, xmat, sizes):
            return (torch.full(o.shape[:2], R3.INF, device=o.device, dtype=o.dtype),
                    torch.zeros_like(o),
                    torch.zeros(o.shape[:2], dtype=torch.long, device=o.device))

        def _clear(sc, p, L, sl, xpos, xmat, sizes, skip=None, eps=1e-6):
            return torch.zeros(p.shape[:-1], dtype=torch.bool, device=p.device)

        floors, floor_kern = {}, {}
        try:
            cls._cast, cls._occluded = staticmethod(_miss), staticmethod(_clear)
            for ss in (2, 1):
                env.ss = ss
                fl, _ = timed(lambda: env.render(s, out=buf), max(3, a.reps // 2))
                floors[ss] = a.batch / fl
                floor_kern[ss] = a.batch / kernel_seconds(lambda: env.render(s, out=buf))
        except Exception as e:                                            # noqa: BLE001
            print(f"   ({name}: floor probe unavailable: {type(e).__name__}: {e})")
        finally:
            cls._cast, cls._occluded, env.ss = keep_cast, keep_occ, keep_ss
        floor = floors.get(2, float("nan"))

        P = len(env._scene(a.size).prims.kind)
        rec = dict(fps=a.batch / wall, wall_ms=wall * 1e3, kernel_ms=kern * 1e3,
                   kernel_frac=kern / wall, cpu_frac=cpu, fps_no_python=a.batch / kern,
                   fps_floor=floor, fps_floor_ss1=floors.get(1),
                   fps_floor_kernel=floor_kern.get(2), fps_floor_kernel_ss1=floor_kern.get(1),
                   prims=P, node=out["node"], loadavg=la)
        out["envs"][name] = rec
        print(f"{name:30s} {a.batch / wall:8,.0f} {wall * 1e3:8.1f} {kern * 1e3:8.1f} "
              f"{100 * kern / wall:5.0f}% {100 * cpu:4.0f}% {a.batch / kern:13,.0f} "
              f"{floor:10,.0f} {floors.get(1, float('nan')):10,.0f} "
              f"{floor_kern.get(2, float('nan')):11,.0f} {P:3d}")
        del env, s, buf
        torch.cuda.empty_cache()
    except Exception as e:                                                # noqa: BLE001
        print(f"{name:30s} FAILED {type(e).__name__}: {e}")
        torch.cuda.empty_cache()

if a.json:
    with open(a.json, "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote", a.json)
