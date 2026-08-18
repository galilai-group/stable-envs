"""Throughput, every cell and every configuration, in ONE process on ONE node.

These families are host-sensitive -- the same env has measured 3,664 f/s idle and 271 contended -- and
the parity survey deliberately packed several array tasks onto a node, so its f/s column is not
comparable across cells. This script exists to produce the column that is: one process, one node, all
rows back to back, ``--exclusive`` so nothing else is on the host.

``cpu_frac`` is not reported and should not be: the render spin-waits, so it reads 1.0 whether the GPU
is saturated or idle. What is reported instead is the fraction of wall time covered by CUDA kernels,
from the profiler, which separates "this node is contended" from "this env is launch bound".

Three kinds of row:

* ``render`` -- ``env.render`` into a pre-allocated buffer at ss=1 and ss=2. The repo's own bench
  protocol (``scratch/perf/bench.py``): five warm-ups, median of ``--reps``.
* ``corrector`` -- inference alone, replayed from a CUDA graph, fp16, channels_last.
* ``end_to_end`` -- render at ss=1 into a buffer and then run the corrector over it, both inside the
  timed region, because that is what a consumer pays. Composing the two measured costs is checked
  against this rather than substituted for it.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402

import numpy as np
import torch

import nets_ss  # noqa: E402

#: The 8-worker JPEG-in-Lance median this repo measures every renderer against
#: (``scratch/perf/bench.py``).
BASELINE = 16889.0


def _timed(fn, reps: int, warm: int = 5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float((np.percentile(ts, 75) - np.percentile(ts, 25))
                                       / np.median(ts))


def _device_share(fn):
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        return sum(int(e.self_device_time_total) for e in prof.key_averages()) / 1e6 / wall
    except Exception:
        return None


def graph_runner(model, size: int, batch: int, shrink: float):
    """A CUDA-graph replay of the corrector, or the eager call if capture fails.

    Capture is worth having and is not worth insisting on: at batch 1024 the corrector is a couple of
    dozen kernels and the launch overhead is a real fraction of a 20 us budget, but a capture failure
    must degrade to a measured eager number rather than to no number.
    """
    x = torch.randint(0, 256, (batch, size, size, 3), dtype=torch.uint8, device="cuda")
    for _ in range(3):
        model(x, shrink)
    torch.cuda.synchronize()
    try:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                model(x, shrink)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            model(x, shrink)
        return x, (lambda: g.replay()), True
    except Exception:
        return x, (lambda: model(x, shrink)), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--sizes", default="default,256")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--nets", default="espcn_w32_d4_u4,espcn_w64_d4_u4,espcn_w64_d6_u4,"
                                      "espcn_w96_d6_u4,unet_w48_d2_u4,unet_w64_d3_u4")
    ap.add_argument("--out", default=os.path.join(K.LOGS, "fps.json"))
    a = ap.parse_args()

    cells = a.cells.split(",") if a.cells else [
        "DMCAcrobotSwingup", "DMCBallInCupCatch", "DMCCartpoleBalance", "DMCCartpoleSwingup",
        "DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard", "DMCPendulumSwingup",
        "DMCPointMassEasy", "DMCReacherEasy", "DMCReacherHard",
        "MujocoReacher", "MujocoPusher", "MujocoInvertedPendulum", "MujocoInvertedDoublePendulum"]
    res = {"host": os.uname().nodename, "device": torch.cuda.get_device_name(0),
           "batch": a.batch, "reps": a.reps, "baseline_8worker_jpeg": BASELINE,
           "loadavg_start": K.loadavg(), "corrector": {}, "cells": {}}

    # ---- corrector cost, which is a property of (architecture, size) and not of the cell ----
    for size in (84, 240, 256):
        for name in a.nets.split(","):
            if not nets_ss.compatible(name, size):
                print(f"corrector {name}@{size}px: skipped (grid does not divide)", flush=True)
                continue
            m = nets_ss.build(name).cuda().half().to(memory_format=torch.channels_last).eval()
            with torch.no_grad():
                x, run, captured = graph_runner(m, size, a.batch, 0.0)
                dt, iqr = _timed(run, a.reps)
            res["corrector"][f"{name}@{size}"] = {
                "us_per_frame": 1e6 * dt / a.batch, "iqr_frac": iqr, "cuda_graph": captured,
                "params": nets_ss.n_params(m), "gflops_per_frame": nets_ss.gflops_per_frame(name, size)}
            print(f"corrector {name}@{size}px: {1e6*dt/a.batch:.2f} us/frame "
                  f"({nets_ss.n_params(m)/1e3:.0f}k params, graph={captured})", flush=True)
            del m, x, run
            torch.cuda.empty_cache()

    # ---- render, and render + corrector end to end ----
    for cell in cells:
        cls = K.cls_of(cell)
        sizes = []
        for s in a.sizes.split(","):
            sizes.append(int(cls.default_size) if s == "default" else int(s))
        row = {"default_size": int(cls.default_size), "sizes": {}}
        for size in sorted(set(sizes)):
            env = K.make(cell, batch=1, size=size)
            pool = np.asarray(env.sample_states(64, rng=np.random.default_rng(0)), np.float32)
            st = torch.as_tensor(np.resize(pool, (a.batch, pool.shape[1])), device="cuda")
            out = torch.empty((a.batch, size, size, 3), dtype=torch.uint8, device="cuda")
            e = {}
            for ss in (1, 2):
                fn = lambda ss=ss: env.render(st, size=size, supersample=ss, out=out)
                dt, iqr = _timed(fn, a.reps)
                e[f"render_ss{ss}"] = {"fps": a.batch / dt, "us_per_frame": 1e6 * dt / a.batch,
                                       "iqr_frac": iqr, "device_kernel_share": _device_share(fn),
                                       "x_baseline": a.batch / dt / BASELINE}
                print(f"{cell} {size}px ss={ss}: {a.batch/dt:.0f} f/s "
                      f"(x{a.batch/dt/BASELINE:.2f} vs 8-worker JPEG, "
                      f"device_share {e[f'render_ss{ss}']['device_kernel_share']})", flush=True)
            for net in [n for n in a.nets.split(",") if nets_ss.compatible(n, size)]:
                m = nets_ss.build(net).cuda().half().to(memory_format=torch.channels_last).eval()
                with torch.no_grad():
                    for _ in range(3):
                        m(out)
                    torch.cuda.synchronize()
                    fn = lambda m=m: m(env.render(st, size=size, supersample=1, out=out))
                    dt, iqr = _timed(fn, a.reps)
                e.setdefault("end_to_end_ss1_plus_corrector", {})[net] = {
                    "fps": a.batch / dt, "us_per_frame": 1e6 * dt / a.batch,
                    "iqr_frac": iqr, "x_baseline": a.batch / dt / BASELINE,
                    "implied_corrector_us": 1e6 * dt / a.batch - e["render_ss1"]["us_per_frame"]}
                print(f"{cell} {size}px ss=1 + {net}: {a.batch/dt:.0f} f/s "
                      f"(x{a.batch/dt/BASELINE:.2f}, implied corrector "
                      f"{e['end_to_end_ss1_plus_corrector'][net]['implied_corrector_us']:.2f} us)",
                      flush=True)
                del m
                torch.cuda.empty_cache()
            row["sizes"][str(size)] = e
            del env, st, out
            torch.cuda.empty_cache()
        res["cells"][cell] = row
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    res["loadavg_end"] = K.loadavg()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
