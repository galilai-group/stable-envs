"""What does supersampling actually buy, and what does it cost, per cell.

Everything downstream depends on four numbers that nobody has written down together:

1. ``ss=2`` against the OFFICIAL frame -- the shipped row, and the one the parity assertion is about.
2. ``ss=1`` against the OFFICIAL frame -- the "do nothing" row a corrector has to beat. If its
   interior-exactness is already 1.0 then anti-aliasing is not what the parity claim rests on and the
   whole question narrows to the all-pixel floor; if it is below 1.0 the corrector has to REPAIR
   interior pixels, which is a different and much harder job.
3. ``ss=1`` against ``ss=2`` -- the residual the corrector is being asked to learn, and how sparse it
   is. A residual that touches 3 % of pixels is a different learning problem from one that touches
   40 %.
4. Render throughput at ``ss=1`` and ``ss=2``, 256 px, batch 1024, with the device-side kernel share
   from the profiler alongside -- because these families spin-wait and ``cpu_frac`` reads 1.0 whether
   the GPU is saturated or idle.

The states are the parity test's own draw (``default_rng(20260811)``, 64 states) so every fidelity
number here is directly comparable to the assertion the cell has to pass.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402  (sets MUJOCO_GL before torch/mujoco import)

import numpy as np
import torch

import stable_envs as se  # noqa: E402


def throughput(env, states_pool: np.ndarray, size: int, batch: int, ss: int, reps: int = 11):
    """Median f/s of ``env.render`` into a pre-allocated buffer -- the repo's own bench protocol."""
    st = torch.as_tensor(np.resize(states_pool, (batch, states_pool.shape[1])), device="cuda")
    out = torch.empty((batch, size, size, 3), dtype=torch.uint8, device="cuda")
    for _ in range(5):
        env.render(st, size=size, supersample=ss, out=out)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        env.render(st, size=size, supersample=ss, out=out)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    res = {"fps": batch / dt, "seconds_per_batch": dt, "us_per_frame": 1e6 * dt / batch,
           "iqr_frac": float((np.percentile(ts, 75) - np.percentile(ts, 25)) / dt),
           "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "loadavg": K.loadavg()}
    # Device-side kernel share. cpu_frac is 1.0 on these families whether the GPU is saturated or
    # idle, because the render spin-waits; the fraction of wall time covered by CUDA kernels is the
    # quantity that separates "this node is contended" from "this env is launch bound".
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            env.render(st, size=size, supersample=ss, out=out)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        dev = sum(int(e.self_device_time_total) for e in prof.key_averages())
        res.update(device_kernel_share=dev / 1e6 / wall, prof_wall_s=wall, n_kernels=len(prof.key_averages()))
    except Exception as e:                                # a profiler failure must not lose the f/s
        res.update(device_kernel_share=None, prof_error=repr(e))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=",".join(K.CELLS))
    ap.add_argument("--sizes", default="256")
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--out", default=os.path.join(K.LOGS, "facts.json"))
    a = ap.parse_args()

    res = {"device": torch.cuda.get_device_name(0), "host": os.uname().nodename,
           "batch": a.batch, "parity_state_seed": K.PARITY_STATE_SEED,
           "loadavg_start": K.loadavg(), "cells": {}}
    for name in a.cells.split(","):
        cls = K.cls_of(name)
        row = {"default_size": int(cls.default_size),
               "default_supersample": int(getattr(cls, "default_supersample", 1)),
               "parity_interior_min": float(getattr(cls, "parity_interior_min", 1.0)),
               "sizes": {}}
        print(f"\n=== {name} (default {cls.default_size} px, ss={row['default_supersample']}) ===",
              flush=True)
        for size in [int(s) for s in a.sizes.split(",")]:
            t0 = time.perf_counter()
            states = np.asarray(
                K.make(name, batch=1, size=size).sample_states(
                    a.n_states, rng=np.random.default_rng(K.PARITY_STATE_SEED)), np.float32)
            official = K.official_frames(name, states, size, log_every=0)
            env = K.make(name, batch=1, size=size)
            ss2 = K.our_frames(name, states, size, 2, env=env)
            ss1 = K.our_frames(name, states, size, 1, env=env)
            ss4 = K.our_frames(name, states, size, 4, env=env)
            e = {"n_states": len(states),
                 "ss2_vs_official": K.agreement(ss2, official),
                 "ss1_vs_official": K.agreement(ss1, official),
                 "ss4_vs_official": K.agreement(ss4, official),
                 "ss1_vs_ss2": K.agreement(ss1, ss2),
                 "gen_seconds": time.perf_counter() - t0}
            # How much of the frame the residual touches at all, and how much of it is >1/255. This is
            # the shape of the learning problem: a sparse edge residual is learnable, a dense one is a
            # different claim.
            d = np.abs(ss1.astype(np.int16) - ss2.astype(np.int16)).max(-1)
            e["residual"] = {"frac_nonzero": float((d > 0).mean()), "frac_gt1": float((d > 1).mean()),
                             "frac_gt8": float((d > 8).mean()), "mean": float(d.mean()),
                             "max": int(d.max())}
            for ss in (1, 2):
                e[f"fps_ss{ss}"] = throughput(env, states, size, a.batch, ss, a.reps)
                print(f"  {size}px ss={ss}: {e[f'fps_ss{ss}']['fps']:.0f} f/s  "
                      f"device_share={e[f'fps_ss{ss}']['device_kernel_share']}", flush=True)
            print(f"  {size}px  ss2 vs official: interior_exact="
                  f"{e['ss2_vs_official']['interior_exact_frac_pooled']}  "
                  f"exact={e['ss2_vs_official']['exact_frac']:.4f}  "
                  f"mean={e['ss2_vs_official']['mean_abs']:.3f}", flush=True)
            print(f"  {size}px  ss1 vs official: interior_exact="
                  f"{e['ss1_vs_official']['interior_exact_frac_pooled']}  "
                  f"exact={e['ss1_vs_official']['exact_frac']:.4f}  "
                  f"mean={e['ss1_vs_official']['mean_abs']:.3f}", flush=True)
            print(f"  {size}px  residual ss1->ss2: {e['residual']}", flush=True)
            row["sizes"][str(size)] = e
            del env
            torch.cuda.empty_cache()
        res["cells"][name] = row
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
    res["loadavg_end"] = K.loadavg()
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
