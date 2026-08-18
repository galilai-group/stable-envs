"""The headline number, measured exactly the way ``examples/bench_storage.py`` measures it.

Same protocol so the result is comparable to ``examples/results/storage/storage_*.json`` without
re-running the storage side: one env, ``size`` px, ``batch`` frames, a pre-allocated ``out=`` buffer,
five warm-up renders, then the MEDIAN of ``reps`` timed renders.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se

#: The 8-worker JPEG-in-Lance baseline this repo has to beat, from examples/results/storage/lance.json.
BASELINE = 16889.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--chunk", type=int, default=0, help="override the family's MJW_CHUNK")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.chunk:
        import importlib
        for m in ("metaworld", "humanoidbench", "fetch", "robocasa"):
            try:
                importlib.import_module("stable_envs.envs." + m).MJW_CHUNK = a.chunk
            except Exception:
                pass
        print(f"# MJW_CHUNK overridden to {a.chunk}", flush=True)
    res = {"device": torch.cuda.get_device_name(0), "size": a.size, "batch": a.batch, "envs": {}}
    for name in a.envs.split(","):
        env = st = out = None
        try:
            env = se.make(name, batch=a.batch, size=a.size, device="cuda")
            pool = env.sample_states(min(512, a.batch))
            st = torch.as_tensor(np.resize(pool, (a.batch, pool.shape[1])), device="cuda")
            out = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
            for _ in range(5):
                env.render(st, out=out)
            torch.cuda.synchronize()
            ts = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                env.render(st, out=out)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            dt = float(np.median(ts))
            res["envs"][name] = {"fps": a.batch / dt, "seconds_per_batch": dt,
                                 "ratio_vs_8worker_jpeg": a.batch / dt / BASELINE}
            print(f"{name:34s} {a.batch/dt:10,.0f} f/s   {a.batch/dt/BASELINE:5.2f}x baseline",
                  flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            res["envs"][name] = {"error": f"{type(e).__name__}: {e}"}
        finally:
            del env, st, out
            torch.cuda.empty_cache()
    print("JSON " + json.dumps(res), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
