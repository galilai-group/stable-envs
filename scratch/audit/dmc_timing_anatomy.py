"""What actually decides DMCBallInCupCatch's frames/s? Supersample, warmup, or the HOST?

Three numbers exist for the same configuration (84 px, batch 4096) and they span 13x:

    271-274 f/s   scratch/census/bottom_sweep.py     (env default ss=2, 3 warm-ups)
      356 f/s     scratch/audit/ss_discrepancy.py    (ss=2 forced, 3 warm-ups, node 074)
      978 f/s     examples/results/decode/ALL.json   (ss=1 forced, 10 renders, node at load 24)
    3,488 f/s     the same script re-run at ss=2     (10 renders, node 086)

Supersample is established as a real 3.2-3.8x by an in-process A/B, and it cannot explain a 3,488
measured at ss=2 against a 978 at ss=1. So one process measures every axis at once, on one node:

  * ss=1 against ss=2, everything else held;
  * the per-call series, so a cost that amortises over warm-ups is visible instead of inferred;
  * ENQUEUE time (no synchronize) against wall time (synchronized). If the launch alone costs what
    the whole render costs, the number is bounded by the HOST, and a table that says "the GPU column
    reproduces across nodes to 0.4%" is making a claim about it that is not true.

    python scratch/audit/dmc_timing_anatomy.py --env DMCBallInCupCatch
"""
from __future__ import annotations

import argparse
import os
import statistics
import time

import torch

import stable_envs as se


def series(env, st, buf, n: int) -> list[float]:
    out = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        env.render(st, out=buf)
        torch.cuda.synchronize()
        out.append(time.perf_counter() - t0)
    return out


def enqueue_only(env, st, buf, n: int) -> float:
    """Median time for the PYTHON CALL to return, without waiting for the device.

    NOT a clean host/device split on its own, and the reason is worth stating rather than reading the
    ratio as one: ``DMControlSuite.render`` starts by pulling the style key to the host
    (``.cpu().numpy()``), which synchronises on the previous call's kernels. So this returning "100%
    of wall" is what a GPU-bound render with an internal sync looks like too. ``cpu_fraction`` is the
    measurement that separates them.
    """
    torch.cuda.synchronize()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        env.render(st, out=buf)
        out.append(time.perf_counter() - t0)
    torch.cuda.synchronize()
    return statistics.median(out)


def cpu_fraction(env, st, buf, n: int) -> tuple[float, float]:
    """``(process CPU seconds, wall seconds)`` over ``n`` synchronised renders.

    The discriminator. ``time.process_time`` counts only seconds this process spent ON a CPU, so a
    render that is waiting for the device contributes wall time and no CPU time. A ratio near 1 means
    the host is busy for the whole render and the device is not the limit — which makes the number a
    property of the node's CPU and of what else is running on it, not of the GPU, and a table that
    calls it "the GPU column" and says it travels across nodes to 0.4% is describing something else.
    """
    torch.cuda.synchronize()
    c0, w0 = time.process_time(), time.perf_counter()
    for _ in range(n):
        env.render(st, out=buf)
    torch.cuda.synchronize()
    return time.process_time() - c0, time.perf_counter() - w0


def one(name: str, batch: int, size: int, ss, pool) -> None:
    kw = {} if ss is None else {"supersample": int(ss)}
    env = se.make(name, batch=batch, size=size, device="cuda", **kw)
    st = torch.as_tensor(pool[:batch], device="cuda")
    f = env.render(st[:1])
    buf = torch.empty((batch, f.shape[1], f.shape[2], 3), dtype=torch.uint8, device="cuda")
    warm = series(env, st, buf, 3)
    mid = series(env, st, buf, 7)
    late = series(env, st, buf, 5)
    enq = enqueue_only(env, st, buf, 5)
    cpu_s, wall_s = cpu_fraction(env, st, buf, 5)
    fps = lambda t: batch / t
    print(f"  ss={env.ss}  first3 {[f'{fps(t):,.0f}' for t in warm]}  "
          f"next7 median {fps(statistics.median(mid)):,.0f}  "
          f"late5 median {fps(statistics.median(late)):,.0f}  "
          f"enqueue-only {fps(enq):,.0f} f/s "
          f"({100 * enq / statistics.median(late):.0f}% of wall)  "
          f"CPU {100 * cpu_s / wall_s:.0f}% of wall", flush=True)
    del env, st, buf, f
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="DMCBallInCupCatch")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    print(f"node {os.uname().nodename}  load {os.getloadavg()}  "
          f"cores visible {len(os.sched_getaffinity(0))}  gpu {torch.cuda.get_device_name(0)}",
          flush=True)
    for name in a.env.split(","):
        if name not in se.REGISTRY:
            print(f"{name}: not registered", flush=True)
            continue
        size = a.size if a.size > 0 else None
        probe = se.make(name, batch=1, device="cuda", **({} if size is None else {"size": size}))
        size = probe.size
        pool = probe.sample_states(a.batch)
        del probe
        torch.cuda.empty_cache()
        for r in range(a.repeat):
            print(f"{name} @{size} pass {r}: load {os.getloadavg()[0]:.2f}", flush=True)
            for ss in (2, 1, None):
                try:
                    one(name, a.batch, size, ss, pool)
                except Exception as e:                                   # noqa: BLE001
                    print(f"  ss={ss}: {type(e).__name__}: {e}", flush=True)
                    torch.cuda.empty_cache()
        del pool
