"""Does ``supersample=1`` still pass ``test_render_parity_with_official_renderer``? Over many draws.

The premise this study started from is that the 15 MSAA'd parity cells need ``supersample=2`` to hold
their parity claim, and that the claim is therefore what costs them their throughput. The single-draw
facts in ``probe_facts.py`` did not support it, so the premise has to be tested the way this repo
already tests a rare pixel event: over MANY seeded draws, not one.

That discipline is not optional here and the repo has the worked example. ``MujocoPusher`` declared
``parity_interior_min = 1.0`` on one unseeded draw that came back clean; seeded and repeated it fails
15 draws in 101, at 1-2 pixels of 3.03 M. A single draw of a quantity that is USUALLY zero, reported
as if it were the quantity, is exactly the failure mode this script exists to avoid -- in the other
direction this time, because a single clean draw at ss=1 would be equally worthless as evidence that
ss=1 is safe.

So: for every cell, ``n_draws`` independent 64-state draws at seeds ``PARITY_STATE_SEED + k`` (k=0 is
the test's own draw), each scored at ss=1 and ss=2 against the official renderer, against the three
thresholds the test actually asserts and the per-class floors it reads off the env:

    interior_exact_frac >= parity_interior_min   (default 1.0, skipped when there are no interior px)
    exact_frac          >= parity_all_exact_min  (default 0.90)
    mean_abs            <= 4.0

A cell "passes at ss=1" only if every draw passes all three.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402

import numpy as np
import torch

import stable_envs as se  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402

MEAN_ABS_MAX = 4.0
ALL_EXACT_MIN = 0.90
INTERIOR_EXACT_MIN = 1.0


def floors(cls) -> dict:
    return {"interior": float(getattr(cls, "parity_interior_min", INTERIOR_EXACT_MIN)),
            "all_exact": float(getattr(cls, "parity_all_exact_min", None) or ALL_EXACT_MIN),
            "mean_abs": MEAN_ABS_MAX,
            "parity_note": str(getattr(cls, "parity_note", ""))}


def verdict(r: dict, f: dict) -> dict:
    """Which of the three assertions this draw would have passed, and by how much."""
    ie = r["interior_exact_frac"]
    return {"interior_ok": True if ie is None else bool(ie >= f["interior"]),
            "all_exact_ok": bool(r["exact_frac"] >= f["all_exact"]),
            "mean_ok": bool(r["mean_abs"] <= f["mean_abs"]),
            "interior_exact_frac": ie, "interior_px": r["interior_px"],
            "interior_bad_px": r["interior_bad_px"], "exact_frac": r["exact_frac"],
            "mean_abs": r["mean_abs"], "max_abs": r["max_abs"], "within8_frac": r["within8_frac"]}


def throughput(env, pool, size, batch, ss, reps=11):
    st = torch.as_tensor(np.resize(pool, (batch, pool.shape[1])), device="cuda")
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
    res = {"fps": batch / dt, "us_per_frame": 1e6 * dt / batch,
           "iqr_frac": float((np.percentile(ts, 75) - np.percentile(ts, 25)) / dt),
           "loadavg": K.loadavg()}
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            env.render(st, size=size, supersample=ss, out=out)
            torch.cuda.synchronize()
            wall = time.perf_counter() - t0
        res["device_kernel_share"] = sum(int(e.self_device_time_total)
                                         for e in prof.key_averages()) / 1e6 / wall
    except Exception as e:
        res["device_kernel_share"] = None
        res["prof_error"] = repr(e)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--draws", type=int, default=41)
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--sizes", default="", help="default: the cell's own default_size, plus 256")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--ss", default="1,2")
    ap.add_argument("--outdir", default=os.path.join(K.LOGS, "parity_draws"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cls = K.cls_of(a.cell)
    f = floors(cls)
    sizes = ([int(s) for s in a.sizes.split(",")] if a.sizes
             else sorted({int(cls.default_size), 256}))
    sslist = [int(s) for s in a.ss.split(",")]
    res = {"cell": a.cell, "host": os.uname().nodename, "device": torch.cuda.get_device_name(0),
           "default_size": int(cls.default_size),
           "default_supersample": int(getattr(cls, "default_supersample", 1)),
           "floors": f, "draws": a.draws, "n_states": a.n_states,
           "parity_state_seed": K.PARITY_STATE_SEED, "loadavg_start": K.loadavg(), "sizes": {}}

    for size in sizes:
        env = K.make(a.cell, batch=1, size=size)
        ref = cls.reference_env(int(size))
        per = {ss: [] for ss in sslist}
        t0 = time.perf_counter()
        pool = None
        for k in range(a.draws):
            seed = K.PARITY_STATE_SEED + k
            st = np.asarray(env.sample_states(a.n_states, rng=np.random.default_rng(seed)),
                            np.float32)
            if pool is None:
                pool = st
            official = cls.reference_render(ref, st)
            for ss in sslist:
                ours = env.render(torch.as_tensor(st, device="cuda"), size=size,
                                  supersample=ss).cpu().numpy()
                per[ss].append({"seed": seed, **verdict(C.pixel_agreement(ours, official), f)})
            if (k + 1) % 5 == 0:
                print(f"  {size}px draw {k+1}/{a.draws}  {(k+1)/(time.perf_counter()-t0):.2f} draws/s",
                      flush=True)
        row = {"per_ss": {}}
        for ss in sslist:
            v = per[ss]
            row["per_ss"][str(ss)] = {
                "n_draws": len(v),
                "n_fail_interior": sum(not d["interior_ok"] for d in v),
                "n_fail_all_exact": sum(not d["all_exact_ok"] for d in v),
                "n_fail_mean": sum(not d["mean_ok"] for d in v),
                "n_fail_any": sum(not (d["interior_ok"] and d["all_exact_ok"] and d["mean_ok"])
                                  for d in v),
                "interior_px_total": int(sum(d["interior_px"] or 0 for d in v)),
                "interior_bad_px_total": int(sum(d["interior_bad_px"] or 0 for d in v)),
                "worst_interior_exact": (None if all(d["interior_exact_frac"] is None for d in v)
                                         else min(d["interior_exact_frac"] for d in v
                                                  if d["interior_exact_frac"] is not None)),
                "worst_exact_frac": min(d["exact_frac"] for d in v),
                "median_exact_frac": float(np.median([d["exact_frac"] for d in v])),
                "worst_mean_abs": max(d["mean_abs"] for d in v),
                "median_mean_abs": float(np.median([d["mean_abs"] for d in v])),
                "worst_max_abs": max(d["max_abs"] for d in v),
                "worst_within8": min(d["within8_frac"] for d in v),
                "draw0": v[0],
                "draws": v,
            }
            s = row["per_ss"][str(ss)]
            print(f"{a.cell} {size}px ss={ss}: fail any {s['n_fail_any']}/{s['n_draws']} "
                  f"(interior {s['n_fail_interior']}, exact {s['n_fail_all_exact']}, "
                  f"mean {s['n_fail_mean']}) | worst exact {s['worst_exact_frac']:.4f} "
                  f"worst mean {s['worst_mean_abs']:.3f} worst interior "
                  f"{s['worst_interior_exact']} bad_px {s['interior_bad_px_total']}"
                  f"/{s['interior_px_total']}", flush=True)
            row["per_ss"][str(ss)]["fps"] = throughput(env, pool, size, a.batch, ss)
            print(f"    fps {row['per_ss'][str(ss)]['fps']['fps']:.0f} "
                  f"device_share {row['per_ss'][str(ss)]['fps']['device_kernel_share']}", flush=True)
        res["sizes"][str(size)] = row
        del env
        torch.cuda.empty_cache()
        with open(os.path.join(a.outdir, f"{a.cell}.json"), "w") as fh:
            json.dump(res, fh, indent=1)
    res["loadavg_end"] = K.loadavg()
    with open(os.path.join(a.outdir, f"{a.cell}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.join(a.outdir, f"{a.cell}.json"), flush=True)


if __name__ == "__main__":
    main()
