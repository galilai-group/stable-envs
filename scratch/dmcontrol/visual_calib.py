"""Calibrate the VISUAL tier for the trackcom DM Control cells, through the SHIPPED env.

The tier's contract wants three measured numbers and three thresholds, and the thresholds have to
come from the spread over INDEPENDENT state draws rather than from one -- a threshold equal to a
single measurement flakes on the next draw, and a measurement inferred from a threshold is not a
measurement. So each cell is drawn ``--draws`` times with different seeds and the worst draw is what
the ceiling has to clear.

Measured through ``se.make(...)`` and ``cls.reference_render`` rather than through a probe harness,
because the number that belongs on the class is the one the conformance suite will compute, and those
two paths differ by everything the env does that a probe does not (the per-world model upload, the
zero-alpha pre-pass, the CUDA-graph replay, chunking).

One cell per process: a CUDA 700 poisons the interpreter, and two MuJoCo envs alive together can
return byte-identical frames from cameras that provably differ.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/dmc2")
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.testing import conformance as C

    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    ref = C.reference_env(cls, a.size)
    rows = []
    for k in range(a.draws):
        states = env.sample_states(a.n, rng=np.random.default_rng(100 + k))
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        r["ssim"] = C.ssim(ours, official)
        rows.append(r)
        print(f"{a.env} draw {k}: mean={r['mean_abs']:.3f} within8={r['within8_frac']:.4f} "
              f"ssim={r['ssim']:.4f} interior={r['interior_exact_frac']} "
              f"interior_frac={r['interior_frac']:.4f} exact={r['exact_frac']:.4f}", flush=True)
    agg = {
        "env": a.env, "n": a.n, "draws": a.draws, "size": a.size,
        "mean_abs": round(float(np.mean([r["mean_abs"] for r in rows])), 3),
        "mean_abs_worst": round(float(np.max([r["mean_abs"] for r in rows])), 3),
        "within8_frac": round(float(np.mean([r["within8_frac"] for r in rows])), 4),
        "within8_worst": round(float(np.min([r["within8_frac"] for r in rows])), 4),
        "ssim": round(float(np.mean([r["ssim"] for r in rows])), 4),
        "ssim_worst": round(float(np.min([r["ssim"] for r in rows])), 4),
        "exact_frac": round(float(np.mean([r["exact_frac"] for r in rows])), 4),
        "interior_frac": round(float(np.mean([r["interior_frac"] for r in rows])), 6),
        "interior_exact": [r["interior_exact_frac"] for r in rows],
        "rows": rows,
    }
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"visual_{a.env}.json")
    with open(path, "w") as f:
        json.dump(agg, f, indent=1, default=float)
    print(json.dumps({k: v for k, v in agg.items() if k != "rows"}, indent=1, default=float))
    print(f"@@wrote {path}")


if __name__ == "__main__":
    main()
