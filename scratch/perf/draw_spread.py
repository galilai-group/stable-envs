"""What IS a visual-tier cell's fidelity distribution over independent seeded draws?

A declared number that holds on the draw it was measured on and not on the next one is the same defect
as a parity floor set from one lucky draw. ``check_visual_fidelity`` draws its states from
``env.sample_states``, which reads the env's OWN generator, so a fresh env at a different seed is an
independent draw of the same measurement -- and the spread over those draws is the thing that belongs
in ``visual_fidelity["draw_spread"]``.

Reports every metric the conformance check computes, per draw, per env, plus min/median/max. With
``--legacy`` the graph is captured OUTSIDE ``gl_rounding`` (the pre-fix path, from
``scratch/perf/fidelity.py``), which is the control for "did the rounding fix improve a number that was
never re-declared?".
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KEYS = ("mean_abs", "within8_frac", "ssim", "interior_mean_abs", "interior_exact_frac",
        "exact_frac", "max_abs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--seeds", default="", help="explicit seed list; overrides --draws. "
                    "The ORDER matters if the reference env turns out to be history-dependent, "
                    "which is exactly what an out-of-order run tests.")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--legacy", action="store_true")
    a = ap.parse_args()
    if a.legacy:
        from fidelity import legacy_truncating_capture
        legacy_truncating_capture()
        print("# LEGACY: graph captured outside gl_rounding (truncating kernel)", flush=True)

    out = {}
    for name in a.envs.split(","):
        per = []
        seeds = ([int(x) for x in a.seeds.split(",")] if a.seeds else list(range(a.draws)))
        for seed in seeds:
            try:
                env = se.make(name, batch=16, device="cuda", seed=seed)
                r = C.check_visual_fidelity(env, n=a.n)
                per.append({k: (None if r.get(k) is None else float(r[k])) for k in KEYS})
                print(f"  {name:34s} seed {seed}: " + " ".join(
                    f"{k}={per[-1][k]:.4f}" for k in ("mean_abs", "within8_frac", "ssim",
                                                      "interior_exact_frac")
                    if per[-1][k] is not None), flush=True)
                del env
                torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                traceback.print_exc()
                per.append({"error": f"{type(e).__name__}: {e}"})
        good = [p for p in per if "error" not in p]
        summ = {}
        for k in KEYS:
            vals = [p[k] for p in good if p.get(k) is not None]
            if vals:
                summ[k] = {"min": min(vals), "median": float(np.median(vals)), "max": max(vals),
                           "n": len(vals)}
        try:
            decl = dict(se.make(name, batch=1, device="cuda").visual_fidelity)
        except Exception:
            decl = {}
        out[name] = {"draws": per, "summary": summ, "declared": decl}
        if summ:
            print(f"{name:34s} over {len(good)} draws: " + ", ".join(
                f"{k} {summ[k]['min']:.4f}..{summ[k]['max']:.4f} (med {summ[k]['median']:.4f}"
                f", declared {decl.get(k, 'n/a')})" for k in
                ("mean_abs", "within8_frac", "ssim", "interior_exact_frac") if k in summ), flush=True)
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
