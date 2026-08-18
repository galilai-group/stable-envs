"""Parity / visual fidelity for the MJWarp families, through the same helpers the suite uses.

Reports the numbers the two conformance tests assert, next to what the class DECLARES, so a change can
be judged as "moved which way, by how much" rather than "still passes".
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np

import stable_envs as se
from stable_envs.testing import conformance as C

SEED = 20260811


def legacy_truncating_capture():
    """Put every family's graph back OUTSIDE ``gl_rounding``, i.e. the pre-fix shipped path.

    Not a knob anyone should have: it exists so the before/after of the capture-site fix is a measured
    pair on one interpreter and one state draw, rather than two numbers from two commits.
    """
    import importlib
    seen = set()
    for name in ("metaworld", "humanoidbench", "fetch", "robocasa", "dmcontrol"):
        try:
            mod = importlib.import_module("stable_envs.envs." + name)
        except Exception:
            continue
        frame = getattr(mod, "_frame", None) or getattr(mod, "_tc_frame", None)
        orig = getattr(mod, "_context", None) or getattr(mod, "_tc_context", None)
        if frame is None or orig is None:
            continue

        def wrap(*args, _orig=orig, _frame=frame):
            c = _orig(*args)
            if id(c) not in seen:
                seen.add(id(c))
                with c.wp.ScopedCapture() as cap:      # outside the patch: the truncating kernel
                    _frame(c)
                c.graph = cap.graph
            return c
        if hasattr(mod, "_context"):
            mod._context = wrap
        else:
            mod._tc_context = wrap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--legacy", action="store_true",
                    help="capture the graph outside gl_rounding, reproducing the pre-fix path")
    a = ap.parse_args()
    if a.legacy:
        legacy_truncating_capture()
        print("# LEGACY: graph captured outside gl_rounding (truncating kernel)", flush=True)
    out = {}
    for name in a.envs.split(","):
        try:
            env = se.make(name, batch=16, device="cuda")
            cls = type(env)
            if env.fidelity_tier == "visual":
                r = C.check_visual_fidelity(env, n=a.n)
                decl = dict(env.visual_fidelity)
                ok = (r["mean_abs"] <= decl["mean_abs_max"] and
                      r["within8_frac"] >= decl["within8_min"] and r["ssim"] >= decl["ssim_min"])
                out[name] = {"tier": "visual", "measured": r, "declared": decl, "passes": bool(ok)}
                print(f"{name:34s} VISUAL mean {r['mean_abs']:.3f} (<= {decl['mean_abs_max']}) "
                      f"within8 {r['within8_frac']:.4f} (>= {decl['within8_min']}) "
                      f"ssim {r['ssim']:.4f} (>= {decl['ssim_min']})  {'PASS' if ok else 'FAIL'}",
                      flush=True)
            else:
                ref = C.reference_env(cls, env.size)
                states = env.sample_states(a.n, rng=np.random.default_rng(SEED))
                official = cls.reference_render(ref, states)
                ours = env.render(states).cpu().numpy()
                r = C.pixel_agreement(ours, official)
                floor = float(getattr(env, "parity_interior_min", 1.0))
                allf = getattr(env, "parity_all_exact_min", None)
                allf = 0.90 if allf is None else float(allf)
                ok = (r["interior_exact_frac"] >= floor and r["exact_frac"] >= allf
                      and r["mean_abs"] <= 4.0)
                out[name] = {"tier": "parity", "measured": r,
                             "declared": {"interior_min": floor, "all_min": allf},
                             "passes": bool(ok)}
                print(f"{name:34s} PARITY interior {r['interior_exact_frac']:.5f} (>= {floor}) "
                      f"all {r['exact_frac']:.4f} (>= {allf}) mean {r['mean_abs']:.3f} "
                      f"{'PASS' if ok else 'FAIL'}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
