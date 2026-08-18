"""What would HumanoidBench's only remaining 1.5x cost in the numbers its tier declares?

``scratch/perf/knobs.py`` puts ``use_shadows=False`` at 1.54x on this family's megakernel and
``scratch/perf/window_dyn.py`` shows no screen window can touch the rest (78% of the frame's cost is
inside the 25% of it the robot occupies). So the shadow ray IS the family's remaining lever, and
HumanoidBench is VISUAL tier, which means dropping it is a question about declared numbers rather
than about parity.

This measures both sides of that question at once: the throughput with the shadow ray off, and
``check_visual_fidelity`` -- the exact quantities ``test_visual_tier_matches_its_declared_fidelity``
asserts -- against the thresholds the classes ALREADY declare. Nothing is relaxed; the point is to
find out whether the cheaper picture still clears the floors that are there, and by how much the
measured values move.
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
import stable_envs.render.mjwarp as MW
from stable_envs.testing import conformance as C


def timed(fn, reps=9, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="HumanoidBenchH1Walk,HumanoidBenchH1HandBalanceSimple,"
                                     "HumanoidBenchH1SitHard,HumanoidBenchH1HandSpoon")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()

    from stable_envs.envs import humanoidbench as HB

    orig = MW.create_render_context
    out = {}
    for name in a.envs.split(","):
        row = {}
        for shadows in (True, False):
            def crc(mm, _s=shadows, **kw):
                kw["use_shadows"] = _s
                return orig(mm, **kw)
            MW.create_render_context = crc
            HB._CTX_CACHE.clear()
            torch.cuda.empty_cache()
            try:
                env = se.make(name, batch=a.batch, size=a.size, device="cuda")
                pool = env.sample_states(min(512, a.batch))
                st = torch.as_tensor(np.resize(pool, (a.batch, pool.shape[1])), device="cuda")
                buf = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
                dt = timed(lambda: env.render(st, out=buf))
                r = C.check_visual_fidelity(env, n=a.n)
                decl = dict(env.visual_fidelity)
                row["shadows" if shadows else "noshadow"] = {
                    "fps": a.batch / dt,
                    "mean_abs": r["mean_abs"], "within8_frac": r["within8_frac"], "ssim": r["ssim"],
                    "interior_exact_frac": r.get("interior_exact_frac"),
                    "passes": bool(r["mean_abs"] <= decl["mean_abs_max"]
                                   and r["within8_frac"] >= decl["within8_min"]
                                   and r["ssim"] >= decl["ssim_min"]),
                    "declared": {k: decl[k] for k in ("mean_abs", "within8_frac", "ssim",
                                                      "mean_abs_max", "within8_min", "ssim_min")},
                }
                del env, st, buf
            except Exception as e:
                import traceback
                traceback.print_exc()
                row["shadows" if shadows else "noshadow"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                MW.create_render_context = orig
                HB._CTX_CACHE.clear()
                torch.cuda.empty_cache()
        out[name] = row
        s, ns = row.get("shadows", {}), row.get("noshadow", {})
        if "fps" in s and "fps" in ns:
            print(f"{name:34s} {s['fps']:8,.0f} -> {ns['fps']:8,.0f} f/s (x{ns['fps']/s['fps']:.2f})  "
                  f"mean {s['mean_abs']:.2f}->{ns['mean_abs']:.2f} (<= {s['declared']['mean_abs_max']}) "
                  f"w8 {s['within8_frac']:.3f}->{ns['within8_frac']:.3f} "
                  f"(>= {s['declared']['within8_min']})  ssim {s['ssim']:.4f}->{ns['ssim']:.4f} "
                  f"(>= {s['declared']['ssim_min']})  "
                  f"{'PASSES' if ns['passes'] else 'FAILS THE DECLARED FLOOR'}", flush=True)
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
