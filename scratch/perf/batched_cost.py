"""What does making a Model array PER-WORLD cost the ray-cast?

Meta-World's megakernel is 7.6 ms per 64 worlds at 256 px where Fetch's is 2.3 and HumanoidBench's 2.9,
on scenes of the same order (13 vs 11 enabled geoms, one casting light each). The one structural
difference is the batch set: Meta-World gives ELEVEN Model arrays a per-world row so that an episode's
placement and a per-environment appearance can differ across the batch, and several of those arrays --
``geom_size``, ``geom_pos``, ``geom_quat``, ``geom_matid`` -- are read INSIDE the BVH traversal, once
per candidate geom per ray. Batched, each is 64 distinct rows instead of one shared row.

This drops one array at a time from ``batch_sizes`` and re-times the graph. DIAGNOSTIC ONLY: a dropped
array stops varying per environment, so nothing here is a shipping configuration.
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


def timed(fn, reps=15, warm=3):
    import warp as wp
    for _ in range(warm):
        fn()
    wp.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        wp.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MetaWorldReach")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    a = ap.parse_args()

    import mujoco_warp as mjw

    from stable_envs.envs import metaworld as MWE

    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    task = type(env).TASK
    b = MWE._build_bundle(task)
    print(f"# {a.env}: batched = {sorted(b.batched)}", flush=True)

    orig = mjw.put_model
    rows = {}

    def run(label, drop):
        MWE._CTX_CACHE.clear()
        torch.cuda.empty_cache()

        def pm(m, batch_sizes=None, **kw):
            bs = {k: v for k, v in (batch_sizes or {}).items() if k not in drop}
            return orig(m, batch_sizes=bs, **kw)
        mjw.put_model = pm
        try:
            c = MWE._context(task, a.size, a.nworld)
            dt = timed(lambda: c.wp.capture_launch(c.graph))
            rows[label] = {"ms": dt * 1e3, "fps": a.nworld / dt}
            base = rows.get("baseline", {}).get("ms", dt * 1e3)
            print(f"{label:28s} {dt*1e3:8.3f} ms  {a.nworld/dt:9,.0f} f/s  x{base/(dt*1e3):5.2f}",
                  flush=True)
        except Exception as e:
            rows[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{label:28s} ERROR {type(e).__name__}: {e}", flush=True)
        finally:
            mjw.put_model = orig
            MWE._CTX_CACHE.clear()
            torch.cuda.empty_cache()

    run("baseline", set())
    for k in sorted(b.batched):
        run(f"drop {k}", {k})
    run("drop all but rgba/light/cam", set(b.batched) - {"mat_rgba", "geom_rgba", "light_diffuse",
                                                         "cam_pos", "cam_quat"})
    run("drop everything", set(b.batched))
    print("JSON " + json.dumps({"env": a.env, "batched": sorted(b.batched), "rows": rows}),
          flush=True)


if __name__ == "__main__":
    main()
