"""Render-only throughput for the resolutions a corrector might be fed, on the node it will run on.

``probe_budget_frontier.py`` swept the resolutions that made sense when the frame it produced was the
final one. A learned corrector changes which resolutions are interesting -- 96 px and 80 px become
candidates because the network, not the upsample filter, is what has to recover the detail -- and it
adds a question that sweep never asked: **what do the auxiliary buffers cost?** ``mujoco_warp`` can
write depth and a per-pixel ``(objid, objtype)`` segmentation out of the same megakernel, and in
real-time graphics those buffers are what make a learned upsampler work. They are only worth having
if they are nearly free, so they are priced here before anything is trained on them.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from probe_budget_frontier import context_with_tex_lod, ctx_cache, env_module


def patched_ctx(mod, overrides, lod=1):
    """Like ``probe_budget_frontier.patched`` but able to add ``render_depth`` / ``render_seg``."""
    import stable_envs.render.mjwarp as MW
    original = MW.create_render_context

    def wrap(m, **kw):
        kw.update(overrides)
        if lod > 1:
            return context_with_tex_lod(m, lod, False, **kw)
        return original(m, **kw)

    mod.MW.create_render_context = wrap
    ctx_cache(mod).clear()
    return original


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--sizes", default="256,160,144,128,112,96,80,64")
    ap.add_argument("--tex-lod", type=int, default=1)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    import stable_envs as se
    mod = env_module(a.env)
    sizes = [int(x) for x in a.sizes.split(",")]
    rows = []
    for aux in (False, True):
        ov = dict(enable_backface_culling=False)
        if aux:
            ov.update(render_depth=True, render_seg=True)
        orig = patched_ctx(mod, ov, a.tex_lod)
        try:
            env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
            states = env.sample_states(64)
            st = torch.as_tensor(states, device="cuda", dtype=torch.float32)
            idx = torch.as_tensor(np.random.default_rng(0).integers(0, len(states), a.batch),
                                  device="cuda")
            st = st[idx]
            for r in sizes:
                try:
                    out = torch.empty((a.batch, r, r, 3), dtype=torch.uint8, device="cuda")
                    env.render(st, size=r, out=out)
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(a.reps):
                        t0 = time.perf_counter()
                        env.render(st, size=r, out=out)
                        torch.cuda.synchronize()
                        ts.append(time.perf_counter() - t0)
                    dt = float(np.median(ts))
                    rows.append({"env": a.env, "render_size": r, "tex_lod": a.tex_lod, "aux": aux,
                                 "fps": a.batch / dt, "seconds_per_batch": dt})
                    del out
                except Exception as e:                            # noqa: BLE001
                    rows.append({"env": a.env, "render_size": r, "tex_lod": a.tex_lod, "aux": aux,
                                 "error": f"{type(e).__name__}: {e}"})
                torch.cuda.empty_cache()
                print(json.dumps(rows[-1]), flush=True)
            del env
        except Exception as e:                                    # noqa: BLE001
            rows.append({"env": a.env, "aux": aux, "error": f"{type(e).__name__}: {e}"})
            print(json.dumps(rows[-1]), flush=True)
        finally:
            mod.MW.create_render_context = orig
            ctx_cache(mod).clear()
            torch.cuda.empty_cache()

    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/"
         f"renderfps_{a.env}_tex{a.tex_lod}.json")
    with open(p, "w") as f:
        json.dump({"gpu": torch.cuda.get_device_name(0), "batch": a.batch, "rows": rows}, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
