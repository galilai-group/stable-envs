"""The re-declaration: ``visual_fidelity`` for a candidate configuration, in the repo's own shape.

A configuration that ships has to declare what IT produces, measured the way every other visual-tier
class in this repo was measured -- **64 states x 4 independent draws from ``sample_states``, and the
declared value is the worst of the four**, not the mean and not one lucky draw. The frontier sweep
used a single draw, which is the right cost for exploring and the wrong one for declaring, so this
exists separately and its output is meant to be pasted.

It also emits the JPEG q95 budget for the same cell on the same draws, because the whole argument for
the configuration is a comparison against that number and a declaration that arrives without it is
just a bigger error with no justification attached.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from probe_budget_frontier import (config_list, ctx_cache, env_module, family_of,
                                   jpeg_roundtrip, metrics, patched, throughput, upsample)

KEYS = ("mean_abs", "within8_frac", "ssim", "interior_mean_abs", "interior_exact_frac")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--config", required=True)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--size", type=int, default=256)
    # The interpreter that can render DM Control TrackCom (dm_control + mujoco_warp) has no
    # torchvision, which is the repo's JPEG read path. The declaration itself does not need JPEG;
    # the budget is measured separately by ``jpeg_budget_from_npy.py`` where torchvision exists.
    ap.add_argument("--no-jpeg", action="store_true")
    a = ap.parse_args()

    import stable_envs as se
    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)
    family = family_of(a.env)
    r, mode, ov, lod = config_list(family)[a.config]
    r = r or a.size                      # 0 means "render natively at the declaration's own size"

    ref = cls.reference_env(a.size)
    draws, jdraws, ship = [], [], []
    fps = None
    for k in range(a.draws):
        probe = se.make(a.env, batch=1, size=a.size, device="cuda", seed=k)
        states = probe.sample_states(a.states)
        del probe
        ctx_cache(mod).clear()
        torch.cuda.empty_cache()
        official = cls.reference_render(ref, states)
        if not a.no_jpeg:
            jdraws.append(metrics(jpeg_roundtrip(official, 95)[0], official))

        for tag, (rr, mm, oo, ll), sink in ((a.config, (r, mode, ov, lod), draws),
                                            ("shipped", (a.size, "bilinear", {}, 1), ship)):
            orig = patched(mod, oo, ll)
            try:
                env = se.make(a.env, batch=1, size=a.size, device="cuda", seed=k)
                ours = upsample(env.render(states, size=rr), a.size, mm).cpu().numpy()
                sink.append(metrics(ours, official))
                del ours
                torch.cuda.empty_cache()
                if k == 0:
                    t = throughput(env, states, a.batch, rr, a.size, mm)
                    if tag == a.config:
                        fps = t["fps"]
                    else:
                        fps_ship = t["fps"]
                del env
            finally:
                mod.MW.create_render_context = orig
                ctx_cache(mod).clear()
                torch.cuda.empty_cache()
        print(json.dumps({"draw": k, a.config: draws[-1], "shipped": ship[-1],
                          "jpeg_q95": jdraws[-1] if jdraws else None}), flush=True)

    def worst(rows):
        return {"mean_abs": round(max(x["mean_abs"] for x in rows), 2),
                "within8_frac": round(min(x["within8_frac"] for x in rows), 3),
                "ssim": round(min(x["ssim"] for x in rows), 3),
                "interior_mean_abs": round(max(x["interior_mean_abs"] for x in rows), 2),
                "interior_exact_frac": round(min(x["interior_exact_frac"] for x in rows), 3)}

    import mujoco
    import warp
    vf = worst(draws)
    vf["draw_spread"] = {k: [round(min(x[k] for x in draws), 3),
                             round(max(x[k] for x in draws), 3)] for k in KEYS}
    vf["measured_at"] = (
        f"256 px, {a.states} states x {a.draws} independent draws from sample_states, one "
        f"{torch.cuda.get_device_name(0)}, mujoco {mujoco.__version__} / warp {warp.__version__}; "
        f"render at {r} px upsampled bilinear to 256, texture LOD {lod}, "
        f"{'backface culling off' if ov.get('enable_backface_culling') is False else 'shipped flags'}"
        f"; the values above are the WORST of the {a.draws}")
    out = {"env": a.env, "config": a.config, "render_size": r, "upsample": mode, "tex_lod": lod,
           "overrides": {k: str(v) for k, v in ov.items()},
           "visual_fidelity": vf,
           "shipped_worst": worst(ship),
           "jpeg_q95_worst": worst(jdraws) if jdraws else None,
           "fps": fps, "fps_shipped": fps_ship, "speedup": fps / fps_ship}
    print(json.dumps(out, indent=1), flush=True)
    p = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
         f"declare_{a.env}_{a.config}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
