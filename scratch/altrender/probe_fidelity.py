"""Throughput AND the four declared fidelity metrics, for one env under several render configs.

A visual-tier class declares ``visual_fidelity`` = mean |diff|, within-8, SSIM and interior-exact
against the official renderer, and ``test_visual_tier_matches_its_declared_fidelity`` asserts against
it. So a candidate configuration is only a candidate once it has been scored the same way, on the
same states, in the same process as the shipped one -- which is what this does. Nothing here is a
speed number on its own; every row carries what the speed cost in pixels.

The configs are applied by patching the ONE call each env module makes to
``stable_envs.render.mjwarp.create_render_context``, so the env's own chunking, graph capture,
rounding patch and channel swap are all still the shipped ones and the only difference is the
switches under test.
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

#: name -> the create_render_context overrides it forces.
CONFIGS = {
    "shipped": {},
    "no_shadows": dict(use_shadows=False),
    "no_backface_cull": dict(enable_backface_culling=False),
    "no_specular": dict(enable_specular=False),
    "no_skybox": dict(render_skybox=False),
    "no_shadows_no_specular": dict(use_shadows=False, enable_specular=False),
    "no_shadows_no_backface_cull": dict(use_shadows=False, enable_backface_culling=False),
}


def env_module(name):
    import importlib
    if name.startswith("RoboCasa"):
        return importlib.import_module("stable_envs.envs.robocasa")
    if name.startswith("HumanoidBench"):
        return importlib.import_module("stable_envs.envs.humanoidbench")
    raise SystemExit(f"unhandled {name}")


def patched(mod, overrides):
    """Force ``overrides`` onto every render context this module builds, and drop its cache."""
    import stable_envs.render.mjwarp as MW
    original = MW.create_render_context

    def wrap(m, **kw):
        kw.update(overrides)
        return original(m, **kw)

    mod.MW.create_render_context = wrap
    mod._CTX_CACHE.clear()
    return original


def unpatch(mod, original):
    mod.MW.create_render_context = original
    mod._CTX_CACHE.clear()


def throughput(env, states, batch, reps=3):
    st = torch.as_tensor(states[:batch], device="cuda", dtype=torch.float32)
    if len(st) < batch:
        st = st[np.random.default_rng(0).integers(0, len(st), batch)]
    out = torch.empty((batch, env.size, env.size, 3), dtype=torch.uint8, device="cuda")
    env.render(st, out=out)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        env.render(st, out=out)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    dt = float(np.median(ts))
    del out, st
    torch.cuda.empty_cache()
    return {"fps": batch / dt, "seconds_per_batch": dt, "batch": batch}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--configs", default="shipped,no_shadows")
    ap.add_argument("--states", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()

    import stable_envs as se
    from stable_envs.testing.conformance import pixel_agreement, ssim

    cls = se.REGISTRY[a.env]
    mod = env_module(a.env)

    # One reference EGL context for the whole process; closing one corrupts every other.
    ref = cls.reference_env(a.size)
    probe = se.make(a.env, batch=1, size=a.size, device="cuda")
    states = probe.sample_states(a.states)
    official = cls.reference_render(ref, states)
    declared = dict(getattr(cls, "visual_fidelity", {}) or {})
    del probe
    mod._CTX_CACHE.clear()
    torch.cuda.empty_cache()

    res = {"env": a.env, "size": a.size, "batch": a.batch, "n_states": a.states,
           "gpu": torch.cuda.get_device_name(0), "declared": declared, "rows": []}
    print(json.dumps({"declared": declared}), flush=True)

    for cfg in a.configs.split(","):
        ov = CONFIGS[cfg]
        orig = patched(mod, ov)
        try:
            env = se.make(a.env, batch=1, size=a.size, device="cuda")
            ours = env.render(states).cpu().numpy()
            row = {"config": cfg, "overrides": {k: str(v) for k, v in ov.items()}}
            row.update(pixel_agreement(ours, official))
            row["ssim"] = ssim(ours, official)
            del ours
            torch.cuda.empty_cache()
            row.update(throughput(env, states, a.batch))
            del env
        except Exception as e:                                        # noqa: BLE001
            row = {"config": cfg, "error": f"{type(e).__name__}: {e}"}
        finally:
            unpatch(mod, orig)
            torch.cuda.empty_cache()
        res["rows"].append(row)
        print(json.dumps(row), flush=True)

    out = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
           f"fidelity_{a.env}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
