"""Where does a 256px/batch-1024 render actually spend its time, and what is the raytracer's ceiling?

Two questions, one script, because the second is only interesting given the first:

  ``--mode sections``  wall-clock attribution of ``env.render`` to upload / launch / sync / composite,
                       by wrapping the methods rather than by re-implementing the loop.
  ``--mode nworld``    the bare cost of one ``capture_launch`` on the env's own context at a sweep of
                       world counts, with nothing else running. This is the number that says whether
                       ``MJW_CHUNK = 64`` is leaving the GPU idle or is already the saturation point.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se

T = collections.defaultdict(float)
N = collections.Counter()


def wrap(obj, name, label=None):
    f = getattr(obj, name)
    label = label or name

    def g(*a, **k):
        t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            T[label] += time.perf_counter() - t0
            N[label] += 1
    setattr(obj, name, g)
    return f


def family(name: str) -> str:
    for f in ("MetaWorld", "HumanoidBench", "Fetch", "RoboCasa"):
        if name.startswith(f):
            return f
    return "?"


def module_for(name: str):
    import importlib
    return importlib.import_module({
        "MetaWorld": "stable_envs.envs.metaworld",
        "HumanoidBench": "stable_envs.envs.humanoidbench",
        "Fetch": "stable_envs.envs.fetch",
        "RoboCasa": "stable_envs.envs.robocasa"}[family(name)])


def timed(fn, reps: int, warm: int = 3) -> float:
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


def mode_sections(a):
    import warp as wp
    env = se.make(a.env, batch=a.batch, size=a.size, device="cuda")
    pool = env.sample_states(min(512, a.batch))
    st = torch.as_tensor(np.resize(pool, (a.batch, pool.shape[1])), device="cuda")
    out = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")

    for _ in range(3):
        env.render(st, out=out)
    torch.cuda.synchronize()

    cls = type(env)
    for meth in ("_upload", "_movers_off", "_layer_off", "_plate", "resolve_style", "_full_qpos",
                 "_camera_pose", "_camera"):
        if hasattr(cls, meth):
            wrap(cls, meth)
    wrap(wp, "capture_launch")
    wrap(wp, "synchronize", "wp.synchronize")
    wrap(torch.cuda, "synchronize", "torch.sync")

    T.clear()
    N.clear()
    reps = a.reps
    t0 = time.perf_counter()
    for _ in range(reps):
        env.render(st, out=out)
    torch.cuda.synchronize()
    total = (time.perf_counter() - t0) / reps

    parts = {k: v / reps for k, v in sorted(T.items(), key=lambda kv: -kv[1])}
    counts = {k: N[k] // reps for k in parts}
    # _upload contains its own syncs on some families; report the flat sum and the residual both.
    flat = sum(parts.values())
    print(json.dumps({"env": a.env, "size": a.size, "batch": a.batch,
                      "fps": a.batch / total, "seconds": total,
                      "ms": {k: round(v * 1e3, 3) for k, v in parts.items()},
                      "calls_per_render": counts,
                      "sum_ms": round(flat * 1e3, 3),
                      "total_ms": round(total * 1e3, 3)}, indent=1), flush=True)


def mode_nworld(a):
    """Bare ``capture_launch`` cost on the env's own context, per world count."""
    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    fam = family(a.env)
    rows = {}
    for nw in [int(x) for x in a.nworlds.split(",")]:
        try:
            t_build = time.perf_counter()
            if fam == "MetaWorld":
                c = mod._context(type(env).TASK, a.size, nw)
            elif fam == "HumanoidBench":
                c = mod._context(type(env).ROBOT, type(env).TASK, a.size, nw)
            elif fam == "Fetch":
                c = mod._context(type(env).TASK, a.size, nw)
            else:
                c = mod._context(type(env).KITCHEN, type(env).CAMERA, a.size, nw)
            t_build = time.perf_counter() - t_build
            wp = c.wp
            dt = timed(lambda: (wp.capture_launch(c.graph), wp.synchronize()), a.reps)
            rows[nw] = {"launch_ms": dt * 1e3, "fps": nw / dt, "build_s": t_build}
            print(f"nworld={nw:5d}  launch {dt*1e3:8.3f} ms  -> {nw/dt:10,.0f} f/s  "
                  f"(context build {t_build:.1f} s)", flush=True)
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"nworld={nw}: {type(e).__name__}: {e}", flush=True)
            rows[nw] = {"error": f"{type(e).__name__}: {e}"}
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()
    print(json.dumps({"env": a.env, "size": a.size, "nworld": rows}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MetaWorldReach")
    ap.add_argument("--mode", default="sections", choices=["sections", "nworld"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--nworlds", default="16,32,64,128,256,512")
    a = ap.parse_args()
    print(f"# {a.env} {a.mode} on {torch.cuda.get_device_name(0)}", flush=True)
    (mode_sections if a.mode == "sections" else mode_nworld)(a)


if __name__ == "__main__":
    main()
