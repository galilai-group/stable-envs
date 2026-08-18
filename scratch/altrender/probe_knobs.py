"""The render-context switches ``scratch/perf/knobs.py`` did not sweep, and the world-count sweep.

``knobs.py`` swept ``block_dim`` and the chunk size. ``create_render_context`` carries six more
compile-time switches that change the megakernel's body -- ``enable_backface_culling``,
``enable_specular``, ``enable_emission``, ``enable_per_light_ambient``, ``use_ambient_lighting``,
``use_precomputed_rays`` -- and the first of them is documented upstream as a *performance* switch
whose effect on pixels is nil when no camera is inside a geom. Each is timed AND diffed against the
shipped configuration on the same states, so a saving is only reported next to what it costs in
pixels.

Also sweeps ``nworld``, because the shipped chunk is 256 for RoboCasa and 64 for HumanoidBench and
the free control this whole investigation starts from (the wrist camera) was quoted at the env
surface, not at 64 worlds.
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

from probe_decompose import load, timed


def build(m, cam_id, res, nworld, groups, qpos0, block_dim, **kw):
    import mujoco
    import warp as wp

    import mujoco_warp as mjw
    import stable_envs.render.mjwarp as MW

    d0 = mujoco.MjData(m)
    d0.qpos[:] = qpos0
    mujoco.mj_forward(m, d0)
    wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in
                                       ("cam_pos", "cam_quat", "light_diffuse")})
    if block_dim is not None:
        wm.block_dim.render = int(block_dim)
    wd = mjw.put_data(m, d0, nworld=int(nworld))
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True,
            cam_active=[i == int(cam_id) for i in range(m.ncam)],
            enabled_geom_groups=list(groups), **kw)

        def frame():
            mjw.kinematics(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc)
            mjw.render(wm, wd, rc)

        frame()
        wp.synchronize()
        with wp.ScopedCapture() as cap:
            frame()
        graph = cap.graph
    return wm, wd, rc, graph, wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()

    import stable_envs as se                                        # noqa: F401
    m, cid, groups, pool, camname = load(a.env)
    rc_bd = 32 if a.env.startswith("RoboCasa") else None
    base = dict(use_textures=True, use_shadows=True, render_skybox=True)

    res = {"env": a.env, "camera": camname, "res": a.res,
           "gpu": torch.cuda.get_device_name(0), "rows": []}

    def run(label, nworld, **kw):
        wm, wd, rc, graph, wp = build(m, cid, a.res, nworld, groups, pool[0], rc_bd, **kw)
        ms = 1e3 * timed(lambda: (wp.capture_launch(graph), wp.synchronize()), a.reps)
        fr = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(
            nworld, a.res, a.res, 4)[..., [2, 1, 0]].clone()
        del wm, wd, rc, graph
        torch.cuda.empty_cache()
        return {"label": label, "nworld": nworld, "ms": ms, "fps": nworld / (ms / 1e3)}, fr

    row, ref = run("shipped", 64, **base)
    res["rows"].append(row)
    print(json.dumps(row), flush=True)

    knobs = [("no_backface_cull", dict(enable_backface_culling=False)),
             ("no_specular", dict(enable_specular=False)),
             ("no_emission", dict(enable_emission=False)),
             ("no_per_light_ambient", dict(enable_per_light_ambient=False)),
             ("no_ambient", dict(use_ambient_lighting=False)),
             ("no_precomputed_rays", dict(use_precomputed_rays=False)),
             ("no_fast_math", dict(use_fast_math=False))]
    for label, kw in knobs:
        try:
            row, fr = run(label, 64, **base, **kw)
        except Exception as e:                                       # noqa: BLE001
            row = {"label": label, "error": f"{type(e).__name__}: {e}"}
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
            continue
        d = (fr.to(torch.float32) - ref.to(torch.float32)).abs()
        row["mean_abs_diff"] = float(d.mean())
        row["max_abs_diff"] = float(d.max())
        row["frac_exact"] = float((d == 0).all(-1).to(torch.float32).mean())
        row["speedup"] = res["rows"][0]["ms"] / row["ms"]
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        del fr, d
        torch.cuda.empty_cache()

    for nw in (16, 32, 128, 256):
        try:
            row, fr = run(f"nworld{nw}", nw, **base)
            del fr
        except Exception as e:                                       # noqa: BLE001
            row = {"label": f"nworld{nw}", "error": f"{type(e).__name__}: {e}"}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()

    out = (f"/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/"
           f"knobs_{a.env}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
