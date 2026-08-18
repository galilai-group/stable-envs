"""What the MJWarp megakernel costs, kernel by kernel, and what each of its knobs is worth.

The context is built through the ENV's own ``_context`` so the scene, the groups, the pre-pass and the
rounding patch are exactly the ones the env ships; only the knob under test is overridden. Two classes
of knob, and the distinction is the whole point of measuring them separately:

  **free**   ``block_dim`` is a launch geometry, not arithmetic. Same threads, same order, same floats.
  **not free** ``enable_specular`` / ``enable_emission`` / ``enable_per_light_ambient`` /
             ``render_seg`` / ``use_shadows`` drop a term at compile time. Bit-identical ONLY if the
             scene's corresponding model fields are zero, which this script also reports so the
             claim can be checked rather than assumed.
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

FAMILIES = ("MetaWorld", "HumanoidBench", "Fetch", "RoboCasa")


def family(name):
    for f in FAMILIES:
        if name.startswith(f):
            return f
    raise SystemExit(f"unknown family for {name}")


def module_for(name):
    import importlib
    return importlib.import_module({
        "MetaWorld": "stable_envs.envs.metaworld",
        "HumanoidBench": "stable_envs.envs.humanoidbench",
        "Fetch": "stable_envs.envs.fetch",
        "RoboCasa": "stable_envs.envs.robocasa"}[family(name)])


def build(mod, env, res, nworld):
    fam = family(type(env).__name__)
    cls = type(env)
    if fam == "MetaWorld":
        return mod._context(cls.TASK, res, nworld)
    if fam == "HumanoidBench":
        return mod._context(cls.ROBOT, cls.TASK, res, nworld)
    if fam == "Fetch":
        return mod._context(cls.SCENE if hasattr(cls, "SCENE") else cls.TASK, res, nworld)
    return mod._context(cls.KITCHEN, cls.CAMERA, res, nworld)


def timed(fn, reps, warm=3):
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


def rgb_of(c, n):
    return c.bgra[:n][..., [2, 1, 0]].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="MetaWorldReach")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--blockdims", default="32,64,128,256,512")
    a = ap.parse_args()

    import warp as wp

    import mujoco_warp as mjw

    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    print(f"# {a.env} knobs, nworld={a.nworld} res={a.size} on {torch.cuda.get_device_name(0)}",
          flush=True)

    # ---- scene facts that decide whether a compile-time drop is bit-identical -------------------
    b = getattr(env, "_b", None)
    m = None
    for attr in ("model",):
        if b is not None and hasattr(b, attr):
            m = getattr(b, attr)
    if m is None and hasattr(mod, "_scene"):
        m = mod._scene(type(env).KITCHEN).model
    facts = {}
    if m is not None:
        facts = {"nlight": int(m.nlight), "ngeom": int(m.ngeom), "nmesh": int(m.nmesh),
                 "ntex": int(m.ntex),
                 "mat_emission_max": float(np.max(m.mat_emission)) if m.nmat else 0.0,
                 "mat_specular_max": float(np.max(m.mat_specular)) if m.nmat else 0.0,
                 "mat_shininess_max": float(np.max(m.mat_shininess)) if m.nmat else 0.0,
                 "light_ambient_max": float(np.max(m.light_ambient)) if m.nlight else 0.0,
                 "light_castshadow": [bool(x) for x in np.asarray(m.light_castshadow).ravel()]}
    print("scene:", json.dumps(facts), flush=True)

    # ---- kernel-by-kernel decomposition at the shipped settings ---------------------------------
    mod._CTX_CACHE.clear()
    c = build(mod, env, a.size, a.nworld)
    ref = rgb_of(c, a.nworld)
    per = {}
    per["kinematics"] = timed(lambda: c.mjw.kinematics(c.wm, c.wd), a.reps)
    if getattr(c, "needs_com", False):
        per["com_pos"] = timed(lambda: c.mjw.com_pos(c.wm, c.wd), a.reps)
    per["camlight"] = timed(lambda: c.mjw.camlight(c.wm, c.wd), a.reps)
    per["refit_bvh"] = timed(lambda: c.mjw.refit_bvh(c.wm, c.wd, c.rc), a.reps)
    per["render"] = timed(lambda: c.mjw.render(c.wm, c.wd, c.rc), a.reps)
    per["graph"] = timed(lambda: c.wp.capture_launch(c.graph), a.reps)
    print("kernels (ms): " + json.dumps({k: round(v * 1e3, 3) for k, v in per.items()}), flush=True)
    base_graph = per["graph"]

    rows = {"baseline": {"ms": base_graph * 1e3, "fps": a.nworld / base_graph, "maxdiff": 0}}

    # ---- block_dim: a launch geometry, so the frame must be bit-identical -----------------------
    orig_put_model = mjw.put_model

    def sweep(label, patch_put_model=None, rc_over=None):
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()
        orig_crc = MW.create_render_context
        if rc_over:
            def crc(mm, **kw):
                kw.update(rc_over)
                return orig_crc(mm, **kw)
            MW.create_render_context = crc
        if patch_put_model:
            mjw.put_model = patch_put_model
        try:
            cc = build(mod, env, a.size, a.nworld)
            dt = timed(lambda: cc.wp.capture_launch(cc.graph), a.reps)
            try:
                d = int((rgb_of(cc, a.nworld).to(torch.int16) - ref.to(torch.int16)).abs().max())
            except Exception:
                d = -1
            rows[label] = {"ms": dt * 1e3, "fps": a.nworld / dt, "maxdiff": d,
                           "speedup": base_graph / dt}
            print(f"{label:34s} {dt*1e3:8.3f} ms  {a.nworld/dt:9,.0f} f/s  "
                  f"x{base_graph/dt:5.2f}  maxdiff {d}", flush=True)
        except Exception as e:
            rows[label] = {"error": f"{type(e).__name__}: {e}"}
            print(f"{label:34s} ERROR {type(e).__name__}: {e}", flush=True)
        finally:
            MW.create_render_context = orig_crc
            mjw.put_model = orig_put_model
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()

    for bd in [int(x) for x in a.blockdims.split(",")]:
        def pm(mm, _bd=bd, **kw):
            wm = orig_put_model(mm, **kw)
            wm.block_dim.render = _bd
            return wm
        sweep(f"block_dim={bd}", patch_put_model=pm)

    for over in ({"enable_specular": False}, {"enable_emission": False},
                 {"enable_per_light_ambient": False}, {"enable_backface_culling": False},
                 {"render_seg": False}, {"use_shadows": False}, {"render_skybox": False},
                 {"use_textures": False}):
        sweep(",".join(f"{k}={v}" for k, v in over.items()), rc_over=over)

    print(json.dumps({"env": a.env, "nworld": a.nworld, "size": a.size, "scene": facts,
                      "kernels_ms": {k: v * 1e3 for k, v in per.items()}, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
