"""How much of an MJWarp frame is the STATIC scene, and does a two-pass split reproduce it?

The premise every family here shares: a kitchen, a table or a floor does not move, and a robot does.
``mujoco_warp`` re-casts every ray against the whole scene anyway, so a batch of 1024 frames pays for
the same static geometry 1024 times. This script measures what that costs and whether removing it is
free in pixels.

Three contexts are built from the SAME model with only ``geom_group`` changed:

  ``base``    the family's own visible set
  ``static``  movers (a geom whose body is not welded to the world, or is a mocap body) hidden
  ``dyn``     everything else hidden

and, because a hit is settled by distance alone whenever no secondary ray crosses the split, the two
halves are composited by DEPTH and compared against ``base`` pixel for pixel. On a scene with no
shadow-casting light that composite is exact by construction; on one with shadows it is not, and the
disagreement this prints is the size of the term a split would have to put back.
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


def model_of(mod, env):
    fam = family(type(env).__name__)
    cls = type(env)
    if fam == "MetaWorld":
        return mod._build_bundle(cls.TASK).model
    if fam == "HumanoidBench":
        return mod._build_bundle(cls.ROBOT, cls.TASK).model
    if fam == "Fetch":
        return mod._build_bundle(cls.SCENE if hasattr(cls, "SCENE") else cls.TASK).model
    return mod._scene(cls.KITCHEN).model


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


def movers_of(m) -> np.ndarray:
    """Exactly ``metaworld.py``'s rule: not welded to the world, or a mocap body."""
    gbody = np.asarray(m.geom_bodyid, np.int64)
    weld = np.asarray(m.body_weldid, np.int64)
    mocap = np.asarray(m.body_mocapid, np.int64)
    return np.flatnonzero((weld[gbody] != 0) | (mocap[gbody] >= 0)).astype(np.int64)


def moving_geoms(mod, env, m, k: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Geoms whose world transform MOVES under the env's OWN declared state. Measured, not inferred.

    ``body_weldid != 0`` is the wrong question for a pinned scene: a RoboCasa kitchen hangs every
    cabinet door and drawer off a hinge, so 176 of 344 visible geoms are 'not welded' while the env
    pins their joints to a constant and no state it can be handed will ever move them. What a static
    plate can hold is decided by the state, so it is the state that is varied here.
    """
    import mujoco
    fam = family(type(env).__name__)
    d = mujoco.MjData(m)
    st = env.sample_states(int(k))
    pos, mat = [], []

    def snap():
        pos.append(np.array(d.geom_xpos, np.float64))
        mat.append(np.array(d.geom_xmat, np.float64))

    if fam == "RoboCasa":
        s = mod._scene(type(env).KITCHEN)
        for row in np.asarray(st, np.float64):
            q = s.pinned.copy()
            q[env._live] = row
            d.qpos[:] = q
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            snap()
    elif fam == "MetaWorld":
        b = mod._build_bundle(type(env).TASK)
        for row in np.asarray(st, np.float64):
            mod._apply_state(b, m, d, row)
            snap()
        for arr in mod.MODEL_ARRAYS:
            getattr(m, arr)[:] = b.pristine[arr]
    else:                                                    # HumanoidBench, Fetch: the state IS qpos
        for row in np.asarray(st, np.float64):
            d.qpos[:] = row[:m.nq]
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            snap()
    p = np.stack(pos)
    q = np.stack(mat)
    moved = ((p - p[0]).any(0).any(-1) | (q - q[0]).any(0).any(-1))
    return np.flatnonzero(moved).astype(np.int64), np.asarray(st, np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--movers", default="weld", choices=["weld", "moved"],
                    help="'weld': body not welded to world. 'moved': measured against the env's "
                         "own sample_states, which is what a pinned scene actually moves.")
    a = ap.parse_args()

    import warp as wp

    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    m = model_of(mod, env)
    vis_groups = set(int(g) for g in mod.VISIBLE_GROUPS)
    g0 = np.array(m.geom_group, copy=True)

    mv = movers_of(m) if a.movers == "weld" else moving_geoms(mod, env, m)[0]
    mv_vis = np.array([g for g in mv if int(g0[g]) in vis_groups], np.int64)
    vis = np.array([g for g in range(m.ngeom) if int(g0[g]) in vis_groups], np.int64)
    st_vis = np.array([g for g in vis if g not in set(mv_vis.tolist())], np.int64)
    import mujoco
    ntri = lambda gs: int(sum(int(m.mesh_facenum[int(m.geom_dataid[g])])
                              for g in gs if int(m.geom_type[g]) == 7))
    facts = {"env": a.env, "movers_rule": a.movers, "ngeom": int(m.ngeom), "visible": len(vis),
             "visible_movers": len(mv_vis), "visible_static": len(st_vis),
             "tri_movers": ntri(mv_vis), "tri_static": ntri(st_vis),
             "nlight": int(m.nlight),
             "castshadow": [bool(x) for x in np.asarray(m.light_castshadow).ravel()]}
    print("scene: " + json.dumps(facts), flush=True)

    orig_crc = MW.create_render_context
    rows = {}
    frames = {}

    def one(label, hide, skybox=None):
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()

        def crc(mm, **kw):
            kw["render_depth"] = True
            if skybox is not None:
                kw["render_skybox"] = skybox
            return orig_crc(mm, **kw)
        MW.create_render_context = crc
        m.geom_group[:] = g0
        if len(hide):
            m.geom_group[hide] = MW.HIDDEN_GROUP
        try:
            c = build(mod, env, a.size, a.nworld)
            dt = timed(lambda: c.wp.capture_launch(c.graph), a.reps)
            rgb = c.bgra[:a.nworld][..., [2, 1, 0]].clone()
            dep = wp.to_torch(c.rc.depth_data).reshape(a.nworld, a.size, a.size).clone()
            frames[label] = (rgb, dep)
            rows[label] = {"ms": dt * 1e3, "fps": a.nworld / dt}
            print(f"{label:12s} {dt*1e3:8.3f} ms  {a.nworld/dt:10,.0f} f/s", flush=True)
        finally:
            MW.create_render_context = orig_crc
            m.geom_group[:] = g0
            mod._CTX_CACHE.clear()
            torch.cuda.empty_cache()

    one("base", np.array([], np.int64))
    one("static", mv_vis)
    one("dyn", st_vis)
    one("dyn_nosky", st_vis, skybox=False)

    if "base" in frames and "static" in frames and "dyn" in frames:
        rgb_b, _ = frames["base"]
        rgb_s, dep_s = frames["static"]
        rgb_d, dep_d = frames["dyn"]
        # depth_out is 0.0 on a miss; a hit is the planar distance, always > 0.
        hit_d = dep_d > 0.0
        hit_s = dep_s > 0.0
        take_d = hit_d & (~hit_s | (dep_d < dep_s))
        comp = torch.where(take_d.unsqueeze(-1), rgb_d, rgb_s)
        diff = (comp.to(torch.int16) - rgb_b.to(torch.int16)).abs()
        rows["composite"] = {
            "max": int(diff.max()), "mean": float(diff.to(torch.float32).mean()),
            "exact_frac": float((diff.amax(-1) == 0).to(torch.float32).mean()),
            "within8_frac": float((diff.amax(-1) <= 8).to(torch.float32).mean()),
            "mover_pixel_frac": float(take_d.to(torch.float32).mean()),
            "miss_frac": float((~hit_d & ~hit_s).to(torch.float32).mean()),
        }
        print("composite vs base: " + json.dumps(rows["composite"]), flush=True)
        base = rows["base"]["ms"]
        for k in ("static", "dyn", "dyn_nosky"):
            if k in rows:
                rows[k]["speedup_vs_base"] = base / rows[k]["ms"]
        print(f"per-frame speedup if only the DYN pass is paid: "
              f"x{base / rows['dyn_nosky']['ms']:.2f}", flush=True)

    print("JSON " + json.dumps({"scene": facts, "size": a.size, "nworld": a.nworld,
                                "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
