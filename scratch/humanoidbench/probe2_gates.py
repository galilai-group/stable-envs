"""G1 (purity), G2 (compression), G3 (decoder stability) and the frame budget, one task per process.

Everything here is measured against MuJoCo's own rasterizer at the task's model camera, and the FIRST
thing measured is whether that rasterizer reproduces gymnasium's own frame -- otherwise every later
number is a comparison against a different reference.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, VISIBLE_GROUPS, cam_id, gym_shot, interior_mask, jsave, make_env,
                       mj_render, mj_segment, sample_qpos, set_state)

SIZE = 256
NSTATE = 12


def model_arrays(m):
    out = {}
    for name in dir(m):
        if name.startswith("_"):
            continue
        try:
            v = getattr(m, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.size:
            out[name] = np.array(v, copy=True)
    return out


def main():
    task = sys.argv[1]
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
    import mujoco
    res = {"task": task, "robot": robot, "size": SIZE}

    env, u = make_env(task, robot, SIZE)
    m, d = u.model, u.data
    cam = cam_id(u)
    res["camera"] = dict(name=u.camera_name, id=int(cam), mode=int(m.cam_mode[cam]))
    res["nq"], res["nv"] = int(m.nq), int(m.nv)
    res["state_bytes_vs_frame"] = round(int(m.nq) * 8 / (SIZE * SIZE * 3), 6)

    # ---- the reference: is mujoco.Renderer at the model camera gymnasium's own frame? -------
    q0 = np.array(d.qpos, np.float64)
    a = gym_shot(u)
    b = mj_render(m, d, cam, SIZE, q0[None], tag="ref")[0]
    dd = np.abs(a.astype(np.int16) - b.astype(np.int16))
    res["reference_matches_gym"] = dict(exact=round(float((dd.max(-1) == 0).mean()), 6),
                                        max_abs=int(dd.max()),
                                        n_colors=int(len(np.unique(a.reshape(-1, 3), axis=0))))

    # ---- G3: is the compiled model a constant of the family? --------------------------------
    A = model_arrays(m)
    np.random.seed(1)
    env.reset(seed=1)
    for _ in range(5):
        env.step(env.action_space.sample())
    B = model_arrays(m)
    np.random.seed(2)
    env.reset(seed=2)
    B2 = model_arrays(m)
    changed_step = sorted(k for k in A if A[k].shape != B[k].shape or not np.array_equal(A[k], B[k]))
    changed_reset = sorted(k for k in A if A[k].shape != B2[k].shape or not np.array_equal(A[k], B2[k]))
    res["G3"] = dict(model_arrays_changed_by_5_steps=changed_step,
                     model_arrays_changed_by_reset=changed_reset,
                     nq=int(m.nq), ngeom=int(m.ngeom), nmesh=int(m.nmesh))

    # ---- states from the env's own distribution ---------------------------------------------
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
    res["qpos_spread"] = round(float(np.abs(Q - Q[0]).max()), 4)

    # ---- G1: purity --------------------------------------------------------------------------
    f1 = mj_render(m, d, cam, SIZE, Q, tag="ref")
    f2 = mj_render(m, d, cam, SIZE, Q, tag="ref")
    res["G1_rerender_max"] = int(np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max())

    # qvel randomised x100: the frame must not depend on it
    rng = np.random.default_rng(0)
    worst = 0
    for i in range(100):
        q = Q[i % len(Q)]
        d.qpos[:] = q
        d.qvel[:] = rng.uniform(-100, 100, size=int(m.nv))
        mujoco.mj_forward(m, d)
        r = mj_render.__globals__["renderer"](m, SIZE, "ref")
        r.update_scene(d, camera=int(cam))
        g = np.ascontiguousarray(r.render())
        worst = max(worst, int(np.abs(g.astype(np.int16) - f1[i % len(Q)].astype(np.int16)).max()))
    res["G1_qvel_x100_max"] = worst

    # same qpos reached through a different history
    np.random.seed(7)
    env.reset(seed=7)
    for _ in range(23):
        env.step(env.action_space.sample())
    f3 = mj_render(m, d, cam, SIZE, Q, tag="ref")
    res["G1_different_history_max"] = int(np.abs(f1.astype(np.int16) - f3.astype(np.int16)).max())

    # a fresh MjData, i.e. no carried derived state at all
    d2 = mujoco.MjData(m)
    f4 = []
    r = mj_render.__globals__["renderer"](m, SIZE, "ref")
    for q in Q:
        set_state(m, d2, q)
        r.update_scene(d2, camera=int(cam))
        f4.append(np.ascontiguousarray(r.render()))
    res["G1_fresh_data_max"] = int(np.abs(np.stack(f4).astype(np.int16) - f1.astype(np.int16)).max())

    # ---- reset staleness (the Meta-World protocol rule) --------------------------------------
    np.random.seed(3)
    env.reset(seed=3)
    s_reset = gym_shot(u)
    mujoco.mj_forward(m, d)
    s_fwd = gym_shot(u)
    res["stale_after_reset_max"] = int(np.abs(s_reset.astype(np.int16) - s_fwd.astype(np.int16)).max())

    # ---- frame budget ------------------------------------------------------------------------
    seg = mj_segment(m, d, cam, SIZE, Q, tag="seg")
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    issite = objtype == int(mujoco.mjtObj.mjOBJ_SITE)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    mesh = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 7], np.int64)
    prim = np.array([i for i in range(int(m.ngeom))
                     if int(m.geom_type[i]) not in (0, 7)], np.int64)
    textured_prim = []
    for g in prim.tolist():
        mid = int(m.geom_matid[g])
        if mid < 0:
            continue
        t = m.mat_texid[mid]
        t = int(np.max(t)) if np.ndim(t) else int(t)
        if t >= 0:
            textured_prim.append(g)
    res["budget"] = dict(
        background=round(float((~isgeom & ~issite).mean()), 5),
        floor=round(float((isgeom & np.isin(objid, plane)).mean()), 5),
        mesh=round(float((isgeom & np.isin(objid, mesh)).mean()), 5),
        primitive=round(float((isgeom & np.isin(objid, prim)).mean()), 5),
        textured_primitive=round(float((isgeom & np.isin(objid, np.array(textured_prim, np.int64))).mean()), 5)
        if textured_prim else 0.0,
        sites=round(float(issite.mean()), 5))

    # ---- shadow + reflection share, on the reference itself ----------------------------------
    def flag(scn_flag, value):
        """``MjvScene.flags`` persists across ``update_scene``, so it is restored here rather than
        left set -- leaving it set made the reflection measurement below report the SHADOW number."""
        r = mj_render.__globals__["renderer"](m, SIZE, "ref")
        out = []
        try:
            for q in Q:
                set_state(m, d, q)
                r.update_scene(d, camera=int(cam))
                r.scene.flags[scn_flag] = value
                out.append(np.ascontiguousarray(r.render()))
        finally:
            r.update_scene(d, camera=int(cam))
            r.scene.flags[scn_flag] = 1 - value
        return np.stack(out)

    noshadow = flag(int(mujoco.mjtRndFlag.mjRND_SHADOW), 0)
    ds = np.abs(noshadow.astype(np.int16) - f1.astype(np.int16)).max(-1)
    im = interior_mask(f1)
    res["shadow_pass"] = dict(pixels_changed=round(float((ds > 0).mean()), 5),
                              max=int(ds.max()), mean=round(float(ds.mean()), 4),
                              interior_pixels_changed=round(float((ds[:, :-1, :-1][im] > 0).mean()), 5))
    keep = np.array(m.mat_reflectance, copy=True)
    m.mat_reflectance[:] = 0.0
    norefl = mj_render(m, d, cam, SIZE, Q, tag="ref")
    m.mat_reflectance[:] = keep
    dr = np.abs(norefl.astype(np.int16) - f1.astype(np.int16)).max(-1)
    res["reflection_pass"] = dict(pixels_changed=round(float((dr > 0).mean()), 5),
                                  max=int(dr.max()),
                                  interior_pixels_changed=round(float((dr[:, :-1, :-1][im] > 0).mean()), 5))

    # ---- the plate question: with every mover hidden, is the frame state-independent? --------
    keepg = np.array(m.geom_group, copy=True)
    keeps = np.array(m.site_group, copy=True)
    movers = [g for g in range(m.ngeom) if int(m.body_weldid[m.geom_bodyid[g]]) != 0]
    m.geom_group[movers] = 7
    m.site_group[:] = 7
    plate = mj_render(m, d, cam, SIZE, Q[:8], tag="plate")
    m.geom_group[:] = keepg
    m.site_group[:] = keeps
    dp = np.abs(plate.astype(np.int16) - plate[0][None].astype(np.int16)).max(-1)
    res["static_plate"] = dict(n_movers=len(movers), max=int(dp.max()),
                               pixels_changed=round(float((dp > 0).mean()), 5))

    jsave(f"{LOGS}/probe2_{robot}_{task}.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
