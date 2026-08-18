"""What the twelve non-landing DM Control domains actually need, measured one domain per process.

Three questions, and the answers decide where each domain can land:

  1. **What does ``initialize_episode`` write into the MODEL?** ``docs/research/dmcontrol.md`` S8 says
     "reacher / finger / fish / swimmer move a target geom or site per EPISODE", which is true and is
     four different facts. Read straight off the source, ``finger-spin`` only ever writes ``site_rgba``
     alpha and ``dof_damping`` -- CONSTANTS, not a per-episode draw -- and ``fish-upright`` only hides
     its target. So the census is taken over real episodes rather than transcribed.

  2. **Is the frame stale straight out of ``reset()``?** ``docs/research/metaworld.md`` S1.1: several
     Meta-World tasks write ``model.body_pos`` AFTER their last forward, so ``data.site_xpos`` holds
     the previous draw and no stored state can reproduce that frame. Whether DM Control does the same
     is a property of ``initialize_episode``'s write order and has to be measured, not assumed --
     ``Task.after_reset`` may or may not forward after the model edits.

  3. **Is there a static plate once the per-episode elements are treated as MOVERS?** The shipped
     renderer bakes the static half of the scene once. A geom the task moves per episode sits on the
     WORLD body, so ``_movers`` (body != 0) calls it static and it goes into the plate -- which is why
     the plate is only constant within an episode. Treating it as a mover with its pose in the state
     costs no bake, and this measures whether the plate is then constant ACROSS episodes.

Plus two mover kinds the shipped ray-caster does not have and that decide two domains:

  * **zero-alpha geometry.** ``mujoco.Renderer`` draws by GROUP; MuJoCo skips alpha == 0. ``finger``
     and ``fish`` hide elements that way, so a renderer that filters on group alone paints geometry the
     reference does not have.
  * **the spatial tendon.** ``ball_in_cup`` draws one, and this reads its geometry out of the live
     ``mjvScene`` (type, size, pose) rather than guessing it from the XML, plus whether
     ``mjVIS_TENDON`` removes it from the plate.

One domain per process: two MuJoCo envs alive together can return byte-identical frames from cameras
that provably differ (docs/research/nvdiffrast.md), and an EGL context is never released.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HIDDEN_GROUP = 5
VISIBLE_GROUPS = (0, 1, 2)

#: The model arrays a DM Control ``initialize_episode`` could plausibly touch and the renderer reads.
#: Deliberately wider than what the source shows, so an edit nobody transcribed still shows up.
MODEL_ARRAYS = ("geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_matid",
                "site_pos", "site_quat", "site_size", "site_rgba", "site_matid",
                "body_pos", "body_quat", "light_pos", "light_dir", "light_diffuse",
                "mat_rgba", "cam_pos", "cam_quat", "cam_fovy", "tendon_rgba", "tendon_width")


def model_of(p):
    return p.model.ptr if hasattr(p.model, "ptr") else p.model


def data_of(p):
    """The raw ``mujoco.MjData``. ``dm_control`` wraps it, and the wrapper is NOT accepted by the
    pybind signatures -- ``mjv_updateScene`` raises ``TypeError`` on it rather than duck-typing."""
    return p.data.ptr if hasattr(p.data, "ptr") else p.data


def effective_alpha(m, kind: str) -> np.ndarray:
    """Alpha MuJoCo actually draws with: the element's own, unless the material supplies it.

    Same resolution ``stable_envs.render.mjwarp.effective_alpha`` performs, repeated here so the probe
    does not depend on the module it is meant to inform.
    """
    rgba = np.asarray(getattr(m, f"{kind}_rgba"), np.float64)
    matid = np.asarray(getattr(m, f"{kind}_matid"), np.int64)
    a = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    for i in np.flatnonzero(default & (matid >= 0)):
        a[i] = float(m.mat_rgba[matid[i], 3])
    return a


def visible(m, kind: str) -> list[int]:
    grp = np.asarray(getattr(m, f"{kind}_group"), np.int64)
    alpha = effective_alpha(m, kind)
    n = int(m.ngeom if kind == "geom" else m.nsite)
    return [i for i in range(n) if int(grp[i]) in VISIBLE_GROUPS and alpha[i] > 0]


def main() -> None:
    import mujoco
    from dm_control import suite

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--states", type=int, default=6)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/dmc2")
    a = ap.parse_args()
    out: dict = {"domain": a.domain, "task": a.task, "camera": a.camera, "size": a.size}

    env = suite.load(a.domain, a.task, task_kwargs={"random": 0})
    p = env.physics
    m = model_of(p)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), a.size)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), a.size)

    # ---- 0. model facts -------------------------------------------------------------------------
    cam_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(m.ncam)]
    out["model"] = {
        "nq": int(m.nq), "ngeom": int(m.ngeom), "nsite": int(m.nsite), "nlight": int(m.nlight),
        "ntendon": int(m.ntendon), "nmesh": int(m.nmesh), "ncam": int(m.ncam),
        "cameras": cam_names,
        "cam_mode": [int(x) for x in m.cam_mode],
        "cam0_is_trackcom": bool(int(m.cam_mode[a.camera]) ==
                                 int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM)),
        "light_mode": [int(x) for x in m.light_mode],
        "geom_types": [int(x) for x in m.geom_type],
        "site_types": [int(x) for x in m.site_type],
    }

    # ---- 1. what initialize_episode writes into the model ---------------------------------------
    # Snapshot straight after load (no reset yet), then after each of N resets. Two different
    # comparisons come out of it: what the FIRST reset changes (task constants, applied once) and what
    # varies BETWEEN resets (the per-episode draw). Conflating them is what made "finger moves its
    # target site" read as a per-episode fact when for `spin` it is neither.
    at_load = {k: np.array(getattr(m, k), copy=True) for k in MODEL_ARRAYS if hasattr(m, k)}
    draws, qposes = [], []
    for _ in range(a.episodes):
        env.reset()
        draws.append({k: np.array(getattr(m, k), copy=True) for k in at_load})
        qposes.append(np.array(p.data.qpos, np.float64))

    def entries(ref, cur):
        """Which (array, flat index) entries differ, with their spread."""
        got = {}
        for k in ref:
            d = np.abs(np.asarray(cur[k], np.float64) - np.asarray(ref[k], np.float64))
            idx = np.flatnonzero(d.reshape(-1) > 0)
            if idx.size:
                got[k] = [int(i) for i in idx]
        return got

    out["reset_changes_model"] = entries(at_load, draws[0])
    per_ep, spread = {}, {}
    for k in at_load:
        stack = np.stack([np.asarray(d[k], np.float64).reshape(-1) for d in draws])
        var = stack.max(0) - stack.min(0)
        idx = np.flatnonzero(var > 0)
        if idx.size:
            per_ep[k] = [int(i) for i in idx]
            spread[k] = [float(var[i]) for i in idx]
    out["per_episode_model_entries"] = per_ep
    out["per_episode_spread"] = spread
    out["per_episode_floats"] = int(sum(len(v) for v in per_ep.values()))

    # name the elements, because "geom_pos entry 12" is not a finding
    named = {}
    for k, idx in per_ep.items():
        obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE,
               "body": mujoco.mjtObj.mjOBJ_BODY, "light": mujoco.mjtObj.mjOBJ_LIGHT,
               "mat": mujoco.mjtObj.mjOBJ_MATERIAL, "cam": mujoco.mjtObj.mjOBJ_CAMERA,
               "tendon": mujoco.mjtObj.mjOBJ_TENDON}.get(k.split("_")[0])
        w = int(np.asarray(getattr(m, k)).reshape(np.asarray(getattr(m, k)).shape[0], -1).shape[1])
        named[k] = sorted({f"{mujoco.mj_id2name(m, obj, i // w)}[{i % w}]" for i in idx})
    out["per_episode_named"] = named

    # ---- 2. zero-alpha census -------------------------------------------------------------------
    out["zero_alpha"] = {}
    for kind in ("geom", "site"):
        al = effective_alpha(m, kind)
        grp = np.asarray(getattr(m, f"{kind}_group"), np.int64)
        n = int(m.ngeom if kind == "geom" else m.nsite)
        hid = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM if kind == "geom"
                                 else mujoco.mjtObj.mjOBJ_SITE, i)
               for i in range(n) if al[i] <= 0 and int(grp[i]) in VISIBLE_GROUPS]
        out["zero_alpha"][kind] = hid

    # ---- 3. tendons, read off the live mjvScene -------------------------------------------------
    r = mujoco.Renderer(m, height=a.size, width=a.size)          # ONE, never closed
    dd = data_of(p)
    opt = mujoco.MjvOption()
    env.reset()
    for _ in range(5):
        env.step(np.zeros(env.action_spec().shape))
    r.update_scene(dd, camera=a.camera, scene_option=opt)
    ten = []
    for i in range(int(r.scene.ngeom)):
        g = r.scene.geoms[i]
        if int(g.objtype) == int(mujoco.mjtObj.mjOBJ_TENDON):
            ten.append({"objid": int(g.objid), "type": int(g.type),
                        "size": [float(x) for x in g.size], "pos": [float(x) for x in g.pos],
                        "mat": [float(x) for x in np.asarray(g.mat).reshape(-1)],
                        "rgba": [float(x) for x in g.rgba]})
    out["tendon_scene_geoms"] = ten
    if int(m.ntendon):
        out["tendon"] = {
            "width": [float(x) for x in m.tendon_width],
            "rgba": [[float(y) for y in x] for x in np.asarray(m.tendon_rgba)],
            "num_wrap": [int(x) for x in dd.ten_wrapnum],
            "wrap_adr": [int(x) for x in dd.ten_wrapadr],
            "wrap_xpos": np.asarray(dd.wrap_xpos, np.float64).reshape(-1, 3)[:8].tolist(),
        }
    out["has_mjVIS_TENDON"] = hasattr(mujoco.mjtVisFlag, "mjVIS_TENDON")

    # ---- 4. reset staleness (the Meta-World protocol rule) --------------------------------------
    def shot():
        r.update_scene(dd, camera=a.camera, scene_option=opt)
        return r.render().copy()

    stale = []
    for _ in range(4):
        env.reset()
        f0 = shot()
        mujoco.mj_forward(m, dd)
        f1 = shot()
        stale.append(int(np.abs(f0.astype(np.int16) - f1.astype(np.int16)).max()))
    out["reset_staleness_max"] = stale

    # ---- 5. is there a plate, and what has to be hidden for it ----------------------------------
    # Four ladders, each strictly wider than the last, over states drawn from DIFFERENT episodes.
    gvis, svis = visible(m, "geom"), visible(m, "site")
    gmov = [i for i in gvis if int(m.geom_bodyid[i]) != 0]
    smov = [i for i in svis if int(m.site_bodyid[i]) != 0]
    ep_geom = sorted({i // 3 for i in per_ep.get("geom_pos", [])}
                     | {i // 4 for i in per_ep.get("geom_quat", [])}
                     | {i // 3 for i in per_ep.get("geom_size", [])})
    ep_site = sorted({i // 3 for i in per_ep.get("site_pos", [])}
                     | {i // 4 for i in per_ep.get("site_quat", [])}
                     | {i // 3 for i in per_ep.get("site_size", [])})
    out["movers"] = {"geom": gmov, "site": smov, "episode_geom": ep_geom, "episode_site": ep_site,
                     "visible_geom": gvis, "visible_site": svis}

    def plate_series(hide_g, hide_s, tendon: bool, per_episode: bool):
        """Render the static half over states, optionally one state per episode."""
        o = mujoco.MjvOption()
        if not tendon and hasattr(mujoco.mjtVisFlag, "mjVIS_TENDON"):
            o.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = 0
        keep_g = np.array(m.geom_group, copy=True)
        keep_s = np.array(m.site_group, copy=True)
        frames = []
        try:
            if hide_g:
                m.geom_group[hide_g] = HIDDEN_GROUP
            if hide_s:
                m.site_group[hide_s] = HIDDEN_GROUP
            for k in range(a.states):
                if per_episode:
                    env.reset()
                else:
                    if k == 0:
                        env.reset()
                    env.step(np.zeros(env.action_spec().shape))
                r.update_scene(dd, camera=a.camera, scene_option=o)
                frames.append(r.render().copy())
        finally:
            m.geom_group[:] = keep_g
            m.site_group[:] = keep_s
        f = np.stack(frames).astype(np.int16)
        d = np.abs(f - f[0]).max(-1)
        return {"max": int(d.max()), "frac_changed": float((d > 0).mean())}

    out["plate"] = {
        "movers_only_within_episode": plate_series(gmov, smov, True, False),
        "movers_only_across_episodes": plate_series(gmov, smov, True, True),
        "movers_plus_tendon_across": plate_series(gmov, smov, False, True),
        "movers_plus_episode_plus_tendon_across":
            plate_series(sorted(set(gmov) | set(ep_geom)), sorted(set(smov) | set(ep_site)),
                         False, True),
    }

    # ---- 6. the state ladder: does qpos + the per-episode slice reproduce a frame? ---------------
    # Episode A's state is captured, a DIFFERENT episode B is restored to it rung by rung, and the
    # frames are diffed. The control is R0 (qpos only across episodes), which must FAIL wherever the
    # model carries a per-episode draw, or the ladder proves nothing.
    def capture():
        return (np.array(dd.qpos, np.float64),
                {k: np.array(getattr(m, k), copy=True) for k in at_load})

    def restore(q, mod, arrays):
        for k in arrays:
            getattr(m, k)[:] = mod[k]
        dd.qpos[:] = q
        dd.qvel[:] = 0
        mujoco.mj_forward(m, dd)

    env.reset()
    for _ in range(7):
        env.step(np.zeros(env.action_spec().shape))
    qa, moda = capture()
    ref = shot()
    ladder = {}
    for name, arrays in (("R0_qpos_only", ()),
                         ("R1_qpos_plus_episode_slice", tuple(per_ep.keys())),
                         ("R2_qpos_plus_all_arrays", tuple(at_load.keys()))):
        env.reset()                                   # a genuinely different episode
        for _ in range(3):
            env.step(np.zeros(env.action_spec().shape))
        restore(qa, moda, arrays)
        got = shot()
        d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
        ladder[name] = {"max": int(d.max()), "frac": float((d.max(-1) > 0).mean())}
    out["state_ladder"] = ladder

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"widen_{a.domain}-{a.task}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    print(f"@@wrote {path}")


if __name__ == "__main__":
    main()
