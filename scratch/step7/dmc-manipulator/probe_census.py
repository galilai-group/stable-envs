"""Phase-0 census + state sufficiency for manipulator-bring_ball (Step 7, W3).

Nothing here touches ``stable_envs/``. It reads the compiled model, the live ``mjvScene``, and
``physics.render`` at the SWM-registered size, and answers:

  1. what the scene is made of (geoms, materials, textures, lights, cameras, reflective planes);
  2. state sufficiency the Step-1 way (re-render / qvel x100 / same qpos different history);
  3. the per-episode draw, via the SHIPPED ``_episode_slots`` census (so its guards are exercised);
  4. whether ``visualize_reward`` or a reward-coloured material is on by default;
  5. what the plate covers: hide every mover and diff the plate against the full frame, within an
     episode and across episodes.

Run:  MUJOCO_GL=glfw python scratch/step7/dmc-manipulator/probe_census.py --size 224
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np

DOMAIN, TASK, CAMERA = "manipulator", "bring_ball", 0
VISIBLE_GROUPS = (0, 1, 2)
HIDDEN_GROUP = 5
EPISODE_ARRAYS = ("geom_pos", "geom_quat", "geom_size", "site_pos", "site_quat", "site_size",
                  "body_pos", "body_quat", "light_pos", "light_dir")
N_DRAWS = 6


def load(seed=0):
    from dm_control import suite
    env = suite.load(DOMAIN, TASK, task_kwargs={"random": seed})
    p = env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    return env, p, m


def name(m, obj, i):
    import mujoco
    return mujoco.mj_id2name(m, obj, i) or f"{obj}{i}"


def effective_alpha(m, kind):
    rgba = np.asarray(getattr(m, f"{kind}_rgba"), np.float64)
    mat = np.asarray(m.mat_rgba, np.float64)
    matid = np.asarray(getattr(m, f"{kind}_matid"), np.int64)
    a = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    for i in np.flatnonzero(default & (matid >= 0)):
        a[i] = float(mat[matid[i], 3])
    return a


def census(m, size):
    import mujoco
    GEOM, MAT, LIGHT, CAM, BODY = (mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_MATERIAL,
                                   mujoco.mjtObj.mjOBJ_LIGHT, mujoco.mjtObj.mjOBJ_CAMERA,
                                   mujoco.mjtObj.mjOBJ_BODY)
    kind = {0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE", 4: "ELLIPSOID", 5: "CYLINDER",
            6: "BOX", 7: "MESH", 8: "SDF"}
    alpha_g = effective_alpha(m, "geom")
    alpha_s = effective_alpha(m, "site")
    geoms = []
    for i in range(m.ngeom):
        mid = int(m.geom_matid[i])
        geoms.append(dict(
            id=i, name=name(m, GEOM, i), type=kind.get(int(m.geom_type[i]), int(m.geom_type[i])),
            body=name(m, BODY, int(m.geom_bodyid[i])), bodyid=int(m.geom_bodyid[i]),
            group=int(m.geom_group[i]), alpha=float(alpha_g[i]),
            material=(name(m, MAT, mid) if mid >= 0 else None),
            reflectance=(float(m.mat_reflectance[mid]) if mid >= 0 else 0.0),
            texid=(int(np.max(m.mat_texid[mid])) if mid >= 0 and np.ndim(m.mat_texid[mid])
                   else (int(m.mat_texid[mid]) if mid >= 0 else -1)),
            size=[float(x) for x in m.geom_size[i]],
            drawn=bool(int(m.geom_group[i]) in VISIBLE_GROUPS and alpha_g[i] > 0),
        ))
    sites = [dict(id=i, name=name(m, mujoco.mjtObj.mjOBJ_SITE, i), group=int(m.site_group[i]),
                  alpha=float(alpha_s[i]), bodyid=int(m.site_bodyid[i]),
                  drawn=bool(int(m.site_group[i]) in VISIBLE_GROUPS and alpha_s[i] > 0))
             for i in range(m.nsite)]
    mats = [dict(id=i, name=name(m, MAT, i), rgba=[float(x) for x in m.mat_rgba[i]],
                 reflectance=float(m.mat_reflectance[i]),
                 specular=float(m.mat_specular[i]), shininess=float(m.mat_shininess[i]),
                 texid=(int(np.max(m.mat_texid[i])) if np.ndim(m.mat_texid[i])
                        else int(m.mat_texid[i])))
            for i in range(m.nmat)]
    lights = [dict(id=i, name=name(m, LIGHT, i),
                   directional=int(getattr(m, "light_directional", np.zeros(m.nlight))[i])
                   if hasattr(m, "light_directional") else None,
                   type=(int(m.light_type[i]) if hasattr(m, "light_type") else None),
                   mode=int(m.light_mode[i]), castshadow=int(m.light_castshadow[i]),
                   pos=[float(x) for x in m.light_pos[i]],
                   dir=[float(x) for x in m.light_dir[i]],
                   diffuse=[float(x) for x in m.light_diffuse[i]],
                   specular=[float(x) for x in m.light_specular[i]],
                   ambient=[float(x) for x in m.light_ambient[i]],
                   cutoff=float(m.light_cutoff[i]), exponent=float(m.light_exponent[i]))
              for i in range(m.nlight)]
    cams = [dict(id=i, name=name(m, CAM, i), mode=int(m.cam_mode[i]), fovy=float(m.cam_fovy[i]),
                 pos=[float(x) for x in m.cam_pos[i]]) for i in range(m.ncam)]
    return dict(
        nq=int(m.nq), nv=int(m.nv), ngeom=int(m.ngeom), nsite=int(m.nsite), nmesh=int(m.nmesh),
        nhfield=int(m.nhfield), ntex=int(m.ntex), nlight=int(m.nlight), ncam=int(m.ncam),
        ntendon=int(m.ntendon), nbody=int(m.nbody),
        shadowsize=int(m.vis.quality.shadowsize), shadowclip=float(m.vis.map.shadowclip),
        offsamples=int(m.vis.quality.offsamples),
        headlight_active=int(m.vis.headlight.active),
        geoms=geoms, sites=sites, materials=mats, lights=lights, cameras=cams,
        drawn_geoms=[g["name"] for g in geoms if g["drawn"]],
        drawn_sites=[s["name"] for s in sites if s["drawn"]],
        reflective_planes=[g["name"] for g in geoms
                           if g["drawn"] and g["type"] == "PLANE" and g["reflectance"] > 0],
    )


def episode_slots(env, m):
    """The SHIPPED census, transcribed, plus the raw per-array diff so nothing is hidden."""
    have = [k for k in EPISODE_ARRAYS if hasattr(m, k)]
    draws = []
    for _ in range(N_DRAWS):
        env.reset()
        draws.append({k: np.array(getattr(m, k), copy=True).reshape(-1) for k in have})
    out = []
    for k in have:
        stack = np.stack([dr[k] for dr in draws]).astype(np.float64)
        for i in np.flatnonzero(stack.max(0) - stack.min(0) > 0):
            out.append((k, int(i), float(stack.max(0)[i] - stack.min(0)[i])))
    return out


def slot_label(m, arr, i):
    import mujoco
    kind = arr.split("_")[0]
    a = np.asarray(getattr(m, arr))
    w = int(a.reshape(a.shape[0], -1).shape[1]) if a.ndim > 1 else 1
    eid = i // w
    obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE,
           "body": mujoco.mjtObj.mjOBJ_BODY, "light": mujoco.mjtObj.mjOBJ_LIGHT}[kind]
    comp = {3: ("x", "y", "z"), 4: ("qw", "qx", "qy", "qz")}.get(w, tuple(map(str, range(w))))
    return f"{name(m, obj, eid)}_{arr.split('_', 1)[1]}_{comp[i % w]}"


def diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(max=int(d.max()), n_bad=int((d.max(-1) > 0).sum()),
                frac_bad=float((d.max(-1) > 0).mean()), mean_abs=float(d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/phase0")
    a = ap.parse_args()
    import mujoco
    S = a.size
    env, p, m = load(0)
    res = dict(domain=DOMAIN, task=TASK, camera=CAMERA, size=S,
               mujoco=mujoco.__version__)
    from importlib.metadata import version as _v
    res["dm_control"] = _v("dm_control")

    res["census"] = census(m, S)

    # ---- state sufficiency -------------------------------------------------
    env.reset()
    for _ in range(20):
        env.step(np.random.default_rng(0).uniform(-1, 1, env.action_spec().shape))
    q = np.array(p.data.qpos, copy=True)
    v = np.array(p.data.qvel, copy=True)
    f0 = p.render(S, S, camera_id=CAMERA).copy()
    f1 = p.render(S, S, camera_id=CAMERA).copy()
    with p.reset_context():
        p.data.qpos[:] = q
        p.data.qvel[:] = v * 100.0
    f2 = p.render(S, S, camera_id=CAMERA).copy()
    # same qpos through a different history: run more steps, then restore
    for _ in range(37):
        env.step(np.random.default_rng(7).uniform(-1, 1, env.action_spec().shape))
    with p.reset_context():
        p.data.qpos[:] = q
        p.data.qvel[:] = 0
    f3 = p.render(S, S, camera_id=CAMERA).copy()
    with p.reset_context():
        p.data.qpos[:] = q
        p.data.qvel[:] = 0
    f4 = p.render(S, S, camera_id=CAMERA).copy()
    res["state_sufficiency"] = dict(
        rerender=diff(f0, f1), qvel_x100=diff(f2, f4), different_history=diff(f3, f4))
    # every qpos entry moves pixels?
    moved = []
    for j in range(m.nq):
        qq = q.copy()
        qq[j] += 0.15
        with p.reset_context():
            p.data.qpos[:] = qq
            p.data.qvel[:] = 0
        moved.append(dict(j=j, n_bad=diff(p.render(S, S, camera_id=CAMERA), f4)["n_bad"]))
    res["qpos_entries_moving_pixels"] = moved
    res["dead_qpos_dims"] = [d["j"] for d in moved if d["n_bad"] == 0]

    # ---- per-episode draw --------------------------------------------------
    slots = episode_slots(env, m)
    res["episode_slots"] = [dict(array=k, index=i, span=s, label=slot_label(m, k, i))
                            for k, i, s in slots]
    res["episode_slot_guard_trips"] = sorted({k for k, _, _ in slots
                                              if k.endswith(("_size", "_quat", "_dir"))})

    # ---- visualize_reward --------------------------------------------------
    res["visualize_reward"] = dict(
        task_attr=bool(getattr(env.task, "_visualize_reward", False)),
        control_env_flag=bool(getattr(env, "_flat_observation", False)),
        highlight_materials=[mm["name"] for mm in res["census"]["materials"]
                             if mm["name"].endswith("_highlight")],
        self_rgba=[float(x) for x in m.mat_rgba[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_MATERIAL, "self")]],
        self_default_rgba=[float(x) for x in m.mat_rgba[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_MATERIAL, "self_default")]],
        note=("dm_control.suite.base.Task.after_step only recolours when "
              "physics.model.mat_rgba is touched by _set_reward_colors, which Task guards on "
              "the 'visualize_reward' flag set by suite.load(..., visualize_reward=True)"))

    # ---- what the plate covers --------------------------------------------
    # movers = anything off the world body, plus anything the episode draws
    ep_bodies = {int(i // 3) if k == "body_pos" else int(i // 4)
                 for k, i, _ in slots if k in ("body_pos", "body_quat")}
    gmov, smov = [], []
    for g in res["census"]["geoms"]:
        if not g["drawn"]:
            continue
        if g["bodyid"] != 0 or g["bodyid"] in ep_bodies:
            gmov.append(g["id"])
    for s in res["census"]["sites"]:
        if s["drawn"] and (s["bodyid"] != 0 or s["bodyid"] in ep_bodies):
            smov.append(s["id"])
    res["mover_geoms"] = [res["census"]["geoms"][i]["name"] for i in gmov]
    res["mover_sites"] = [res["census"]["sites"][i]["name"] for i in smov]
    res["static_geoms"] = [g["name"] for g in res["census"]["geoms"]
                           if g["drawn"] and g["id"] not in gmov]

    r = mujoco.Renderer(m, height=S, width=S)
    dptr = p.data.ptr if hasattr(p.data, 'ptr') else p.data
    opt = mujoco.MjvOption()
    keep_gg = np.array(m.geom_group, copy=True)
    keep_sg = np.array(m.site_group, copy=True)

    def plate():
        m.geom_group[gmov] = HIDDEN_GROUP
        if smov:
            m.site_group[smov] = HIDDEN_GROUP
        r.update_scene(dptr, camera=CAMERA, scene_option=opt)
        img = r.render().copy()
        m.geom_group[:] = keep_gg
        m.site_group[:] = keep_sg
        return img

    def full():
        r.update_scene(dptr, camera=CAMERA, scene_option=opt)
        return r.render().copy()

    # within an episode
    env.reset()
    pl_a, fr_a = [], []
    for _ in range(4):
        for _ in range(5):
            env.step(np.random.default_rng(3).uniform(-1, 1, env.action_spec().shape))
        pl_a.append(plate())
        fr_a.append(full())
    res["plate_within_episode"] = [diff(pl_a[0], x) for x in pl_a[1:]]
    res["plate_vs_full"] = [diff(pl_a[i], fr_a[i]) for i in range(len(pl_a))]
    # across episodes
    pl_b = []
    for _ in range(4):
        env.reset()
        pl_b.append(plate())
    res["plate_across_episodes"] = [diff(pl_b[0], x) for x in pl_b[1:]]

    # what the REFERENCE bridge is: Renderer at default MjvOption vs physics.render
    env.reset()
    res["renderer_vs_physics_render"] = diff(full(), p.render(S, S, camera_id=CAMERA))

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"census_{S}.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "census"}, indent=1, default=float))
    print("\nwrote", path)


if __name__ == "__main__":
    main()
