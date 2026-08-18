"""Fetch probe 1: what is actually in the model, what is actually in the picture.

Nothing here reads a constant off the source and believes it. Every question is asked by
constructing a frame in which only the quantity of interest can be responsible for a pixel.

  A. model census per scene (G3 across resets)
  B. the camera: the official render uses a FREE camera; recover its world pose by measurement
     (mjv_updateScene, average of the two GL eyes) and prove it by adding a MODEL camera at that
     pose and diffing 0/255 against the official frame
  C. the translucent mocap markers: how much of the frame do they touch, and by how much
  D. alpha census + textured-primitive census (what MJWarp would get wrong)
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import gymnasium
import gymnasium_robotics  # noqa: F401

gymnasium.register_envs(gymnasium_robotics)

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
os.makedirs(OUT, exist_ok=True)

SCENES = {
    "Reach": "FetchReach-v4",
    "Push": "FetchPush-v4",
    "Slide": "FetchSlide-v4",
    "PickAndPlace": "FetchPickAndPlace-v4",
}
SIZE = 480


def census(m):
    return dict(nq=int(m.nq), nv=int(m.nv), ngeom=int(m.ngeom), nmesh=int(m.nmesh),
                nbody=int(m.nbody), ntex=int(m.ntex), nmat=int(m.nmat), nsite=int(m.nsite),
                ncam=int(m.ncam), nlight=int(m.nlight), nmocap=int(m.nmocap))


def eff_alpha(m, kind="geom"):
    rgba = np.asarray(getattr(m, f"{kind}_rgba"), np.float64)
    matid = np.asarray(getattr(m, f"{kind}_matid"), np.int64)
    out = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    take = default & (matid >= 0)
    if take.any():
        out[take] = np.asarray(m.mat_rgba, np.float64)[matid[take], 3]
    return out


def free_cam_pose(m, d, cam):
    """World pose of the FREE camera, measured off MuJoCo's own scene, not derived from az/el."""
    scn = mujoco.MjvScene(m, maxgeom=1000)
    opt = mujoco.MjvOption()
    pert = mujoco.MjvPerturb()
    mujoco.mjv_updateScene(m, d, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    e0, e1 = scn.camera[0], scn.camera[1]
    pos = 0.5 * (np.array(e0.pos, np.float64) + np.array(e1.pos, np.float64))
    fwd = np.array(e0.forward, np.float64)
    up = np.array(e0.up, np.float64)
    return pos, fwd / np.linalg.norm(fwd), up / np.linalg.norm(up), \
        dict(eye0=np.array(e0.pos).tolist(), eye1=np.array(e1.pos).tolist(),
             frustum=[float(e0.frustum_bottom), float(e0.frustum_top), float(e0.frustum_center),
                      float(e0.frustum_near), float(e0.frustum_far)],
             ipd=float(m.vis.global_.ipd), fovy_global=float(m.vis.global_.fovy),
             orthographic=int(m.vis.global_.orthographic))


def mat_from_fwd_up(fwd, up):
    z = -fwd
    x = np.cross(fwd, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)          # columns


def main():
    report = {}
    for name, env_id in SCENES.items():
        rep = {}
        env = gymnasium.make(env_id, render_mode="rgb_array", width=SIZE, height=SIZE)
        u = env.unwrapped
        shapes = []
        for s in range(3):
            env.reset(seed=s)
            shapes.append(census(u.model))
        rep["census"] = shapes[0]
        rep["G3_shape_constant"] = all(x == shapes[0] for x in shapes)
        m, d = u.model, u.data

        # ---- geoms/sites
        ga = eff_alpha(m, "geom")
        sa = eff_alpha(m, "site")
        rep["geoms"] = [dict(
            i=g, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
            type=int(m.geom_type[g]), group=int(m.geom_group[g]), matid=int(m.geom_matid[g]),
            rgba=np.asarray(m.geom_rgba[g]).round(4).tolist(), alpha=float(ga[g]),
            dataid=int(m.geom_dataid[g])) for g in range(m.ngeom)]
        rep["sites"] = [dict(
            i=s, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s),
            type=int(m.site_type[s]), group=int(m.site_group[s]), matid=int(m.site_matid[s]),
            size=np.asarray(m.site_size[s]).tolist(),
            rgba=np.asarray(m.site_rgba[s]).round(4).tolist(), alpha=float(sa[s]),
            bodyid=int(m.site_bodyid[s]), pos=np.asarray(m.site_pos[s]).round(5).tolist())
            for s in range(m.nsite)]
        rep["alpha_census"] = dict(
            geom_zero=[int(g) for g in np.flatnonzero(ga <= 0)],
            geom_partial=[[int(g), float(ga[g])] for g in np.flatnonzero((ga > 0) & (ga < 1))],
            site_zero=[int(s) for s in np.flatnonzero(sa <= 0)],
            site_partial=[[int(s), float(sa[s])] for s in np.flatnonzero((sa > 0) & (sa < 1))])
        rep["materials"] = [dict(
            i=k, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k),
            rgba=np.asarray(m.mat_rgba[k]).round(4).tolist(),
            texid=np.asarray(m.mat_texid[k]).tolist() if np.ndim(m.mat_texid[k]) else int(m.mat_texid[k]),
            specular=float(m.mat_specular[k]), shininess=float(m.mat_shininess[k]),
            reflectance=float(m.mat_reflectance[k]), emission=float(m.mat_emission[k]))
            for k in range(m.nmat)]
        rep["textures"] = [dict(i=t, type=int(m.tex_type[t]), h=int(m.tex_height[t]),
                                w=int(m.tex_width[t]),
                                name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t))
                           for t in range(m.ntex)]
        # textured primitives MJWarp cannot texgen (not PLANE=0 / MESH=7)
        gaps = []
        for g in range(m.ngeom):
            mid = int(m.geom_matid[g])
            if mid < 0 or int(m.geom_type[g]) in (0, 7):
                continue
            t = m.mat_texid[mid]
            t = int(np.max(t)) if np.ndim(t) else int(t)
            if t >= 0:
                gaps.append(int(g))
        rep["textured_primitive_geoms"] = gaps
        rep["lights"] = [dict(i=l, type=int(m.light_type[l]) if hasattr(m, "light_type") else None,
                              directional=int(m.light_directional[l]) if hasattr(m, "light_directional") else None,
                              castshadow=int(m.light_castshadow[l]),
                              pos=np.asarray(m.light_pos[l]).tolist(),
                              dir=np.asarray(m.light_dir[l]).tolist(),
                              ambient=np.asarray(m.light_ambient[l]).tolist(),
                              diffuse=np.asarray(m.light_diffuse[l]).tolist(),
                              specular=np.asarray(m.light_specular[l]).tolist())
                         for l in range(m.nlight)]
        hl = m.vis.headlight
        rep["headlight"] = dict(active=int(hl.active), ambient=np.asarray(hl.ambient).tolist(),
                                diffuse=np.asarray(hl.diffuse).tolist(),
                                specular=np.asarray(hl.specular).tolist())
        rep["cameras"] = [dict(i=c, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, c),
                               bodyid=int(m.cam_bodyid[c]), fovy=float(m.cam_fovy[c]),
                               mode=int(m.cam_mode[c])) for c in range(m.ncam)]
        rep["joints"] = [dict(i=j, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                              type=int(m.jnt_type[j]), qposadr=int(m.jnt_qposadr[j]))
                         for j in range(m.njnt)]
        rep["offsamples"] = int(m.vis.quality.offsamples)

        # ---- the camera actually used by the official renderer
        env.reset(seed=0)
        env.render()                                     # force viewer creation
        vcam = u.mujoco_renderer.viewer.cam
        rep["viewer_cam"] = dict(type=int(vcam.type), fixedcamid=int(vcam.fixedcamid),
                                 distance=float(vcam.distance), azimuth=float(vcam.azimuth),
                                 elevation=float(vcam.elevation),
                                 lookat=np.asarray(vcam.lookat).tolist())
        pos, fwd, up, extra = free_cam_pose(m, d, vcam)
        rep["free_cam"] = dict(pos=pos.tolist(), forward=fwd.tolist(), up=up.tolist(), **extra)

        # ---- reference frames on real reset states
        frames, states = [], []
        for s in range(8):
            env.reset(seed=100 + s)
            frames.append(np.ascontiguousarray(env.render()))
            states.append(dict(qpos=u.data.qpos.copy(), mocap_pos=u.data.mocap_pos.copy(),
                               mocap_quat=u.data.mocap_quat.copy(), goal=np.asarray(u.goal).copy()))
        frames = np.stack(frames)
        np.save(f"{OUT}/ref_{name}.npy", frames)
        np.savez(f"{OUT}/states_{name}.npz",
                 qpos=np.stack([s["qpos"] for s in states]),
                 mocap_pos=np.stack([s["mocap_pos"] for s in states]),
                 mocap_quat=np.stack([s["mocap_quat"] for s in states]),
                 goal=np.stack([s["goal"] for s in states]))
        rep["frame_stats"] = dict(distinct_colors=int(len(np.unique(frames[0].reshape(-1, 3), axis=0))),
                                  mean=float(frames.mean()))

        # ---- translucent mocap markers: how much of the picture are they?
        keep = m.geom_rgba.copy()
        mocap_geoms = [g for g in range(m.ngeom)
                       if int(m.body_mocapid[m.geom_bodyid[g]]) >= 0]
        rep["mocap_geoms"] = mocap_geoms
        per_geom = {}
        env.reset(seed=100)
        base = np.ascontiguousarray(env.render())
        for g in mocap_geoms + [None]:
            m.geom_rgba[:] = keep
            hide = mocap_geoms if g is None else [g]
            m.geom_rgba[hide, 3] = 0.0
            f = np.ascontiguousarray(env.render())
            dd = np.abs(base.astype(np.int16) - f.astype(np.int16)).max(-1)
            per_geom["all" if g is None else str(g)] = dict(
                touched_frac=float((dd > 0).mean()), mean_abs=float(dd.mean()),
                max_abs=int(dd.max()))
        m.geom_rgba[:] = keep
        rep["mocap_marker_impact"] = per_geom

        # ---- sites: how much of the picture are they?
        skeep = m.site_rgba.copy()
        site_imp = {}
        for s in range(m.nsite):
            m.site_rgba[:] = skeep
            m.site_rgba[s, 3] = 0.0
            f = np.ascontiguousarray(env.render())
            dd = np.abs(base.astype(np.int16) - f.astype(np.int16)).max(-1)
            site_imp[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, s) or str(s)] = dict(
                touched_frac=float((dd > 0).mean()), max_abs=int(dd.max()))
        m.site_rgba[:] = skeep
        rep["site_impact"] = site_imp

        report[name] = rep
        print(name, "census", rep["census"], "G3", rep["G3_shape_constant"], flush=True)
        print("  markers:", per_geom["all"], flush=True)
        print("  sites:", site_imp, flush=True)
        print("  textured primitives:", gaps, "alpha:", rep["alpha_census"], flush=True)

    with open(f"{OUT}/probe1.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("WROTE", f"{OUT}/probe1.json")


if __name__ == "__main__":
    main()
