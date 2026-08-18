"""Facts a visual-tier locomotion env needs before it can be written, one env per process.

Answers, for ONE gymnasium locomotion domain:

  1. is gymnasium's own ``render()`` the same picture as ``mujoco.Renderer`` at the ``track`` camera?
     (``docs/research/mujoco_locomotion.md`` asserts it; the visual tier's reference is gymnasium's,
     so it has to hold at the DEFAULT ``offsamples=4`` rather than at the probes' 0.)
  2. which camera do we drive nvdiffrast with -- ``d.cam_xpos``/``cam_xmat`` (cheap, per frame) or
     ``mjv_updateScene``'s mono GL camera (what ``mjr_render`` projects with)? Both are built and the
     gap is printed rather than assumed.
  3. is ``mj_kinematics + mj_comPos + mj_camlight`` enough, i.e. can the frame loop skip the dynamics?
     ``mj_camlight`` needs ``subtree_com`` for a ``trackcom`` camera and ``mj_kinematics`` does not
     write it -- the trap ``mujoco_locomotion.md`` Section 6.2 records for MJWarp.
  4. the model's own texture/material facts, including ``mat_texuniform`` on ``MatPlane``, because the
     texgen law's scale term depends on it and ``mujoco_warp``'s implementation omits it.
  5. what nvdiffrast scores against THE OFFICIAL FRAME (not against a de-multisampled stand-in), at
     supersample 1 and 2: mean abs, within 8, SSIM, exact, interior exact.

Run one env per process: two MuJoCo envs alive together can return byte-identical frames from cameras
that provably differ.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdloco"


def emit(tag, obj):
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--states", type=int, default=16)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import gymnasium
    import mujoco
    import torch

    from stable_envs.render import nvdiff as ND
    from stable_envs.testing import conformance as C

    S = int(args.size)
    env = gymnasium.make(args.env, render_mode="rgb_array", width=S, height=S)
    u = env.unwrapped
    m, d = u.model, u.data
    u.model.vis.global_.offwidth = max(int(m.vis.global_.offwidth), S)
    u.model.vis.global_.offheight = max(int(m.vis.global_.offheight), S)
    env.reset(seed=0)
    u.render()

    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    emit("model", {
        "env": args.env, "nq": int(m.nq), "ngeom": int(m.ngeom), "nmesh": int(m.nmesh),
        "ntex": int(m.ntex), "nlight": int(m.nlight), "track_cam": int(cam_id),
        "cam_mode": int(m.cam_mode[cam_id]), "offsamples": int(m.vis.quality.offsamples),
        "geom_types": [int(t) for t in m.geom_type],
        "floor_size": np.asarray(m.geom_size[floor]).tolist(),
        "extent": float(m.stat.extent),
        "lights": [{"directional": int(m.light_directional[i]) if hasattr(m, "light_directional")
                    else int(getattr(m, "light_type", [0])[i]),
                    "castshadow": int(m.light_castshadow[i]),
                    "pos": np.asarray(m.light_pos[i]).tolist(),
                    "dir": np.asarray(m.light_dir[i]).tolist(),
                    "diffuse": np.asarray(m.light_diffuse[i]).tolist(),
                    "specular": np.asarray(m.light_specular[i]).tolist(),
                    "cutoff": float(m.light_cutoff[i]), "exponent": float(m.light_exponent[i])}
                   for i in range(int(m.nlight))],
        "headlight_active": int(m.vis.headlight.active),
        "shadowsize": int(m.vis.quality.shadowsize), "shadowclip": float(m.vis.map.shadowclip),
        "shadowscale": float(m.vis.map.shadowscale),
        "materials": [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k),
                       "texuniform": int(m.mat_texuniform[k]),
                       "texrepeat": np.asarray(m.mat_texrepeat[k]).tolist(),
                       "reflectance": float(m.mat_reflectance[k]),
                       "specular": float(m.mat_specular[k]),
                       "shininess": float(m.mat_shininess[k])} for k in range(int(m.nmat))],
        "textures": [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t),
                      "type": int(m.tex_type[t]), "w": int(m.tex_width[t]),
                      "h": int(m.tex_height[t])} for t in range(int(m.ntex))]})

    # ---- states from the env's own reset + random-policy rollout
    rng = np.random.default_rng(0)
    lo, hi = env.action_space.low, env.action_space.high
    states = []
    for k in range(int(args.states)):
        env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        for _ in range(int(rng.integers(0, 61))):
            _, _, term, trunc, _ = env.step(rng.uniform(lo, hi).astype(np.float32))
            if term or trunc:
                env.reset(seed=int(rng.integers(0, 2 ** 31 - 1)))
        states.append(np.asarray(d.qpos, np.float64).copy())
    states = np.stack(states)

    nv = int(m.nv)

    def official(q):
        u.set_state(q, np.zeros(nv))
        u.mujoco_renderer.viewer.make_context_current()
        return u.render().copy()

    off = np.stack([official(q) for q in states])

    # ---- 1. gymnasium's frame vs a bare mujoco.Renderer at the same camera
    ren = mujoco.Renderer(m, height=S, width=S)
    bare = []
    for q in states:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=int(cam_id))
        bare.append(ren.render().copy())
    bare = np.stack(bare)
    emit("gym_vs_renderer", {"max_abs": int(np.abs(off.astype(np.int16) - bare.astype(np.int16)).max()),
                             "mean_abs": float(np.abs(off.astype(np.int16)
                                                      - bare.astype(np.int16)).mean())})

    # The SHADOW BUDGET: the same official pipeline with mjRND_SHADOW off. Scoring our (shadowless)
    # render against both references brackets exactly what a shadow pass could buy, before writing one.
    noshadow = []
    for q in states:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=int(cam_id))
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        noshadow.append(ren.render().copy())
    noshadow = np.stack(noshadow)
    dd = np.abs(off.astype(np.int16) - noshadow.astype(np.int16)).max(-1)
    emit("shadow_pixels", {"frac_changed": float((dd > 0).mean()), "mean_abs": float(dd.mean()),
                           "max_abs": int(dd.max()),
                           "mean_where_changed": float(dd[dd > 0].mean()) if (dd > 0).any() else 0.0})

    # ---- 2/3. cameras and the cheap kinematics path
    scn = mujoco.MjvScene(m, maxgeom=max(2 * int(m.ngeom), 1000))
    opt = mujoco.MjvOption()
    mjc = mujoco.MjvCamera()
    mjc.type = mujoco.mjtCamera.mjCAMERA_FIXED
    mjc.fixedcamid = int(cam_id)
    gpos, gmat, gfovy, gnear = [], [], [], []
    dpos, dmat = [], []
    kpos, kmat = [], []
    gx_f, gm_f, gx_k, gm_k = [], [], [], []
    for q in states:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        mujoco.mjv_updateScene(m, d, opt, None, mjc, mujoco.mjtCatBit.mjCAT_ALL, scn)
        c = ND.Camera.from_glcam(m, scn)
        gpos.append(c.pos)
        gmat.append(c.mat)
        gfovy.append(c.fovy)
        gnear.append(c.znear)
        dpos.append(np.array(d.cam_xpos[cam_id], np.float64, copy=True))
        dmat.append(np.array(d.cam_xmat[cam_id], np.float64, copy=True).reshape(3, 3))
        gx_f.append(np.array(d.geom_xpos, np.float64, copy=True))
        gm_f.append(np.array(d.geom_xmat, np.float64, copy=True))
        # the cheap path
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_camlight(m, d)
        kpos.append(np.array(d.cam_xpos[cam_id], np.float64, copy=True))
        kmat.append(np.array(d.cam_xmat[cam_id], np.float64, copy=True).reshape(3, 3))
        gx_k.append(np.array(d.geom_xpos, np.float64, copy=True))
        gm_k.append(np.array(d.geom_xmat, np.float64, copy=True))
    emit("camera", {
        "glcam_vs_camxpos_pos": float(np.abs(np.stack(gpos) - np.stack(dpos)).max()),
        "glcam_vs_camxpos_mat": float(np.abs(np.stack(gmat) - np.stack(dmat)).max()),
        "glcam_fovy": float(np.mean(gfovy)), "model_fovy": float(m.cam_fovy[cam_id]),
        "glcam_fovy_spread": float(np.ptp(gfovy)),
        "glcam_near": float(np.mean(gnear)),
        "model_near": float(m.vis.map.znear) * float(m.stat.extent),
        "kin_vs_forward_cam_pos": float(np.abs(np.stack(kpos) - np.stack(dpos)).max()),
        "kin_vs_forward_cam_mat": float(np.abs(np.stack(kmat) - np.stack(dmat)).max()),
        "kin_vs_forward_geom_pos": float(np.abs(np.stack(gx_k) - np.stack(gx_f)).max()),
        "kin_vs_forward_geom_mat": float(np.abs(np.stack(gm_k) - np.stack(gm_f)).max())})

    # ---- 5. nvdiffrast against the OFFICIAL frame
    glcam = ND.Camera(np.stack(gpos), np.stack(gmat), float(np.mean(gfovy)), float(np.mean(gnear)),
                      float(m.vis.map.zfar) * float(m.stat.extent))
    dcam = ND.Camera(np.stack(dpos), np.stack(dmat), float(m.cam_fovy[cam_id]),
                     float(m.vis.map.znear) * float(m.stat.extent),
                     float(m.vis.map.zfar) * float(m.stat.extent))
    xpos_all = np.stack(gx_f)
    xmat_all = np.stack(gm_f).reshape(len(states), -1, 3, 3)

    res = {}
    for ss in (1, 2):
        br = ND.BatchRenderer(m, res=S, device="cuda", supersample=ss)
        br.resolve_from_mujoco(m, d, camera=mjc)
        if ss == 1:
            emit("scene", {"geoms_drawn": br.scene.G, "verts": br.scene.V,
                           "tris": int(br.scene.face.shape[0]),
                           "textures": {int(t): [list(v["shape"]), str(v["resampled"]), v["mipmap_ok"],
                                                 bool(v["cube"])]
                                        for t, v in br.scene.textures.items()},
                           "skybox": None if br.skybox_tex is None else
                           [list(br.skybox_tex["shape"]), str(br.skybox_tex["resampled"]),
                            bool(br.skybox_tex["cube"])]})
        xp = torch.as_tensor(xpos_all[:, br.scene.geoms], dtype=torch.float32, device="cuda")
        xm = torch.as_tensor(xmat_all[:, br.scene.geoms], dtype=torch.float32, device="cuda")
        for cname, cam in (("glcam", glcam), ("camxpos", dcam)):
            t0 = time.perf_counter()
            ours = br.render(xp, xm, cam).cpu().numpy()
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            for ref_name, ref in (("official", off), ("noshadow", noshadow)):
                r = C.pixel_agreement(ours, ref)
                r["ssim"] = C.ssim(ours, ref)
                r["fps_batch"] = len(states) / dt
                res[f"ss{ss}.{cname}.{ref_name}"] = r
                emit(f"score.ss{ss}.{cname}.{ref_name}", r)
        if ss == 1:
            np.save(f"{OUT}/{args.env}_ours.npy", ours[:3])
            np.save(f"{OUT}/{args.env}_off.npy", off[:3])
        del br
        torch.cuda.empty_cache()

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/facts_{args.env}.json", "w") as f:
        json.dump({"env": args.env, "size": S, "scores": res}, f, indent=1, default=float)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
