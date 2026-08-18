"""The deciding measurement: does a prefiltered sampler close Gymnasium's minified checker floor?

``docs/research/mujoco_locomotion.md`` Section 6 rules the MJWarp backend out on one number. The far
tenth of the floor has a reference luminance spread of 1.5-3.4 about a single grey and MJWarp's has
92, because ``mujoco_warp`` builds every texture at one mip level and samples with no level of
detail; **0.0017-0.0069 % of those pixels are bit-exact**, and that floor carries 99.45-100.00 % of
the interior error on all six domains.

So the far band is scored on its own here, not folded into a whole-frame average -- a whole-frame
number would let a 55 %-of-frame skybox hide the one surface the question is about. Reported per
domain: far-band exactness and luminance spread on both sides (the statistic that separates a
prefiltered sampler from a point-sampled one), then the near band, the whole floor, and the frame.

The reference is ``mujoco.Renderer`` at the model's own ``track`` camera, which
``mujoco_locomotion.md`` measures bit-identical to gymnasium's own frame. One env per process.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save, score  # noqa: E402

LUM = np.array([0.299, 0.587, 0.114])


def states_for(env, n, seed=0):
    """The env's own reset distribution plus a short random-policy rollout, as Section 6 uses."""
    rng = np.random.default_rng(seed)
    out = []
    env.reset(seed=seed)
    u = env.unwrapped
    for k in range(n):
        env.reset(seed=seed + k)
        for _ in range(int(rng.integers(0, 60))):
            env.step(env.action_space.sample())
        out.append(np.asarray(u.data.qpos, np.float64).copy())
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HalfCheetah-v5")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--states", type=int, default=8)
    ap.add_argument("--offsamples", type=int, default=0)
    ap.add_argument("--refl0", action="store_true", help="zero mat_reflectance on BOTH sides")
    ap.add_argument("--shadow0", action="store_true", help="mjRND_SHADOW off on the reference")
    ap.add_argument("--bodytex0", action="store_true", help="drop the body material's cube texture")
    ap.add_argument("--floortex0", action="store_true",
                    help="drop MatPlane's texture on BOTH sides: the symmetric shading control")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import gymnasium
    import mujoco
    import torch

    from stable_envs.render import nvdiff as ND

    env = gymnasium.make(args.env, render_mode="rgb_array", width=args.size, height=args.size)
    u = env.unwrapped
    m, d = u.model, u.data
    m.vis.quality.offsamples = int(args.offsamples)
    if args.refl0:
        m.mat_reflectance[:] = 0.0
    if args.floortex0:
        for k in range(m.nmat):
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or "") == "MatPlane":
                if np.ndim(m.mat_texid[k]):
                    m.mat_texid[k, :] = -1
                else:
                    m.mat_texid[k] = -1
    if args.bodytex0:
        for k in range(m.nmat):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or ""
            if nm != "MatPlane":
                if np.ndim(m.mat_texid[k]):
                    m.mat_texid[k, :] = -1
                else:
                    m.mat_texid[k] = -1
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    if cam_id < 0:
        cam_id = 0
    states = states_for(env, args.states)

    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    emit("model", {"env": args.env, "ngeom": int(m.ngeom), "ntex": int(m.ntex),
                   "cam": int(cam_id), "cam_mode": int(m.cam_mode[cam_id]),
                   "floor_geom": int(floor),
                   "floor_size": np.asarray(m.geom_size[floor]).tolist() if floor >= 0 else None,
                   "textures": [{"id": t, "type": int(m.tex_type[t]), "w": int(m.tex_width[t]),
                                 "h": int(m.tex_height[t]),
                                 "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, t)}
                                for t in range(m.ntex)],
                   "materials": [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k),
                                  "texrepeat": np.asarray(m.mat_texrepeat[k]).tolist(),
                                  "texuniform": int(m.mat_texuniform[k]),
                                  "specular": float(m.mat_specular[k]),
                                  "shininess": float(m.mat_shininess[k]),
                                  "reflectance": float(m.mat_reflectance[k])}
                                 for k in range(m.nmat)]})

    ren = mujoco.Renderer(m, height=args.size, width=args.size)
    ref, seg, dep = [], [], []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=int(cam_id))
        if args.shadow0:
            ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        if args.refl0:
            ren.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        ref.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=int(cam_id))
        seg.append(ren.render()[..., 0].copy())
        ren.disable_segmentation_rendering()
        ren.enable_depth_rendering()
        ren.update_scene(d, camera=int(cam_id))
        dep.append(ren.render().copy())
        ren.disable_depth_rendering()
    ref, seg, dep = np.stack(ref), np.stack(seg), np.stack(dep)

    # Two ways to name the same camera, and they are not the same camera. `d.cam_xpos`/`cam_xmat`
    # is the pose MuJoCo computed; `mjv_updateScene`'s two mjvGLCamera eyes are what mjr_render
    # actually projects with. Build both, print the gap, and render with the second.
    scn = mujoco.MjvScene(m, maxgeom=max(2 * m.ngeom, 1000))
    opt = mujoco.MjvOption()
    mjc = mujoco.MjvCamera()
    mjc.type = mujoco.mjtCamera.mjCAMERA_FIXED
    mjc.fixedcamid = int(cam_id)
    gpos, gmat, gfovy, gnear = [], [], [], []
    dpos, dmat = [], []
    for q in states:
        d.qpos[:] = q
        mujoco.mj_forward(m, d)
        mujoco.mjv_updateScene(m, d, opt, None, mjc, mujoco.mjtCatBit.mjCAT_ALL, scn)
        c = ND.Camera.from_glcam(m, scn)
        gpos.append(c.pos)
        gmat.append(c.mat)
        gfovy.append(c.fovy)
        gnear.append(c.znear)
        dpos.append(np.asarray(d.cam_xpos[cam_id], np.float64).copy())
        dmat.append(np.asarray(d.cam_xmat[cam_id], np.float64).reshape(3, 3).copy())
    glcam = ND.Camera(np.stack(gpos), np.stack(gmat), float(np.mean(gfovy)),
                      float(np.mean(gnear)), float(m.vis.map.zfar) * float(m.stat.extent))
    emit("camera", {"cam_fovy_model": float(m.cam_fovy[cam_id]),
                    "glcam_fovy": float(np.mean(gfovy)),
                    "glcam_near": float(np.mean(gnear)),
                    "model_near": float(m.vis.map.znear) * float(m.stat.extent),
                    "pos_max_delta": float(np.abs(np.stack(gpos) - np.stack(dpos)).max()),
                    "mat_max_delta": float(np.abs(np.stack(gmat) - np.stack(dmat)).max()),
                    "cam_resolution": np.asarray(m.cam_resolution[cam_id]).tolist()
                    if hasattr(m, "cam_resolution") else None,
                    "cam_sensorsize": np.asarray(m.cam_sensorsize[cam_id]).tolist()
                    if hasattr(m, "cam_sensorsize") else None,
                    "cam_intrinsic": np.asarray(m.cam_intrinsic[cam_id]).tolist()
                    if hasattr(m, "cam_intrinsic") else None})

    fl = seg == floor
    far = np.zeros_like(fl)
    near = np.zeros_like(fl)
    for k in range(len(fl)):
        dk = dep[k][fl[k]]
        if dk.size:
            hi = np.quantile(dk, 0.9)
            lo = np.quantile(dk, 0.1)
            far[k] = fl[k] & (dep[k] >= hi)
            near[k] = fl[k] & (dep[k] <= lo)
    regions = {"frame": np.ones_like(fl), "floor": fl, "far": far, "near": near,
               "bodies": (seg >= 0) & ~fl, "sky": seg < 0}

    qt = torch.as_tensor(states, dtype=torch.float32)
    out = {}
    for name, kw in [("mip", dict(mip=True, tex=True, pol="pot", aniso=1)),
                     ("nomip", dict(mip=False, tex=True, pol="pot", aniso=1))]:
        br = ND.BatchRenderer(m, res=args.size, device="cuda", tex_policy=kw["pol"],
                              aniso=kw.get("aniso", 1))
        br.resolve_from_mujoco(m, d, camera=None)
        br.use_tex, br.mipmap = kw["tex"], kw["mip"]
        if name == "mip":
            emit("scene", {"geoms_drawn": br.scene.G, "verts": br.scene.V,
                           "tris": int(br.scene.face.shape[0]),
                           "textures": {int(t): [v["shape"], str(v["resampled"]), v["mipmap_ok"]]
                                        for t, v in br.scene.textures.items()},
                           "skybox": None if br.skybox_tex is None
                           else [br.skybox_tex["shape"], str(br.skybox_tex["resampled"])]})
        ours = br.render_qpos(qt, cam=glcam).cpu().numpy()
        row = {}
        for rname, rmask in regions.items():
            s = score(ours, ref, rmask)
            if s.get("n", 0):
                a = ours[rmask] @ LUM
                b = ref[rmask] @ LUM
                s["lum_std_ours"] = float(a.std())
                s["lum_std_ref"] = float(b.std())
            row[rname] = s
        out[name] = row
        emit(f"score.{name}", {k: {kk: v.get(kk) for kk in
                                   ("n", "mean_abs", "exact_frac", "within8_frac", "interior_exact",
                                    "lum_std_ours", "lum_std_ref")} for k, v in row.items()})
        if name == "mip":
            np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
                    f"loco_{args.env}{args.tag}_ours.npy", ours[:2])
            np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
                    f"loco_{args.env}{args.tag}_ref.npy", ref[:2])
        del br
    save(f"loco_{args.env}{args.tag}.json", {"args": vars(args), "scores": out})


if __name__ == "__main__":
    main()
