"""Facts the MJWarp re-measurement rests on, each measured rather than read off the XML.

Answers, per domain:
  1. Which camera does gymnasium actually render through, and what is its mode?
  2. Does ``mujoco.Renderer`` at that camera reproduce gymnasium's own frame BIT-EXACTLY? If yes the
     whole ablation ladder can be driven through ``mujoco.Renderer``, which is controllable; if no,
     every ablation number would be measured against a different reference than the headline.
  3. Does MJWarp's trackcom camera land where MuJoCo's does — WITH ``com_pos`` and WITHOUT it?
  4. Does the reflection pass reach the frame at all? (RoboCasa declared reflectance 0.5-1.0 on six
     materials and MuJoCo drew it on 0.0000 of pixels. Declared is not drawn.)
  5. Model census: textures, materials, reflectance, alpha, geom types, lights, offsamples.

One env per process is NOT required here because nothing compares two envs; the reference identity
check in (2) is re-run in a clean subprocess by ``--one`` anyway.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/mujoco_locomotion")

from mjw_common import (ENVS, VISIBLE_GROUPS, agreement, gym_shot, jsave, make_env, mj_render,
                        mj_segment, sample_qpos, track_cam)

SIZE = 240
NSTATE = 8


def census(m):
    import mujoco
    from stable_envs.render import mjwarp as MW
    ga = MW.effective_alpha(m, "geom")
    sa = MW.effective_alpha(m, "site")
    mats = []
    for i in range(int(m.nmat)):
        tid = np.asarray(m.mat_texid[i]).reshape(-1)
        mats.append(dict(
            name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}",
            reflectance=round(float(m.mat_reflectance[i]), 4),
            specular=round(float(m.mat_specular[i]), 4),
            shininess=round(float(m.mat_shininess[i]), 4),
            emission=round(float(m.mat_emission[i]), 4),
            texrepeat=np.asarray(m.mat_texrepeat[i]).tolist(),
            texuniform=int(m.mat_texuniform[i]),
            texid=[int(t) for t in tid if int(t) >= 0]))
    texs = [dict(id=i, type=int(m.tex_type[i]), w=int(m.tex_width[i]), h=int(m.tex_height[i]),
                 nchannel=int(m.tex_nchannel[i]),
                 name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, i) or f"tex{i}")
            for i in range(int(m.ntex))]
    return dict(
        nq=int(m.nq), ngeom=int(m.ngeom), nsite=int(m.nsite), nmesh=int(m.nmesh),
        ntex=int(m.ntex), nmat=int(m.nmat), nlight=int(m.nlight), ncam=int(m.ncam),
        offsamples=int(m.vis.quality.offsamples),
        headlight_active=int(m.vis.headlight.active),
        geom_types=sorted({int(t) for t in m.geom_type}),
        geom_groups=sorted({int(g) for g in m.geom_group}),
        zero_alpha_geoms=int(((ga <= 0.0) & np.isin(m.geom_group, VISIBLE_GROUPS)).sum()),
        zero_alpha_sites=int(((sa <= 0.0) & np.isin(m.site_group, VISIBLE_GROUPS)).sum()),
        translucent_geoms=int((((ga > 0.0) & (ga < 1.0))
                               & np.isin(m.geom_group, VISIBLE_GROUPS)).sum()),
        visible_sites=int(np.isin(m.site_group, VISIBLE_GROUPS).sum()),
        lights=[dict(directional=int(m.light_directional[i]) if hasattr(m, "light_directional")
                     else int(m.light_type[i]),
                     castshadow=int(m.light_castshadow[i]),
                     mode=int(m.light_mode[i])) for i in range(int(m.nlight))],
        materials=mats, textures=texs,
        texture_gaps=[list(x) for x in __import__(
            "stable_envs.render.mjwarp", fromlist=["x"]).texture_gaps(m, VISIBLE_GROUPS)],
    )


def run(env_id):
    import mujoco

    import mujoco_warp as mjw
    from mjw_common import MjwRenderer

    out = {"env": env_id}
    env, u = make_env(env_id, SIZE)
    m, d = u.model, u.data
    cam = track_cam(m)
    out["track_cam_id"] = int(cam)
    out["renderer_camera_id"] = int(u.mujoco_renderer.camera_id)
    out["cam_mode"] = int(m.cam_mode[cam]) if cam >= 0 else None
    out["cam_mode_name"] = (str(mujoco.mjtCamLight(int(m.cam_mode[cam]))) if cam >= 0 else None)
    out["cam_bodyid"] = int(m.cam_bodyid[cam]) if cam >= 0 else None
    out["census"] = census(m)

    st = sample_qpos(env_id, NSTATE, seed=0)

    # --- (2) is mujoco.Renderer at the track camera the same reference gymnasium produces? -----
    ref_gym = []
    nv = int(m.nv)
    for q in st:
        u.set_state(q, np.zeros(nv))
        ref_gym.append(gym_shot(u))
    ref_gym = np.stack(ref_gym)
    ref_mjr = mj_render(m, d, cam, SIZE, st)
    out["mjRenderer_vs_gymnasium"] = agreement(ref_mjr, ref_gym)

    # --- (4) does the reflection pass reach the frame? ----------------------------------------
    refl = np.array(m.mat_reflectance, copy=True)
    declared = [i for i in range(int(m.nmat)) if float(refl[i]) > 0]
    out["materials_declaring_reflectance"] = [
        dict(id=i, name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i),
             reflectance=round(float(refl[i]), 4)) for i in declared]
    if declared:
        m.mat_reflectance[:] = 0.0
        ref_norefl = mj_render(m, d, cam, SIZE, st, tag="norefl")
        m.mat_reflectance[:] = refl
        dd = np.abs(ref_mjr.astype(np.int16) - ref_norefl.astype(np.int16)).max(-1)
        im = __import__("mjw_common", fromlist=["x"]).interior_mask(ref_mjr)
        out["reflection_reaches_frame"] = dict(
            pixels_changed_frac=round(float((dd > 0).mean()), 6),
            max_abs=int(dd.max()), mean_abs=round(float(dd.mean()), 4),
            interior_pixels_changed_frac=round(float((dd[:, :-1, :-1][im] > 0).mean()), 6),
            interior_mean_abs=round(float(dd[:, :-1, :-1][im].mean()), 4))
    else:
        out["reflection_reaches_frame"] = None

    # --- (3) MJWarp's trackcom camera, with and without com_pos --------------------------------
    mjr_cam = []
    for q in st:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        mjr_cam.append(np.array(d.cam_xpos[cam], np.float64))
    mjr_cam = np.stack(mjr_cam)

    W = MjwRenderer(m, cam, SIZE, NSTATE)
    out["mjwarp_prepared"] = dict(hidden_geoms=len(W.prepared.hidden_geoms),
                                  hidden_sites=len(W.prepared.hidden_sites),
                                  texture_gaps=[list(x) for x in W.texture_gaps])
    for tag, com in (("with_com_pos", True), ("without_com_pos", False)):
        W.render(st, com=com)
        got = W.cam_xpos()[:NSTATE, cam]
        out[f"cam_xpos_{tag}"] = dict(
            max_err=round(float(np.abs(got - mjr_cam).max()), 8),
            spread_over_states=round(float(np.abs(got - got[0]).max()), 8))
    out["cam_xpos_reference_spread"] = round(float(np.abs(mjr_cam - mjr_cam[0]).max()), 6)
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or ENVS
    res = []
    for e in ids:
        try:
            res.append(run(e))
            print(e, "ok", flush=True)
        except Exception as ex:                                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            res.append({"env": e, "error": f"{type(ex).__name__}: {ex}"})
    jsave("/mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/probe1_facts.json", res)
    for r in res:
        print("\n===", r["env"])
        for k in ("track_cam_id", "renderer_camera_id", "cam_mode_name",
                  "mjRenderer_vs_gymnasium", "materials_declaring_reflectance",
                  "reflection_reaches_frame", "cam_xpos_with_com_pos",
                  "cam_xpos_without_com_pos", "cam_xpos_reference_spread", "mjwarp_prepared",
                  "error"):
            if k in r:
                print(" ", k, "=", r[k])
