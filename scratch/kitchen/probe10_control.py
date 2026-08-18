"""Probe 10: is the 0.158 interior figure the KITCHEN, or my harness?

A negative result has to survive the same scrutiny as a positive one, so two controls:

  A. the same kitchen comparison with shadows off on BOTH sides. If interior exactness jumps, the
     residual is specifically the shadow MODEL -- mujoco_warp casts hard any-hit shadow rays where
     MuJoCo filters a 4096 shadow map -- and not the harness.
  B. the same harness pointed at Fetch's reach.xml, where the shipped env measures interior 0.9988.
     If the harness cannot reproduce a known-good number, nothing it says about the kitchen counts.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_envs.render import mjwarp as MW
from stable_envs.testing.conformance import pixel_agreement

RES = 480
DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION = 0.5, 0.5, 0.0


def effective_rgba(m, g):
    rgba = np.asarray(m.geom_rgba[g], np.float64)
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[mid], np.float64)
    return rgba


def surgical(xml, lookat, distance, azimuth, elevation, keep_tex):
    from probe1_model import free_cam_pose
    base = mujoco.MjModel.from_xml_path(xml)
    d0 = mujoco.MjData(base)
    mujoco.mj_forward(base, d0)
    pos, quat = free_cam_pose(base, d0, lookat, distance, azimuth, elevation)
    spec = mujoco.MjSpec.from_file(xml)
    for i in range(base.ngeom + 1):
        spec.add_material(name=f"se:mat{i}")
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = float(base.vis.global_.fovy)
    m = spec.compile()
    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")
    trans = []
    for g in range(m.ngeom):
        k = mat0 + g
        rgba = effective_rgba(base, g)
        mid = int(base.geom_matid[g])
        if mid >= 0:
            sp, sh, em = (float(base.mat_specular[mid]), float(base.mat_shininess[mid]),
                          float(base.mat_emission[mid]))
            tid, trep, tuni = (np.array(base.mat_texid[mid], copy=True),
                               np.array(base.mat_texrepeat[mid], copy=True),
                               int(base.mat_texuniform[mid]))
            refl = float(base.mat_reflectance[mid])
        else:
            sp, sh, em, tid, trep, tuni, refl = (DEFAULT_SPECULAR, DEFAULT_SHININESS,
                                                 DEFAULT_EMISSION, None, None, 0, 0.0)
        if rgba[3] < 1.0 and int(m.geom_group[g]) in MW.DEFAULT_GROUPS:
            trans.append(g)
        m.mat_rgba[k], m.mat_specular[k], m.mat_shininess[k] = rgba, sp, sh
        m.mat_emission[k], m.mat_reflectance[k] = em, refl
        if keep_tex and tid is not None:
            m.mat_texid[k], m.mat_texrepeat[k], m.mat_texuniform[k] = tid, trep, tuni
        else:
            m.mat_texid[k] = -1
        m.geom_matid[g], m.geom_rgba[g] = k, rgba
    m.site_group[:] = MW.HIDDEN_GROUP
    if trans:
        m.geom_group[trans] = MW.HIDDEN_GROUP
    m.geom_margin[:] = 0.0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    return m, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")


def compare(m, camid, states, shadows_ours, shadows_ref, textures, nworld):
    import warp as wp
    import mujoco_warp as mjw
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, width=RES, height=RES)
    wm = mjw.put_model(m)
    wd = mjw.put_data(m, d, nworld=nworld)
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=nworld, cam_res=(RES, RES), render_rgb=True, render_seg=False,
            cam_active=[i == camid for i in range(m.ncam)],
            enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=textures,
            use_shadows=shadows_ours, render_skybox=True, enable_specular=True)

        def frame():
            mjw.kinematics(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc)
            mjw.render(wm, wd, rc)
        frame()
        wp.synchronize()
    bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(nworld, RES, RES, 4)
    qp = wp.to_torch(wd.qpos)
    n = len(states)
    qp[:n] = torch.as_tensor(states, dtype=qp.dtype, device=qp.device)
    torch.cuda.synchronize()
    frame()
    wp.synchronize()
    ours = bgra[:n, ..., [2, 1, 0]].cpu().numpy()
    ref = []
    for q in states:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camid)
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows_ref)
        ref.append(r.render().copy())
    return pixel_agreement(ours, np.stack(ref))


def main():
    which = sys.argv[1]
    if which == "kitchen":
        from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
        from gymnasium_robotics.envs.franka_kitchen import franka_env
        from probe5_contamination import pool
        xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                           "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
        m, camid = surgical(xml, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"],
                            keep_tex=False)
        st = pool(8)
        print("kitchen, textures off both sides, SHADOWS ON both:")
        print(f"  {compare(m, camid, st, True, True, False, 8)}")
    elif which == "kitchen_noshadow":
        from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
        from gymnasium_robotics.envs.franka_kitchen import franka_env
        from probe5_contamination import pool
        xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                           "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
        m, camid = surgical(xml, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"],
                            keep_tex=False)
        st = pool(8)
        print("kitchen, textures off both sides, SHADOWS OFF both:")
        print(f"  {compare(m, camid, st, False, False, False, 8)}")
    elif which == "fetch":
        import gymnasium_robotics as gr
        from gymnasium_robotics.envs.fetch.fetch_env import DEFAULT_CAMERA_CONFIG as CC
        xml = os.path.join(os.path.dirname(gr.__file__), "envs", "assets", "fetch", "reach.xml")
        m, camid = surgical(xml, np.asarray(CC["lookat"]), float(CC["distance"]),
                            float(CC["azimuth"]), float(CC["elevation"]), keep_tex=True)
        d = mujoco.MjData(m)
        rng = np.random.default_rng(0)
        st = np.stack([np.asarray(d.qpos, np.float64) + rng.uniform(-0.05, 0.05, m.nq)
                       for _ in range(8)])
        print("fetch reach.xml through the SAME harness (control):")
        print(f"  {compare(m, camid, st, True, True, True, 8)}")


if __name__ == "__main__":
    main()
