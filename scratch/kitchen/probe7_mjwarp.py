"""Probe 7: the only design left. Render the WHOLE kitchen with mujoco_warp -- shadows included -- and
measure it against MuJoCo.

probe6 killed the plate: mover shadows land on static geometry over 85.6% of the frame, and a baked
plate scores 0.834 of non-mover INTERIOR pixels with shadows on against 1.000000 with them off. So the
plate can only carry this scene if the shadows go, and they are the scene. The remaining design is the
one fetch.py did not need: no plate at all, mujoco_warp drawing everything and ray-casting its own
shadows.

The model surgery is fetch.py's, with one deliberate difference: the cloned per-geom material KEEPS its
texture id. fetch.py sets ``mat_texid = -1`` because a Fetch scene has no textured surfaces worth
having; here the marble floor, the marble counters and the wood cabinets are most of the picture.
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import time

import numpy as np
import mujoco
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stable_envs.render import mjwarp as MW
from stable_envs.testing.conformance import pixel_agreement
from probe1_model import free_cam_pose
from probe5_contamination import pool

RES = 480
DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION = 0.5, 0.5, 0.0


def effective_rgba(m, g):
    rgba = np.asarray(m.geom_rgba[g], np.float64)
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[mid], np.float64)
    return rgba


def build_model(keep_tex=True):
    from gymnasium_robotics.envs.franka_kitchen.franka_env import DEFAULT_CAMERA_CONFIG as CC
    from gymnasium_robotics.envs.franka_kitchen import franka_env
    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    base = mujoco.MjModel.from_xml_path(xml)
    d0 = mujoco.MjData(base)
    mujoco.mj_forward(base, d0)
    pos, quat = free_cam_pose(base, d0, CC["lookat"], CC["distance"], CC["azimuth"], CC["elevation"])

    spec = mujoco.MjSpec.from_file(xml)
    for i in range(base.ngeom + 1):
        spec.add_material(name=f"se:mat{i}")
    cam = spec.worldbody.add_camera()
    cam.name, cam.pos, cam.quat = "se:cam", pos, quat
    cam.fovy = float(base.vis.global_.fovy)
    m = spec.compile()

    gname = lambda mm, i: mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
    assert m.ngeom == base.ngeom
    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")

    translucent = []
    for g in range(m.ngeom):
        k = mat0 + g
        rgba = effective_rgba(base, g)
        mid = int(base.geom_matid[g])
        if mid >= 0:
            spec_, shin, emis = (float(base.mat_specular[mid]), float(base.mat_shininess[mid]),
                                 float(base.mat_emission[mid]))
            texid = np.array(base.mat_texid[mid], copy=True)
            texrep = np.array(base.mat_texrepeat[mid], copy=True)
            texuni = int(base.mat_texuniform[mid])
            refl = float(base.mat_reflectance[mid])
        else:
            spec_, shin, emis = DEFAULT_SPECULAR, DEFAULT_SHININESS, DEFAULT_EMISSION
            texid, texrep, texuni, refl = None, None, 0, 0.0
        if rgba[3] < 1.0:
            translucent.append(g)
        m.mat_rgba[k] = rgba
        m.mat_specular[k] = spec_
        m.mat_shininess[k] = shin
        m.mat_emission[k] = emis
        m.mat_reflectance[k] = refl
        if keep_tex and texid is not None:
            m.mat_texid[k] = texid
            m.mat_texrepeat[k] = texrep
            m.mat_texuniform[k] = texuni
        else:
            m.mat_texid[k] = -1
        m.geom_matid[g] = k
        m.geom_rgba[g] = rgba

    m.site_group[:] = MW.HIDDEN_GROUP
    m.geom_group[translucent] = MW.HIDDEN_GROUP
    m.geom_margin[:] = 0.0
    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 4 * RES)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 4 * RES)
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "se:cam")
    return m, cam_id, translucent


def main():
    import warp as wp
    import mujoco_warp as mjw

    m, camid, translucent = build_model(keep_tex=True)
    print(f"translucent hidden: {translucent}")
    print(f"texture gaps after surgery: {len(MW.texture_gaps(m))}")

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, width=RES, height=RES)

    NW = 16
    wm = mjw.put_model(m)
    wd = mjw.put_data(m, d, nworld=NW)
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=NW, cam_res=(RES, RES), render_rgb=True, render_seg=False,
            cam_active=[i == camid for i in range(m.ncam)],
            enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=True, use_shadows=True,
            render_skybox=True, enable_specular=True)

        def frame():
            mjw.kinematics(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc)
            mjw.render(wm, wd, rc)

        frame()
        wp.synchronize()
    bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(NW, RES, RES, 4)

    states = pool(NW)

    # ours
    qp = wp.to_torch(wd.qpos)
    qp[:] = torch.as_tensor(states, dtype=qp.dtype, device=qp.device)
    torch.cuda.synchronize()
    frame()
    wp.synchronize()
    ours = bgra[..., [2, 1, 0]].cpu().numpy()

    # reference
    ref = []
    for q in states:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camid)
        ref.append(r.render().copy())
    ref = np.stack(ref)

    print("\n== full-scene mujoco_warp (shadows ON) vs MuJoCo, same surgical model ==")
    print(f"  {pixel_agreement(ours, ref)}")

    # what does shadows-off cost on the mjwarp side (i.e. is the shadow term the residual)?
    with MW.gl_rounding():
        rc2 = MW.create_render_context(
            m, nworld=NW, cam_res=(RES, RES), render_rgb=True, render_seg=False,
            cam_active=[i == camid for i in range(m.ncam)],
            enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=True, use_shadows=False,
            render_skybox=True, enable_specular=True)

        def frame2():
            mjw.kinematics(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc2)
            mjw.render(wm, wd, rc2)
        frame2()
        wp.synchronize()
    b2 = wp.to_torch(rc2.rgb_data).view(torch.uint8).reshape(NW, RES, RES, 4)
    frame2()
    wp.synchronize()
    ours_ns = b2[..., [2, 1, 0]].cpu().numpy()
    print(f"\n== mujoco_warp shadows OFF vs MuJoCo shadows ON ==")
    print(f"  {pixel_agreement(ours_ns, ref)}")

    # throughput, for the record
    t0 = time.perf_counter()
    for _ in range(20):
        frame()
    wp.synchronize()
    dt = time.perf_counter() - t0
    print(f"\n  mjwarp {NW} worlds x 20 iters: {20*NW/dt:,.0f} f/s at {RES}px")


if __name__ == "__main__":
    main()
