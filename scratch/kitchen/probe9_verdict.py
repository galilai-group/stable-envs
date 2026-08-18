"""Probe 9: pin the texture crash to a texture TYPE, then measure the best configuration that survives.

Two questions, both decisive for the family:

  1. probe8 showed every ``use_textures=True`` configuration faults with CUDA 700 and every
     ``use_textures=False`` one renders. This narrows it: the kitchen carries one 2-D texture
     (``texplane``), one skybox, and TWELVE cube textures (``mjTEXTURE_CUBE``). Strip only the cube
     textures and see whether the fault goes away.

  2. With textures off -- the only configuration mujoco_warp can run -- how close is the frame? That is
     the ceiling for this family under the remaining design, and the plate design is already dead
     (probe6: mover shadows put 85.6% of the frame under a state-dependent shadow, and a baked plate
     scores 0.834 of non-mover interior pixels against 1.000000 with shadows off).
"""
from __future__ import annotations

import os
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES = 480


def strip(m, which):
    """Remove textures of the given mjtTexture types from every material."""
    import mujoco as mj
    n = 0
    for i in range(m.nmat):
        t = np.asarray(m.mat_texid[i]).reshape(-1).copy()
        for k, tid in enumerate(t):
            if tid >= 0 and int(m.tex_type[tid]) in which:
                t[k] = -1
                n += 1
        m.mat_texid[i] = t.reshape(np.asarray(m.mat_texid[i]).shape)
    return n


def run_case(case):
    from probe7_mjwarp import build_model
    from stable_envs.render import mjwarp as MW
    import warp as wp
    import mujoco_warp as mjw
    import torch

    m, camid, _ = build_model(keep_tex=True)
    if case == "no_cube":
        n = strip(m, {int(mujoco.mjtTexture.mjTEXTURE_CUBE)})
        print(f"  stripped {n} cube texture slots")
    elif case == "cube_only":
        n = strip(m, {int(mujoco.mjtTexture.mjTEXTURE_2D),
                      int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)})
        print(f"  stripped {n} 2d/skybox texture slots")
    elif case == "all_tex":
        pass
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    wd = mjw.put_data(m, d, nworld=4)
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=4, cam_res=(RES, RES), render_rgb=True, render_seg=False,
            cam_active=[i == camid for i in range(m.ncam)],
            enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=True, use_shadows=True,
            render_skybox=(case != "no_cube_noskybox"), enable_specular=True)
        mjw.kinematics(wm, wd)
        mjw.camlight(wm, wd)
        mjw.refit_bvh(wm, wd, rc)
        mjw.render(wm, wd, rc)
        wp.synchronize()
    a = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(4, RES, RES, 4)[..., [2, 1, 0]].cpu().numpy()
    print(f"OK {case}: mean={a.mean():.2f}")


def measure_notex():
    """The best mujoco_warp can do here: textures off, shadows on, against MuJoCo's own frame."""
    from probe7_mjwarp import build_model
    from probe5_contamination import pool
    from stable_envs.render import mjwarp as MW
    from stable_envs.testing.conformance import pixel_agreement
    import warp as wp
    import mujoco_warp as mjw
    import torch
    import time

    NW = 16
    m, camid, _ = build_model(keep_tex=False)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, width=RES, height=RES)
    wm = mjw.put_model(m)
    wd = mjw.put_data(m, d, nworld=NW)
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=NW, cam_res=(RES, RES), render_rgb=True, render_seg=False,
            cam_active=[i == camid for i in range(m.ncam)],
            enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=False, use_shadows=True,
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
    qp = wp.to_torch(wd.qpos)
    qp[:] = torch.as_tensor(states, dtype=qp.dtype, device=qp.device)
    torch.cuda.synchronize()
    frame()
    wp.synchronize()
    ours = bgra[..., [2, 1, 0]].cpu().numpy()

    ref = []
    for q in states:
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=camid)
        ref.append(r.render().copy())
    ref = np.stack(ref)
    print("\n== mujoco_warp, TEXTURES OFF (the only configuration that runs), vs MuJoCo ==")
    print(f"  same model, textures off on BOTH sides is not possible -- MuJoCo keeps them, so this")
    print(f"  is our best frame against the frame the family actually produces:")
    print(f"  {pixel_agreement(ours, ref)}")

    # and against a MuJoCo render with the textures stripped too, to separate the texture term from
    # everything else (shading, shadow-edge rule, quantisation)
    m2, camid2, _ = build_model(keep_tex=False)
    d2 = mujoco.MjData(m2)
    r2 = mujoco.Renderer(m2, width=RES, height=RES)
    ref2 = []
    for q in states:
        d2.qpos[:] = q
        d2.qvel[:] = 0
        mujoco.mj_forward(m2, d2)
        r2.update_scene(d2, camera=camid2)
        ref2.append(r2.render().copy())
    ref2 = np.stack(ref2)
    print(f"\n== the same, against MuJoCo with textures ALSO stripped (isolates shading/shadow) ==")
    print(f"  {pixel_agreement(ours, ref2)}")

    t0 = time.perf_counter()
    for _ in range(20):
        frame()
    wp.synchronize()
    print(f"\n  mjwarp {NW} worlds: {20*NW/(time.perf_counter()-t0):,.0f} f/s at {RES}px (textures off)")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] == "measure":
            measure_notex()
        else:
            run_case(sys.argv[1])
        return
    for case in ("all_tex", "no_cube", "cube_only"):
        p = subprocess.run([sys.executable, os.path.abspath(__file__), case],
                           capture_output=True, text=True, timeout=900)
        ok = [ln for ln in p.stdout.splitlines() if ln.startswith("OK ")]
        extra = [ln for ln in p.stdout.splitlines() if ln.strip().startswith("stripped")]
        tag = "PASS " + ok[0][3:] if (p.returncode == 0 and ok) else "CRASH"
        print(f"  {case:12s} {tag}   {extra[0].strip() if extra else ''}")
    p = subprocess.run([sys.executable, os.path.abspath(__file__), "measure"],
                       capture_output=True, text=True, timeout=1800)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr[-2000:])


if __name__ == "__main__":
    main()
