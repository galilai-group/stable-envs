"""Does neutralising ``mjTEXTURE_CUBE`` slots lift Franka Kitchen's Gate 1?

``docs/research/kitchen.md`` Gate 1 records ``mujoco_warp`` hard-faulting with CUDA 700 on this
model, on 9 of 9 configurations that enable textures, and pins the cause to the 67 cube-texture
slots. It concludes the scene "cannot be loaded at all", which is what removes the second-plate
shadow rescue as a side effect (that rescue needs the same backend's mesh BVH).

``scratch/metaworld/probe_backend3.py`` found the fault is avoidable rather than fatal: pointing
every MATERIAL away from a cube texture — keeping the skybox, keeping ``mat_rgba`` — makes MJWarp
render a scene it previously faulted on. The kitchen's cube textures are the counters, cabinetry,
walls and oven, which are STATIC and therefore baked through MuJoCo anyway, so their MJWarp colour
never reaches a composited pixel.

So the question this asks is narrow and worth its own number: with the cube slots neutralised, does
the kitchen LOAD and RENDER? If it does, Gate 1 is a workaround rather than a wall, and the verdict
rests on Gate 2 (four casting lights over 85.6 % of the frame) alone.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import mujoco

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from stable_envs.render import mjwarp as MW                                 # noqa: E402

RES = 224
OUT = "/mnt/vast/home/randall.balestriero/se_logs/kitchen"


def main(neutralise: bool):
    import torch
    import warp as wp

    import mujoco_warp as mjw
    import gymnasium
    import gymnasium_robotics                                               # noqa: F401
    from gymnasium_robotics.envs.franka_kitchen import franka_env
    gymnasium.register_envs(gymnasium_robotics)

    xml = os.path.join(os.path.dirname(os.path.realpath(franka_env.__file__)),
                       "../assets/kitchen_franka/kitchen_assets/kitchen_env_model.xml")
    m = mujoco.MjModel.from_xml_path(xml)
    rec = {"neutralise": neutralise, "ngeom": int(m.ngeom), "ntex": int(m.ntex)}

    ttype = np.asarray(m.tex_type, np.int64)
    rec["n_tex_by_type"] = {int(t): int((ttype == t).sum()) for t in np.unique(ttype)}

    if neutralise:
        flat = m.mat_texid.reshape(m.nmat, -1) if m.mat_texid.ndim > 1 else \
            m.mat_texid.reshape(-1, 1)
        hit = 0
        for i in range(flat.shape[0]):
            for j in range(flat.shape[1]):
                t = int(flat[i, j])
                if t >= 0 and ttype[t] == int(mujoco.mjtTexture.mjTEXTURE_CUBE):
                    flat[i, j] = -1
                    hit += 1
        rec["n_neutralised"] = hit

    m.geom_margin[:] = 0.0
    m.pair_margin[:] = 0.0
    MW.prepare_model(m, MW.DEFAULT_GROUPS)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    try:
        wm = mjw.put_model(m)
        wd = mjw.put_data(m, d0, nworld=4)
        with MW.gl_rounding():
            rc = MW.create_render_context(
                m, nworld=4, cam_res=(RES, RES), render_rgb=True,
                cam_active=[i == 0 for i in range(max(m.ncam, 1))],
                enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=True,
                use_shadows=True, render_skybox=True, enable_specular=True)
            mjw.kinematics(wm, wd)
            mjw.camlight(wm, wd)
            mjw.refit_bvh(wm, wd, rc)
            mjw.render(wm, wd, rc)
            wp.synchronize()
        img = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(4, RES, RES, 4)[
            0, ..., [2, 1, 0]].cpu().numpy()
        rec["rendered"] = True
        rec["unique_colours"] = int(len(np.unique(img.reshape(-1, 3), axis=0)))
        rec["range"] = [int(img.min()), int(img.max())]
    except Exception as e:                                                  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rec["rendered"] = False
        rec["error"] = f"{type(e).__name__}: {e}"
    print("RESULT " + json.dumps(rec), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"probe12_nocube_{int(neutralise)}.json"), "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] == "1")
