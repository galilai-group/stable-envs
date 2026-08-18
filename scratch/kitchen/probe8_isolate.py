"""Probe 8: which mujoco_warp feature makes the kitchen megakernel fault?

One configuration per process, because a CUDA illegal access poisons the context for everything after
it. ``--all`` re-runs this file as a subprocess per configuration and prints a table.
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

CONFIGS = [
    # name              tex    shadow  spec   skybox nworld  keep_lights
    ("baseline",        True,  True,   True,  True,  16,     None),
    ("no_shadow",       True,  False,  True,  True,  16,     None),
    ("no_tex",          False, True,   True,  True,  16,     None),
    ("no_spec",         True,  True,   False, True,  16,     None),
    ("no_skybox",       True,  True,   True,  False, 16,     None),
    ("nworld1",         True,  True,   True,  True,  1,      None),
    ("dirlights_only",  True,  True,   True,  True,  16,     3),
    ("one_light",       True,  True,   True,  True,  16,     1),
    ("no_shadow_notex", False, False,  True,  True,  16,     None),
]


def run_one(name):
    from probe7_mjwarp import build_model
    from stable_envs.render import mjwarp as MW
    import warp as wp
    import mujoco_warp as mjw
    import torch

    cfg = [c for c in CONFIGS if c[0] == name][0]
    _, tex, shadow, spec, skybox, nworld, keep_lights = cfg
    m, camid, _ = build_model(keep_tex=tex)
    if keep_lights is not None:
        # drop lights by moving them far away and zeroing their contribution is not enough: the count
        # itself is what is being tested, so recompile from a spec with the extras removed instead.
        m.light_diffuse[keep_lights:] = 0.0
        m.light_specular[keep_lights:] = 0.0
        m.light_castshadow[keep_lights:] = 0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    wd = mjw.put_data(m, d, nworld=nworld)
    with MW.gl_rounding():
        rc = MW.create_render_context(
            m, nworld=nworld, cam_res=(RES, RES), render_rgb=True, render_seg=False,
            cam_active=[i == camid for i in range(m.ncam)],
            enabled_geom_groups=list(MW.DEFAULT_GROUPS), use_textures=tex, use_shadows=shadow,
            render_skybox=skybox, enable_specular=spec)
        mjw.kinematics(wm, wd)
        mjw.camlight(wm, wd)
        mjw.refit_bvh(wm, wd, rc)
        mjw.render(wm, wd, rc)
        wp.synchronize()
    img = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(nworld, RES, RES, 4)
    arr = img[..., [2, 1, 0]].cpu().numpy()
    print(f"OK {name}: mean={arr.mean():.2f} nonzero={float((arr>0).mean()):.4f}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--all":
        run_one(sys.argv[1])
        return
    for c in CONFIGS:
        p = subprocess.run([sys.executable, os.path.abspath(__file__), c[0]],
                           capture_output=True, text=True, timeout=900)
        out = [ln for ln in p.stdout.splitlines() if ln.startswith("OK ")]
        if p.returncode == 0 and out:
            print(f"  {c[0]:18s} PASS   {out[0][3:]}")
        else:
            err = ""
            for ln in (p.stderr or "").splitlines():
                if "Error" in ln or "error" in ln or "Exception" in ln:
                    err = ln.strip()
                    break
            print(f"  {c[0]:18s} CRASH  {err[:110]}")


if __name__ == "__main__":
    main()
