"""Corrected scene complexity and shading ablations, using the geoms RoboCasa actually renders.

``robosuite/environments/base.py`` sets ``vopt.geomgroup[0] = render_collision_mesh (False)`` and
``vopt.geomgroup[1] = render_visual_mesh (True)``, so an official RoboCasa observation draws **group
1 only**. A bare ``mujoco.Renderer`` defaults to showing groups 0-2, which paints robosuite's
collision primitives over the visual meshes and produces a flat, garish frame that looks exactly
like a textures-ablated one.

Earlier probes here measured against that bare-Renderer baseline, so their geom/triangle counts and
their ablation percentages describe a picture RoboCasa never shows. Everything is recomputed against
the correct visibility mask, and the corrected baseline is diffed against robosuite's own
``sim.render`` to prove the mask is right rather than merely plausible.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out"
RES = 224


def main():
    import mujoco
    import imageio.v2 as imageio
    from robocasa.utils.env_utils import create_env

    env = create_env(env_name="PickPlaceCounterToSink", robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=RES, camera_heights=RES, seed=0, render_onscreen=False)
    env.reset()
    m = env.sim.model._model
    d = env.sim.data._data
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_agentview_left")
    rec = {}

    # ---- visible-geometry accounting ---------------------------------------------------------
    vis = [i for i in range(m.ngeom) if int(m.geom_group[i]) == 1]
    tris = sum(int(m.mesh_facenum[int(m.geom_dataid[i])])
               for i in vis if int(m.geom_type[i]) == 7)
    n_mesh_vis = sum(1 for i in vis if int(m.geom_type[i]) == 7)
    used_meshes = {int(m.geom_dataid[i]) for i in vis if int(m.geom_type[i]) == 7}
    used_mats = {int(m.geom_matid[i]) for i in vis if int(m.geom_matid[i]) >= 0}
    used_tex = set()
    for mt in used_mats:
        tid = m.mat_texid[mt]
        for t in (np.atleast_1d(tid)):
            if int(t) >= 0:
                used_tex.add(int(t))
    tex_bytes = 0
    for t in used_tex:
        tex_bytes += int(m.tex_width[t]) * int(m.tex_height[t]) * 3

    rec["visible"] = {
        "ngeom_total": int(m.ngeom),
        "ngeom_group1_rendered": len(vis),
        "ngeom_group0_collision_hidden": int(np.sum(m.geom_group == 0)),
        "n_mesh_geoms_rendered": n_mesh_vis,
        "n_unique_meshes_rendered": len(used_meshes),
        "triangles_rendered": tris,
        "triangles_all_groups": sum(int(m.mesh_facenum[int(m.geom_dataid[i])])
                                    for i in range(m.ngeom) if int(m.geom_type[i]) == 7),
        "n_materials_used": len(used_mats),
        "n_textures_used": len(used_tex),
        "texture_MB_used": tex_bytes / 1e6,
        "texture_MB_total": float(m.tex_data.size / 1e6),
    }
    print("visible:", json.dumps(rec["visible"], indent=2))

    # ---- the correct baseline, and proof it is correct ----------------------------------------
    r = mujoco.Renderer(m, height=RES, width=RES)
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1

    def shot(rnd_off=None):
        r.update_scene(d, camera=cam, scene_option=opt)
        if rnd_off is not None:
            r.scene.flags[rnd_off] = 0
        return r.render().copy()

    base = shot()
    official = env.sim.render(width=RES, height=RES, camera_name="robot0_agentview_left")[::-1]
    e = np.abs(base.astype(np.int16) - official.astype(np.int16)).max(-1)
    rec["baseline_matches_robosuite"] = {
        "mean": float(e.mean()), "max": int(e.max()), "exact": float((e == 0).mean())}
    print("corrected baseline vs robosuite sim.render:", json.dumps(rec["baseline_matches_robosuite"]))
    imageio.imwrite(os.path.join(OUT, "vis_base.png"), base)
    imageio.imwrite(os.path.join(OUT, "vis_official.png"), official)

    def diff(a, b):
        x = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
        return {"mean": float(x.mean()), "max": int(x.max()), "frac_changed": float((x > 2).mean())}

    abl = {}
    for tag, flag in (("shadows", mujoco.mjtRndFlag.mjRND_SHADOW),
                      ("reflections", mujoco.mjtRndFlag.mjRND_REFLECTION),
                      ("skybox", mujoco.mjtRndFlag.mjRND_SKYBOX),
                      ("haze", mujoco.mjtRndFlag.mjRND_HAZE)):
        abl[tag] = diff(base, shot(flag))
    r.close()

    texid = m.mat_texid.copy()
    m.mat_texid[:] = -1
    r2 = mujoco.Renderer(m, height=RES, width=RES)
    r2.update_scene(d, camera=cam, scene_option=opt)
    flat = r2.render().copy()
    r2.close()
    m.mat_texid[:] = texid
    abl["textures"] = diff(base, flat)
    imageio.imwrite(os.path.join(OUT, "vis_no_texture.png"), flat)

    rec["ablations_corrected"] = abl
    print("ablations (group-1 only):", json.dumps(abl, indent=2))

    with open(os.path.join(OUT, "visible.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
