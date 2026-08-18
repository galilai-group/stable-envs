"""What would a from-scratch rasterizer have to implement to match MuJoCo pixel-for-pixel here?

PushT was tractable because the official renderer draws a dozen flat polygons with no lighting: the
GPU port only had to match a transform and a palette. RoboCasa's frames come out of MuJoCo's OpenGL
pipeline, so "matching the official renderer" means matching whatever subset of that pipeline the
scene actually switches on. This inventories exactly that subset from the compiled model -- shadows,
skybox, reflections, specular/shininess, texture mapping modes, multisampling -- so the port cost is
counted rather than guessed.

Also measures the two things that decide whether a *reduced* target is honest:
  * how much of the frame is textured surface vs flat-coloured surface (ablate textures, diff);
  * how much of the frame shadows and reflections are worth (ablate each, diff).
If flat shading changes almost every pixel, "same geometry, simplified shading" is not a small
approximation and must not be sold as one.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

TEX_TYPE = {0: "2d", 1: "cube", 2: "skybox"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    import imageio.v2 as imageio
    from robocasa.utils.env_utils import create_env

    env = create_env(env_name=args.env_name, robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=args.res, camera_heights=args.res,
                     seed=0, render_onscreen=False)
    env.reset()
    m = env.sim.model._model
    d = env.sim.data._data
    rec = {}

    # ---- feature inventory -------------------------------------------------------------------
    rec["lights"] = {
        "nlight": int(m.nlight),
        "n_castshadow": int(np.sum(m.light_castshadow)),
        "n_directional": int(np.sum(m.light_directional)) if hasattr(m, "light_directional") else -1,
        "headlight_ambient": m.vis.headlight.ambient.tolist(),
        "headlight_diffuse": m.vis.headlight.diffuse.tolist(),
        "headlight_specular": m.vis.headlight.specular.tolist(),
        "headlight_active": int(m.vis.headlight.active),
    }
    rec["quality"] = {
        "shadowsize": int(m.vis.quality.shadowsize),
        "offsamples": int(m.vis.quality.offsamples),
        "numslices": int(m.vis.quality.numslices),
        "numquads": int(m.vis.quality.numquads),
        "shadowclip": float(m.vis.map.shadowclip),
        "shadowscale": float(m.vis.map.shadowscale),
    }
    tt = Counter(TEX_TYPE.get(int(t), str(t)) for t in m.tex_type)
    rec["textures"] = {
        "ntex": int(m.ntex), "by_type": dict(tt),
        "total_MB": float(m.tex_data.size / 1e6),
        "has_skybox": bool(tt.get("skybox", 0) > 0),
        "n_nearest_filtered": None,
    }
    rec["materials"] = {
        "nmat": int(m.nmat),
        "n_with_texture": int(np.sum(m.mat_texid.max(axis=1) >= 0)) if m.mat_texid.ndim > 1
                          else int(np.sum(m.mat_texid >= 0)),
        "n_reflective": int(np.sum(m.mat_reflectance > 0)),
        "max_reflectance": float(m.mat_reflectance.max()) if m.nmat else 0.0,
        "n_specular_gt0": int(np.sum(m.mat_specular > 0)),
        "shininess_range": [float(m.mat_shininess.min()), float(m.mat_shininess.max())] if m.nmat else [],
        "n_texuniform": int(np.sum(m.mat_texuniform)),
        "texrepeat_unique": len({tuple(r) for r in m.mat_texrepeat}) if m.nmat else 0,
        "n_emission_gt0": int(np.sum(m.mat_emission > 0)),
    }
    rec["geoms"] = {
        "ngeom": int(m.ngeom),
        "n_with_material": int(np.sum(m.geom_matid >= 0)),
        "n_plain_rgba": int(np.sum(m.geom_matid < 0)),
        "n_transparent": int(np.sum((m.geom_rgba[:, 3] < 1.0) & (m.geom_rgba[:, 3] > 0))),
    }

    # ---- what each feature is worth, in pixels -----------------------------------------------
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_agentview_left")
    r = mujoco.Renderer(m, height=args.res, width=args.res)

    # NOTE: mjRND_* are *scene* flags (mjtRndFlag on MjvScene.flags), not mjtVisFlag on MjvOption.
    # Setting them on the option object silently does nothing and every ablation reads as a
    # perfect 0.0 -- which looks like "this feature is unused" rather than "the probe missed".
    def shot(rnd_off=None):
        r.update_scene(d, camera=cam)
        if rnd_off is not None:
            r.scene.flags[rnd_off] = 0
        return r.render().copy()

    base = shot()
    imageio.imwrite(os.path.join(args.out, "feat_base.png"), base)
    rec["scene_default_rnd_flags"] = {
        name: int(r.scene.flags[val])
        for name, val in mujoco.mjtRndFlag.__members__.items()
        if int(val) < len(r.scene.flags)
    }

    def diff(a, b):
        e = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
        return {"mean": float(e.mean()), "max": int(e.max()),
                "frac_changed": float((e > 2).mean())}

    abl = {}
    no_shadow = shot(mujoco.mjtRndFlag.mjRND_SHADOW)
    abl["shadows_off"] = diff(base, no_shadow)
    abl["reflections_off"] = diff(base, shot(mujoco.mjtRndFlag.mjRND_REFLECTION))
    abl["skybox_off"] = diff(base, shot(mujoco.mjtRndFlag.mjRND_SKYBOX))
    imageio.imwrite(os.path.join(args.out, "feat_no_shadow.png"), no_shadow)

    # textures off: strip every material's texture binding and recompile the render context
    texid = m.mat_texid.copy()
    m.mat_texid[:] = -1
    r2 = mujoco.Renderer(m, height=args.res, width=args.res)
    r2.update_scene(d, camera=cam)
    flat = r2.render().copy()
    m.mat_texid[:] = texid
    r2.close()
    abl["textures_off"] = diff(base, flat)
    imageio.imwrite(os.path.join(args.out, "feat_no_texture.png"), flat)
    imageio.imwrite(os.path.join(args.out, "feat_texture_diff_x4.png"),
                    np.clip(np.abs(base.astype(np.int16) - flat.astype(np.int16)) * 4, 0, 255).astype(np.uint8))

    rec["ablations"] = abl
    r.close()
    print(json.dumps(rec, indent=2))
    with open(os.path.join(args.out, "render_features.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
