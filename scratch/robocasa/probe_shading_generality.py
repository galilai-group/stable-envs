"""Does the "simple shading" finding hold, or was it one lucky camera?

The single-camera probe said shadows, reflections and the skybox are worth exactly zero pixels --
which, if true everywhere, is the difference between porting a textured-Phong rasterizer and porting
a full graphics engine. That is too load-bearing to rest on one viewpoint of one kitchen: the scene
carries 49 materials with reflectance up to 0.8, and a camera that sees the floor or a window could
light up every term that this one did not.

So the ablation is repeated over every camera in the model and over several resampled kitchens, and
the *worst* case is reported, not the mean. Textures are ablated the same way, to check that the one
feature that did matter keeps mattering.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--n-scenes", type=int, default=4)
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

    rows = []
    worst = {}
    for si in range(args.n_scenes):
        env.reset()
        m = env.sim.model._model
        d = env.sim.data._data
        r = mujoco.Renderer(m, height=args.res, width=args.res)
        names = []
        for i in range(m.ncam):
            adr = int(m.name_camadr[i])
            names.append(m.names[adr:m.names.index(b"\0", adr)].decode())

        def shot(cam, rnd_off=None, no_tex=False):
            r.update_scene(d, camera=cam)
            if rnd_off is not None:
                r.scene.flags[rnd_off] = 0
            return r.render().copy()

        for ci, cname in enumerate(names):
            base = shot(ci)

            def dd(x):
                e = np.abs(base.astype(np.int16) - x.astype(np.int16)).max(-1)
                return float(e.mean()), int(e.max()), float((e > 2).mean())

            row = {"scene": si, "layout": int(env.layout_id), "style": int(env.style_id),
                   "cam": cname,
                   "n_castshadow": int(np.sum(m.light_castshadow)), "nlight": int(m.nlight),
                   "n_reflective_mat": int(np.sum(m.mat_reflectance > 0))}
            for tag, flag in (("shadow", mujoco.mjtRndFlag.mjRND_SHADOW),
                              ("reflection", mujoco.mjtRndFlag.mjRND_REFLECTION),
                              ("skybox", mujoco.mjtRndFlag.mjRND_SKYBOX),
                              ("haze", mujoco.mjtRndFlag.mjRND_HAZE)):
                mean, mx, frac = dd(shot(ci, rnd_off=flag))
                row[f"{tag}_mean"] = mean
                row[f"{tag}_max"] = mx
                row[f"{tag}_frac"] = frac
            rows.append(row)
            print(json.dumps(row))
        r.close()

        # textures off needs a second context (material bindings are baked at context build)
        texid = m.mat_texid.copy()
        r2 = mujoco.Renderer(m, height=args.res, width=args.res)
        r2.update_scene(d, camera=0)
        b0 = r2.render().copy()
        r2.close()
        m.mat_texid[:] = -1
        r3 = mujoco.Renderer(m, height=args.res, width=args.res)
        r3.update_scene(d, camera=0)
        flat = r3.render().copy()
        r3.close()
        m.mat_texid[:] = texid
        e = np.abs(b0.astype(np.int16) - flat.astype(np.int16)).max(-1)
        rows[-1]["textures_off_mean"] = float(e.mean())
        rows[-1]["textures_off_frac"] = float((e > 2).mean())
        print(f"scene {si} textures_off: mean {e.mean():.2f}  frac {(e>2).mean():.3f}")
        if si == 0:
            imageio.imwrite(os.path.join(args.out, "gen_base.png"), b0)
            imageio.imwrite(os.path.join(args.out, "gen_flat.png"), flat)

    for tag in ("shadow", "reflection", "skybox", "haze"):
        worst[tag] = {
            "worst_mean": max(r[f"{tag}_mean"] for r in rows),
            "worst_max": max(r[f"{tag}_max"] for r in rows),
            "worst_frac": max(r[f"{tag}_frac"] for r in rows),
        }
    out = {"n_rows": len(rows), "worst_case_over_all_cameras_and_scenes": worst,
           "textures_off": [r["textures_off_mean"] for r in rows if "textures_off_mean" in r],
           "rows": rows}
    print("\n" + json.dumps(worst, indent=2))
    with open(os.path.join(args.out, "shading_generality.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
