"""The constants a DM Control renderer needs, read off the compiled model and the live mjvScene.

Everything here is a thing that reads plausibly from the XML and is wrong, or is simply not in the
XML at all:

  * what ``dm_control``'s own ``physics.render()`` passes for scene options (which geom and SITE
    groups are visible) -- sites are drawn by default and several domains hang one off a MOVING
    body, so a plate baked by hiding only the moving GEOMS is not constant (measured: 28/255 on
    acrobot, 103/255 on ball_in_cup);
  * the camera mode of camera 0 -- ``trackcom`` cameras move with the body, so those domains have
    no fixed plate at all;
  * the two ``mjvGLCamera`` eyes MuJoCo actually renders from, and the lights as they appear in
    ``mjvScene`` (the headlight is not in the model).
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

CAMLIGHT = {0: "fixed", 1: "track", 2: "trackcom", 3: "targetbody", 4: "targetbodycom"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=84)
    ap.add_argument("--out", default="")
    ap.add_argument("--domains", default="cartpole:swingup,cartpole:balance,pendulum:swingup,"
                                         "acrobot:swingup,point_mass:easy,ball_in_cup:catch,"
                                         "reacher:easy,finger:spin,cheetah:run,walker:walk,"
                                         "hopper:hop,swimmer:swimmer6,fish:swim,humanoid:stand")
    args = ap.parse_args()

    import mujoco
    from dm_control import suite

    dflt = mujoco.MjvOption()
    out = {"default_geomgroup": [int(x) for x in dflt.geomgroup],
           "default_sitegroup": [int(x) for x in dflt.sitegroup],
           "domains": {}}
    print("default geomgroup", out["default_geomgroup"], "sitegroup", out["default_sitegroup"])

    for spec in args.domains.split(","):
        dom, task = spec.split(":")
        key = f"{dom}-{task}"
        try:
            env = suite.load(dom, task, task_kwargs={"random": 0})
            p = env.physics
            mjm = p.model.ptr if hasattr(p.model, "ptr") else p.model
            mjd = p.data.ptr if hasattr(p.data, "ptr") else p.data
            mujoco.mj_forward(mjm, mjd)

            mjm.vis.global_.offwidth = max(int(mjm.vis.global_.offwidth), args.res)
            mjm.vis.global_.offheight = max(int(mjm.vis.global_.offheight), args.res)
            r = mujoco.Renderer(mjm, height=args.res, width=args.res)
            r.update_scene(mjd, camera=0)
            sc = r.scene
            mine = r.render().copy()
            theirs = p.render(height=args.res, width=args.res, camera_id=0).copy()

            rec = {
                "render_matches_dm_control": int(np.abs(mine.astype(np.int32)
                                                        - theirs.astype(np.int32)).max()),
                "cam0_mode": CAMLIGHT.get(int(mjm.cam_mode[0]), int(mjm.cam_mode[0])),
                "cam_modes": [CAMLIGHT.get(int(x), int(x)) for x in mjm.cam_mode],
                "moving_geoms": [int(g) for g in range(mjm.ngeom) if int(mjm.geom_bodyid[g]) != 0],
                "moving_sites": [int(s) for s in range(mjm.nsite) if int(mjm.site_bodyid[s]) != 0],
                "site_groups": [int(x) for x in mjm.site_group],
                "site_types": [int(x) for x in mjm.site_type],
                "geom_types": [int(x) for x in mjm.geom_type],
                "geom_names": [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, g)
                               for g in range(mjm.ngeom)],
                "site_names": [mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_SITE, s)
                               for s in range(mjm.nsite)],
                "ngeom_in_scene": int(sc.ngeom),
                "nlight_in_scene": int(sc.nlight),
                "scene_lights": [{
                    "pos": [round(float(x), 5) for x in sc.lights[i].pos],
                    "dir": [round(float(x), 5) for x in sc.lights[i].dir],
                    "directional": int(sc.lights[i].type) if hasattr(sc.lights[i], "type")
                    else int(getattr(sc.lights[i], "directional", -1)),
                    "ambient": [round(float(x), 4) for x in sc.lights[i].ambient],
                    "diffuse": [round(float(x), 4) for x in sc.lights[i].diffuse],
                    "specular": [round(float(x), 4) for x in sc.lights[i].specular],
                    "attenuation": [round(float(x), 5) for x in sc.lights[i].attenuation],
                    "cutoff": float(sc.lights[i].cutoff), "exponent": float(sc.lights[i].exponent),
                    "castshadow": int(sc.lights[i].castshadow),
                } for i in range(int(sc.nlight))],
                "eyes": [{"pos": [round(float(x), 6) for x in sc.camera[i].pos],
                          "forward": [round(float(x), 6) for x in sc.camera[i].forward],
                          "up": [round(float(x), 6) for x in sc.camera[i].up],
                          "near": float(sc.camera[i].frustum_near),
                          "top": float(sc.camera[i].frustum_top),
                          "bottom": float(sc.camera[i].frustum_bottom),
                          "center": float(sc.camera[i].frustum_center)} for i in range(2)],
                "scene_geom_kinds": sorted({int(sc.geoms[i].type) for i in range(int(sc.ngeom))}),
            }
            # plate constancy with BOTH moving geoms and moving sites hidden
            g0 = np.array(mjm.geom_group, copy=True)
            s0 = np.array(mjm.site_group, copy=True)
            rng = np.random.default_rng(0)
            plates = []
            for _ in range(4):
                mjm.geom_group[rec["moving_geoms"]] = 5
                mjm.site_group[rec["moving_sites"]] = 5
                mjd.qpos[:] = np.array(mjm.qpos0, np.float64) + rng.normal(0, .4, mjm.nq)
                mujoco.mj_forward(mjm, mjd)
                r.update_scene(mjd, camera=0)
                plates.append(r.render().copy())
                mjm.geom_group[:] = g0
                mjm.site_group[:] = s0
            plates = np.stack(plates)
            rec["plate_constant_max"] = int(np.abs(plates.astype(np.int32)
                                                   - plates[0].astype(np.int32)).max())
            out["domains"][key] = rec
            print(f"{key:22s} dm_ctrl_diff={rec['render_matches_dm_control']:3d} "
                  f"cam0={rec['cam0_mode']:12s} plate_const={rec['plate_constant_max']:3d} "
                  f"nlight={rec['nlight_in_scene']} kinds={rec['scene_geom_kinds']} "
                  f"msites={len(rec['moving_sites'])}", flush=True)
        except Exception as e:                                             # noqa: BLE001
            import traceback
            traceback.print_exc()
            out["domains"][key] = {"error": f"{type(e).__name__}: {e}"}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
