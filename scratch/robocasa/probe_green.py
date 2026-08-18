"""Why is MJWarp's RoboCasa frame green even with every texture baked to a flat colour?

`probe_texgen.py` established that the flat mean-colour bake is nearly free on MuJoCo's side
(mean 7.99/255) but that MJWarp renders the *baked* kitchen just as green as the textured one. A
flat-material scene is exactly the case backend3d measured at 98.9 % pixel-exact on OGBench, so
either this model breaks something structural or the probe harness is wrong.

Bisected here, with a known-good scene as the control:

  control  : an OGBench scene through the identical harness (expect near-exact)
  const    : kitchen with every mat_rgba and geom_rgba forced to one grey
  lighting : ambient only / no headlight / no specular / no emission, one at a time
  groups   : does the geom-group filter change which surfaces are painted
  values   : the actual unique colours MJWarp emits, and MuJoCo's, side by side
"""
from __future__ import annotations

import argparse
import collections
import json
import os

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def score(a, b):
    e = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return {"mean": round(float(e.mean()), 3), "exact": round(float((e == 0).mean()), 4),
            "within8": round(float((e <= 8).mean()), 4)}


def mj_render(mjm, mjd, res, cam_id, groups=(1,)):
    import mujoco
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    r = mujoco.Renderer(mjm, height=res, width=res)
    r.update_scene(mjd, camera=cam_id, scene_option=opt)
    out = r.render().copy()
    r.close()
    return out


def warp_render(mjm, mjd, res, cam_id, groups=(1,), **kw):
    import warp as wp
    import mujoco_warp as mjw
    cfg = dict(render_rgb=True, render_depth=False, render_seg=False, use_textures=True,
               use_shadows=False, render_skybox=True)
    cfg.update(kw)
    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=1)
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(res, res),
        enabled_geom_groups=list(groups),
        cam_active=[i == cam_id for i in range(mjm.ncam)], **cfg)
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    out = wp.zeros((1, res, res), dtype=wp.vec3)
    mjw.get_rgb(rc, 0, out)
    wp.synchronize()
    px = np.clip(out.numpy()[0] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    del rc, d, m
    return np.ascontiguousarray(px)


def top_colours(img, k=6):
    c = collections.Counter(map(tuple, img.reshape(-1, 3).tolist()))
    n = img.shape[0] * img.shape[1]
    return [{"rgb": list(v), "frac": round(cnt / n, 4)} for v, cnt in c.most_common(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(HERE, "out", "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "out", "qpos_bank.npy"))
    ap.add_argument("--control", default="/mnt/vast/home/randall.balestriero/code/stable-envs/"
                                         "scratch/backend3d/scenes/antmaze_medium.xml")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    import mujoco
    import imageio.v2 as imageio
    rec = {}

    # ---- control: a scene the 3-D backend survey scored at 98.9 % ------------------------------
    if os.path.exists(args.control):
        cm = mujoco.MjModel.from_xml_path(args.control)
        cd = mujoco.MjData(cm)
        mujoco.mj_forward(cm, cd)
        groups = tuple(sorted({int(g) for g in cm.geom_group}))
        ref = mj_render(cm, cd, args.res, 0, groups)
        got = warp_render(cm, cd, args.res, 0, groups)
        rec["control_ogbench"] = {"scene": os.path.basename(args.control), "groups": list(groups),
                                  "score": score(got, ref)}
        print(json.dumps(rec["control_ogbench"]), flush=True)

    # ---- the kitchen ----------------------------------------------------------------------------
    def fresh():
        m = mujoco.MjModel.from_xml_path(args.xml)
        d = mujoco.MjData(m)
        d.qpos[:] = np.load(args.qpos)[0]
        mujoco.mj_forward(m, d)
        return m, d

    mjm, mjd = fresh()
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if cam_id < 0:
        cam_id = 0
    ref = mj_render(mjm, mjd, args.res, cam_id)

    rec["model"] = {
        "nmat": int(mjm.nmat), "ntex": int(mjm.ntex), "nlight": int(mjm.nlight),
        "ncam": int(mjm.ncam), "ngeom": int(mjm.ngeom), "nmesh": int(mjm.nmesh),
        "nmeshtexcoord": int(mjm.nmeshtexcoord),
        "meshes_with_texcoord": int((np.diff(np.append(mjm.mesh_texcoordadr, mjm.nmeshtexcoord))
                                     > 0).sum()) if hasattr(mjm, "mesh_texcoordadr") else -1,
        "mat_rgba_min": mjm.mat_rgba.min(0).tolist(), "mat_rgba_max": mjm.mat_rgba.max(0).tolist(),
        "mat_rgba_mean": mjm.mat_rgba.mean(0).tolist(),
        "mat_specular_mean": float(mjm.mat_specular.mean()),
        "mat_shininess_mean": float(mjm.mat_shininess.mean()),
        "mat_emission_max": float(mjm.mat_emission.max()),
        "mat_reflectance_max": float(mjm.mat_reflectance.max()),
        "light_diffuse": mjm.light_diffuse.tolist(), "light_ambient": mjm.light_ambient.tolist(),
        "light_specular": mjm.light_specular.tolist(),
        "light_type": mjm.light_type.tolist(), "light_castshadow": mjm.light_castshadow.tolist(),
        "headlight_ambient": mjm.vis.headlight.ambient.tolist(),
        "headlight_diffuse": mjm.vis.headlight.diffuse.tolist(),
        "headlight_specular": mjm.vis.headlight.specular.tolist(),
        "headlight_active": int(mjm.vis.headlight.active),
        "geom_group_hist": {int(k): int(v) for k, v in
                            zip(*np.unique(mjm.geom_group, return_counts=True))},
    }
    print(json.dumps(rec["model"], indent=1), flush=True)

    variants = {}
    base = warp_render(mjm, mjd, args.res, cam_id)
    variants["default"] = {"score": score(base, ref), "top": top_colours(base)}
    imageio.imwrite(os.path.join(args.out, "green_default.png"), base)

    for name, kw in [
        ("no_specular", dict(enable_specular=False)),
        ("no_emission", dict(enable_emission=False)),
        ("no_specular_no_emission", dict(enable_specular=False, enable_emission=False)),
        ("no_ambient", dict(use_ambient_lighting=False)),
        ("no_skybox", dict(render_skybox=False)),
        ("no_textures", dict(use_textures=False)),
        ("no_tex_no_spec_no_emis", dict(use_textures=False, enable_specular=False,
                                        enable_emission=False)),
    ]:
        px = warp_render(mjm, mjd, args.res, cam_id, **kw)
        variants[name] = {"score": score(px, ref), "top": top_colours(px)}
        print(name, json.dumps(variants[name]), flush=True)

    # ---- constant-colour kitchen: removes every material question -------------------------------
    mjm_c, mjd_c = fresh()
    mjm_c.mat_rgba[:] = np.array([0.7, 0.7, 0.7, 1.0])
    mjm_c.geom_rgba[:] = np.array([0.7, 0.7, 0.7, 1.0])
    if mjm_c.mat_texid.ndim > 1:
        mjm_c.mat_texid[:] = -1
    else:
        mjm_c.mat_texid[:] = -1
    mjm_c.mat_specular[:] = 0.0
    mjm_c.mat_shininess[:] = 0.0
    mjm_c.mat_emission[:] = 0.0
    mjm_c.mat_reflectance[:] = 0.0
    mujoco.mj_forward(mjm_c, mjd_c)
    ref_c = mj_render(mjm_c, mjd_c, args.res, cam_id)
    got_c = warp_render(mjm_c, mjd_c, args.res, cam_id, use_textures=False)
    variants["const_grey"] = {"score": score(got_c, ref_c),
                              "top_mjwarp": top_colours(got_c), "top_mujoco": top_colours(ref_c)}
    imageio.imwrite(os.path.join(args.out, "green_const_mujoco.png"), ref_c)
    imageio.imwrite(os.path.join(args.out, "green_const_mjwarp.png"), got_c)
    print("const_grey", json.dumps(variants["const_grey"]), flush=True)

    # ---- and with the collision groups drawn too --------------------------------------------------
    got_g = warp_render(mjm_c, mjd_c, args.res, cam_id, groups=(0, 1, 2), use_textures=False)
    ref_g = mj_render(mjm_c, mjd_c, args.res, cam_id, groups=(0, 1, 2))
    variants["const_grey_groups012"] = {"score": score(got_g, ref_g)}
    print("const_grey_groups012", json.dumps(variants["const_grey_groups012"]), flush=True)

    rec["variants"] = variants
    rec["mujoco_top_colours"] = top_colours(ref)
    with open(os.path.join(args.out, "green.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({"mujoco_top": rec["mujoco_top_colours"]}, indent=1))


if __name__ == "__main__":
    main()
