"""How close can MJWarp get to MuJoCo on RoboCasa once the invisible geoms are excluded?

`probe_alpha.py` removed the cause of the old 77.8/255 result (zero-alpha marker geoms painted
opaque green) and landed at mean 7.5/255 on `robot0_agentview_left`. This re-runs the calibration
sweep the original probe ran -- but on the corrected model, with shadows in the grid this time, and
scoring interior pixels separately from silhouettes, which is the convention `backend3d.md` uses
for every other scene.

The wrist camera (`robot0_eye_in_hand`) was 7x worse than the two agent views, so it gets its own
breakdown rather than being averaged away.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def mj_render(mjm, mjd, res, cam_id, groups=(1,), seg=False, shadows=True):
    import mujoco
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[:] = 0
    for g in groups:
        opt.geomgroup[g] = 1
    r = mujoco.Renderer(mjm, height=res, width=res)
    if seg:
        r.enable_segmentation_rendering()
    r.update_scene(mjd, camera=cam_id, scene_option=opt)
    if not seg:
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = int(shadows)
    out = r.render().copy()
    r.close()
    return out


def interior_mask(seg):
    """Pixels whose 3x3 neighbourhood belongs to one object -- excludes silhouettes."""
    ids = seg[..., 0].astype(np.int32) * 16 + seg[..., 1].astype(np.int32)
    m = np.ones(ids.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            m &= (ids == np.roll(np.roll(ids, dy, 0), dx, 1))
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = False
    return m


def score(a, b, mask=None):
    e = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    if mask is not None:
        e = e[mask]
    if e.size == 0:
        return {}
    return {"mean": round(float(e.mean()), 3), "max": int(e.max()),
            "exact": round(float((e == 0).mean()), 4),
            "within2": round(float((e <= 2).mean()), 4),
            "within8": round(float((e <= 8).mean()), 4),
            "within32": round(float((e <= 32).mean()), 4)}


def effective_alpha(mjm):
    a = mjm.geom_rgba[:, 3].copy()
    hm = mjm.geom_matid >= 0
    a[hm] = mjm.mat_rgba[mjm.geom_matid[hm], 3]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(HERE, "out", "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "out", "qpos_bank.npy"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cams", nargs="+",
                    default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"])
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    import mujoco
    import warp as wp
    import mujoco_warp as mjw
    import imageio.v2 as imageio

    mjm = mujoco.MjModel.from_xml_path(args.xml)          # reference: untouched model
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = np.load(args.qpos)[0]
    mujoco.mj_forward(mjm, mjd)

    fix = mujoco.MjModel.from_xml_path(args.xml)          # render model: invisible geoms hidden
    fixd = mujoco.MjData(fix)
    fixd.qpos[:] = np.load(args.qpos)[0]
    a = effective_alpha(fix)
    fix.geom_group[a <= 0] = 3
    mujoco.mj_forward(fix, fixd)

    m = mjw.put_model(fix)
    d = mjw.put_data(fix, fixd, nworld=1)

    rec = {"res": args.res, "mujoco_warp": getattr(mjw, "__version__", "?")}
    out = wp.zeros((1, args.res, args.res), dtype=wp.vec3)

    def warp_px(rc):
        mjw.kinematics(m, d)
        mjw.camlight(m, d)
        mjw.refit_bvh(m, d, rc)
        mjw.render(m, d, rc)
        mjw.get_rgb(rc, 0, out)
        wp.synchronize()
        return np.ascontiguousarray(np.clip(out.numpy()[0] * 255.0 + 0.5, 0, 255).astype(np.uint8))

    grid = list(itertools.product([True, False], [True, False], [True, False], [True, False]))
    per_cam = {}
    for cam in args.cams:
        cid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cid < 0:
            continue
        seg = mj_render(mjm, mjd, args.res, cid, seg=True)
        inter = interior_mask(seg)
        results = []
        best = None
        for shadow, spec, sky, amb in grid:
            ref = mj_render(mjm, mjd, args.res, cid, shadows=shadow)
            rc = mjw.create_render_context(
                fix, nworld=1, cam_res=(args.res, args.res), render_rgb=True,
                render_depth=False, render_seg=False, use_textures=True,
                use_shadows=shadow, enable_specular=spec, render_skybox=sky,
                use_ambient_lighting=amb, enabled_geom_groups=[1],
                cam_active=[i == cid for i in range(fix.ncam)])
            px = warp_px(rc)
            s = {"shadows": shadow, "specular": spec, "skybox": sky, "ambient": amb,
                 "full": score(px, ref), "interior": score(px, ref, inter)}
            results.append(s)
            if best is None or s["interior"]["mean"] < best[0]["interior"]["mean"]:
                best = (s, px.copy(), ref.copy())
            del rc
        per_cam[cam] = {"interior_frac": round(float(inter.mean()), 4),
                        "best": best[0], "sweep": results}
        print(cam, json.dumps(best[0]), flush=True)
        pad = np.zeros((args.res, 3, 3), np.uint8)
        imageio.imwrite(os.path.join(args.out, f"parity_{cam}.png"),
                        np.concatenate([best[2], pad, best[1], pad,
                                        np.abs(best[1].astype(np.int16) - best[2].astype(np.int16))
                                        .clip(0, 255).astype(np.uint8)], axis=1))
    rec["per_cam"] = per_cam
    with open(os.path.join(args.out, "parity.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({c: v["best"] for c, v in per_cam.items()}, indent=2))


if __name__ == "__main__":
    main()
