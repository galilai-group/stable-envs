"""The green is invisible geometry, not missing textures.

`probe_green.py` narrowed it: forcing every material and geom colour in the kitchen to one grey
makes MJWarp agree with MuJoCo at mean 1.85/255 (95.1 % within 8). The renderer, the lighting, the
camera and the geometry are all fine. What is wrong is *which geoms get drawn*.

MJWarp's shading reads `color[0..2]` and packs alpha as a constant 255
(`render.py`: `base_color = wp.vec3(color[0], color[1], color[2])` ... `pack_rgba_to_uint32(..., 255.0)`).
Alpha is never consulted. RoboCasa's kitchens carry a large number of **invisible marker geoms** --
placement regions and gripper sites written as `rgba="0 1 0 0"` in geom group 1 -- which MuJoCo
draws as fully transparent (i.e. not at all) and MJWarp paints as opaque pure green.

That is a model-side fix, not an upstream one: move the zero-alpha geoms into a geom group the
render context does not enable. This probe measures how much of the disagreement that recovers, and
what is left afterwards.
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
    return {"mean": round(float(e.mean()), 3), "max": int(e.max()),
            "exact": round(float((e == 0).mean()), 4),
            "within8": round(float((e <= 8).mean()), 4),
            "within32": round(float((e <= 32).mean()), 4)}


def mj_render(mjm, mjd, res, cam_id, groups=(1,), seg=False):
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
    rc = mjw.create_render_context(mjm, nworld=1, cam_res=(res, res),
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


def effective_alpha(mjm):
    """MuJoCo takes a geom's colour from its material when one is bound, else from geom_rgba."""
    a = mjm.geom_rgba[:, 3].copy()
    has_mat = mjm.geom_matid >= 0
    a[has_mat] = mjm.mat_rgba[mjm.geom_matid[has_mat], 3]
    return a


def hide_invisible(mjm, hidden_group=3, thresh=0.0):
    """Move geoms MuJoCo would draw as fully transparent out of the enabled render group."""
    a = effective_alpha(mjm)
    sel = a <= thresh
    mjm.geom_group[sel] = hidden_group
    return sel


def whiten_textured(mjm):
    """MuJoCo replaces a textured material's base colour; MJWarp modulates by mat_rgba.

    Neutralising the base colour makes the two agree on textured surfaces. Returns n changed.
    """
    texid = mjm.mat_texid
    n = 0
    for mi in range(mjm.nmat):
        row = np.atleast_1d(texid[mi])
        if any(int(x) >= 0 for x in row):
            if not np.allclose(mjm.mat_rgba[mi, :3], 1.0):
                mjm.mat_rgba[mi, :3] = 1.0
                n += 1
    return n


def top_colours(img, k=5):
    c = collections.Counter(map(tuple, img.reshape(-1, 3).tolist()))
    n = img.shape[0] * img.shape[1]
    return [{"rgb": list(v), "frac": round(cnt / n, 4)} for v, cnt in c.most_common(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(HERE, "out", "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "out", "qpos_bank.npy"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cams", nargs="+",
                    default=["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"])
    ap.add_argument("--nqpos", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()

    import mujoco
    import imageio.v2 as imageio
    rec = {"res": args.res, "mujoco": mujoco.__version__}

    def fresh():
        m = mujoco.MjModel.from_xml_path(args.xml)
        d = mujoco.MjData(m)
        d.qpos[:] = np.load(args.qpos)[0]
        mujoco.mj_forward(m, d)
        return m, d

    mjm, mjd = fresh()
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cams[0])

    a = effective_alpha(mjm)
    g1 = mjm.geom_group == 1
    rec["alpha_census"] = {
        "ngeom": int(mjm.ngeom),
        "group1": int(g1.sum()),
        "group1_alpha_zero": int((g1 & (a <= 0)).sum()),
        "group1_alpha_partial": int((g1 & (a > 0) & (a < 1)).sum()),
        "group1_alpha_opaque": int((g1 & (a >= 1)).sum()),
        "alpha_hist": {str(round(float(k), 3)): int(v) for k, v in
                       zip(*np.unique(np.round(a[g1], 3), return_counts=True))},
    }
    print(json.dumps(rec["alpha_census"], indent=1), flush=True)

    ref = mj_render(mjm, mjd, args.res, cam_id)
    variants = {}
    variants["as_is"] = score(warp_render(mjm, mjd, args.res, cam_id), ref)

    # A: hide the zero-alpha marker geoms
    m_a, d_a = fresh()
    sel = hide_invisible(m_a)
    mujoco.mj_forward(m_a, d_a)
    px_a = warp_render(m_a, d_a, args.res, cam_id)
    variants["hide_alpha0"] = score(px_a, ref)
    rec["hidden_geoms"] = int(sel.sum())
    imageio.imwrite(os.path.join(args.out, "alpha_hidden_mjwarp.png"), px_a)

    # B: hide + neutralise the base colour under textures
    m_b, d_b = fresh()
    hide_invisible(m_b)
    rec["whitened_materials"] = whiten_textured(m_b)
    mujoco.mj_forward(m_b, d_b)
    px_b = warp_render(m_b, d_b, args.res, cam_id)
    variants["hide_alpha0+whiten"] = score(px_b, ref)
    imageio.imwrite(os.path.join(args.out, "alpha_hidden_whitened_mjwarp.png"), px_b)

    # C: also drop the partially transparent geoms MJWarp would draw opaque
    m_c, d_c = fresh()
    hide_invisible(m_c, thresh=0.999)
    whiten_textured(m_c)
    mujoco.mj_forward(m_c, d_c)
    px_c = warp_render(m_c, d_c, args.res, cam_id)
    variants["hide_alpha_lt1+whiten"] = score(px_c, ref)

    rec["variants"] = variants
    rec["top_colours"] = {"mujoco": top_colours(ref), "as_is": top_colours(
        warp_render(mjm, mjd, args.res, cam_id)), "fixed": top_colours(px_b)}
    print(json.dumps(variants, indent=1), flush=True)

    # ---- where the residual lives, by geom type -------------------------------------------------
    seg = mj_render(mjm, mjd, args.res, cam_id, seg=True)
    isgeom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    gids = np.where(isgeom, seg[..., 0], -1)
    texid = mjm.mat_texid

    def mat_tex(mi):
        if mi < 0:
            return -1
        row = np.atleast_1d(texid[mi])
        c = [int(x) for x in row if int(x) >= 0]
        return c[0] if c else -1

    tname = {int(getattr(mujoco.mjtGeom, n)): n.replace("mjGEOM_", "")
             for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")}
    err = np.abs(px_b.astype(np.int16) - ref.astype(np.int16)).max(-1)
    npx = args.res * args.res
    by = {}
    for g in np.unique(gids):
        if g < 0:
            continue
        key = f"{tname.get(int(mjm.geom_type[g]), '?')}{'+tex' if mat_tex(int(mjm.geom_matid[g])) >= 0 else ''}"
        mk = gids == g
        v = by.setdefault(key, [0, 0.0, 0])
        v[0] += int(mk.sum()); v[1] += float(err[mk].sum()); v[2] += int((err[mk] == 0).sum())
    tot = max(float(err.sum()), 1.0)
    rec["residual_by_geom_type"] = {
        k: {"px_frac": round(v[0] / npx, 4), "mean_err": round(v[1] / max(v[0], 1), 2),
            "exact_frac": round(v[2] / max(v[0], 1), 4), "share_of_abs_err": round(v[1] / tot, 4)}
        for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])}
    print(json.dumps(rec["residual_by_geom_type"], indent=1), flush=True)

    # ---- generality: more cameras, more poses ----------------------------------------------------
    bank = np.load(args.qpos)
    gen = []
    for cam in args.cams:
        cid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if cid < 0:
            continue
        for qi in range(min(args.nqpos, len(bank))):
            mjm.geom_group[:] = m_b.geom_group  # reuse the fixed grouping for the warp side
            mjd.qpos[:] = bank[qi]; mujoco.mj_forward(mjm, mjd)
            d_b.qpos[:] = bank[qi]; mujoco.mj_forward(m_b, d_b)
            r0 = mj_render(mujoco.MjModel.from_xml_path(args.xml), mjd, args.res, cid) \
                if False else None
            mm, dd = fresh()
            dd.qpos[:] = bank[qi]; mujoco.mj_forward(mm, dd)
            r0 = mj_render(mm, dd, args.res, cid)
            p0 = warp_render(m_b, d_b, args.res, cid)
            s = score(p0, r0); s.update({"cam": cam, "qpos": qi})
            gen.append(s)
            print(json.dumps(s), flush=True)
    rec["generality"] = gen
    if gen:
        rec["generality_summary"] = {
            "mean_of_mean": round(float(np.mean([g["mean"] for g in gen])), 3),
            "mean_exact": round(float(np.mean([g["exact"] for g in gen])), 4),
            "mean_within8": round(float(np.mean([g["within8"] for g in gen])), 4)}

    pad = np.zeros((args.res, 3, 3), np.uint8)
    imageio.imwrite(os.path.join(args.out, "alpha_sheet.png"),
                    np.concatenate([ref, pad, px_a, pad, px_b], axis=1))
    with open(os.path.join(args.out, "alpha.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps({k: rec[k] for k in ("variants", "generality_summary") if k in rec}, indent=2))


if __name__ == "__main__":
    main()
