"""MJWarp against MuJoCo on the six locomotion domains, with every known correction applied, and an
ablation ladder that says WHICH of the three declared blockers the residual is.

The reference is ``mujoco.Renderer`` at the model's ``track`` camera, which ``mjw_probe1_facts.py``
measures BIT-IDENTICAL to ``gymnasium``'s own frame (exact_frac 1.0, max_abs 0, on all six). That
matters: it is what lets an ablation modify the model and still be comparable to the headline.

Corrections applied to MJWarp, none optional:
  gl_rounding() . prepare_model() for geoms AND sites . BGRA->RGB . shadows ON . skybox ON .
  geom groups 0-2 on both sides . com_pos in the frame loop (trackcom reads subtree_com).

The ladder, applied IDENTICALLY to both renderers:
  L0  the official scene
  L1  mat_reflectance = 0                 (the mirror off)
  L2  L1 + the checker texels overwritten with their own mean  (the minification filter defeated,
                                                                by making the texture constant)
  L3  L1 + every mat_texid = -1           (a flat-material scene: the regime backend3d.md calls
                                           parity at 98.9% interior)
  L4  L0 with shadows off on both sides   (does the shadow map explain anything?)

Per-class attribution on L0 uses MuJoCo's own segmentation buffer, and the floor is split into a NEAR
and a FAR band by image row, because the minification hypothesis makes a specific prediction: the far
band should be much worse than the near band, and it should be worse in a particular way -- MJWarp
point-samples one texel per ray, so its far field keeps the checker's contrast while MuJoCo's has been
averaged to one grey.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/mujoco_locomotion")

from mjw_common import (ENVS, agreement, interior_mask, jsave, make_env, mj_render, mj_segment,
                        sample_qpos, track_cam)

SIZE = 240
NSTATE = 32


def newton(m):
    """PGS is a CONSTRAINT SOLVER and reaches no pixel; ``mujoco_warp.put_model`` refuses it.

    Humanoid-v5 is the only one of the six that sets it. Swapping it for Newton on the render-only
    model is the same class of edit as Fetch zeroing ``geom_margin``: a field the rasterizer never
    reads, changed so the backend will accept the model at all.
    """
    import mujoco
    old = int(m.opt.solver)
    m.opt.solver = int(mujoco.mjtSolver.mjSOL_NEWTON)
    return old


def build_mjw(m, cam, res, nworld, **kw):
    from mjw_common import MjwRenderer
    old = newton(m)
    try:
        return MjwRenderer(m, cam, res, nworld, **kw)
    finally:
        m.opt.solver = old


def flat_texture(m, texid):
    """Overwrite one texture's texels with their own mean. Returns the original bytes."""
    adr = int(m.tex_adr[texid])
    w, h, c = int(m.tex_width[texid]), int(m.tex_height[texid]), int(m.tex_nchannel[texid])
    n = w * h * c
    old = np.array(m.tex_data[adr:adr + n], copy=True)
    px = old.reshape(-1, c)
    m.tex_data[adr:adr + n] = np.tile(np.round(px.mean(0)).astype(np.uint8), (w * h, 1)).reshape(-1)
    return old


def class_masks(seg, m):
    """(background, floor, bodies) boolean masks from MuJoCo's segmentation buffer."""
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    floor = isgeom & np.isin(objid, plane)
    bg = ~isgeom
    return bg, floor, isgeom & ~floor


def band_stats(img, mask):
    lum = img.astype(np.float64).mean(-1)
    v = lum[mask]
    if v.size < 16:
        return None
    return dict(n=int(v.size), mean=round(float(v.mean()), 2), std=round(float(v.std()), 2),
                p5=round(float(np.percentile(v, 5)), 1), p95=round(float(np.percentile(v, 95)), 1),
                ncolors=int(len(np.unique(np.round(lum[mask], 3)))))


def run(env_id, size=SIZE, n=NSTATE):
    import mujoco

    out = {"env": env_id, "size": size, "n_states": n}
    env, u = make_env(env_id, size)
    m, d = u.model, u.data
    cam = track_cam(m)
    st = sample_qpos(env_id, n, seed=0)

    refl0 = np.array(m.mat_reflectance, copy=True)
    texid0 = np.array(m.mat_texid, copy=True)
    plane_mat = [i for i in range(int(m.nmat)) if float(refl0[i]) > 0]
    plane_tex = sorted({int(t) for i in plane_mat for t in np.asarray(m.mat_texid[i]).reshape(-1)
                        if int(t) >= 0})

    ladder = {}

    # ---------------- L0: the official scene ------------------------------------------------
    ref = mj_render(m, d, cam, size, st, tag="L0")
    W = build_mjw(m, cam, size, n)
    got = W.render(st)
    ladder["L0_official"] = agreement(got, ref)
    out["texture_gaps"] = [list(x) for x in W.texture_gaps]
    out["hidden_by_prepare"] = [len(W.prepared.hidden_geoms), len(W.prepared.hidden_sites)]

    # per-class attribution, and the floor split near/far
    seg = mj_segment(m, d, cam, size, st, tag="L0seg")
    bg, floor, body = class_masks(seg, m)
    im = interior_mask(ref)
    dif = np.abs(got.astype(np.int16) - ref.astype(np.int16)).max(-1)
    di = dif[:, :-1, :-1]
    tot_err = float(di[im].astype(np.float64).sum()) or 1.0
    cls = {}
    for tag, msk in (("background", bg), ("floor", floor), ("bodies", body)):
        mm = msk[:, :-1, :-1] & im
        cls[tag] = dict(
            frame_frac=round(float(msk.mean()), 5),
            all_exact=round(float((dif[msk] == 0).mean()), 6) if msk.any() else None,
            interior_share=round(float(mm.sum() / max(im.sum(), 1)), 5),
            interior_exact=round(float((di[mm] == 0).mean()), 6) if mm.any() else None,
            interior_mean=round(float(di[mm].mean()), 4) if mm.any() else None,
            share_of_interior_error=round(float(di[mm].astype(np.float64).sum() / tot_err), 5))
    out["by_class_L0"] = cls

    rows = np.nonzero(floor.any(0).any(1))[0] if floor.any() else np.array([])
    if rows.size > 20:
        lo, hi = int(rows.min()), int(rows.max())
        band = max(1, (hi - lo) // 10)
        near = floor.copy(); near[:, :hi - band] = False
        far = floor.copy(); far[:, lo + band:] = False
        fb = {}
        for tag, msk in (("near", near), ("far", far)):
            mm = msk[:, :-1, :-1] & im
            fb[tag] = dict(
                frame_frac=round(float(msk.mean()), 5),
                all_exact=round(float((dif[msk] == 0).mean()), 6) if msk.any() else None,
                interior_exact=round(float((di[mm] == 0).mean()), 6) if mm.any() else None,
                interior_mean=round(float(di[mm].mean()), 4) if mm.any() else None,
                reference=band_stats(ref, msk), mjwarp=band_stats(got, msk))
        out["floor_bands_L0"] = fb

    # ---------------- L1: the mirror off ----------------------------------------------------
    m.mat_reflectance[:] = 0.0
    ref1 = mj_render(m, d, cam, size, st, tag="L1")
    W1 = build_mjw(m, cam, size, n)
    ladder["L1_no_reflectance"] = agreement(W1.render(st), ref1)

    # ---------------- L2: the checker made constant -----------------------------------------
    if plane_tex:
        saved = {t: flat_texture(m, t) for t in plane_tex}
        ref2 = mj_render(m, d, cam, size, st, tag="L2")
        W2 = build_mjw(m, cam, size, n)
        ladder["L2_flat_plane_texture"] = agreement(W2.render(st), ref2)
        for t, old in saved.items():
            adr = int(m.tex_adr[t])
            m.tex_data[adr:adr + old.size] = old

    # ---------------- L3: no textures at all ------------------------------------------------
    m.mat_texid[:] = -1
    ref3 = mj_render(m, d, cam, size, st, tag="L3")
    W3 = build_mjw(m, cam, size, n)
    ladder["L3_no_textures"] = agreement(W3.render(st), ref3)
    m.mat_texid[:] = texid0
    m.mat_reflectance[:] = refl0

    # ---------------- L4: shadows off both sides --------------------------------------------
    r = __import__("mjw_common", fromlist=["x"]).renderer(m, size, "L4")
    r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    ref4 = []
    for q in st:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=int(cam))
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        ref4.append(np.ascontiguousarray(r.render()))
    ref4 = np.stack(ref4)
    W4 = build_mjw(m, cam, size, n, shadows=False)
    ladder["L4_no_shadows"] = agreement(W4.render(st), ref4)

    out["ladder"] = ladder
    return out


if __name__ == "__main__":
    ids = sys.argv[1:] or ENVS
    res = []
    for e in ids:
        try:
            res.append(run(e))
        except Exception as ex:                                        # noqa: BLE001
            import traceback
            traceback.print_exc()
            res.append({"env": e, "error": f"{type(ex).__name__}: {ex}"})
        print("done", e, flush=True)
    tag = "_".join(x.split("-")[0] for x in ids) if len(ids) < 6 else "all"
    jsave(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/probe2_parity_{tag}.json", res)
    for r in res:
        print("\n===", r["env"])
        if "error" in r:
            print("  ", r["error"])
            continue
        for k, v in r["ladder"].items():
            print(f"  {k:26s} all={v['exact_frac']:.4f} int={v['interior_exact']} "
                  f"mean={v['mean_abs']:.3f} w8={v['within8_frac']:.4f} intfrac={v['interior_frac']}")
        print("  by class:", r["by_class_L0"])
        if "floor_bands_L0" in r:
            print("  floor bands:", r["floor_bands_L0"])
