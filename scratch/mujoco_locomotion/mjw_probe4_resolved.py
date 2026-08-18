"""The same ladder as ``mjw_probe2_parity.py`` with the FIFTH known correction added: MuJoCo resolves
a geom's colour from the geom's own ``rgba`` unless that is the compiler default, and MJWarp resolves
it from the geom's MATERIAL whenever it has one.

Every one of these six models trips it. ``MatPlane`` carries no ``rgba`` (so 1,1,1,1) and the floor
geom carries ``rgba="0.8 0.9 0.8 1"``; ``material="geom"`` likewise sits under body geoms that set
their own colour. Uncorrected, MJWarp paints the whole floor and (on three of the six) the whole body
one material colour, which shows up as a uniform brightness offset -- the floor's mean luminance was
91-99 in the reference and 106-112 in MJWarp -- and would be charged to the texture filter.

The fix is the one ``fetch.py`` uses: recompile from the env's own MJCF with ONE material per geom
carrying the RESOLVED colour and the original specular/shininess/emission/reflectance/texture. The
surgery is then PROVED inert by rendering the surgical model through MuJoCo and requiring it to be
bit-identical to the official model through MuJoCo.

Ladder (both sides always modified identically):
  L0 official . L1 reflectance 0 . L2 L1 + plane texels flattened to their own mean .
  L3 L1 + every mat_texid = -1 . L4 L0 without shadows . L5 L3 + specular 0
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/mujoco_locomotion")

from mjw_common import (agreement, interior_mask, jsave, make_env, mj_render, mj_segment,
                        renderer, sample_qpos, track_cam)

SIZE = 240
NSTATE = 32
DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])


def resolved_model(u):
    """Recompile the env's MJCF with one material per geom carrying MuJoCo's own resolved colour."""
    import mujoco
    spec = mujoco.MjSpec.from_file(u.fullpath)
    base = spec.compile()
    for i in range(base.ngeom):
        spec.add_material(name=f"se:mat{i}")
    m = spec.compile()
    assert m.ngeom == base.ngeom and m.nq == base.nq, "surgery changed the model"
    mat0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "se:mat0")
    for g in range(m.ngeom):
        k = mat0 + g
        old = int(base.geom_matid[g])
        rgba = np.asarray(base.geom_rgba[g], np.float64)
        if old >= 0 and np.allclose(rgba, DEFAULT_RGBA):
            rgba = np.asarray(base.mat_rgba[old], np.float64)
        if old >= 0:
            m.mat_specular[k] = base.mat_specular[old]
            m.mat_shininess[k] = base.mat_shininess[old]
            m.mat_emission[k] = base.mat_emission[old]
            m.mat_reflectance[k] = base.mat_reflectance[old]
            m.mat_texid[k] = base.mat_texid[old]
            m.mat_texrepeat[k] = base.mat_texrepeat[old]
            m.mat_texuniform[k] = base.mat_texuniform[old]
        else:
            m.mat_specular[k] = 0.5
            m.mat_shininess[k] = 0.5
            m.mat_emission[k] = 0.0
            m.mat_reflectance[k] = 0.0
            m.mat_texid[k] = -1
        m.mat_rgba[k] = rgba
        m.geom_rgba[g] = rgba
        m.geom_matid[g] = k
    m.vis.quality.offsamples = int(base.vis.quality.offsamples)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 4 * SIZE)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 4 * SIZE)
    return m, mat0


def newton(m):
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
    adr = int(m.tex_adr[texid])
    w, h, c = int(m.tex_width[texid]), int(m.tex_height[texid]), int(m.tex_nchannel[texid])
    n = w * h * c
    old = np.array(m.tex_data[adr:adr + n], copy=True)
    px = old.reshape(-1, c)
    m.tex_data[adr:adr + n] = np.tile(np.round(px.mean(0)).astype(np.uint8), (w * h, 1)).reshape(-1)
    return old


def attribute(got, ref, seg, m):
    import mujoco
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    floor = isgeom & np.isin(objid, plane)
    classes = {"background": ~isgeom, "floor": floor, "bodies": isgeom & ~floor}
    im = interior_mask(ref)
    dif = np.abs(got.astype(np.int16) - ref.astype(np.int16)).max(-1)
    di = dif[:, :-1, :-1]
    tot = float(di[im].astype(np.float64).sum()) or 1.0
    out = {}
    for tag, msk in classes.items():
        mm = msk[:, :-1, :-1] & im
        out[tag] = dict(
            frame_frac=round(float(msk.mean()), 5),
            all_exact=round(float((dif[msk] == 0).mean()), 6) if msk.any() else None,
            interior_share=round(float(mm.sum() / max(im.sum(), 1)), 5),
            interior_exact=round(float((di[mm] == 0).mean()), 6) if mm.any() else None,
            interior_mean=round(float(di[mm].mean()), 4) if mm.any() else None,
            share_of_interior_error=round(float(di[mm].astype(np.float64).sum() / tot), 5))
    return out


def run(env_id, size=SIZE, n=NSTATE):
    import mujoco

    out = {"env": env_id, "size": size, "n_states": n}
    env, u = make_env(env_id, size)
    om, od = u.model, u.data
    cam = track_cam(om)
    st = sample_qpos(env_id, n, seed=0)

    m, mat0 = resolved_model(u)
    d = mujoco.MjData(m)

    # The surgery must change nothing MuJoCo draws. If it does, every number below is measured
    # against a scene the official env never renders.
    ref_official = mj_render(om, od, cam, size, st, tag="official")
    ref_surgical = mj_render(m, d, cam, size, st, tag="surgicalL0")
    out["surgery_is_inert"] = agreement(ref_surgical, ref_official)

    refl0 = np.array(m.mat_reflectance, copy=True)
    texid0 = np.array(m.mat_texid, copy=True)
    spec0 = np.array(m.mat_specular, copy=True)
    plane_tex = sorted({int(t) for g in range(m.ngeom) if int(m.geom_type[g]) == 0
                        for t in np.asarray(m.mat_texid[mat0 + g]).reshape(-1) if int(t) >= 0})
    out["plane_textures"] = plane_tex

    from stable_envs.render import mjwarp as MW
    out["texture_gaps"] = [list(x) for x in MW.texture_gaps(m, (0, 1, 2))]

    ladder, attr = {}, {}
    seg = mj_segment(m, d, cam, size, st, tag="seg")

    W = build_mjw(m, cam, size, n)
    got = W.render(st)
    ladder["L0_official"] = agreement(got, ref_surgical)
    attr["L0_official"] = attribute(got, ref_surgical, seg, m)

    m.mat_reflectance[:] = 0.0
    ref1 = mj_render(m, d, cam, size, st, tag="L1")
    g1 = build_mjw(m, cam, size, n).render(st)
    ladder["L1_no_reflectance"] = agreement(g1, ref1)
    attr["L1_no_reflectance"] = attribute(g1, ref1, seg, m)

    if plane_tex:
        saved = {t: flat_texture(m, t) for t in plane_tex}
        ref2 = mj_render(m, d, cam, size, st, tag="L2")
        g2 = build_mjw(m, cam, size, n).render(st)
        ladder["L2_flat_plane_texture"] = agreement(g2, ref2)
        attr["L2_flat_plane_texture"] = attribute(g2, ref2, seg, m)
        for t, old in saved.items():
            adr = int(m.tex_adr[t])
            m.tex_data[adr:adr + old.size] = old

    m.mat_texid[:] = -1
    ref3 = mj_render(m, d, cam, size, st, tag="L3")
    g3 = build_mjw(m, cam, size, n).render(st)
    ladder["L3_no_textures"] = agreement(g3, ref3)
    attr["L3_no_textures"] = attribute(g3, ref3, seg, m)

    m.mat_specular[:] = 0.0
    ref5 = mj_render(m, d, cam, size, st, tag="L5")
    g5 = build_mjw(m, cam, size, n).render(st)
    ladder["L5_no_textures_no_specular"] = agreement(g5, ref5)
    attr["L5_no_textures_no_specular"] = attribute(g5, ref5, seg, m)
    m.mat_specular[:] = spec0
    m.mat_texid[:] = texid0
    m.mat_reflectance[:] = refl0

    r = renderer(m, size, "L4")
    ref4 = []
    for q in st:
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=int(cam))
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        ref4.append(np.ascontiguousarray(r.render()))
    ref4 = np.stack(ref4)
    g4 = build_mjw(m, cam, size, n, shadows=False).render(st)
    ladder["L4_no_shadows"] = agreement(g4, ref4)

    out["ladder"] = ladder
    out["attribution"] = attr

    # The floor near/far bands at L0, which is where the minification claim is decided.
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    floor = isgeom & np.isin(objid, plane)
    rows = np.nonzero(floor.any(0).any(1))[0]
    if rows.size > 20:
        lo, hi = int(rows.min()), int(rows.max())
        band = max(1, (hi - lo) // 10)
        near = floor.copy(); near[:, :hi - band] = False
        far = floor.copy(); far[:, lo + band:] = False
        im = interior_mask(ref_surgical)
        dif = np.abs(got.astype(np.int16) - ref_surgical.astype(np.int16)).max(-1)
        bands = {}
        for tag, msk in (("near", near), ("far", far)):
            mm = msk[:, :-1, :-1] & im
            lum_r = ref_surgical.astype(np.float64).mean(-1)[msk]
            lum_g = got.astype(np.float64).mean(-1)[msk]
            bands[tag] = dict(
                frame_frac=round(float(msk.mean()), 5),
                all_exact=round(float((dif[msk] == 0).mean()), 6),
                interior_exact=round(float((dif[:, :-1, :-1][mm] == 0).mean()), 6) if mm.any() else None,
                ref_mean=round(float(lum_r.mean()), 2), ref_std=round(float(lum_r.std()), 2),
                mjw_mean=round(float(lum_g.mean()), 2), mjw_std=round(float(lum_g.std()), 2))
        out["floor_bands_L0"] = bands
    return out


if __name__ == "__main__":
    e = sys.argv[1]
    sz = int(sys.argv[2]) if len(sys.argv) > 2 else SIZE
    r = run(e, size=sz)
    jsave(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/"
          f"probe4_{e.split('-')[0]}_{sz}.json", r)
    print("\n===", e)
    print("  surgery_is_inert:", r["surgery_is_inert"])
    print("  texture_gaps:", len(r["texture_gaps"]), r["texture_gaps"][:4])
    for k, v in r["ladder"].items():
        print(f"  {k:30s} all={v['exact_frac']:.4f} int={v['interior_exact']} "
              f"mean={v['mean_abs']:.3f} w8={v['within8_frac']:.4f} intfrac={v['interior_frac']}")
    for k, v in r["attribution"].items():
        print(" ", k)
        for c, s in v.items():
            print(f"     {c:11s} frame={s['frame_frac']:.4f} all_exact={s['all_exact']} "
                  f"int_share={s['interior_share']:.4f} int_exact={s['interior_exact']} "
                  f"int_mean={s['interior_mean']} err_share={s['share_of_interior_error']}")
    if "floor_bands_L0" in r:
        for b, v in r["floor_bands_L0"].items():
            print("  band", b, v)
