"""Why MJWarp renders RoboCasa's kitchen flat, and whether baking fixes it.

The prior write-up recorded "MJWarp is largely not applying RoboCasa's materials" as an unexplained
residual. `mujoco_warp/_src/render.py::sample_texture` explains it: it derives UVs for exactly two
geom types.

    if geom_type[geom_id] == GeomType.PLANE:  ...procedural texgen...
    if geom_type[geom_id] == GeomType.MESH:   ...barycentric interpolation of mesh_texcoord...

For every other type -- BOX, SPHERE, CYLINDER, CAPSULE, ELLIPSOID -- `uv` stays (0, 0), so the whole
surface samples the texture's corner texel and comes out one flat colour.

That is real and is confirmed below, but it is NOT what made the RoboCasa frame green: see
`probe_green.py` and `probe_alpha.py`, which trace that to MJWarp drawing zero-alpha marker geoms
opaque. Read this probe for the texgen gap and the bake, not for the headline failure.

This probe tests that mechanistically rather than by inference:

  1. a minimal two-geom scene (textured box, textured plane) rendered by both, so the failure is
     visible with nothing else in the frame;
  2. the same box re-expressed as an explicit mesh with cube texcoords -- if MJWarp then matches,
     the fix is a bake, not an upstream renderer change;
  3. on the real kitchen: what fraction of visible pixels each geom type owns, and how the
     MJWarp-vs-MuJoCo error splits across that partition;
  4. the BGRA channel re-check, scored independently of the earlier probes;
  5. a flat mean-colour bake, priced from both sides -- what MuJoCo loses by baking, and whether
     MJWarp reproduces the baked scene.

Run in the se3d venv.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile

os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def score(a, b):
    e = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return {"mean": float(e.mean()), "max": int(e.max()),
            "exact": float((e == 0).mean()), "within8": float((e <= 8).mean()),
            "within32": float((e <= 32).mean())}


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


def warp_render(mjm, mjd, res, cam_id, groups=(1,), textures=True, read="get_rgb"):
    """One-world MJWarp render. `read` in {get_rgb, bgra_swap, raw_bgr}."""
    import warp as wp
    import mujoco_warp as mjw

    m = mjw.put_model(mjm)
    d = mjw.put_data(mjm, mjd, nworld=1)
    rc = mjw.create_render_context(
        mjm, nworld=1, cam_res=(res, res), render_rgb=True, render_depth=False,
        render_seg=False, use_textures=textures, use_shadows=False, render_skybox=True,
        enabled_geom_groups=list(groups),
        cam_active=[i == cam_id for i in range(mjm.ncam)],
    )
    mjw.kinematics(m, d)
    mjw.camlight(m, d)
    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)
    wp.synchronize()
    if read == "get_rgb":
        out = wp.zeros((1, res, res), dtype=wp.vec3)
        mjw.get_rgb(rc, 0, out)
        wp.synchronize()
        px = np.clip(out.numpy()[0] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        raw = rc.rgb_data.numpy()[: res * res].view(np.uint8).reshape(res, res, 4)
        px = raw[..., [2, 1, 0]] if read == "bgra_swap" else raw[..., :3]
    del rc, d, m
    return np.ascontiguousarray(px)


MINI = """
<mujoco model="texgen_repro">
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="64" height="64"
             rgb1="0.9 0.1 0.1" rgb2="0.1 0.1 0.9"/>
    <material name="checkmat" texture="checker" texrepeat="1 1" texuniform="false"
              specular="0" shininess="0" reflectance="0"/>
    <material name="graymat" rgba="0.6 0.6 0.6 1" specular="0" shininess="0" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="graymat" group="1"/>
    <geom name="tbox" type="box" pos="0 0 0.5" size="0.5 0.5 0.5" material="checkmat" group="1"/>
    <camera name="cam" pos="2.2 -2.2 1.9" xyaxes="0.707 0.707 0 -0.4 0.4 0.82"/>
  </worldbody>
</mujoco>
"""

# Same scene, box re-expressed as an explicit 12-triangle mesh with per-face cube texcoords.
MESH_BOX = """
<mujoco model="texgen_repro_mesh">
  <asset>
    <texture name="checker" type="2d" builtin="checker" width="64" height="64"
             rgb1="0.9 0.1 0.1" rgb2="0.1 0.1 0.9"/>
    <material name="checkmat" texture="checker" texrepeat="1 1" texuniform="false"
              specular="0" shininess="0" reflectance="0"/>
    <material name="graymat" rgba="0.6 0.6 0.6 1" specular="0" shininess="0" reflectance="0"/>
    <mesh name="cube" vertex="{verts}" face="{faces}" texcoord="{tcs}"/>
  </asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="graymat" group="1"/>
    <geom name="tbox" type="mesh" mesh="cube" pos="0 0 0.5" material="checkmat" group="1"/>
    <camera name="cam" pos="2.2 -2.2 1.9" xyaxes="0.707 0.707 0 -0.4 0.4 0.82"/>
  </worldbody>
</mujoco>
"""


def _cube_mesh(half=0.5):
    """Unit cube as 24 verts (4 per face) with per-face [0,1]^2 texcoords, 12 triangles."""
    axes = [
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),   # +X
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),  # -X
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),  # +Y
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),  # -Y
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),   # +Z
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),  # -Z
    ]
    verts, faces, tcs = [], [], []
    for n, u, v in axes:
        n, u, v = np.array(n, float), np.array(u, float), np.array(v, float)
        base = len(verts)
        for su, sv, tu, tv in ((-1, -1, 0, 0), (1, -1, 1, 0), (1, 1, 1, 1), (-1, 1, 0, 1)):
            verts.append(half * (n + su * u + sv * v))
            tcs.append((tu, tv))
        faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    return (" ".join(f"{c:.6f}" for p in verts for c in p),
            " ".join(str(i) for f in faces for i in f),
            " ".join(f"{c:.6f}" for t in tcs for c in t))


def bake_flat(mjm):
    """Replace every textured material with the mean colour of its texture. Returns n changed."""
    texid = mjm.mat_texid
    n = 0
    for mi in range(mjm.nmat):
        row = texid[mi] if texid.ndim > 1 else np.array([texid[mi]])
        ti = int(row[1]) if row.size > 1 else int(row[0])
        if ti < 0 and row.size > 1:
            cand = [int(x) for x in row if int(x) >= 0]
            ti = cand[0] if cand else -1
        if ti < 0:
            continue
        adr, w, h, nc = (int(mjm.tex_adr[ti]), int(mjm.tex_width[ti]),
                         int(mjm.tex_height[ti]), int(mjm.tex_nchannel[ti]))
        data = mjm.tex_data[adr: adr + w * h * nc].reshape(h, w, nc).astype(np.float32) / 255.0
        mean = data[..., :3].reshape(-1, min(3, nc)).mean(0)
        if mean.size < 3:
            mean = np.repeat(mean, 3)[:3]
        mjm.mat_rgba[mi, :3] = mean
        if texid.ndim > 1:
            texid[mi, :] = -1
        else:
            texid[mi] = -1
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=os.path.join(HERE, "out", "pinned_scene.xml"))
    ap.add_argument("--qpos", default=os.path.join(HERE, "out", "qpos_bank.npy"))
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--cam", default="robot0_agentview_left")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    import imageio.v2 as imageio
    rec = {"mujoco": mujoco.__version__}

    # ---- 1. minimal repro: textured box + textured plane ---------------------------------------
    mini = {}
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "mini.xml")
        open(p, "w").write(MINI)
        mm = mujoco.MjModel.from_xml_path(p)
        md = mujoco.MjData(mm)
        mujoco.mj_forward(mm, md)
        ref = mj_render(mm, md, args.res, 0)
        got = warp_render(mm, md, args.res, 0)
        mini["box_primitive"] = score(got, ref)
        imageio.imwrite(os.path.join(args.out, "texgen_mini_mujoco.png"), ref)
        imageio.imwrite(os.path.join(args.out, "texgen_mini_mjwarp.png"), got)

        # how many distinct colours does each renderer paint on the box?
        seg = mj_render(mm, md, args.res, 0, seg=True)
        gid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, "tbox")
        mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == gid)
        mini["box_px"] = int(mask.sum())
        mini["box_unique_colours_mujoco"] = int(len(np.unique(ref[mask], axis=0)))
        mini["box_unique_colours_mjwarp"] = int(len(np.unique(got[mask], axis=0)))
        mini["box_only"] = score(got[mask], ref[mask])
        floor = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        fmask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg[..., 0] == floor)
        mini["floor_only"] = score(got[fmask], ref[fmask])

        # ---- 2. same box as a mesh with cube texcoords -----------------------------------------
        v, f, t = _cube_mesh()
        p2 = os.path.join(td, "mesh.xml")
        open(p2, "w").write(MESH_BOX.format(verts=v, faces=f, tcs=t))
        mm2 = mujoco.MjModel.from_xml_path(p2)
        md2 = mujoco.MjData(mm2)
        mujoco.mj_forward(mm2, md2)
        ref2 = mj_render(mm2, md2, args.res, 0)
        got2 = warp_render(mm2, md2, args.res, 0)
        gid2 = mujoco.mj_name2id(mm2, mujoco.mjtObj.mjOBJ_GEOM, "tbox")
        seg2 = mj_render(mm2, md2, args.res, 0, seg=True)
        mask2 = (seg2[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (seg2[..., 0] == gid2)
        mini["box_as_mesh"] = score(got2, ref2)
        mini["box_as_mesh_box_only"] = score(got2[mask2], ref2[mask2])
        mini["box_as_mesh_unique_colours_mjwarp"] = int(len(np.unique(got2[mask2], axis=0)))
        mini["box_as_mesh_unique_colours_mujoco"] = int(len(np.unique(ref2[mask2], axis=0)))
        # does the mesh bake also reproduce the ORIGINAL primitive-box MuJoCo frame?
        mini["meshbox_mjwarp_vs_primbox_mujoco"] = score(got2, ref)
        imageio.imwrite(os.path.join(args.out, "texgen_mesh_mujoco.png"), ref2)
        imageio.imwrite(os.path.join(args.out, "texgen_mesh_mjwarp.png"), got2)
    rec["minimal_repro"] = mini
    print(json.dumps({"minimal_repro": mini}, indent=2), flush=True)

    # ---- 3. real kitchen: pixel share by geom type ---------------------------------------------
    mjm = mujoco.MjModel.from_xml_path(args.xml)
    mjd = mujoco.MjData(mjm)
    mjd.qpos[:] = np.load(args.qpos)[0]
    mujoco.mj_forward(mjm, mjd)
    cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, args.cam)
    if cam_id < 0:
        cam_id = 0

    ref = mj_render(mjm, mjd, args.res, cam_id)
    seg = mj_render(mjm, mjd, args.res, cam_id, seg=True)
    isgeom = seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM
    gids = np.where(isgeom, seg[..., 0], -1)

    texid = mjm.mat_texid
    def mat_tex(mi):
        if mi < 0:
            return -1
        row = texid[mi] if texid.ndim > 1 else np.array([texid[mi]])
        if row.size > 1:
            cand = [int(x) for x in row if int(x) >= 0]
            return cand[0] if cand else -1
        return int(row[0])

    tname = {int(getattr(mujoco.mjtGeom, n)): n.replace("mjGEOM_", "")
             for n in dir(mujoco.mjtGeom) if n.startswith("mjGEOM_")}
    share = {}
    total = int(isgeom.sum())
    for g in np.unique(gids):
        if g < 0:
            continue
        gt = tname.get(int(mjm.geom_type[g]), str(int(mjm.geom_type[g])))
        textured = mat_tex(int(mjm.geom_matid[g])) >= 0
        key = f"{gt}{'+tex' if textured else ''}"
        share[key] = share.get(key, 0) + int((gids == g).sum())
    npx = args.res * args.res
    rec["kitchen_pixel_share"] = {
        "visible_geom_px_frac": total / npx,
        "background_px_frac": 1.0 - total / npx,
        "by_type_frac_of_frame": {k: v / npx for k, v in sorted(share.items(), key=lambda kv: -kv[1])},
    }
    tex_nonmesh = sum(v for k, v in share.items()
                      if k.endswith("+tex") and not k.startswith("MESH") and not k.startswith("PLANE"))
    rec["kitchen_pixel_share"]["textured_non_mesh_non_plane_frac"] = tex_nonmesh / npx

    # ---- 4. BGRA re-check, and where the error lives --------------------------------------------
    reads = {}
    for mode in ("raw_bgr", "bgra_swap", "get_rgb"):
        px = warp_render(mjm, mjd, args.res, cam_id, read=mode)
        reads[mode] = score(px, ref)
        if mode == "get_rgb":
            best_px = px
    rec["bgra_recheck"] = reads

    err = np.abs(best_px.astype(np.int16) - ref.astype(np.int16)).max(-1)
    by = {}
    for g in np.unique(gids):
        if g < 0:
            continue
        gt = tname.get(int(mjm.geom_type[g]), str(int(mjm.geom_type[g])))
        textured = mat_tex(int(mjm.geom_matid[g])) >= 0
        key = f"{gt}{'+tex' if textured else ''}"
        mm_ = gids == g
        a = by.setdefault(key, [0, 0.0, 0])
        a[0] += int(mm_.sum())
        a[1] += float(err[mm_].sum())
        a[2] += int((err[mm_] == 0).sum())
    rec["error_by_geom_type"] = {
        k: {"px_frac": v[0] / npx, "mean_err": v[1] / max(v[0], 1), "exact_frac": v[2] / max(v[0], 1),
            "share_of_total_abs_err": v[1] / max(float(err.sum()), 1.0)}
        for k, v in sorted(by.items(), key=lambda kv: -kv[1][1])
    }
    imageio.imwrite(os.path.join(args.out, "texgen_kitchen_mujoco.png"), ref)
    imageio.imwrite(os.path.join(args.out, "texgen_kitchen_mjwarp.png"), best_px)

    # ---- 5. flat mean-colour bake ---------------------------------------------------------------
    mjm_b = mujoco.MjModel.from_xml_path(args.xml)
    mjd_b = mujoco.MjData(mjm_b)
    mjd_b.qpos[:] = np.load(args.qpos)[0]
    n_baked = bake_flat(mjm_b)
    mujoco.mj_forward(mjm_b, mjd_b)
    ref_baked = mj_render(mjm_b, mjd_b, args.res, cam_id)
    warp_baked = warp_render(mjm_b, mjd_b, args.res, cam_id, textures=False)
    rec["bake"] = {
        "materials_baked": n_baked,
        "nmat": int(mjm.nmat),
        "mujoco_baked_vs_mujoco_textured": score(ref_baked, ref),
        "mjwarp_baked_vs_mujoco_baked": score(warp_baked, ref_baked),
        "mjwarp_baked_vs_mujoco_textured": score(warp_baked, ref),
    }
    imageio.imwrite(os.path.join(args.out, "texgen_kitchen_baked_mujoco.png"), ref_baked)
    imageio.imwrite(os.path.join(args.out, "texgen_kitchen_baked_mjwarp.png"), warp_baked)
    pad = np.zeros((args.res, 3, 3), np.uint8)
    imageio.imwrite(os.path.join(args.out, "texgen_sheet.png"),
                    np.concatenate([ref, pad, best_px, pad, ref_baked, pad, warp_baked], axis=1))

    print(json.dumps(rec, indent=2))
    with open(os.path.join(args.out, "texgen.json"), "w") as f:
        json.dump(rec, f, indent=2)


if __name__ == "__main__":
    main()
