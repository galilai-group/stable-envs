"""Nail the colour model: per-face-class statistics, a correct per-flag ablation, and the shadow term.

``shade.py`` established that wall pixels already fall out of a plain Blinn-Phong evaluation to <1/255,
and that the floor does not -- by ~157/255. Two candidate causes: the floor's base colour comes from a
*texture* (the decoloured 'grid' texture), not from ``mat_rgba``; and MuJoCo's shadow pass moves 66% of
the pixels. Both are measured here rather than argued about:

  * the observed shading scalar ``obs / base_colour`` is tabulated per face class, so the actual value of
    ``ambient + Σ diffuse`` is read off the image instead of derived from the light list;
  * the shadow render is diffed against the no-shadow render and dumped as an image, so its *spatial*
    structure (a real occlusion, or a shadow-map coverage artefact) is visible rather than inferred.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_scene import make  # noqa: E402
import raycast as RC  # noqa: E402
from shade import normals  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF


def base_colour(m, g):
    """The (r,g,b) MuJoCo modulates with light for geom ``g``: texture if the material has one, else rgba."""
    mid = int(m.geom_matid[g])
    rgba = m.geom_rgba[g].astype(np.float64)[:3]
    tex = None
    if mid >= 0:
        mr = m.mat_rgba[mid].astype(np.float64)[:3]
        tid = m.mat_texid[mid]
        tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
        if np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
            rgba = mr
        if tid >= 0:
            attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
            buf = getattr(m, attr)
            a0 = int(m.tex_adr[tid])
            h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
            reg = buf[a0:a0 + 3 * h * w].reshape(h, w, 3).astype(np.float64) / 255.0
            tex = reg
            rgba = rgba * reg.reshape(-1, 3).mean(0)
    return rgba, tex


def run(loco, size, img=224, seed=0):
    import mujoco
    import imageio.v2 as imageio
    env, u = make(loco, size, img)
    m, d = u.model, u.data
    env.reset(options={"task_id": 1})
    for _ in range(20):
        env.step(u.action_space.sample())
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    base_flags = np.array(r.scene.flags, copy=True)
    obs = r.render().astype(np.float64)

    res = dict(env=f"{loco}maze-{size}", img=img)

    # ---- per-flag ablation, baseline restored every time ---------------------
    abl = {}
    for nm in ["mjRND_SHADOW", "mjRND_REFLECTION", "mjRND_SKYBOX", "mjRND_FOG", "mjRND_HAZE",
               "mjRND_CULL_FACE", "mjRND_WIREFRAME", "mjRND_ADDITIVE"]:
        f = getattr(mujoco.mjtRndFlag, nm, None)
        if f is None:
            continue
        r.update_scene(d, camera=u.custom_camera)
        for k in range(len(base_flags)):
            r.scene.flags[k] = int(base_flags[k])
        r.scene.flags[int(f)] = 0 if base_flags[int(f)] else 1
        im2 = r.render().astype(np.float64)
        dd = np.abs(im2 - obs)
        abl[nm] = dict(default_on=bool(base_flags[int(f)]),
                       frac=float((dd.max(-1) > 0).mean()), mean=float(dd.mean()), max=float(dd.max()))
        if nm == "mjRND_SHADOW":
            noshadow = im2.copy()
    res["flag_ablation"] = abl
    for k in range(len(base_flags)):
        r.scene.flags[k] = int(base_flags[k])

    # ---- geometry -------------------------------------------------------------
    cam = RC.avgcam(RC.glcam(r.scene, 0), RC.glcam(r.scene, 1))
    gid, t, fwd, o, dr = RC.cast(m, d, cam, img, img)
    gid = np.where(t >= INF / 2, -1, gid)
    p = o + t[..., None] * dr
    nrm = normals(m, d, gid, p)
    names = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}

    cls = np.full(gid.shape, "bg", dtype=object)
    is_wall = np.isin(gid, [g for g in range(m.ngeom) if (names[g] or "").startswith("block_")])
    cls[gid == 0] = "floor"
    cls[is_wall & (nrm[..., 2] > 0.5)] = "wall_top"
    cls[is_wall & (nrm[..., 2] <= 0.5)] = "wall_side"
    agent = [g for g in range(m.ngeom)
             if not (names[g] or "").startswith(("block_", "teleport")) and names[g] != "floor"]
    cls[np.isin(gid, agent)] = "agent"
    cls[np.char.startswith(np.array([names.get(int(x), "bg") or "bg" for x in gid.ravel()]).reshape(gid.shape),
                           "teleport")] = "teleport"

    # ---- observed shading scalar per class -----------------------------------
    bc = np.zeros((img, img, 3))
    for g in np.unique(gid):
        if g < 0:
            continue
        c, _ = base_colour(m, int(g))
        bc[gid == g] = c
    res["base_colours"] = {names.get(int(g), "bg"): [round(x, 4) for x in base_colour(m, int(g))[0]]
                           for g in np.unique(gid) if g >= 0}
    if len(res["base_colours"]) > 8:
        keep = ["floor"] + [k for k in res["base_colours"] if not k.startswith("block_")][:8]
        blocks = [k for k in res["base_colours"] if k.startswith("block_")][:2]
        res["base_colours"] = {k: v for k, v in res["base_colours"].items() if k in keep + blocks}

    out_cls = {}
    for c in ["floor", "wall_top", "wall_side", "agent", "teleport", "bg"]:
        msk = cls == c
        if not msk.any():
            continue
        v = obs[msk]
        b = bc[msk]
        sat = (v >= 254.5).any(-1)
        s = np.where(b > 1e-6, v / np.maximum(b * 255.0, 1e-9), np.nan)
        e = dict(n=int(msk.sum()),
                 obs_mean=[round(x, 2) for x in v.mean(0)],
                 obs_std=[round(x, 3) for x in v.std(0)],
                 obs_uniq=int(len(np.unique(v, axis=0))),
                 frac_saturated=float(sat.mean()))
        good = ~sat & np.isfinite(s).all(-1)
        if good.any():
            sg = s[good]
            e["shade_scalar_mean"] = [round(x, 5) for x in sg.mean(0)]
            e["shade_scalar_p01"] = [round(x, 5) for x in np.percentile(sg, 1, axis=0)]
            e["shade_scalar_p50"] = [round(x, 5) for x in np.percentile(sg, 50, axis=0)]
            e["shade_scalar_p99"] = [round(x, 5) for x in np.percentile(sg, 99, axis=0)]
            # modes of the scalar (rounded) -> is it binary (lit/shadow)?
            q = np.round(sg[:, 0], 3)
            vals, cnt = np.unique(q, return_counts=True)
            top = np.argsort(-cnt)[:6]
            e["shade_scalar_modes"] = [[float(vals[i]), int(cnt[i])] for i in top]
        # same, on the NO-SHADOW render
        v2 = noshadow[msk]
        good2 = (~(v2 >= 254.5).any(-1)) & (b > 1e-6).all(-1)
        if good2.any():
            s2 = v2[good2] / (b[good2] * 255.0)
            e["noshadow_scalar_p50"] = [round(x, 5) for x in np.percentile(s2, 50, axis=0)]
            q = np.round(s2[:, 0], 3)
            vals, cnt = np.unique(q, return_counts=True)
            top = np.argsort(-cnt)[:4]
            e["noshadow_scalar_modes"] = [[float(vals[i]), int(cnt[i])] for i in top]
        out_cls[c] = e
    res["by_class"] = out_cls

    # ---- analytic shadow: is any geom above the hit point? -------------------
    lit = np.ones(gid.shape, bool)
    up = np.array([0.0, 0.0, 1.0])
    flat = p.reshape(-1, 3) + 1e-4 * nrm.reshape(-1, 3)
    dirs = np.broadcast_to(up, flat.shape).copy()
    occ = np.zeros(len(flat), bool)
    for g in range(m.ngeom):
        if m.geom_rgba[g][3] <= 0 or int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = d.geom_xpos[g].astype(np.float64)
        mat = d.geom_xmat[g].reshape(3, 3).astype(np.float64)
        sz = m.geom_size[g].astype(np.float64)
        gt = int(m.geom_type[g])
        if gt == mujoco.mjtGeom.mjGEOM_BOX:
            tt = RC.hit_box(flat, dirs, pos, mat, sz)
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            tt = RC.hit_sphere(flat, dirs, pos, sz[0])
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            tt = RC.hit_capsule(flat, dirs, pos, mat, sz[0], sz[1])
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            tt = RC.hit_cylinder(flat, dirs, pos, mat, sz[0], sz[1])
        else:
            continue
        occ |= (tt < INF / 2) & (tt > 1e-3)
    lit = (~occ).reshape(gid.shape)
    res["analytic_lit_frac"] = float(lit.mean())
    # observed shadow mask: where does turning shadows off brighten the pixel?
    obs_shadowed = (noshadow - obs).max(-1) > 1.5
    res["observed_shadow_frac"] = float(obs_shadowed.mean())
    res["shadow_mask_agreement"] = float(((~lit) == obs_shadowed).mean())
    res["shadow_by_class"] = {c: dict(observed=float(obs_shadowed[cls == c].mean()),
                                      analytic=float((~lit)[cls == c].mean()))
                              for c in np.unique(cls) if (cls == c).any()}

    # ---- images ---------------------------------------------------------------
    sheet = np.concatenate([obs, noshadow, np.clip(np.abs(noshadow - obs) * 4, 0, 255)], 1).astype(np.uint8)
    imageio.imwrite(os.path.join(OUT, f"shadow_{loco}_{size}.png"), sheet)
    sm = np.stack([obs_shadowed * 255, (~lit) * 255, np.zeros_like(obs_shadowed) * 255], -1).astype(np.uint8)
    imageio.imwrite(os.path.join(OUT, f"shadowmask_{loco}_{size}.png"), sm)

    env.close()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = [("point", "medium"), ("point", "large"), ("ant", "large")]
    allr = []
    for loco, size in cells:
        try:
            r = run(loco, size)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            continue
        allr.append(r)
        print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, "shade2.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
