"""What does an OGBench maze frame's COLOUR depend on? Four measurements, no guessing.

  A. render-flag ablation -- toggle MuJoCo's own shadow / skybox / reflection / fog / haze / MSAA and
     count the pixels each one moves. Whatever moves ~0 pixels does not have to be reimplemented.
  B. flatness -- group pixels by (geom, face normal) and report the spread of RGB inside each group.
     If the spread is ~0 the picture is flat-shaded and a 2-D rasterizer with a tiny palette suffices;
     if it is large the shading is a real per-pixel function and must be modelled.
  C. analytic Blinn-Phong residual -- evaluate MuJoCo's documented lighting model per pixel from the
     ray-cast normal and the scene's own light parameters, and diff against the real frame.
  D. footprint -- how many pixels the agent actually occupies (the thing a state->frame renderer has to
     get right), per family.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_scene import make  # noqa: E402
import raycast as RC  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = RC.INF


def normals(model, data, gid, p):
    """Analytic surface normal at hit point ``p`` for the geom in ``gid`` (-1 -> zeros)."""
    import mujoco
    H, W = gid.shape
    n = np.zeros((H, W, 3))
    for g in np.unique(gid):
        if g < 0:
            continue
        msk = gid == g
        pos = data.geom_xpos[g].astype(np.float64)
        mat = data.geom_xmat[g].reshape(3, 3).astype(np.float64)
        sz = model.geom_size[g].astype(np.float64)
        gt = int(model.geom_type[g])
        q = p[msk] - pos
        loc = q @ mat
        if gt == mujoco.mjtGeom.mjGEOM_PLANE:
            nn = np.broadcast_to(mat[:, 2], (msk.sum(), 3))
        elif gt == mujoco.mjtGeom.mjGEOM_BOX:
            r = np.abs(loc) / sz[None, :]
            ax = r.argmax(1)
            nl = np.zeros_like(loc)
            nl[np.arange(len(ax)), ax] = np.sign(loc[np.arange(len(ax)), ax])
            nn = nl @ mat.T
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            nn = q / np.linalg.norm(q, axis=1, keepdims=True)
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            hh = sz[1]
            zc = np.clip(loc[:, 2], -hh, hh)
            nl = loc - np.stack([np.zeros_like(zc), np.zeros_like(zc), zc], 1)
            nn = (nl / np.linalg.norm(nl, axis=1, keepdims=True)) @ mat.T
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            hh = sz[1]
            cap = np.abs(np.abs(loc[:, 2]) - hh) < 1e-6
            nl = np.stack([loc[:, 0], loc[:, 1], np.zeros_like(loc[:, 2])], 1)
            nl = nl / np.maximum(np.linalg.norm(nl, axis=1, keepdims=True), 1e-12)
            nl[cap] = np.stack([np.zeros(cap.sum()), np.zeros(cap.sum()),
                                np.sign(loc[cap, 2])], 1)
            nn = nl @ mat.T
        else:
            nn = np.zeros((msk.sum(), 3))
        n[msk] = nn
    return n


def flag_ablation(u, img):
    """Toggle each mjtRndFlag and count moved pixels."""
    import mujoco
    m, d = u.model, u.data
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    base = r.render().copy()
    # mjv_updateScene does NOT reset scn.flags, so a toggle persists into the next test unless the
    # baseline is restored explicitly -- without this the ablation is cumulative, not per-flag.
    base_flags = np.array(r.scene.flags, copy=True)
    out = {}
    names = ["mjRND_SHADOW", "mjRND_WIREFRAME", "mjRND_REFLECTION", "mjRND_ADDITIVE",
             "mjRND_SKYBOX", "mjRND_FOG", "mjRND_HAZE", "mjRND_CULL_FACE"]
    for nm in names:
        f = getattr(mujoco.mjtRndFlag, nm, None)
        if f is None:
            continue
        r.update_scene(d, camera=u.custom_camera)
        for k in range(len(base_flags)):
            r.scene.flags[k] = int(base_flags[k])
        was = int(base_flags[int(f)])
        r.scene.flags[int(f)] = 0 if was else 1
        img2 = r.render().copy()
        dd = np.abs(img2.astype(np.int16) - base.astype(np.int16))
        out[nm] = dict(default_on=bool(was), frac_changed=float((dd.max(-1) > 0).mean()),
                       mean=float(dd.mean()), max=int(dd.max()))
    for k in range(len(base_flags)):
        r.scene.flags[k] = int(base_flags[k])
    return out, base


def run(loco, size, img=224, seed=0):
    import mujoco
    env, u = make(loco, size, img)
    m, d = u.model, u.data
    env.reset(options={"task_id": 1})
    for _ in range(20):
        env.step(u.action_space.sample())

    res = dict(env=f"{loco}maze-{size}", img=img)

    # ---- A. render-flag ablation -----------------------------------------
    res["flag_ablation"], rgb = flag_ablation(u, img)

    # ---- scene lights (what mjv actually hands the rasterizer) ------------
    r = mujoco.Renderer(m, width=img, height=img)
    r.update_scene(d, camera=u.custom_camera)
    scn = r.scene
    lights = []
    for i in range(int(scn.nlight)):
        L = scn.lights[i]
        e = dict(pos=L.pos.tolist(), dir=L.dir.tolist(), ambient=L.ambient.tolist(),
                 diffuse=L.diffuse.tolist(), specular=L.specular.tolist(),
                 attenuation=L.attenuation.tolist(), cutoff=float(L.cutoff),
                 exponent=float(L.exponent), castshadow=int(L.castshadow))
        for a in ("directional", "type", "headlight"):
            if hasattr(L, a):
                e[a] = int(getattr(L, a))
        lights.append(e)
    res["scene_lights"] = lights
    res["headlight_model"] = dict(ambient=m.vis.headlight.ambient.tolist(),
                                  diffuse=m.vis.headlight.diffuse.tolist(),
                                  specular=m.vis.headlight.specular.tolist(),
                                  active=int(m.vis.headlight.active))
    cam = RC.avgcam(RC.glcam(scn, 0), RC.glcam(scn, 1))
    res["camera"] = cam

    # ---- ray-cast geometry ------------------------------------------------
    gid, t, fwd, o, dr = RC.cast(m, d, cam, img, img)
    gid = np.where(t >= INF / 2, -1, gid)
    p = o + t[..., None] * dr
    nrm = normals(m, d, gid, p)

    # ---- D. footprints -----------------------------------------------------
    names = {g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(m.ngeom)}
    agent = np.array([g for g in range(m.ngeom)
                      if not (names[g] or "").startswith("block_") and names[g] != "floor"
                      and not (names[g] or "").startswith("teleport")])
    res["agent_geom_names"] = [names[g] for g in agent]
    res["agent_pixels"] = int(np.isin(gid, agent).sum())
    res["agent_pixel_frac"] = float(np.isin(gid, agent).mean())
    res["floor_pixel_frac"] = float((gid == 0).mean())
    res["bg_pixel_frac"] = float((gid < 0).mean())

    # ---- B. flatness: RGB spread within (geom, face) groups ---------------
    key = gid.astype(np.int64) * 100 + (np.round(nrm * 4).astype(np.int64) @ np.array([1, 7, 49]) + 50)
    groups = []
    f = rgb.astype(np.float64)
    for k in np.unique(key):
        msk = key == k
        if msk.sum() < 50:
            continue
        v = f[msk]
        groups.append(dict(n=int(msk.sum()), gid=int(np.unique(gid[msk])[0]),
                           name=names.get(int(np.unique(gid[msk])[0]), "bg"),
                           std=float(v.std(0).max()), ptp=float(np.ptp(v, 0).max()),
                           mean=[round(x, 2) for x in v.mean(0)]))
    groups.sort(key=lambda g: -g["n"])
    res["face_groups_top"] = groups[:12]
    res["n_face_groups"] = len(groups)
    tot = sum(g["n"] for g in groups)
    res["flatness_weighted_std"] = float(sum(g["n"] * g["std"] for g in groups) / max(tot, 1))
    res["flatness_weighted_ptp"] = float(sum(g["n"] * g["ptp"] for g in groups) / max(tot, 1))
    # how well does per-face CONSTANT colour do?
    approx = np.zeros_like(f)
    for k in np.unique(key):
        msk = key == k
        approx[msk] = f[msk].mean(0)
    dd = np.abs(approx - f).max(-1)
    res["flat_shade_err"] = dict(mean=float(dd.mean()), p99=float(np.percentile(dd, 99)),
                                 max=float(dd.max()), frac_le1=float((dd <= 1).mean()),
                                 frac_le4=float((dd <= 4).mean()))

    # ---- C. analytic Blinn-Phong ------------------------------------------
    V = -dr / np.linalg.norm(dr, axis=-1, keepdims=True)
    rgba = np.zeros((img, img, 4))
    spec = np.zeros((img, img))
    shin = np.zeros((img, img))
    for g in np.unique(gid):
        if g < 0:
            continue
        msk = gid == g
        mid = int(m.geom_matid[g])
        c = m.geom_rgba[g].astype(np.float64)
        s, sh = 0.0, 0.0
        if mid >= 0:
            mr = m.mat_rgba[mid].astype(np.float64)
            if not np.allclose(m.geom_rgba[g], [0.5, 0.5, 0.5, 1.0]):
                c = m.geom_rgba[g].astype(np.float64)
            else:
                c = mr
            s = float(m.mat_specular[mid])
            sh = float(m.mat_shininess[mid])
        rgba[msk] = c
        spec[msk] = s
        shin[msk] = sh
    col = np.zeros((img, img, 3))
    for L in lights:
        ld = np.asarray(L["dir"], np.float64)
        lp = np.asarray(L["pos"], np.float64)
        directional = bool(L.get("directional", L.get("type", 0) == 1))
        if directional:
            Lv = np.broadcast_to(-ld / np.linalg.norm(ld), (img, img, 3))
            att = np.ones((img, img))
        else:
            dv = lp - p
            dist = np.linalg.norm(dv, axis=-1, keepdims=True)
            Lv = dv / np.maximum(dist, 1e-9)
            a = np.asarray(L["attenuation"], np.float64)
            att = 1.0 / np.maximum(a[0] + a[1] * dist[..., 0] + a[2] * dist[..., 0] ** 2, 1e-9)
        ndl = np.clip((nrm * Lv).sum(-1), 0, None)
        col += np.asarray(L["ambient"]) * rgba[..., :3]
        col += np.asarray(L["diffuse"]) * rgba[..., :3] * (ndl * att)[..., None]
        Hv = Lv + V
        Hv = Hv / np.maximum(np.linalg.norm(Hv, axis=-1, keepdims=True), 1e-9)
        ndh = np.clip((nrm * Hv).sum(-1), 0, None)
        col += np.asarray(L["specular"]) * (spec * ndh ** np.maximum(shin * 128, 1))[..., None]
    pred = np.clip(col, 0, 1) * 255
    dd = np.abs(pred - rgb.astype(np.float64)).max(-1)
    vis = gid >= 0
    res["phong_err"] = dict(mean=float(dd[vis].mean()), p50=float(np.percentile(dd[vis], 50)),
                            p99=float(np.percentile(dd[vis], 99)), max=float(dd[vis].max()),
                            frac_le2=float((dd[vis] <= 2).mean()), frac_le8=float((dd[vis] <= 8).mean()))
    # per-geom-class residual
    cls = {"floor": gid == 0,
           "wall": np.isin(gid, [g for g in range(m.ngeom) if (names[g] or "").startswith("block_")]),
           "agent": np.isin(gid, agent)}
    res["phong_err_by_class"] = {k: (dict(n=int(v.sum()), mean=float(dd[v].mean()),
                                          p99=float(np.percentile(dd[v], 99)))
                                     if v.sum() else None) for k, v in cls.items()}

    # save an inspectable sheet
    import imageio.v2 as imageio
    sheet = np.concatenate([rgb, np.clip(pred, 0, 255).astype(np.uint8),
                            np.clip(np.abs(pred - rgb.astype(np.float64)) * 4, 0, 255).astype(np.uint8)], 1)
    imageio.imwrite(os.path.join(OUT, f"shade_{loco}_{size}.png"), sheet)

    env.close()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = [("point", "medium"), ("point", "large"), ("point", "teleport"),
             ("ant", "large"), ("humanoid", "large")]
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
    with open(os.path.join(OUT, "shade.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
