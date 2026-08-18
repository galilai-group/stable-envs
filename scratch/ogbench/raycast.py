"""Analytic ray-cast of an OGBench maze scene, checked against MuJoCo's own segmentation + depth buffers.

The question this answers is *geometric*, and it is deliberately separated from shading: if an independent
pinhole ray-caster over {finite plane, axis-aligned box, sphere, capsule, cylinder} reproduces MuJoCo's
per-pixel geom-id map exactly, then the scene is analytically representable and the only remaining work is
colour. If it does not, no amount of shading work will save it.

Camera is *measured*, not assumed. ``mjv_updateScene`` fills two ``mjvGLCamera`` (left/right eye, separated
by ``vis.global_.ipd``); which one -- or the average -- ``mjr_render`` actually uses for a mono offscreen
buffer is an implementation detail, so all three candidates are ray-cast and the one that reproduces the
segmentation buffer wins. Same for the frustum left/right convention and for what MuJoCo's depth buffer
means (perpendicular vs radial distance): both are scored.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_scene import make  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"
INF = 1e30


# ----------------------------------------------------------------------------- primitives
def hit_plane(o, dr, pos, mat, size):
    """Finite rectangle (MuJoCo renders a plane geom of nonzero size as a size[0] x size[1] rect)."""
    n = mat[:, 2]
    den = dr @ n
    t = np.where(np.abs(den) > 1e-12, ((pos - o) @ n) / np.where(np.abs(den) > 1e-12, den, 1.0), INF)
    p = o + t[:, None] * dr
    loc = (p - pos) @ mat
    ok = (t > 0) & (np.abs(loc[:, 0]) <= size[0]) & (np.abs(loc[:, 1]) <= size[1])
    return np.where(ok, t, INF)


def hit_box(o, dr, pos, mat, size):
    ol = (o - pos) @ mat
    dl = dr @ mat
    inv = 1.0 / np.where(np.abs(dl) < 1e-12, 1e-12, dl)
    t1 = (-size[None, :] - ol) * inv
    t2 = (size[None, :] - ol) * inv
    tmin = np.minimum(t1, t2).max(1)
    tmax = np.maximum(t1, t2).min(1)
    ok = (tmax >= np.maximum(tmin, 0.0))
    return np.where(ok, np.where(tmin > 0, tmin, tmax), INF)


def hit_sphere(o, dr, pos, r):
    oc = o - pos
    b = (oc * dr).sum(1)
    c = (oc * oc).sum(1) - r * r
    disc = b * b - c
    ok = disc > 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t = -b - sq
    t = np.where(t > 0, t, -b + sq)
    return np.where(ok & (t > 0), t, INF)


def hit_cylinder(o, dr, pos, mat, r, hh):
    """Z-aligned (in geom frame) capped cylinder."""
    ol = (o - pos) @ mat
    dl = dr @ mat
    a = dl[:, 0] ** 2 + dl[:, 1] ** 2
    b = ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1]
    c = ol[:, 0] ** 2 + ol[:, 1] ** 2 - r * r
    disc = b * b - a * c
    best = np.full(len(o), INF)
    ok = (disc > 0) & (a > 1e-12)
    sq = np.sqrt(np.maximum(disc, 0.0))
    for t in ((-b - sq) / np.where(a > 1e-12, a, 1.0), (-b + sq) / np.where(a > 1e-12, a, 1.0)):
        z = ol[:, 2] + t * dl[:, 2]
        good = ok & (t > 0) & (np.abs(z) <= hh)
        best = np.where(good & (t < best), t, best)
    # caps
    for zc in (-hh, hh):
        t = (zc - ol[:, 2]) / np.where(np.abs(dl[:, 2]) > 1e-12, dl[:, 2], 1e-12)
        px = ol[:, 0] + t * dl[:, 0]
        py = ol[:, 1] + t * dl[:, 1]
        good = (t > 0) & (px * px + py * py <= r * r) & (np.abs(dl[:, 2]) > 1e-12)
        best = np.where(good & (t < best), t, best)
    return best


def hit_capsule(o, dr, pos, mat, r, hh):
    """Z-aligned capsule = cylinder(|z|<=hh) + two end spheres."""
    ol = (o - pos) @ mat
    dl = dr @ mat
    a = dl[:, 0] ** 2 + dl[:, 1] ** 2
    b = ol[:, 0] * dl[:, 0] + ol[:, 1] * dl[:, 1]
    c = ol[:, 0] ** 2 + ol[:, 1] ** 2 - r * r
    disc = b * b - a * c
    best = np.full(len(o), INF)
    sq = np.sqrt(np.maximum(disc, 0.0))
    okc = (disc > 0) & (a > 1e-12)
    for t in ((-b - sq) / np.where(a > 1e-12, a, 1.0), (-b + sq) / np.where(a > 1e-12, a, 1.0)):
        z = ol[:, 2] + t * dl[:, 2]
        good = okc & (t > 0) & (np.abs(z) <= hh)
        best = np.where(good & (t < best), t, best)
    for zc in (-hh, hh):
        oc = ol - np.array([0.0, 0.0, zc])
        bb = (oc * dl).sum(1)
        cc = (oc * oc).sum(1) - r * r
        dd = bb * bb - cc
        s = np.sqrt(np.maximum(dd, 0.0))
        for t in (-bb - s, -bb + s):
            good = (dd > 0) & (t > 0)
            best = np.where(good & (t < best), t, best)
    return best


# ----------------------------------------------------------------------------- camera
def rays(cam, H, W, mode="lr_from_tb"):
    """Pixel-centre rays for an mjvGLCamera-style pinhole. Row 0 is the TOP of the image."""
    pos = np.asarray(cam["pos"], np.float64)
    fwd = np.asarray(cam["forward"], np.float64)
    up = np.asarray(cam["up"], np.float64)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    near = cam["frustum_near"]
    bot, top, ctr = cam["frustum_bottom"], cam["frustum_top"], cam["frustum_center"]
    half_w = 0.5 * (top - bot) * (W / H)
    left, rgt = ctr - half_w, ctr + half_w
    j = (np.arange(W) + 0.5) / W
    i = (np.arange(H) + 0.5) / H
    xs = left + j * (rgt - left)
    ys = top - i * (top - bot)
    X, Y = np.meshgrid(xs, ys)
    d = (X[..., None] * right + Y[..., None] * up + near * fwd)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    o = np.broadcast_to(pos, d.shape)
    return o.reshape(-1, 3).copy(), d.reshape(-1, 3).copy(), fwd


def cast(model, data, cam, H, W):
    """Return (geom_id map, t map) for the whole scene, all geoms with group visible."""
    import mujoco
    o, dr, fwd = rays(cam, H, W)
    best_t = np.full(len(o), INF)
    best_g = np.full(len(o), -1, np.int32)
    for g in range(model.ngeom):
        if model.geom_rgba[g][3] <= 0:
            continue
        gt = int(model.geom_type[g])
        pos = data.geom_xpos[g].astype(np.float64)
        mat = data.geom_xmat[g].reshape(3, 3).astype(np.float64)
        sz = model.geom_size[g].astype(np.float64)
        if gt == mujoco.mjtGeom.mjGEOM_PLANE:
            t = hit_plane(o, dr, pos, mat, sz)
        elif gt == mujoco.mjtGeom.mjGEOM_BOX:
            t = hit_box(o, dr, pos, mat, sz)
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            t = hit_sphere(o, dr, pos, sz[0])
        elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
            t = hit_capsule(o, dr, pos, mat, sz[0], sz[1])
        elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
            t = hit_cylinder(o, dr, pos, mat, sz[0], sz[1])
        else:
            continue
        upd = t < best_t
        best_t = np.where(upd, t, best_t)
        best_g = np.where(upd, g, best_g)
    return best_g.reshape(H, W), best_t.reshape(H, W), fwd, o.reshape(H, W, 3), dr.reshape(H, W, 3)


def glcam(scn, i):
    c = scn.camera[i]
    return dict(pos=c.pos.tolist(), forward=c.forward.tolist(), up=c.up.tolist(),
                frustum_near=float(c.frustum_near), frustum_far=float(c.frustum_far),
                frustum_bottom=float(c.frustum_bottom), frustum_top=float(c.frustum_top),
                frustum_center=float(c.frustum_center))


def avgcam(a, b):
    out = {}
    for k in a:
        out[k] = ((np.asarray(a[k]) + np.asarray(b[k])) / 2).tolist() if isinstance(a[k], list) \
            else (a[k] + b[k]) / 2
    return out


def run(loco, size, img=224, seed=0, nstates=4):
    import mujoco
    env, u = make(loco, size, img)
    m, d = u.model, u.data
    # segmentation is corrupted by MSAA -> turn multisampling off for the geometry probe
    m.vis.quality.offsamples = 0
    u.custom_renderer = mujoco.Renderer(m, width=img, height=img)
    res = dict(env=f"{loco}maze-{size}", img=img)

    rng = np.random.default_rng(seed)
    env.reset(options={"task_id": 1})
    for _ in range(20):
        env.step(u.action_space.sample())

    r = u.custom_renderer
    r.update_scene(d, camera=u.custom_camera)
    cams = dict(eye0=glcam(r.scene, 0), eye1=glcam(r.scene, 1))
    cams["avg"] = avgcam(cams["eye0"], cams["eye1"])
    res["cameras"] = cams
    res["cam_eyes_identical"] = bool(np.allclose(np.asarray(cams["eye0"]["pos"]),
                                                 np.asarray(cams["eye1"]["pos"])))
    res["nlight_scene"] = int(r.scene.nlight)

    # ------- MuJoCo ground truth: segmentation + depth ----------------------
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=u.custom_camera)
    seg = r.render()                      # (H,W,2): objid, objtype
    r.disable_segmentation_rendering()
    r.enable_depth_rendering()
    r.update_scene(d, camera=u.custom_camera)
    dep = r.render().astype(np.float64)
    r.disable_depth_rendering()

    geom_gt = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1).astype(np.int32)
    res["gt_geom_ids"] = sorted(int(x) for x in np.unique(geom_gt))
    res["gt_bg_frac"] = float((geom_gt < 0).mean())

    # ------- analytic cast, all three camera candidates ---------------------
    scores = {}
    for name, cam in cams.items():
        gid, t, fwd, o, dr = cast(m, d, cam, img, img)
        gid = np.where(t >= INF / 2, -1, gid)
        agree = float((gid == geom_gt).mean())
        # perpendicular vs radial depth convention
        zperp = (t * (dr @ fwd))
        finite = (geom_gt >= 0) & (t < INF / 2) & (dep > 0) & (dep < dep.max() * 0.999)
        e_perp = float(np.abs(zperp - dep)[finite].max()) if finite.any() else float("nan")
        e_rad = float(np.abs(t - dep)[finite].max()) if finite.any() else float("nan")
        scores[name] = dict(id_agree=agree, depth_maxerr_perp=e_perp, depth_maxerr_radial=e_rad)
    res["camera_scores"] = scores
    best = max(scores, key=lambda k: scores[k]["id_agree"])
    res["best_camera"] = best

    # ------- with the winning camera: interior vs boundary breakdown --------
    cam = cams[best]
    gid, t, fwd, o, dr = cast(m, d, cam, img, img)
    gid = np.where(t >= INF / 2, -1, gid)
    ne = gid != geom_gt
    # a boundary pixel is one whose 4-neighbourhood in the GT id map is not uniform
    b = np.zeros_like(ne)
    b[1:, :] |= geom_gt[1:, :] != geom_gt[:-1, :]
    b[:-1, :] |= geom_gt[1:, :] != geom_gt[:-1, :]
    b[:, 1:] |= geom_gt[:, 1:] != geom_gt[:, :-1]
    b[:, :-1] |= geom_gt[:, 1:] != geom_gt[:, :-1]
    res["id_agree_all"] = float((~ne).mean())
    res["id_boundary_frac"] = float(b.mean())
    res["id_agree_interior"] = float((~ne[~b]).mean())
    res["id_mismatch_interior_count"] = int((ne & ~b).sum())
    zperp = t * (dr @ fwd)
    fin = (geom_gt >= 0) & (t < INF / 2) & ~b
    res["depth_interior_maxerr"] = float(np.abs(zperp - dep)[fin].max())
    res["depth_interior_p999"] = float(np.percentile(np.abs(zperp - dep)[fin], 99.9))

    # ------- repeat over several states (does it hold as the agent moves?) ---
    per_state = []
    for s in range(nstates):
        for _ in range(25):
            env.step(u.action_space.sample())
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=u.custom_camera)
        sg = r.render()
        r.disable_segmentation_rendering()
        gt2 = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1).astype(np.int32)
        g2, t2, _, _, _ = cast(m, d, cam, img, img)
        g2 = np.where(t2 >= INF / 2, -1, g2)
        bb = np.zeros_like(gt2, bool)
        bb[1:, :] |= gt2[1:, :] != gt2[:-1, :]
        bb[:-1, :] |= gt2[1:, :] != gt2[:-1, :]
        bb[:, 1:] |= gt2[:, 1:] != gt2[:, :-1]
        bb[:, :-1] |= gt2[:, 1:] != gt2[:, :-1]
        per_state.append(dict(all=float((g2 == gt2).mean()),
                              interior=float((g2 == gt2)[~bb].mean()),
                              interior_bad=int(((g2 != gt2) & ~bb).sum())))
    res["per_state"] = per_state

    # ------- MSAA cost on RGB: offsamples 4 vs 0 ---------------------------
    m.vis.quality.offsamples = 4
    r4 = mujoco.Renderer(m, width=img, height=img)
    r4.update_scene(d, camera=u.custom_camera)
    f4 = r4.render().copy()
    m.vis.quality.offsamples = 0
    r0 = mujoco.Renderer(m, width=img, height=img)
    r0.update_scene(d, camera=u.custom_camera)
    f0 = r0.render().copy()
    dd = np.abs(f4.astype(np.int16) - f0.astype(np.int16))
    res["msaa_effect"] = dict(frac_pixels_changed=float((dd.max(-1) > 0).mean()),
                              mean=float(dd.mean()), max=int(dd.max()))

    env.close()
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    cells = [("point", "medium"), ("point", "large"), ("point", "giant"), ("point", "teleport"),
             ("ant", "large"), ("humanoid", "large")]
    if len(sys.argv) > 1:
        cells = [tuple(sys.argv[1].split("-"))]
    allr = []
    for loco, size in cells:
        try:
            r = run(loco, size)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"FAIL {loco}-{size}: {e}", flush=True)
            continue
        allr.append(r)
        print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, "raycast.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
