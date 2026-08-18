"""Pin the texgen constants: slopes as a function of size, texrepeat, texuniform and numquads.

``probe_texgen2`` showed the map is ``u = f(x)``, ``v = g(y)`` for every primitive -- a single planar
projection along the geom's local z, not a per-face unwrap and not a cube map. What is left is the
scale, and the scale is where a read of the source and a measurement disagree: MJWarp's plane rule
(``0.5*repeat*x``, no size term) is twice the slope this measures, and MuJoCo's own ``texuniform``
switch decides whether the size term is there at all.

``numquads`` is in the sweep because a box's slope came out three times the others', which is what a
per-sub-quad texcoord assignment would look like: if the slope tracks ``numquads`` it is tessellation,
and if it does not it is a constant.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save  # noqa: E402
from probe_texgen2 import ONE, paint  # noqa: E402


def profile(coord, val, nb=400):
    """Slope of ``val`` against ``coord`` when ``val`` depends on ``coord`` alone, unwrapping [0,1)."""
    if coord.size < 64:
        return float("nan"), float("nan"), float("nan")
    lo, hi = float(coord.min()), float(coord.max())
    if hi - lo < 1e-6:
        return float("nan"), float("nan"), float("nan")
    b = np.clip(((coord - lo) / (hi - lo) * nb).astype(int), 0, nb - 1)
    c, w = [], []
    for k in range(nb):
        msk = b == k
        if msk.sum() > 4:
            c.append(lo + (k + 0.5) * (hi - lo) / nb)
            w.append(np.median(val[msk]))
    c, w = np.array(c), np.array(w)
    if len(c) < 8:
        return float("nan"), float("nan"), float("nan")
    du = np.diff(w)
    du = du - np.round(du)
    uw = np.concatenate([[w[0]], w[0] + np.cumsum(du)])
    A = np.stack([c, np.ones_like(c)], -1)
    s, b0 = np.linalg.lstsq(A, uw, rcond=None)[0]
    return float(s), float(np.abs(A @ [s, b0] - uw).max()), float(b0)


def sample(gtype, size, rx=1.0, ry=1.0, uni=False, numquads=None, res=320, ttype="2d",
           az=35.0, el=25.0, dist=3.0):
    import mujoco
    xml = ONE.format(gtype=gtype, size=size, ttype=ttype, tw=256, th=256, rx=rx, ry=ry,
                     uni="true" if uni else "false", quat="1 0 0 0")
    if numquads is not None:
        xml = xml.replace('<quality offsamples="0"/>', f'<quality offsamples="0" numquads="{numquads}"/>')
    m = mujoco.MjModel.from_xml_string(xml)
    paint(m)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0, 0, 0)
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    opt = mujoco.MjvOption()
    r = mujoco.Renderer(m, height=res, width=res)
    r.update_scene(d, camera=cam, scene_option=opt)
    for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
              mujoco.mjtRndFlag.mjRND_SKYBOX):
        r.scene.flags[f] = 0
    rgb = r.render().copy()
    gl = r.scene.camera
    campos = 0.5 * (np.asarray(gl[0].pos, np.float64) + np.asarray(gl[1].pos, np.float64))
    fwd = 0.5 * (np.asarray(gl[0].forward, np.float64) + np.asarray(gl[1].forward, np.float64))
    up = 0.5 * (np.asarray(gl[0].up, np.float64) + np.asarray(gl[1].up, np.float64))
    near = 0.5 * (gl[0].frustum_near + gl[1].frustum_near)
    top = 0.5 * (gl[0].frustum_top + gl[1].frustum_top)
    bot = 0.5 * (gl[0].frustum_bottom + gl[1].frustum_bottom)
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam, scene_option=opt)
    dep = r.render().copy()
    r.disable_depth_rendering()
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    hh, mid = 0.5 * (top - bot), 0.5 * (top + bot)
    xs = (2.0 * (np.arange(res) + 0.5) / res - 1.0) * hh
    ys = mid + (1.0 - 2.0 * (np.arange(res) + 0.5) / res) * hh
    X, Y = np.meshgrid(xs, ys)
    dirs = X[..., None] * right + Y[..., None] * up + near * fwd
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    tpar = dep / np.einsum("hwk,k->hw", dirs, fwd)
    P = campos + tpar[..., None] * dirs
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g")
    loc = np.einsum("ji,hwj->hwi", np.asarray(d.geom_xmat[gid], np.float64).reshape(3, 3),
                    P - np.asarray(d.geom_xpos[gid], np.float64))
    hit = dep < (0.9 * float(m.vis.map.zfar) * float(m.stat.extent))
    u = (rgb[..., 0].astype(np.float64) + 0.5) / 256.0
    v = (rgb[..., 1].astype(np.float64) + 0.5) / 256.0
    del r
    return u[hit], v[hit], loc[..., 0][hit], loc[..., 1][hit], loc[..., 2][hit]


def row(label, gtype, size, **kw):
    u, v, x, y, z = sample(gtype, size, **kw)
    su, eu, bu = profile(x, u)
    sv, ev, bv = profile(y, v)
    out = {"case": label, "geom": gtype, "size": size, "n": int(len(u)), **{k: kw[k] for k in kw},
           "du_dx": su, "du_dx_res": eu, "u0": bu, "dv_dy": sv, "dv_dy_res": ev, "v0": bv}
    emit("slope", out)
    return out


def main():
    rows = []
    # small geoms so the map never wraps: then the OFFSET is measurable, not just the slope
    for sz, uni, rx in [("0.4 0.4 0.1", False, 1.0), ("0.4 0.4 0.1", True, 1.0),
                        ("0.4 0.4 0.1", False, 2.0), ("0.4 0.4 0.1", True, 0.5),
                        ("0.8 0.4 0.1", False, 1.0)]:
        rows.append(row("plane_small", "plane", sz, uni=uni, rx=rx, ry=rx, dist=2.0, el=55.0))
    for sz, uni, rx in [("0.4 0.3 0.2", False, 1.0), ("0.4 0.3 0.2", True, 1.0),
                        ("0.4 0.3 0.2", False, 0.5)]:
        rows.append(row("box_small", "box", sz, uni=uni, rx=rx, ry=rx, dist=2.0))
    for gt, sz in (("sphere", "0.4"), ("cylinder", "0.4 0.6"), ("capsule", "0.3 0.4")):
        rows.append(row("curved_small", gt, sz, rx=0.5, ry=0.5, dist=2.5))
    # planes: size scaling, repeat scaling, and what texuniform changes
    for sz, uni, rx in [("1 1 0.1", False, 1.0), ("2 1 0.1", False, 1.0), ("4 1 0.1", False, 1.0),
                        ("1 1 0.1", True, 1.0), ("2 1 0.1", True, 1.0), ("4 1 0.1", True, 1.0),
                        ("1 1 0.1", False, 2.0), ("1 1 0.1", True, 3.0)]:
        rows.append(row("plane", "plane", sz, uni=uni, rx=rx, ry=rx, dist=6.0, el=60.0))
    # box: does the slope track numquads?
    for nq in (1, 2, 4, 8):
        rows.append(row(f"box_nq{nq}", "box", "0.5 0.5 0.5", numquads=nq))
    for sz in ("0.5 0.5 0.5", "1 1 1"):
        rows.append(row("box_size", "box", sz))
        rows.append(row("box_size_uni", "box", sz, uni=True))
    # curved primitives: size and repeat scaling
    for gt, sz in (("sphere", "0.5"), ("sphere", "1"), ("cylinder", "0.5 0.5"),
                   ("capsule", "0.3 0.5")):
        rows.append(row("curved", gt, sz))
        rows.append(row("curved_uni", gt, sz, uni=True))
        rows.append(row("curved_rep2", gt, sz, rx=2.0, ry=2.0))
    save("texgen3.json", rows)


if __name__ == "__main__":
    main()
