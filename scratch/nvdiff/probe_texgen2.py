"""Invert MuJoCo's primitive texgen exactly: per-pixel local hit point vs per-pixel texel index.

The first probe read texel indices off a frame and guessed which world coordinate they tracked. This
one does not guess. MuJoCo's own depth buffer gives the hit point of every pixel, the segmentation
buffer says which geom it belongs to, and the texture is painted with its own (i, j), so each pixel is
one equation ``uv = A @ local + b`` and the whole map falls out of a least-squares solve with a
residual that says whether the affine form is even right.

One geom per scene, one scene per process-free call: a second textured geom in the frame is what made
the first pass read a plane's mapping as a box's.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save  # noqa: E402

ONE = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/><quality offsamples="0"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/><map znear="0.05" zfar="20"/></visual>
  <asset>
    <texture name="idx" type="{ttype}" builtin="flat" width="{tw}" height="{th}" rgb1="1 1 1"/>
    <material name="mm" texture="idx" texrepeat="{rx} {ry}" texuniform="{uni}"
              specular="0" shininess="0" emission="1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true" ambient="0 0 0" diffuse="0 0 0"
           specular="0 0 0" castshadow="false"/>
    <body name="b" pos="0 0 0" quat="{quat}">
      <geom name="g" type="{gtype}" size="{size}" material="mm"/>
    </body>
  </worldbody>
</mujoco>
"""


def paint(m, name="idx"):
    import mujoco
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
    w, h = int(m.tex_width[t]), int(m.tex_height[t])
    nc = int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[t])
    jj, ii = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    a = np.zeros((h, w, nc), np.uint8)
    a[..., 0] = ii % 256
    a[..., 1] = jj % 256
    a[..., 2] = 40 * (ii // 256) + 10 * (jj // 256)
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    buf[a0:a0 + w * h * nc] = a.reshape(-1)
    return t, w, h


def run_one(gtype, size, ttype="2d", rx=1.0, ry=1.0, uni=False, tw=256, th=256, res=256,
            az=35.0, el=25.0, dist=3.0, quat="1 0 0 0"):
    import mujoco
    xml = ONE.format(gtype=gtype, size=size, ttype=ttype, tw=tw, th=th, rx=rx, ry=ry,
                     uni="true" if uni else "false", quat=quat)
    m = mujoco.MjModel.from_xml_string(xml)
    t, w, h = paint(m)
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
    hh = 0.5 * (top - bot)
    mid = 0.5 * (top + bot)
    xs = (2.0 * (np.arange(res) + 0.5) / res - 1.0) * hh
    ys = mid + (1.0 - 2.0 * (np.arange(res) + 0.5) / res) * hh
    X, Y = np.meshgrid(xs, ys)
    dirs = X[..., None] * right + Y[..., None] * up + near * fwd
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    # MuJoCo's depth buffer is PERPENDICULAR distance, so the ray parameter is dep / (dir . fwd)
    tpar = dep / np.einsum("hwk,k->hw", dirs, fwd)
    P = campos + tpar[..., None] * dirs

    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "g")
    gpos = np.asarray(d.geom_xpos[gid], np.float64)
    gmat = np.asarray(d.geom_xmat[gid], np.float64).reshape(3, 3)
    loc = np.einsum("ji,hwj->hwi", gmat, P - gpos)

    hit = dep < (0.9 * float(m.vis.map.zfar) * float(m.stat.extent))
    i = rgb[..., 0].astype(np.float64) + 256 * (rgb[..., 2].astype(np.int64) // 40)
    j = rgb[..., 1].astype(np.float64) + 256 * ((rgb[..., 2].astype(np.int64) % 40) // 10)
    u = (i + 0.5) / w
    v = (j + 0.5) / h

    out = {"geom": gtype, "size": size, "ttype": ttype, "texrepeat": [rx, ry], "texuniform": int(uni),
           "npx": int(hit.sum()), "u_range": [float(u[hit].min()), float(u[hit].max())],
           "v_range": [float(v[hit].min()), float(v[hit].max())],
           "loc_range": [[float(loc[..., k][hit].min()), float(loc[..., k][hit].max())]
                         for k in range(3)]}
    # affine fit on the unwrapped values is only meaningful where the map does not wrap; report the
    # residual so a wrapped case announces itself instead of returning a plausible slope
    A = np.stack([loc[..., 0][hit], loc[..., 1][hit], loc[..., 2][hit],
                  np.ones(int(hit.sum()))], -1)
    for nm, tgt in (("u", u[hit]), ("v", v[hit])):
        c, *_ = np.linalg.lstsq(A, tgt, rcond=None)
        pred = A @ c
        out[nm + "_coef"] = [float(x) for x in c]
        out[nm + "_resid"] = float(np.abs(pred - tgt).mean())
    np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/"
            f"tg2_{gtype}_{ttype}_{str(size).replace(' ', '_')}.npy",
            np.stack([u, v, loc[..., 0], loc[..., 1], loc[..., 2], hit.astype(float)], -1))
    del r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=256)
    args = ap.parse_args()
    rows = []
    cases = [
        dict(gtype="plane", size="1 1 0.1"),
        dict(gtype="plane", size="2 1 0.1"),
        dict(gtype="plane", size="1 1 0.1", uni=True),
        dict(gtype="box", size="0.5 0.5 0.5"),
        dict(gtype="box", size="0.5 0.25 0.75"),
        dict(gtype="box", size="0.5 0.5 0.5", uni=True),
        dict(gtype="box", size="0.5 0.5 0.5", rx=2.0, ry=3.0),
        dict(gtype="cylinder", size="0.5 0.5"),
        dict(gtype="sphere", size="0.5"),
        dict(gtype="capsule", size="0.3 0.5"),
        dict(gtype="box", size="0.5 0.5 0.5", ttype="cube"),
        dict(gtype="sphere", size="0.5", ttype="cube"),
        dict(gtype="cylinder", size="0.5 0.5", ttype="cube"),
    ]
    for c in cases:
        try:
            row = run_one(res=args.res, **c)
        except Exception as e:                                        # noqa: BLE001
            row = dict(c, error=f"{type(e).__name__}: {e}")
        rows.append(row)
        emit("fit", row)
    save("texgen2.json", rows)


if __name__ == "__main__":
    main()
