"""What MuJoCo's texture pipeline actually does, measured on frames rather than read off a source.

Four questions, each answered by a picture MuJoCo draws and nothing else:

1. **Where does a texel land?** A plane is drawn under an orthographic top-down camera, so every
   output pixel has an exactly known world position, and the texture's texels are painted with their
   own (i, j) index. Reading the frame back therefore inverts the texgen directly: no fitting, no
   hypothesis. Repeated over ``texrepeat``, ``texuniform`` and plane size, that pins the affine map.

2. **Does MuJoCo resample a non-power-of-two texture?** nvdiffrast asserts power-of-two dimensions for
   mipmapping and MuJoCo's textures often are not (OGBench's grid is 300x300, Gymnasium's checker
   100x100). If MuJoCo's own upload rescales them, resampling on our side is FAITHFUL rather than a
   compromise -- and if it does not, resampling is a declared residual. The texel-index texture answers
   this too: count the distinct index values across the plane.

3. **The same for a BOX and a CYLINDER**, which is the RoboCasa blocker: MJWarp has no texgen for them
   at all, so there is no prior art to check against and the mapping has to come from here.

4. **What does minification look like?** The same plane at grazing incidence, far from the camera,
   with a checker: the luminance spread per band is the statistic that separates a mipmapped sampler
   from a point-sampled one.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save  # noqa: E402

XML = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/><quality offsamples="0" shadowsize="4096"/><headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <map znear="0.01" zfar="50"/></visual>
  <asset>
    <texture name="idx" type="2d" builtin="flat" width="{tw}" height="{th}" rgb1="1 1 1"/>
    <texture name="cube" type="cube" builtin="flat" width="{tw}" height="{tw}" rgb1="1 1 1"/>
    <material name="mplane" texture="idx" texrepeat="{rx} {ry}" texuniform="{uniname}"
              specular="0" shininess="0" emission="1"/>
    <material name="mbox" texture="idx" texrepeat="{rx} {ry}" texuniform="{uniname}"
              specular="0" shininess="0" emission="1"/>
    <material name="mcube" texture="cube" texrepeat="{rx} {ry}" texuniform="{uniname}"
              specular="0" shininess="0" emission="1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true" ambient="0 0 0" diffuse="0 0 0"
           specular="0 0 0" castshadow="false"/>
    <geom name="plane" type="plane" pos="0 0 0" size="{px} {py} 0.1" material="mplane"/>
    <body pos="{bx} 0 1">
      <geom name="box" type="box" size="{sx} {sy} {sz}" material="mbox"/>
    </body>
    <body pos="{cx} 0 1">
      <geom name="cyl" type="cylinder" size="{sx} {sz}" material="mbox"/>
    </body>
    <body pos="{ex} 0 1">
      <geom name="sph" type="sphere" size="{sx}" material="mcube"/>
    </body>
    <camera name="top" pos="0 0 8" xyaxes="1 0 0 0 1 0" fovy="{fovy}" mode="fixed"/>
  </worldbody>
</mujoco>
"""


def build(tw=300, th=300, rx=1.0, ry=1.0, uni=0, px=1.0, py=1.0, fovy=40.0,
          sx=0.5, sy=0.5, sz=0.5, bx=10.0, cx=13.0, ex=16.0):
    import mujoco
    xml = XML.format(tw=tw, th=th, rx=rx, ry=ry, uniname="true" if uni else "false",
                     px=px, py=py, fovy=fovy, sx=sx, sy=sy, sz=sz, bx=bx, cx=cx, ex=ex)
    m = mujoco.MjModel.from_xml_string(xml)
    return m


def paint_index(m, name="idx"):
    """Overwrite a texture's data with its own texel index, so the frame inverts the texgen.

    R = i % 256 (column), G = j % 256 (row), B = 200 + (i//256) + 2*(j//256) so a wrapped index is
    still distinguishable. Written after compilation and before the Renderer is built, which is when
    the upload happens.
    """
    import mujoco
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
    w, h = int(m.tex_width[t]), int(m.tex_height[t])
    nc = int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[t])
    jj, ii = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    a = np.zeros((h, w, nc), np.uint8)
    a[..., 0] = ii % 256
    if nc > 1:
        a[..., 1] = jj % 256
    if nc > 2:
        a[..., 2] = 200 + (ii // 256) + 2 * (jj // 256)
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    buf[a0:a0 + w * h * nc] = a.reshape(-1)
    return t, w, h


def ortho_render(m, res, cam="top"):
    import mujoco
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=res, width=res)
    o = mujoco.MjvOption()
    r.update_scene(d, camera=cam, scene_option=o)
    for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
              mujoco.mjtRndFlag.mjRND_SKYBOX):
        r.scene.flags[f] = 0
    return r.render().copy(), r, d


def q1_plane(args):
    """Invert the plane texgen from the frame. Orthographic-free: use a very long focal instead."""
    import mujoco
    out = []
    for (rx, ry, uni, px, py) in [(1, 1, 0, 1.0, 1.0), (2, 2, 0, 1.0, 1.0), (1, 1, 1, 1.0, 1.0),
                                  (1, 1, 0, 2.0, 1.0), (1, 1, 1, 2.0, 1.0), (3, 1, 0, 2.0, 1.0)]:
        m = build(tw=args.tw, th=args.th, rx=rx, ry=ry, uni=uni, px=px, py=py, fovy=args.fovy)
        t, w, h = paint_index(m)
        img, r, d = ortho_render(m, args.res)
        cam = r.scene.camera
        campos = 0.5 * (np.asarray(cam[0].pos) + np.asarray(cam[1].pos))
        near = 0.5 * (cam[0].frustum_near + cam[1].frustum_near)
        top = 0.5 * (cam[0].frustum_top + cam[1].frustum_top)
        # perspective from (0,0,z) straight down: world x = (ndc_x * top / near) * z
        f = top / near
        ndc = (2.0 * (np.arange(args.res) + 0.5) / args.res - 1.0) * f
        wx = ndc * campos[2]
        wy = -ndc * campos[2]                       # row 0 is the TOP of the returned image
        X, Y = np.meshgrid(wx, wy)
        inside = (np.abs(X) < px - 1e-3) & (np.abs(Y) < py - 1e-3)
        # texel index recovered from the painted colours
        I = img[..., 0].astype(np.float64)
        J = img[..., 1].astype(np.float64)
        # fit u = a*x + b in TEXEL units, from the recovered column index (mod 256 -> use B channel)
        hi = img[..., 2].astype(np.int64) - 200
        Ifull = I + 256 * (hi % 2 if w > 256 else 0)
        Jfull = J + 256 * ((hi // 2) if h > 256 else 0)
        sel = inside
        ax = np.polyfit(X[sel], Ifull[sel], 1) if sel.sum() > 10 else [np.nan, np.nan]
        ay = np.polyfit(Y[sel], Jfull[sel], 1) if sel.sum() > 10 else [np.nan, np.nan]
        distinct_i = int(len(np.unique(Ifull[sel])))
        distinct_j = int(len(np.unique(Jfull[sel])))
        out.append({"texrepeat": [rx, ry], "texuniform": uni, "plane_half": [px, py],
                    "tex": [w, h], "npx": int(sel.sum()),
                    "du_dx_texels": float(ax[0]), "u0_texels": float(ax[1]),
                    "dv_dy_texels": float(ay[0]), "v0_texels": float(ay[1]),
                    "distinct_i": distinct_i, "distinct_j": distinct_j,
                    "u_per_x": float(ax[0]) / w, "u0": float(ax[1]) / w,
                    "v_per_y": float(ay[0]) / h, "v0": float(ay[1]) / h})
        emit("plane", out[-1])
        del r, m, d
    return out


def q2_prims(args):
    """Box / cylinder / sphere texgen, from a camera looking straight at each one."""
    import mujoco
    m = build(tw=args.tw, th=args.th, rx=1, ry=1, uni=0, px=1.0, py=1.0, fovy=args.fovy)
    paint_index(m)
    paint_index(m, "cube")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=args.res, width=args.res)
    o = mujoco.MjvOption()
    rows = []
    for name, look in [("box", (10.0, 0.0, 1.0)), ("cyl", (13.0, 0.0, 1.0)),
                       ("sph", (16.0, 0.0, 1.0))]:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance = 2.0
        cam.azimuth = 0.0
        cam.elevation = 0.0
        r.update_scene(d, camera=cam, scene_option=o)
        for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
                  mujoco.mjtRndFlag.mjRND_SKYBOX):
            r.scene.flags[f] = 0
        img = r.render().copy()
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        sel = img.sum(-1) > 0
        rows.append({"geom": name, "npx": int(sel.sum()),
                     "distinct_colours": int(len(np.unique(img[sel].reshape(-1, 3), axis=0))),
                     "i_range": [int(img[..., 0][sel].min()), int(img[..., 0][sel].max())],
                     "j_range": [int(img[..., 1][sel].min()), int(img[..., 1][sel].max())]})
        emit("prim", rows[-1])
        np.save(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/texgen_{name}.npy", img)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--tw", type=int, default=300)
    ap.add_argument("--th", type=int, default=300)
    ap.add_argument("--fovy", type=float, default=14.0)
    args = ap.parse_args()
    res = {"args": vars(args), "plane": q1_plane(args), "prims": q2_prims(args)}
    save("texgen.json", res)


if __name__ == "__main__":
    main()
