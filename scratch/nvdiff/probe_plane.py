"""The plane texgen scale as a function of plane size, measured without ever letting the map wrap.

A synthetic 1x1 plane says the scale is ``0.5*repeat/(2*size)``; OGBench's real 16x16 floor says
``0.5*repeat/size``. One of those readings is an artefact, and the way to find out which is to remove
the thing that produces artefacts: with ``texrepeat`` small enough that u never leaves [0,1] there is
no unwrapping step to get wrong, and the slope is a straight least-squares fit.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, save  # noqa: E402

XML = """
<mujoco>
  <visual><global offwidth="1024" offheight="1024"/><quality offsamples="0"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/><map znear="0.01" zfar="200"/></visual>
  <asset>
    <texture name="idx" type="2d" builtin="flat" width="256" height="256" rgb1="1 1 1"/>
    <material name="mm" texture="idx" texrepeat="{rx} {ry}" texuniform="{uni}"
              specular="0" shininess="0" emission="1"/>
  </asset>
  <worldbody>
    <light pos="0 0 5" dir="0 0 -1" directional="true" ambient="0 0 0" diffuse="0 0 0"
           specular="0 0 0" castshadow="false"/>
    <geom name="g" type="plane" pos="{px} {py} 0" size="{sx} {sy} {sz}" material="mm"/>
    {extra}
    <camera name="top" pos="{px} {py} {camz}" xyaxes="1 0 0 0 1 0" fovy="45" mode="fixed"/>
  </worldbody>
</mujoco>
"""


def run(sx, sy, sz=0.1, rx=0.1, ry=0.1, uni=False, px=0.0, py=0.0, extra="", res=384):
    import mujoco
    camz = max(1.2 * max(sx, sy), 1.0)
    xml = XML.format(sx=sx, sy=sy, sz=sz, rx=rx, ry=ry, uni="true" if uni else "false",
                     px=px, py=py, camz=camz, extra=extra)
    m = mujoco.MjModel.from_xml_string(xml)
    t = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, "idx")
    w, h = int(m.tex_width[t]), int(m.tex_height[t])
    a0 = int(m.tex_adr[t])
    nc = int(m.tex_nchannel[t]) if hasattr(m, "tex_nchannel") else 3
    jj, ii = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    pat = np.zeros((h, w, nc), np.uint8)
    pat[..., 0] = np.round(ii * 255.0 / (w - 1))
    pat[..., 1] = np.round(jj * 255.0 / (h - 1))
    pat[..., 2] = 128
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    buf[a0:a0 + w * h * nc] = pat.reshape(-1)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=res, width=res)
    o = mujoco.MjvOption()
    r.update_scene(d, camera="top", scene_option=o)
    for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION,
              mujoco.mjtRndFlag.mjRND_SKYBOX):
        r.scene.flags[f] = 0
    rgb = r.render().copy()
    gl = r.scene.camera
    campos = 0.5 * (np.asarray(gl[0].pos, np.float64) + np.asarray(gl[1].pos, np.float64))
    near = 0.5 * (gl[0].frustum_near + gl[1].frustum_near)
    top = 0.5 * (gl[0].frustum_top + gl[1].frustum_top)
    f = top / near
    ndc = (2.0 * (np.arange(res) + 0.5) / res - 1.0) * f
    wx = campos[0] + ndc * campos[2]
    wy = campos[1] - ndc * campos[2]
    X, Y = np.meshgrid(wx, wy)
    inside = (np.abs(X - px) < sx - 1e-3) & (np.abs(Y - py) < sy - 1e-3)
    lx, ly = (X - px)[inside], (Y - py)[inside]
    u = (rgb[..., 0].astype(np.float64) * (w - 1) / 255.0 + 0.5)[inside] / w
    v = (rgb[..., 1].astype(np.float64) * (h - 1) / 255.0 + 0.5)[inside] / h
    A = np.stack([lx, np.ones_like(lx)], -1)
    cu = np.linalg.lstsq(A, u, rcond=None)[0]
    B = np.stack([ly, np.ones_like(ly)], -1)
    cv = np.linalg.lstsq(B, v, rcond=None)[0]
    out = {"sx": sx, "sy": sy, "sz": sz, "rep": [rx, ry], "uni": int(uni), "extra": bool(extra),
           "extent": float(m.stat.extent), "npx": int(inside.sum()),
           "du_dx": float(cu[0]), "u0": float(cu[1]), "res_u": float(np.abs(A @ cu - u).mean()),
           "dv_dy": float(cv[0]), "v0": float(cv[1]), "res_v": float(np.abs(B @ cv - v).mean())}
    out["scale_x"] = float(0.5 * rx / cu[0]) if cu[0] else float("nan")
    out["scale_y"] = float(-0.5 * ry / cv[0]) if cv[0] else float("nan")
    emit("plane", out)
    del r
    return out


def main():
    rows = []
    for s in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        rows.append(run(s, s))
    rows.append(run(16.0, 8.0))
    rows.append(run(1.0, 1.0, uni=True))
    rows.append(run(16.0, 16.0, uni=True))
    # does a far-away geom (which changes stat.extent) move the scale?
    far = '<body pos="60 0 0"><geom type="sphere" size="0.1"/></body>'
    rows.append(run(1.0, 1.0, extra=far))
    rows.append(run(16.0, 16.0, extra=far))
    # and does the plane's own position matter (OGBench's floor sits at (10,10,0))?
    rows.append(run(16.0, 16.0, px=10.0, py=10.0))
    save("plane.json", rows)


if __name__ == "__main__":
    main()
