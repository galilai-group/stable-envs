"""Invert the texgen on the REAL OGBench floor, not a synthetic plane, and compare it to ours.

The synthetic calibration says a plane's map is ``u = 0.25*repeat*x/size + 0.25``. A law fitted on
toy scenes and applied to a real one is exactly the step that goes wrong silently, so this repaints
the actual ``grid`` texture of the actual env with its own texel index and reads the mapping back off
MuJoCo's frame -- then asks ``stable_envs.render.nvdiff.texgen`` for its answer and prints both.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
from common import emit, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", default="point")
    ap.add_argument("--maze", default="medium")
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    import gymnasium
    import mujoco
    import ogbench  # noqa: F401

    from stable_envs.render import nvdiff as ND

    env = gymnasium.make(f"{args.loco}maze-{args.maze}-v0", render_mode="rgb_array",
                         ob_type="pixels", width=args.size, height=args.size)
    env.reset(options={"task_id": 1})
    u_ = env.unwrapped
    m, d = u_.model, u_.data
    m.vis.quality.offsamples = 0

    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    mid = int(m.geom_matid[floor])
    tex = ND._tex_of(m, mid)
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    emit("floor", {"geom": floor, "type": int(m.geom_type[floor]),
                   "size": np.asarray(m.geom_size[floor]).tolist(),
                   "pos": np.asarray(m.geom_pos[floor]).tolist(),
                   "matid": mid, "texid": tex, "tex_wh": [w, h],
                   "texrepeat": np.asarray(m.mat_texrepeat[mid]).tolist(),
                   "texuniform": int(m.mat_texuniform[mid]),
                   "mat_rgba": np.asarray(m.mat_rgba[mid]).tolist(),
                   "geom_rgba": np.asarray(m.geom_rgba[floor]).tolist()})

    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[tex])
    jj, ii = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    # A LINEAR ramp, not a wrapped index: the floor is minified here, so MuJoCo's sample is an
    # average over many texels, and an average of a linear ramp is still the ramp while an average of
    # a wrapped index is noise. This is the difference between measuring the texgen and measuring the
    # decode.
    pat = np.zeros((h, w, nc), np.uint8)
    pat[..., 0] = np.round(ii * 255.0 / (w - 1)).astype(np.uint8)
    pat[..., 1] = np.round(jj * 255.0 / (h - 1)).astype(np.uint8)
    pat[..., 2] = 128
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    buf[a0:a0 + w * h * nc] = pat.reshape(-1)
    # the shading multiplies the texture, so kill the lights: emission 1 leaves the texel value alone
    m.mat_emission[mid] = 1.0
    m.mat_specular[mid] = 0.0
    for li in range(m.nlight):
        m.light_diffuse[li] = 0
        m.light_specular[li] = 0
        m.light_ambient[li] = 0
    m.vis.headlight.diffuse[:] = 0
    m.vis.headlight.specular[:] = 0
    m.vis.headlight.ambient[:] = 0

    cam_mjv = u_.custom_camera
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=args.size, width=args.size)
    r.update_scene(d, camera=cam_mjv)
    for f in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
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
    r.update_scene(d, camera=cam_mjv)
    dep = r.render().copy()
    r.disable_depth_rendering()
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam_mjv)
    seg = r.render().copy()
    r.disable_segmentation_rendering()

    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, fwd)
    hh, mid_y = 0.5 * (top - bot), 0.5 * (top + bot)
    xs = (2.0 * (np.arange(args.size) + 0.5) / args.size - 1.0) * hh
    ys = mid_y + (1.0 - 2.0 * (np.arange(args.size) + 0.5) / args.size) * hh
    X, Y = np.meshgrid(xs, ys)
    dirs = X[..., None] * right + Y[..., None] * up + near * fwd
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    P = campos + (dep / np.einsum("hwk,k->hw", dirs, fwd))[..., None] * dirs

    sel = seg[..., 0] == floor
    gpos = np.asarray(d.geom_xpos[floor], np.float64)
    gmat = np.asarray(d.geom_xmat[floor], np.float64).reshape(3, 3)
    loc = np.einsum("ji,hwj->hwi", gmat, P - gpos)
    uu = (rgb[..., 0].astype(np.float64) * (w - 1) / 255.0 + 0.5) / w
    vv = (rgb[..., 1].astype(np.float64) * (h - 1) / 255.0 + 0.5) / h

    A = np.stack([loc[..., 0][sel], np.ones(int(sel.sum()))], -1)
    cu = np.linalg.lstsq(A, uu[sel], rcond=None)[0]
    B = np.stack([loc[..., 1][sel], np.ones(int(sel.sum()))], -1)
    cv = np.linalg.lstsq(B, vv[sel], rcond=None)[0]
    Am, bm = ND.texgen(m, floor, drawn_plane=2.0 * 20.0 * float(m.stat.extent))
    emit("fit", {"npx": int(sel.sum()),
                 "measured_du_dx": float(cu[0]), "measured_u0": float(cu[1]),
                 "resid_u": float(np.abs(A @ cu - uu[sel]).mean()),
                 "measured_dv_dy": float(cv[0]), "measured_v0": float(cv[1]),
                 "resid_v": float(np.abs(B @ cv - vv[sel]).mean()),
                 "ours_du_dx": float(Am[0, 0]), "ours_u0": float(bm[0]),
                 "ours_dv_dy": float(Am[1, 1]), "ours_v0": float(bm[1]),
                 "u_range": [float(uu[sel].min()), float(uu[sel].max())],
                 "v_range": [float(vv[sel].min()), float(vv[sel].max())],
                 "loc_x_range": [float(loc[..., 0][sel].min()), float(loc[..., 0][sel].max())],
                 "loc_y_range": [float(loc[..., 1][sel].min()), float(loc[..., 1][sel].max())],
                 "distinct_i": int(len(np.unique(rgb[..., 0][sel]))),
                 "distinct_j": int(len(np.unique(rgb[..., 1][sel])))})
    save(f"ogfloor_{args.loco}_{args.maze}.json", {"ok": True})


if __name__ == "__main__":
    main()
