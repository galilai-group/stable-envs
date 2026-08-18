"""Where does MJWarp's residual disagreement with MuJoCo live?

Whole-image exact-pixel agreement conflates two very different things: a backend that
shades surfaces differently, and a backend that agrees on every surface but resolves
silhouettes differently. MuJoCo's GL renderer multisamples; MJWarp casts one ray per
pixel. That alone guarantees disagreement on a one-pixel band around every edge, and it
is not a fidelity problem -- it vanishes under any downsampling.

So: recompute agreement on interior pixels only, where "edge" is any pixel within
`radius` of a colour discontinuity in the MuJoCo reference.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("WARP_CACHE_PATH", "/mnt/vast/home/randall.balestriero/.cache/warp")

import mujoco
import numpy as np

from compare_agreement import mujoco_egl, mjwarp


def edge_mask(img, radius=1, thresh=8):
    g = img.astype(np.int32).sum(-1)
    d = np.zeros(g.shape, bool)
    d[:-1, :] |= np.abs(g[1:, :] - g[:-1, :]) > thresh
    d[1:, :] |= np.abs(g[1:, :] - g[:-1, :]) > thresh
    d[:, :-1] |= np.abs(g[:, 1:] - g[:, :-1]) > thresh
    d[:, 1:] |= np.abs(g[:, 1:] - g[:, :-1]) > thresh
    out = d.copy()
    for _ in range(radius):
        out[:-1, :] |= out[1:, :]
        out[1:, :] |= out[:-1, :]
        out[:, :-1] |= out[:, 1:]
        out[:, 1:] |= out[:, :-1]
    return out


def report(xml, res, cam, label, shadow=True, skybox=True):
    mjm = mujoco.MjModel.from_xml_path(xml)
    mjd = mujoco.MjData(mjm)
    rng = np.random.default_rng(3)
    mjd.qpos[:2] += rng.uniform(-2, 2, size=2)
    mujoco.mj_forward(mjm, mjd)

    ref = mujoco_egl(mjm, mjd, res, cam, shadow=shadow, reflection=False, skybox=skybox)
    got = mjwarp(xml, mjd, res, cam, use_shadows=shadow, render_skybox=skybox,
                 use_textures=True)[..., [2, 1, 0]]

    e = edge_mask(ref)
    a, b = ref.astype(np.int32), got.astype(np.int32)
    d = np.abs(a - b).max(-1)
    row = {
        "scene": label, "res": res, "edge_frac": float(e.mean()),
        "all": {"exact": float((d == 0).mean()), "mae": float(np.abs(a - b).mean()),
                "within_8": float((d <= 8).mean())},
        "interior": {"exact": float((d[~e] == 0).mean()), "mae": float(np.abs(a - b)[~e].mean()),
                     "within_8": float((d[~e] <= 8).mean())},
        "edges": {"exact": float((d[e] == 0).mean()), "mae": float(np.abs(a - b)[e].mean()),
                  "within_8": float((d[e] <= 8).mean())},
    }
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="antmaze_medium.xml,humanoidmaze_medium.xml,cube_single.xml")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenes"))
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    for scene in args.scenes.split(","):
        for sub, tag in ((args.dir, ""), (os.path.join(args.dir, "flat"), "_FLAT")):
            xml = os.path.join(sub, scene)
            if not os.path.exists(xml):
                continue
            mjm = mujoco.MjModel.from_xml_path(xml)
            cam = 0 if mjm.ncam > 0 else -1
            try:
                rows.append(report(xml, args.res, cam, scene[:-4] + tag))
            except Exception as e:
                print(json.dumps({"scene": scene + tag, "error": f"{type(e).__name__}: {e}"}), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
