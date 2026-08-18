"""The hole fraction as a function of the camera delta, on a CONTROLLED delta rather than the pool's.

``probe_holes.py`` answers the question for the deltas this env's own states happen to produce. This
one decouples it: bake one plate, then move the camera by exactly 1, 2, 5, ... mm along eight random
directions (and, separately, rotate it by exactly 0.1 ... 2 degrees) and measure

  * the fraction of target pixels no source pixel lands on,
  * the error against a TRUE render from that displaced pose,

so the answer is a curve in millimetres that any camera distribution can be read against, and the
quantiser in ``rc_pipeline.py`` can be set from it instead of from a guess.

Placing the camera at an arbitrary pose is the same mechanism the plate cache needs: ``cam_pos`` and
``cam_quat`` are per-world model fields, so the pose is expressed in the camera's parent body frame
and 64 arbitrary viewpoints are one launch. ``camera_error`` asserts that MJWarp put the camera where
it was asked to, because ``camlight``'s convention is a thing to check rather than to assume.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rp_common as RP  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj"


def render_poses(c, qrow, bpos, bmat, xpos, xmat):
    """Render the SAME qpos from ``len(xpos)`` arbitrary camera poses. Returns (rgb, depth, err)."""
    lp, lq = RP.local_from_world(bpos, bmat, xpos, xmat)
    n = len(xpos)
    rgb = torch.empty((n, c.res, c.res, 3), dtype=torch.uint8, device="cuda")
    dep = torch.empty((n, c.res, c.res), dtype=torch.float32, device="cuda")
    q = torch.as_tensor(np.repeat(qrow[None], min(n, c.nworld), 0), dtype=torch.float32,
                        device="cuda")
    err = (0.0, 0.0)
    for lo in range(0, n, c.nworld):
        hi = min(lo + c.nworld, n)
        k = hi - lo
        c.wp.to_torch(c.wd.qpos)[:k] = q[:k]
        RP.set_camera(c, lp[lo:hi], lq[lo:hi], k)
        RP.launch(c)
        e = RP.camera_error(c, xpos[lo:hi], xmat[lo:hi], k)
        err = (max(err[0], e[0]), max(err[1], e[1]))
        rgb[lo:hi] = c.bgra[:k][..., [2, 1, 0]]
        dep[lo:hi] = c.depth[:k]
    return rgb, dep, err


def rodrigues(axis: np.ndarray, deg: float) -> np.ndarray:
    a = axis / np.linalg.norm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--dirs", type=int, default=8)
    ap.add_argument("--mm", default="0,1,2,5,10,20,50,100,300")
    ap.add_argument("--deg", default="0,0.1,0.25,0.5,1,2,5")
    ap.add_argument("--scales", default="1,2")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import stable_envs as se

    if RP.family(a.env) != "RoboCasa":
        raise SystemExit("this probe drives cam_pos/cam_quat, which is the RoboCasa camera model")
    tag = a.tag or a.env
    mod = RP.module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    m = RP.model_of(mod, cls)
    cid = RP.cam_id_of(mod, cls, m)

    st = env.sample_states(64)
    qpos = RP.full_qpos(mod, cls, env, m, st)
    cpos, cmat, gp, gm = RP.kinematics_cpu(m, qpos, cid)
    mv = RP.moving_geoms(m, gp, gm, mod.VISIBLE_GROUPS)
    base = int(np.argmin(np.linalg.norm(cpos - cpos.mean(0), axis=1)))
    qrow = qpos[base]
    bpos, bmat = RP.parent_frame(m, cid, qrow)
    x0, R0 = cpos[base], cmat[base]

    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(a.dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    axes = rng.normal(size=(a.dirs, 3))

    mm = [float(x) for x in a.mm.split(",")]
    dg = [float(x) for x in a.deg.split(",")]
    tgt_pos = np.concatenate([x0[None] + (d / 1000.0) * dirs for d in mm]
                             + [np.repeat(x0[None], a.dirs, 0) for _ in dg])
    tgt_mat = np.concatenate([np.repeat(R0[None], a.dirs, 0) for _ in mm]
                             + [np.stack([rodrigues(axes[j], g) @ R0 for j in range(a.dirs)])
                                for g in dg])
    label = ([("mm", d) for d in mm for _ in range(a.dirs)]
             + [("deg", g) for g in dg for _ in range(a.dirs)])

    c1 = RP.make_ctx(mod, cls, m, a.size, a.nworld, hide=mv, depth=True)
    cam_full = RP.Pinhole(c1.ray0, a.size)
    rt = cam_full.roundtrip_err(c1.ray0)
    assert rt < 1e-3, f"the ray table does not invert: {rt} px"
    truth, _, err = render_poses(c1, qrow, bpos, bmat, tgt_pos, tgt_mat)
    print(f"ray round trip {rt:.2e} px; camera placement error {err[0]:.2e} m / {err[1]:.2e}",
          flush=True)
    assert err[0] < 1e-5 and err[1] < 1e-5, "MJWarp did not put the camera where it was asked to"
    truth_np = truth.cpu().numpy()
    del c1, truth
    mod._CTX_CACHE.clear()
    torch.cuda.empty_cache()

    rows = []
    dumps = {}
    for scale in [int(x) for x in a.scales.split(",")]:
        c = RP.make_ctx(mod, cls, m, a.size * scale, a.nworld, hide=mv, depth=True)
        rgb_p, dep_p, _ = render_poses(c, qrow, bpos, bmat, x0[None], R0[None])
        ray_s = c.ray0.to(torch.float32).reshape(-1, 3)
        xa = torch.as_tensor(x0[None], dtype=torch.float32, device="cuda")
        Ra = torch.as_tensor(R0[None], dtype=torch.float32, device="cuda")
        P = RP.unproject(ray_s, dep_p.reshape(1, -1), xa, Ra)
        src = rgb_p.reshape(1, -1, 3)

        n = len(tgt_pos)
        hf, raw, filled = [], [], []
        for lo in range(0, n, 16):
            hi = min(lo + 16, n)
            k = hi - lo
            xb = torch.as_tensor(tgt_pos[lo:hi], dtype=torch.float32, device="cuda")
            Rb = torch.as_tensor(tgt_mat[lo:hi], dtype=torch.float32, device="cuda")
            rgb, dep, hole = RP.splat(P.expand(k, -1, -1), src.expand(k, -1, -1), xb, Rb,
                                      cam_full, a.size)
            rgbf, _, _ = RP.fill_holes(rgb, dep, hole)
            hf.append(hole.float().mean((1, 2)).cpu().numpy())
            raw.append(rgb.cpu().numpy())
            filled.append(rgbf.cpu().numpy())
        hf = np.concatenate(hf)
        raw = np.concatenate(raw)
        filled = np.concatenate(filled)

        for kind in ("mm", "deg"):
            for v in (mm if kind == "mm" else dg):
                idx = np.array([i for i, l in enumerate(label) if l == (kind, v)])
                row = {"scale": scale, "kind": kind, "delta": v,
                       "hole_frac_mean": float(hf[idx].mean()),
                       "hole_frac_max": float(hf[idx].max()),
                       "raw": RP.metrics(raw[idx], truth_np[idx]),
                       "filled": RP.metrics(filled[idx], truth_np[idx])}
                rows.append(row)
                print(f"s{scale} {kind}={v:g}: hole {row['hole_frac_mean']*100:6.3f}% "
                      f"raw {row['raw']['mean_abs']:6.3f} ssim {row['raw']['ssim']:.4f} | "
                      f"filled {row['filled']['mean_abs']:6.3f} ssim {row['filled']['ssim']:.4f}",
                      flush=True)
        for v in (20.0, 50.0):
            if ("mm", v) in label:
                i = label.index(("mm", v))
                dumps[(scale, v)] = (truth_np[i], filled[i], raw[i])
        del c, P, src
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()

    with open(os.path.join(OUT, f"delta_{tag}.json"), "w") as f:
        json.dump({"env": a.env, "base_state": base, "rows": rows}, f, indent=1)
    from PIL import Image
    for (scale, v), (t, f_, r) in dumps.items():
        d = np.abs(t.astype(np.int32) - f_.astype(np.int32)).max(-1)
        d = np.stack([np.clip(d * 8, 0, 255).astype(np.uint8)] * 3, -1)
        p = os.path.join(OUT, f"delta_{tag}_s{scale}_{int(v)}mm.png")
        Image.fromarray(np.concatenate([t, f_, r, d], 1)).save(p)
        print(f"wrote {p}", flush=True)
    print("JSON " + json.dumps({"env": a.env, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
