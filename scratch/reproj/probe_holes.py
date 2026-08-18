"""THE decisive curve: what does a baked plate cost when the camera moves by a centimetre?

"There is no static view to bake" was concluded from EXACT camera poses -- all four RoboCasa cameras
are bolted to the robot, so ``cam_xpos`` moves 2.2-3.4 cm (room) and 59 cm (wrist) over the env's own
state pool and no cache keyed on the pose can ever hit. This asks the question real-time graphics
asks instead: bake ONE plate with its depth buffer and REPROJECT it to the new camera, and measure

  * the hole fraction -- target pixels no source pixel lands on -- as a function of the camera delta,
  * the error on the pixels that ARE covered, against the true static render from the new pose,

so the viability of the whole idea is decided by a curve rather than by an argument. Nothing is timed
here; ``rc_pipeline.py`` does that. This answers "is the picture right", and only if it is does the
throughput matter.

Plates are baked at several resolutions. At 1x one source pixel lands on one target pixel, so
magnification tears holes even where nothing is disoccluded; at 2x and 4x there are ~4 and ~16
candidates per target pixel, which is the standard answer to that and costs nothing at run time
because the plate is baked once and reused by the whole batch. Each is scored raw and after the
cheapest hole fill there is (3x3 dilation of the filled region into the holes).

``--plates K`` additionally quantises: cluster the pool's camera poses into K cells, bake one plate
per cell, reproject each state from ITS cell. K=1 is one plate for the whole pool.
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


def rot_angle(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """``(N,3,3)`` against one ``(3,3)`` -> degrees between the two orientations."""
    t = np.einsum("nij,ij->n", A, B)
    return np.degrees(np.arccos(np.clip((t - 1.0) / 2.0, -1.0, 1.0)))


def render_states(c, qpos: np.ndarray):
    """Render ``qpos`` through ``c`` in chunks of ``c.nworld``; returns (rgb, depth) on device."""
    wp = c.wp
    n = len(qpos)
    rgb = torch.empty((n, c.res, c.res, 3), dtype=torch.uint8, device="cuda")
    dep = torch.empty((n, c.res, c.res), dtype=torch.float32, device="cuda")
    q = torch.as_tensor(qpos, dtype=torch.float32, device="cuda")
    for lo in range(0, n, c.nworld):
        hi = min(lo + c.nworld, n)
        k = hi - lo
        wp.to_torch(c.wd.qpos)[:k] = q[lo:hi]
        RP.launch(c)
        rgb[lo:hi] = c.bgra[:k][..., [2, 1, 0]]
        dep[lo:hi] = c.depth[:k]
    return rgb, dep


def medoids(x: np.ndarray, k: int) -> np.ndarray:
    """``k`` indices into ``x`` by farthest-point seeding then one Lloyd step. No sklearn."""
    if k <= 1:
        return np.array([int(np.argmin(np.linalg.norm(x - x.mean(0), axis=1)))])
    sel = [int(np.argmin(np.linalg.norm(x - x.mean(0), axis=1)))]
    d = np.linalg.norm(x - x[sel[0]], axis=1)
    while len(sel) < min(k, len(x)):
        sel.append(int(np.argmax(d)))
        d = np.minimum(d, np.linalg.norm(x - x[sel[-1]], axis=1))
    sel = np.array(sel)
    lab = np.argmin(((x[:, None] - x[sel][None]) ** 2).sum(-1), 1)
    for j in range(len(sel)):
        mem = np.flatnonzero(lab == j)
        if len(mem):
            cen = x[mem].mean(0)
            sel[j] = mem[int(np.argmin(np.linalg.norm(x[mem] - cen, axis=1)))]
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--scales", default="1,2,4")
    ap.add_argument("--plates", default="1,4,16")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import stable_envs as se

    tag = a.tag or a.env
    scales = [int(x) for x in a.scales.split(",")]
    kplates = [int(x) for x in a.plates.split(",")]
    mod = RP.module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    m = RP.model_of(mod, cls)
    cid = RP.cam_id_of(mod, cls, m)

    st = env.sample_states(a.n)
    qpos = RP.full_qpos(mod, cls, env, m, st)
    cpos, cmat, gp, gm = RP.kinematics_cpu(m, qpos, cid)
    mv = RP.moving_geoms(m, gp, gm, mod.VISIBLE_GROUPS)

    ref = int(np.argmin(np.linalg.norm(cpos - cpos.mean(0), axis=1)))
    dt = np.linalg.norm(cpos - cpos[ref], axis=1)
    dr = rot_angle(cmat, cmat[ref])
    facts = {"env": a.env, "n": a.n, "size": a.size, "n_movers": int(len(mv)),
             "cam_xpos_spread_m": float(np.abs(cpos - cpos.mean(0)).max()),
             "cam_delta_m": {"mean": float(dt.mean()), "p50": float(np.median(dt)),
                             "max": float(dt.max())},
             "cam_delta_deg": {"mean": float(dr.mean()), "p50": float(np.median(dr)),
                               "max": float(dr.max())},
             "distinct_cam_poses": int(len(np.unique(np.round(cpos, 9), axis=0)))}
    print("facts " + json.dumps(facts), flush=True)

    # The truth to beat, once: the static scene from EVERY state's own camera, at target resolution.
    c1 = RP.make_ctx(mod, cls, m, a.size, a.nworld, hide=mv, depth=True)
    cam_full = RP.Pinhole(c1.ray0, a.size)
    rt = cam_full.roundtrip_err(c1.ray0)
    assert rt < 1e-3, f"the ray table does not invert: {rt} px"
    truth_rgb, _ = render_states(c1, qpos)
    truth = truth_rgb.cpu().numpy()
    del c1
    mod._CTX_CACHE.clear()
    torch.cuda.empty_cache()
    print(f"ray-table round trip {rt:.2e} px; truth rendered", flush=True)

    rows, dumps = [], {}
    for scale in scales:
        res_s = a.size * scale
        c = RP.make_ctx(mod, cls, m, res_s, a.nworld, hide=mv, depth=True)
        ray_s = c.ray0.to(torch.float32).reshape(-1, 3)
        for K in kplates:
            sel = medoids(cpos, K)
            lab = np.argmin(((cpos[:, None] - cpos[sel][None]) ** 2).sum(-1), 1)
            rgb_p, dep_p = render_states(c, qpos[sel])
            xa = torch.as_tensor(cpos[sel], dtype=torch.float32, device="cuda")
            Ra = torch.as_tensor(cmat[sel], dtype=torch.float32, device="cuda")
            P = RP.unproject(ray_s, dep_p.reshape(len(sel), -1), xa, Ra)
            src = rgb_p.reshape(len(sel), -1, 3)

            hf, hfa, raw, filled = [], [], [], []
            for lo in range(0, a.n, 16):
                hi = min(lo + 16, a.n)
                w = torch.as_tensor(lab[lo:hi], dtype=torch.long, device="cuda")
                xb = torch.as_tensor(cpos[lo:hi], dtype=torch.float32, device="cuda")
                Rb = torch.as_tensor(cmat[lo:hi], dtype=torch.float32, device="cuda")
                rgb, dep, hole = RP.splat(P.index_select(0, w), src.index_select(0, w),
                                          xb, Rb, cam_full, a.size)
                rgbf, _, holef = RP.fill_holes(rgb, dep, hole)
                hf.append(hole.float().mean((1, 2)).cpu().numpy())
                hfa.append(holef.float().mean((1, 2)).cpu().numpy())
                raw.append(rgb.cpu().numpy())
                filled.append(rgbf.cpu().numpy())
            hf = np.concatenate(hf)
            hfa = np.concatenate(hfa)
            raw = np.concatenate(raw)
            filled = np.concatenate(filled)
            dt_k = np.linalg.norm(cpos - cpos[sel][lab], axis=1)
            row = {"scale": scale, "plate_res": res_s, "K": K,
                   "delta_to_plate_m": {"mean": float(dt_k.mean()), "max": float(dt_k.max())},
                   "hole_frac": {"mean": float(hf.mean()), "max": float(hf.max())},
                   "hole_frac_after_fill": {"mean": float(hfa.mean()), "max": float(hfa.max())},
                   "raw_vs_true_static": RP.metrics(raw, truth),
                   "filled_vs_true_static": RP.metrics(filled, truth)}
            rows.append(row)
            print(f"scale {scale} K {K}: " + json.dumps(row), flush=True)
            if K == 1:
                worst = int(np.argmax(dt))
                dumps[scale] = {"truth": truth[worst], "filled": filled[worst], "raw": raw[worst],
                                "i": worst, "dt": float(dt[worst])}
            del P, src, rgb_p, dep_p
            torch.cuda.empty_cache()
        del c
        mod._CTX_CACHE.clear()
        torch.cuda.empty_cache()

    curve = [{"i": int(i), "dt_m": float(dt[i]), "dr_deg": float(dr[i])} for i in range(a.n)]
    with open(os.path.join(OUT, f"holes_{tag}.json"), "w") as f:
        json.dump({"facts": facts, "rows": rows, "curve": curve}, f, indent=1)
    for scale, d in dumps.items():
        _dump(os.path.join(OUT, f"holes_{tag}_s{scale}.png"), d)
    print("JSON " + json.dumps({"facts": facts, "rows": rows}), flush=True)


def _dump(path: str, d: dict) -> None:
    """truth | reprojected+filled | reprojected raw | 8x amplified |difference|."""
    from PIL import Image
    t, f_, r = d["truth"], d["filled"], d["raw"]
    diff = np.abs(t.astype(np.int32) - f_.astype(np.int32)).max(-1)
    diff = np.stack([np.clip(diff * 8, 0, 255).astype(np.uint8)] * 3, -1)
    Image.fromarray(np.concatenate([t, f_, r, diff], 1)).save(path)
    print(f"wrote {path} (state {d['i']}, delta {d['dt']*100:.2f} cm)", flush=True)


if __name__ == "__main__":
    main()
