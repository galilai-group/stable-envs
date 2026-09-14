"""Official 3.11.0 vs official 3.12.0 on the same 64 qpos: is quadruped-walk's floor immune to the
3.12 finite-plane texture re-anchoring (texture-fidelity findings.md S2)? Per surface, off the
segmentation buffer (objid < 0 sky, geom 0 floor, geoms > 0 body)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from stable_envs.testing import conformance as C  # noqa: E402


def per_surface(a, b, segs):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    objid, objtype = segs[..., 0], segs[..., 1]
    geom = objtype == 5
    masks = {"sky": objid < 0, "floor": geom & (objid == 0), "body": geom & (objid > 0),
             "ellipsoid": geom & (objid == 3), "site": objtype == 6}
    out = {}
    for k, msk in masks.items():
        n = int(msk.sum())
        out[k] = {"share": float(msk.mean()), "mean_abs": float(d[msk].mean()) if n else None,
                  "exact": float((d[msk] == 0).mean()) if n else None,
                  "within8": float((d[msk] <= 8).mean()) if n else None,
                  "resid_share": float(d[msk].sum() / max(d.sum(), 1))}
    return out


def main():
    corpus, out = sys.argv[1], sys.argv[2]
    res = {}
    for cam in (2, 0):
        for size in (84, 224):
            a = np.load(f"{corpus}/official_mj3110_cam{cam}_{size}px.npz")
            b = np.load(f"{corpus}/official_mj3120_cam{cam}_{size}px.npz")
            r = C.pixel_agreement(a["frames"], b["frames"])
            r["ssim"] = C.ssim(a["frames"], b["frames"])
            r["per_surface_(3.11 segs)"] = per_surface(a["frames"], b["frames"], a["segs"])
            r["segs_identical"] = bool(np.array_equal(a["segs"], b["segs"]))
            res[f"cam{cam}_{size}px"] = r
            print(f"cam{cam} {size}px: 3.11 vs 3.12 mean {r['mean_abs']:.3f} exact {r['exact_frac']:.4f} "
                  f"within8 {r['within8_frac']:.4f} max {r['max_abs']} ssim {r['ssim']:.4f} | floor mean "
                  f"{r['per_surface_(3.11 segs)']['floor']['mean_abs']:.3f} sky "
                  f"{r['per_surface_(3.11 segs)']['sky']['mean_abs']:.3f} body "
                  f"{r['per_surface_(3.11 segs)']['body']['mean_abs']:.3f} segs_identical={r['segs_identical']}")
    json.dump(res, open(f"{out}/official_311_vs_312.json", "w"), indent=1)


if __name__ == "__main__":
    main()
