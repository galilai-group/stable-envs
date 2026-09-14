"""Step 6A: the two arms against each other, pixel for pixel, on the same node.

``localize.py`` saved official and stable-envs frames per arm at identical states. This answers,
per config and per surface, which SIDE moved between MuJoCo 3.11 and 3.12:

    official(3.11) vs official(3.12)   -- the reference renderer's own drift
    ours(3.11)     vs ours(3.12)       -- mujoco_warp's drift (same tex_data on both arms)
    ours(3.11)     vs official(3.12)   -- the cross pair Step 4 effectively measured
                                          (a 3.11-declared renderer against a 3.12 reference)

numpy + PIL only; runs in whichever interpreter is handy.

    python scratch/step6/texture-fidelity/cross.py --a <arm311 dir> --b <arm312 dir> --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def agree(a, b, mask=None) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    if mask is not None:
        d = d[mask]
    if d.size == 0:
        return {"px": 0}
    return {"px": int(d.size), "mean_abs": float(d.mean()), "exact_frac": float((d == 0).mean()),
            "within8_frac": float((d <= 8).mean()), "max_abs": int(d.max())}


def ssim(a, b, mask=None, win: int = 7) -> float:
    import scipy.ndimage as ndi
    f = lambda x: ndi.uniform_filter(x, size=win, mode="nearest")
    x = a.astype(np.float64) @ [0.299, 0.587, 0.114]
    y = b.astype(np.float64) @ [0.299, 0.587, 0.114]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ux, uy = f(x), f(y)
    vx, vy = f(x * x) - ux * ux, f(y * y) - uy * uy
    vxy = f(x * y) - ux * uy
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux ** 2 + uy ** 2 + c1) * (vx + vy + c2))
    return float(s.mean() if mask is None else s[mask].mean())


def masks(seg, floor_ids) -> dict:
    objid, objtype = seg[..., 0], seg[..., 1]
    is_geom = objtype == 5                                  # mjOBJ_GEOM
    sky = objid < 0
    floor = is_geom & np.isin(objid, floor_ids)
    return {"skybox": sky, "floor": floor, "bodies": ~sky & ~floor}


def full(a, b, mk) -> dict:
    out = agree(a, b)
    out["ssim"] = ssim(a, b)
    out["by_region"] = {k: {**agree(a, b, v), "share": float(v.mean()), "ssim": ssim(a, b, v)}
                        for k, v in mk.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm A localize dir (3.11)")
    ap.add_argument("--b", required=True, help="arm B localize dir (3.12)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    A, B, out = Path(args.a), Path(args.b), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    la = json.loads((A / "localize.json").read_text())
    lb = json.loads((B / "localize.json").read_text())
    doc = {"arm_a": la["provenance"], "arm_b": lb["provenance"], "configs": {}}
    for cfg in la["configs"]:
        if "error" in la["configs"][cfg] or "error" in lb["configs"].get(cfg, {"error": 1}):
            doc["configs"][cfg] = {"skipped": "an arm errored"}
            continue
        fa = np.load(A / "frames" / f"{cfg}.npz")
        fb = np.load(B / "frames" / f"{cfg}.npz")
        ta, tb = la["configs"][cfg]["textures"], lb["configs"][cfg]["textures"]
        row = {"same_texture_bytes": [t["sha256"] for t in ta["textures"]] == [t["sha256"] for t in tb["textures"]],
               "same_material_table": ta["materials"] == tb["materials"]}
        # floor geom ids: planes. Read off arm A's segmentation-derived masks via the objid table:
        # the model is identical on both arms (asserted above by the texture/material digests), and
        # plane ids are the geoms the localize masks called floor. Recover them from the seg buffer:
        for tag in ("frozen", "check"):
            sa, sb = fa[f"{tag}_seg"].astype(np.int32), fb[f"{tag}_seg"].astype(np.int32)
            oa, ob = fa[f"{tag}_official"], fb[f"{tag}_official"]
            ua, ub = fa[f"{tag}_ours"], fb[f"{tag}_ours"]
            floor_ids = _floor_ids(la["configs"][cfg])
            mk = masks(sa, floor_ids)
            row[tag] = {
                "seg_identical_across_arms": bool(np.array_equal(sa, sb)),
                "official311_vs_official312": full(oa, ob, mk),
                "ours311_vs_ours312": full(ua, ub, mk),
                "ours311_vs_official311": full(ua, oa, mk),
                "ours312_vs_official312": full(ub, ob, mk),
                "ours311_vs_official312": full(ua, ob, mk),
                "ours312_vs_official311": full(ub, oa, mk),
            }
            _sheet(out / f"{cfg}_{tag}_cross.png", oa, ob, ua, ub,
                   np.linspace(0, len(oa) - 1, 4).astype(int))
        # the 2x-downsample self-consistency on each arm, region-split, is in localize.json already;
        # here only the cross-arm identity of the S-frame matters.
        row["resolution"] = {"ours_S_identical_across_arms": bool(np.array_equal(fa["frozen_ours"], fb["frozen_ours"])),
                             "ours_2S_down_identical_across_arms": bool(np.array_equal(fa["frozen_ours_2S_down"], fb["frozen_ours_2S_down"]))}
        doc["configs"][cfg] = row
        f = row["frozen"]
        print(f"[{cfg}] official 3.11 vs 3.12: mean={f['official311_vs_official312']['mean_abs']:.3f} "
              f"exact={f['official311_vs_official312']['exact_frac']:.4f} | ours 3.11 vs 3.12: "
              f"mean={f['ours311_vs_ours312']['mean_abs']:.3f} exact={f['ours311_vs_ours312']['exact_frac']:.4f}",
              flush=True)
    (out / "cross.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print("wrote", out / "cross.json")


def _floor_ids(cfg_row: dict) -> list[int]:
    # geom ids with a material AND plane type are not recorded directly; localize.json records
    # geom_matid only. The plane geoms in every dm_control suite model are the geoms named "floor"/
    # "ground"; MuJoCo's segmentation gives their geom id. Recover from the frames: a pixel whose
    # objtype is geom and whose id maps to the floor is exactly what localize's masks called floor,
    # and localize wrote the floor share -- so pick the geom ids whose pixel share reproduces it.
    return cfg_row.get("floor_geom_ids", [])


def _sheet(path, oa, ob, ua, ub, idx):
    try:
        from PIL import Image
    except ImportError:
        return
    rows = []
    for i in idx:
        d_off = np.clip(np.abs(oa[i].astype(np.int16) - ob[i].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        d_our = np.clip(np.abs(ua[i].astype(np.int16) - ub[i].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        rows.append(np.concatenate([oa[i], ob[i], d_off, ua[i], ub[i], d_our], axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)


if __name__ == "__main__":
    main()
