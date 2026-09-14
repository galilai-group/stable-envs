"""Step 6G cpu-3d, Phase 1: what the frozen MJWarp frames are made of, per surface, per state.

Reads the frame stacks the texture-fidelity workstream left in its worktree (NOT committed there:
``localize312/frames/<cfg>.npz`` holds ``frozen_ours`` = our MJWarp frames at the frozen Step-4
states, ``frozen_official`` = dm_control's frames at the same states, ``frozen_seg`` = MuJoCo's own
segmentation buffer). It answers the questions a CPU design needs and a GPU cannot answer better:

  * how much of the frame is skybox / floor / body, and how big the movers' screen bbox is;
  * whether the SKYBOX is constant in camera space across states (a trackcom camera translates and
    never rotates, so it should be -- which makes 48-89% of the frame a bakeable plate);
  * whether the FLOOR is (it should not be: the checker phase moves with the camera).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CFGS = ("DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCHumanoidStand")
PLANE = 0


def surface_masks(seg: np.ndarray, plane_ids: set[int]):
    gid = seg[..., 0].astype(np.int64)
    sky = gid < 0
    floor = np.isin(gid, list(plane_ids))
    body = ~sky & ~floor
    return sky, floor, body


def bbox_stats(mask: np.ndarray) -> dict:
    H, W = mask.shape[1:]
    areas, hs, ws = [], [], []
    for mk in mask:
        ys, xs = np.nonzero(mk)
        if len(ys) == 0:
            areas.append(0); hs.append(0); ws.append(0); continue
        h = int(ys.max() - ys.min() + 1); w = int(xs.max() - xs.min() + 1)
        hs.append(h); ws.append(w); areas.append(h * w / (H * W))
    union = mask.any(0)
    ys, xs = np.nonzero(union)
    return {"bbox_share_median": float(np.median(areas)), "bbox_share_max": float(np.max(areas)),
            "bbox_h_median": float(np.median(hs)), "bbox_w_median": float(np.median(ws)),
            "union_bbox_share": float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1) / (H * W))
            if len(ys) else 0.0}


def constancy(frames: np.ndarray, mask: np.ndarray) -> dict:
    """Over pixels that carry this surface in >= 2 states: is the colour the same every time?"""
    N, H, W, _ = frames.shape
    f = frames.astype(np.int16)
    cnt = mask.sum(0)
    ok = cnt >= 2
    mx = np.full((H, W, 3), -1, np.int16)
    mn = np.full((H, W, 3), 999, np.int16)
    for i in range(N):
        mi = mask[i][..., None]
        mx = np.where(mi, np.maximum(mx, f[i]), mx)
        mn = np.where(mi, np.minimum(mn, f[i]), mn)
    spread = (mx - mn).max(-1)
    sp = spread[ok]
    return {"pixels_with_2plus_states": int(ok.sum()), "constant_frac": float((sp == 0).mean()) if sp.size else None,
            "within1_frac": float((sp <= 1).mean()) if sp.size else None,
            "spread_max": int(sp.max()) if sp.size else None, "spread_mean": float(sp.mean()) if sp.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=Path, required=True)
    ap.add_argument("--census-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    res = {}
    for c in CFGS:
        z = np.load(a.frames_dir / f"{c}.npz")
        ours, off, seg = z["frozen_ours"], z["frozen_official"], z["frozen_seg"]
        cen = json.load(open(next(a.census_dir.glob(f"{c}_*.json"))))
        plane_ids = {g["id"] for g in cen["geoms"] if g["type"] == "plane"}
        sky, floor, body = surface_masks(seg, plane_ids)
        d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
        r = {"n": int(ours.shape[0]), "size": int(ours.shape[1]),
             "sha256_frozen_ours": hashlib.sha256(np.ascontiguousarray(ours).tobytes()).hexdigest(),
             "share": {"sky": float(sky.mean()), "floor": float(floor.mean()), "body": float(body.mean())},
             "body_px_per_frame_median": float(np.median(body.reshape(len(body), -1).sum(1))),
             "body_bbox": bbox_stats(body),
             "sky_constant_across_states_ours": constancy(ours, sky),
             "sky_constant_across_states_official": constancy(off, sky),
             "floor_constant_across_states_ours": constancy(ours, floor),
             "distinct_colours_ours": {"sky": int(len(np.unique(ours[sky], axis=0))),
                                       "floor": int(len(np.unique(ours[floor], axis=0))),
                                       "body": int(len(np.unique(ours[body], axis=0)))},
             "ours_vs_official_mean_abs": {"all": float(d.mean()), "sky": float(d[sky].mean()),
                                           "floor": float(d[floor].mean()), "body": float(d[body].mean())},
             "ours_vs_official_within1": {"sky": float((d[sky] <= 1).mean()), "floor": float((d[floor] <= 1).mean()),
                                          "body": float((d[body] <= 1).mean())}}
        # the horizon row: where does floor end (per frame), i.e. how much of the frame is a pure function
        # of the ray direction (sky) vs of (ray direction, camera height) (floor)
        rows = [int(np.nonzero(fl.any(1))[0].min()) if fl.any() else -1 for fl in floor]
        r["floor_top_row"] = {"min": int(min(rows)), "max": int(max(rows))}
        res[c] = r
        print(f"{c:18s} share sky/floor/body = {r['share']['sky']:.3f}/{r['share']['floor']:.3f}/{r['share']['body']:.3f}  "
              f"body px {r['body_px_per_frame_median']:.0f} bbox {r['body_bbox']['bbox_share_median']:.3f} (max {r['body_bbox']['bbox_share_max']:.3f}) "
              f"sky const ours {r['sky_constant_across_states_ours']['constant_frac']} off {r['sky_constant_across_states_official']['constant_frac']} "
              f"floor const {r['floor_constant_across_states_ours']['constant_frac']:.3f} horizon rows {r['floor_top_row']}")
    (a.out / "frames_census.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
