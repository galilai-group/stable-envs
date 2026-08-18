"""Two escape hatches from the checker floor, measured rather than argued.

1. SUPERSAMPLING. MJWarp casts one ray per pixel and never mip-filters; the standard fix for a
   point-sampled minified texture is to cast more rays and box-average. This repo already
   supersamples four other MuJoCo domains, so it is the first thing to try rather than the last.
   The question is not "does it help" (it must) but "does it reach exactness", because the far field
   at 240 px has hundreds of floor texels inside one pixel and 4x4 samples is not hundreds.

2. ``use_fast_math=False``. A compile-time flag on the megakernel that no measurement in this repo
   has ever varied.

Both are scored on the whole frame, on floor pixels, and on the FAR band of the floor, because that
is where the minification claim lives.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/mujoco_locomotion")

from mjw_common import agreement, interior_mask, jsave, make_env, mj_render, mj_segment, sample_qpos, track_cam
from mjw_probe4_resolved import build_mjw, resolved_model

SIZE = 240
NSTATE = 16


def downsample(img, ss):
    if ss == 1:
        return img
    n, r, _, c = img.shape
    s = r // ss
    return (img.astype(np.float32).reshape(n, s, ss, s, ss, c).mean((2, 4))
            .round().clip(0, 255).astype(np.uint8))


def run(env_id, size=SIZE, n=NSTATE):
    import mujoco
    out = {"env": env_id, "size": size, "n_states": n, "runs": {}}
    env, u = make_env(env_id, size)
    cam = track_cam(u.model)
    st = sample_qpos(env_id, n, seed=0)
    m, _ = resolved_model(u)
    d = mujoco.MjData(m)

    ref = mj_render(m, d, cam, size, st, tag="ref")
    seg = mj_segment(m, d, cam, size, st, tag="seg")
    objid, objtype = seg[..., 0], seg[..., 1]
    isgeom = objtype == int(mujoco.mjtObj.mjOBJ_GEOM)
    plane = np.array([i for i in range(int(m.ngeom)) if int(m.geom_type[i]) == 0], np.int64)
    floor = isgeom & np.isin(objid, plane)
    rows = np.nonzero(floor.any(0).any(1))[0]
    lo, hi = int(rows.min()), int(rows.max())
    band = max(1, (hi - lo) // 10)
    far = floor.copy(); far[:, lo + band:] = False
    im = interior_mask(ref)

    for tag, ss, fm in (("ss1", 1, True), ("ss2", 2, True), ("ss4", 4, True),
                        ("ss1_nofastmath", 1, False)):
        W = build_mjw(m, cam, size * ss, n, fast_math=fm)
        got = downsample(W.render(st), ss)
        a = agreement(got, ref)
        dif = np.abs(got.astype(np.int16) - ref.astype(np.int16)).max(-1)
        a["floor_all_exact"] = round(float((dif[floor] == 0).mean()), 6)
        a["floor_mean"] = round(float(dif[floor].mean()), 4)
        a["far_all_exact"] = round(float((dif[far] == 0).mean()), 6)
        a["far_mean"] = round(float(dif[far].mean()), 4)
        a["far_ref_std"] = round(float(ref.astype(np.float64).mean(-1)[far].std()), 3)
        a["far_mjw_std"] = round(float(got.astype(np.float64).mean(-1)[far].std()), 3)
        mm = far[:, :-1, :-1] & im
        a["far_interior_exact"] = (round(float((dif[:, :-1, :-1][mm] == 0).mean()), 6)
                                   if mm.any() else None)
        out["runs"][tag] = a
        print(f"  {tag:16s} all={a['exact_frac']:.4f} int={a['interior_exact']} "
              f"floor={a['floor_all_exact']:.4f} far={a['far_all_exact']:.6f} "
              f"far_mean={a['far_mean']:.2f} far_std ref={a['far_ref_std']} mjw={a['far_mjw_std']}",
              flush=True)
    return out


if __name__ == "__main__":
    e = sys.argv[1]
    print("===", e, flush=True)
    r = run(e)
    jsave(f"/mnt/vast/home/randall.balestriero/stable_envs_logs/mjloco/"
          f"probe5_{e.split('-')[0]}.json", r)
