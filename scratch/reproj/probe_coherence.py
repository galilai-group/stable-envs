"""Is a windowed pass paying for the rays it casts, or for where they point?

``scratch/perf/window_dyn.py`` measures a RECTANGLE window and finds the cost proportional to the ray
count (0.25 of the rays -> 3.01x). ``rc_pipeline.py`` opens the same number of rays as an arbitrary
SET -- the union of the movers' own screen boxes, which is 0.30 of the frame per state against the
0.50 its bounding rectangle needs -- and gets nothing back. The two cannot both be a statement about
ray count, so this holds the count fixed and varies only the SHAPE:

  ``rect``     one rectangle, the coherent case ``window_dyn.py`` measured
  ``mask``     the union of the movers' per-geom boxes, padded to the same count
  ``random``   the same count, drawn uniformly over the frame -- the incoherent limit
  ``strided``  every k-th pixel, coherent in neither axis but perfectly regular

Same context, same scene, same worlds, same number of threads. Whatever separates them is not work.
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
from rc_pipeline import aabb_corners, clip_to_near, geom_boxes, ray_mask  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--keep", default="dyn", choices=["dyn", "full"])
    a = ap.parse_args()

    import stable_envs as se

    mod = RP.module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    m = RP.model_of(mod, cls)
    cid = RP.cam_id_of(mod, cls, m)
    S = a.size

    st = env.sample_states(a.nworld)
    qpos = RP.full_qpos(mod, cls, env, m, st)
    cpos, cmat, gp, gm = RP.kinematics_cpu(m, qpos, cid)
    mv = RP.moving_geoms(m, gp, gm, mod.VISIBLE_GROUPS)
    vis = set(int(g) for g in mod.VISIBLE_GROUPS)
    static = np.array([g for g in range(m.ngeom)
                       if int(m.geom_group[g]) in vis and g not in set(mv.tolist())], np.int64)
    hide = static if a.keep == "dyn" else np.array([], np.int64)

    base = RP.make_ctx(mod, cls, m, S, a.nworld, hide=hide, depth=True, skybox=False)
    cam = RP.Pinhole(base.ray0, S)
    ray = base.ray0.clone()
    q = torch.as_tensor(qpos, dtype=torch.float32, device="cuda")
    base.wp.to_torch(base.wd.qpos)[:] = q
    RP.launch(base)
    t_full = RP.timed(lambda: base.wp.capture_launch(base.graph), a.reps)
    rows = {"full": {"ms": t_full * 1e3, "rays": S * S, "ray_frac": 1.0}}
    print(f"full        {t_full*1e3:8.3f} ms", flush=True)

    xpos = torch.as_tensor(cpos, dtype=torch.float32, device="cuda")
    xmat = torch.as_tensor(cmat, dtype=torch.float32, device="cuda")
    corners = torch.as_tensor(aabb_corners(m, mv), dtype=torch.float32, device="cuda")
    gx = torch.as_tensor(gp[:, mv], dtype=torch.float32, device="cuda")
    gmm = torch.as_tensor(gm[:, mv], dtype=torch.float32, device="cuda")
    cor = (gx.unsqueeze(2) + torch.einsum("ngij,gcj->ngci", gmm, corners)).reshape(len(gp), -1, 3)
    pc = torch.einsum("nji,ncj->nci", xmat, cor - xpos.unsqueeze(1))
    znear = float(getattr(base.rc, "znear", 0.01))
    gb = geom_boxes(clip_to_near(pc, len(mv), znear), cam, S, znear)
    keep = ray_mask(gb.reshape(-1, 4), S)
    mask_idx = torch.nonzero(keep).squeeze(1)
    n_mask = int(mask_idx.numel())
    side = int(np.ceil(np.sqrt(n_mask)) // 8 * 8)
    while side * side < n_mask:
        side += 8
    side = min(side, S)
    print(f"mask rays {n_mask} ({n_mask/(S*S):.3f} of the frame) -> side {side}", flush=True)

    ub = torch.stack([gb[..., 0].amin().reshape(1), gb[..., 1].amax().reshape(1),
                      gb[..., 2].amin().reshape(1), gb[..., 3].amax().reshape(1)]).reshape(-1)
    cc, rc_ = int((ub[0] + ub[1]) // 2), int((ub[2] + ub[3]) // 2)
    x0 = min(max(cc - side // 2, 0), S - side)
    y0 = min(max(rc_ - side // 2, 0), S - side)
    rect = ((torch.arange(side, device="cuda") + y0).view(-1, 1) * S
            + (torch.arange(side, device="cuda") + x0).view(1, -1)).reshape(-1)
    pad = lambda ix: torch.cat([ix, ix[:1].expand(side * side - len(ix))])
    g = torch.Generator(device="cuda").manual_seed(0)
    rnd = torch.randperm(S * S, device="cuda", generator=g)[:side * side]
    # Every second pixel: the SAME ray distribution as the full frame (so the same average per-ray
    # cost as ``random``) but read in order, which is the control that separates "which rays" from
    # "in what order".
    strd = torch.arange(0, S * S, 2, device="cuda")
    top = torch.arange(0, side * side, device="cuda")            # the first rows, a free control

    c = RP.make_ctx(mod, cls, m, side, a.nworld, hide=hide, depth=True, skybox=False, share=base)
    for label, ix in (("rect", rect), ("mask", pad(mask_idx)), ("random", rnd),
                      ("stride2", pad(strd)), ("toprows", top)):
        c.wp.to_torch(c.rc.ray).copy_(ray.index_select(0, ix))
        c.wp.capture_launch(c.graph_draw)
        c.wp.synchronize()
        t = RP.timed(lambda: c.wp.capture_launch(c.graph_draw), a.reps)
        rows[label] = {"ms": t * 1e3, "rays": side * side,
                       "ray_frac": side * side / float(S * S),
                       "speedup_vs_full": t_full / t}
        print(f"{label:11s} {t*1e3:8.3f} ms  rays {side*side/float(S*S):.3f}  "
              f"x{t_full/t:5.2f}", flush=True)

    out = {"env": a.env, "keep": a.keep, "size": S, "nworld": a.nworld, "side": side,
           "mask_frac": n_mask / float(S * S), "rows": rows}
    print("JSON " + json.dumps(out), flush=True)
    with open(os.path.join(OUT, f"coherence_{a.env}_{a.keep}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
