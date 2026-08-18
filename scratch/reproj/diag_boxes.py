"""Why does the mover window never fire? The box the geometry implies against the pixels it paints.

``rc_pipeline.py`` reports ``window_sides {256: 16}`` on every chunk of every draw: the union of the
movers' screen boxes IS the frame, while ``scratch/perf/roi2.py`` measures the movers painting 16 %
of it. Something between those two numbers is where the 2-3x went, and it is one of three things:

1. a handful of movers are huge (a cabinet door, a counter-top) and their boxes cover everything;
2. the movers are small but SCATTERED, so their union is the frame even though each is tiny;
3. ``geom_aabb`` is not what the box arithmetic assumes it is.

This prints the per-geom box areas with names, the union of the boxes against the union of the
PIXELS the same movers paint, and how much of the frame the largest few movers are responsible for.
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
    a = ap.parse_args()

    import mujoco

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

    c = RP.make_ctx(mod, cls, m, S, a.nworld, hide=static, depth=True, skybox=False)
    cam = RP.Pinhole(c.ray0, S)
    q = torch.as_tensor(qpos, dtype=torch.float32, device="cuda")
    c.wp.to_torch(c.wd.qpos)[:] = q
    RP.launch(c)
    painted = (c.depth[:] > 0)                                   # the pixels the movers really paint

    xpos = torch.as_tensor(cpos, dtype=torch.float32, device="cuda")
    xmat = torch.as_tensor(cmat, dtype=torch.float32, device="cuda")
    corners = torch.as_tensor(aabb_corners(m, mv), dtype=torch.float32, device="cuda")
    gx = torch.as_tensor(gp[:, mv], dtype=torch.float32, device="cuda")
    gmm = torch.as_tensor(gm[:, mv], dtype=torch.float32, device="cuda")
    cor = (gx.unsqueeze(2) + torch.einsum("ngij,gcj->ngci", gmm, corners)).reshape(len(gp), -1, 3)
    pc = torch.einsum("nji,ncj->nci", xmat, cor - xpos.unsqueeze(1))
    znear = float(getattr(c.rc, "znear", 0.01))
    gb = geom_boxes(clip_to_near(pc, len(mv), znear), cam, S, znear)

    area = ((gb[..., 1] - gb[..., 0]).float() * (gb[..., 3] - gb[..., 2]).float() / (S * S))
    per_geom = area.mean(0).cpu().numpy()
    order = np.argsort(-per_geom)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"geom{g}" for g in mv]
    from rc_pipeline import local_boxes
    sizes = local_boxes(m, mv)
    mj_ab = np.asarray(m.geom_aabb, np.float64).reshape(-1, 6)[mv]

    union_all = float(ray_mask(gb.reshape(-1, 4), S).float().mean())
    per_state = np.array([float(ray_mask(gb[i], S).float().mean()) for i in range(len(gp))])
    painted_union = float(painted.any(0).float().mean())
    print(json.dumps({
        "env": a.env, "n_movers": int(len(mv)),
        "painted_frac_mean": float(painted.float().mean()),
        "painted_union_over_chunk": painted_union,
        "box_union_per_state_mean": float(per_state.mean()),
        "box_union_over_chunk": union_all,
        "geom_box_frac_mean": float(per_geom.mean()),
    }, indent=1), flush=True)
    print("largest movers by mean screen box:", flush=True)
    for k in order[:14]:
        print(f"  {names[k]:40s} box {per_geom[k]:.4f}  mesh c "
              f"{sizes[k, 0]:+.3f}{sizes[k, 1]:+.3f}{sizes[k, 2]:+.3f} h "
              f"{sizes[k, 3]:.3f} {sizes[k, 4]:.3f} {sizes[k, 5]:.3f} | geom_aabb c "
              f"{mj_ab[k, 0]:+.3f}{mj_ab[k, 1]:+.3f}{mj_ab[k, 2]:+.3f} h "
              f"{mj_ab[k, 3]:.3f} {mj_ab[k, 4]:.3f} {mj_ab[k, 5]:.3f}", flush=True)

    # what the union costs as the biggest movers are dropped, and what dropping them would cost in
    # pixels: a mover excluded from the window has to be drawn by the PLATE, which cannot move it
    keep = list(order)
    for drop in (0, 1, 2, 4, 8, 16):
        sub = np.array(sorted(keep[drop:]), np.int64)
        if len(sub) == 0:
            continue
        u = float(ray_mask(gb[:, torch.as_tensor(sub, device="cuda")].reshape(-1, 4), S).float().mean())
        print(f"  drop {drop:2d} largest -> box union {u:.4f}", flush=True)


if __name__ == "__main__":
    main()
