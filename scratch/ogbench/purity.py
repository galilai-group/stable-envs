"""Is an OGBench maze frame a pure function of a compact state vector? Measured, not assumed.

Four questions, each answered by a pixel diff on the official renderer:
  1. determinism  -- same qpos rendered twice in one process: bit-identical?
  2. qvel-blind   -- same qpos, wildly different qvel: bit-identical? (if yes the frame is f(qpos) only)
  3. path-blind   -- same qpos reached from two different rollouts: bit-identical? (no hidden render state)
  4. sub-state    -- which qpos entries actually move pixels (per-coordinate perturbation sensitivity)

Also reports the edge-pixel fraction (how much of the frame is an anti-aliased boundary), which upper-
bounds what any independent rasterizer can hope to match bit-exactly given MSAA.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_scene import make  # noqa: E402

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"


def diff_stats(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return dict(max=int(d.max()), mean=float(d.mean()), frac_ne=float((d.max(-1) > 0).mean()))


def run(loco, size, img=224, seed=0):
    u_env, u = make(loco, size, img)
    m, d = u.model, u.data
    rng = np.random.default_rng(seed)
    res = {"env": f"{loco}maze-{size}", "nq": int(m.nq), "nv": int(m.nv), "img": img}

    # a reachable state: roll the env a bit
    u_env.reset(options={"task_id": 1})
    for _ in range(20):
        u_env.step(u.action_space.sample())
    qpos = d.qpos.copy()
    qvel = d.qvel.copy()

    def render_at(qp, qv):
        u.set_state(qp, qv)
        return u.render().copy()

    f0 = render_at(qpos, qvel)
    f1 = render_at(qpos, qvel)
    res["determinism"] = diff_stats(f0, f1)

    # qvel-blind
    for scale in (1.0, 10.0, 100.0):
        qv2 = rng.normal(size=m.nv) * scale
        res[f"qvel_blind_x{scale:g}"] = diff_stats(f0, render_at(qpos, qv2))

    # path-blind: get to the same qpos from a different rollout
    u_env.reset(options={"task_id": 2})
    for _ in range(37):
        u_env.step(u.action_space.sample())
    res["path_blind"] = diff_stats(f0, render_at(qpos, qvel))

    # per-qpos-coordinate sensitivity
    sens = []
    for k in range(m.nq):
        qp2 = qpos.copy()
        qp2[k] += 0.05
        s = diff_stats(f0, render_at(qp2, qvel))
        sens.append(dict(k=k, frac_ne=s["frac_ne"], max=s["max"]))
    res["qpos_sensitivity"] = sens
    res["n_qpos_that_move_pixels"] = int(sum(1 for s in sens if s["frac_ne"] > 0))

    # ---- edge-pixel fraction (MSAA-limited ceiling) -------------------------
    f = f0.astype(np.int16)
    gx = np.abs(np.diff(f, axis=1)).sum(-1)[:-1, :]
    gy = np.abs(np.diff(f, axis=0)).sum(-1)[:, :-1]
    grad = gx + gy
    res["edge_frac_any"] = float((grad > 0).mean())
    res["edge_frac_gt8"] = float((grad > 8).mean())

    # ---- palette: how many distinct colours in a frame? ---------------------
    res["unique_colors"] = int(len(np.unique(f0.reshape(-1, 3), axis=0)))

    # ---- what the two rjepa model mutations change --------------------------
    e_raw, u_raw = make(loco, size, img, decolor=False)
    u_raw.set_state(qpos, qvel)
    f_raw = u_raw.render().copy()
    res["decolor_floor_effect"] = diff_stats(f0, f_raw)
    e_raw.close()
    if loco == "humanoid":
        e_nm, u_nm = make(loco, size, img, decolor=True, marker=0.0)
        u_nm.set_state(qpos, qvel)
        res["marker_effect"] = diff_stats(f0, u_nm.render().copy())
        e_nm.close()

    u_env.close()
    return res, f0


def main():
    os.makedirs(OUT, exist_ok=True)
    import imageio.v2 as imageio
    allres = []
    for loco, size in [("point", "medium"), ("point", "large"), ("point", "giant"),
                       ("point", "teleport"), ("ant", "large"), ("ant", "giant"),
                       ("humanoid", "large")]:
        r, f = run(loco, size)
        allres.append(r)
        imageio.imwrite(os.path.join(OUT, f"frame_{loco}_{size}.png"), f)
        print(json.dumps(r, indent=1), flush=True)
    with open(os.path.join(OUT, "purity.json"), "w") as fh:
        json.dump(allres, fh, indent=1)


if __name__ == "__main__":
    main()
