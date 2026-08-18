"""Where the last few levels are: the analytic shading against MuJoCo on FLAT-SHADED faces.

The interior-pixel metric keeps only pixels whose 3x3 neighbourhood in the reference is one colour.
On this family almost nothing qualifies -- the checker floor sits under a spotlight gradient and the
skybox is a speckled gradient -- except the flat faces of cartpole's box cart. Those are exactly the
pixels where Gouraud interpolation is NOT an excuse: a box face has one normal, so per-vertex and
per-pixel shading agree by construction, and any disagreement is the light model.

So this dumps, for those pixels, the reference colour, ours, and every term that goes into ours.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="DMCCartpoleSwingup")
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    import scipy.ndimage as ndi

    import stable_envs as se
    from stable_envs.envs import dmcontrol as DM

    env = se.make(args.env, batch=4, size=args.size, supersample=args.ss, device="cuda")
    cls = type(env)
    ref = cls.reference_env(size=args.size)
    states = env.sample_states(args.n, rng=np.random.default_rng(0))
    official = cls.reference_render(ref, states)
    ours = env.render(states).cpu().numpy()
    sc = env._scene()

    # per-pixel mover identity from OUR cast, at output resolution (ss=1 so the mask is a hard one)
    dev = env.device
    R = args.size
    o, d = sc.cam.rays(R, R, dev)
    N = R * R
    B = len(states)
    s = torch.as_tensor(states, device=dev, dtype=torch.float32)
    xpos, xmat = env._fk(s, sc)
    P = xpos.shape[1]
    sizes = torch.tensor(sc.prims.size, device=dev, dtype=torch.float32).view(1, P, 3).expand(B, P, 3)
    oo = o.expand(B, N, 3)
    dd = d.view(1, N, 3).expand(B, N, 3)
    t_m, n_m, i_m = cls._cast(sc, oo, dd, xpos, xmat, sizes)
    st = env._static_t(sc, 1).view(1, N).expand(B, N)
    cover = (t_m < st).cpu().numpy().reshape(B, R, R)
    idx = i_m.cpu().numpy().reshape(B, R, R)

    g = official.astype(np.int16)
    flat = np.stack([ndi.minimum_filter(
        ((np.abs(np.diff(x, axis=0)).sum(-1)[:, :-1] + np.abs(np.diff(x, axis=1)).sum(-1)[:-1]) == 0),
        size=3) for x in g])
    interior = np.zeros((B, R, R), bool)
    interior[:, :-1, :-1] = flat

    names = ([f"geom{i}" for i in sc.movers[0]] + [f"site{i}" for i in sc.movers[1]])
    print(f"{args.env}: movers {names}  kinds {list(sc.prims.kind)}")
    print(f"lights: kind={list(sc.lights.kind)} amb={sc.lights.ambient[:, 0]} "
          f"dif={sc.lights.diffuse[:, 0]} spec={sc.lights.specular[:, 0]} "
          f"cut={sc.lights.cutoff} exp={sc.lights.exponent} cast={sc.lights.castshadow}")
    print(f"materials: rgb={sc.prims.mat_rgb.round(4).tolist()} spec={sc.prims.spec} "
          f"shin={sc.prims.shininess}")

    diff = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    for p in range(P):
        m = cover & (idx == p) & interior
        if not m.any():
            print(f"  prim {p} ({names[p]}, kind {int(sc.prims.kind[p])}): no interior pixels")
            continue
        print(f"  prim {p} ({names[p]}, kind {int(sc.prims.kind[p])}): {int(m.sum())} interior px, "
              f"exact {(diff[m] == 0).mean():.4f}, mean {diff[m].mean():.3f}, max {diff[m].max()}")
        u_ref, c = np.unique(official[m], axis=0, return_counts=True)
        u_our = np.unique(ours[m], axis=0)
        order = np.argsort(-c)[:6]
        print(f"       reference colours {u_ref[order].tolist()} counts {c[order].tolist()}")
        print(f"       ours             {u_our[:6].tolist()}")
    m_all = cover & interior
    if m_all.any():
        print(f"  ALL mover-interior: {int(m_all.sum())} px exact {(diff[m_all] == 0).mean():.4f} "
              f"mean {diff[m_all].mean():.3f}")
    m_static = (~cover) & interior
    if m_static.any():
        print(f"  static-interior:    {int(m_static.sum())} px exact "
              f"{(diff[m_static] == 0).mean():.4f} mean {diff[m_static].mean():.3f}")


if __name__ == "__main__":
    main()
