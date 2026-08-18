"""Locate the pixels that disagree, in the reference's own coordinates.

Two questions:
  * tic_tac_toe disagrees on 34 pixels whose neighbourhood is FLAT and by the full 255. A flat-area
    disagreement is a geometry error, never anti-aliasing, so it has a location and a cause.
  * go_19x19's residual is entirely on rims but averages ~23/255 on the pixels that differ, which is
    far more than a quantisation difference and says the coverage model is wrong rather than coarse.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se
from stable_envs.testing import conformance as C


def flat_bad(name, n=16, size=None):
    cls = se.REGISTRY[name]
    env = se.make(name, batch=8, size=size, device="cuda")
    st = env.sample_states(n, rng=np.random.default_rng(0))
    ref = cls.reference_env(size=env.size)
    off = cls.reference_render(ref, st)
    ours = env.render(st).cpu().numpy()
    d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
    m = C.interior_mask(off)
    flat = np.zeros_like(d, bool)
    flat[:, :-1, :-1] = m
    bad = (d > 0) & flat
    print(f"=== {name}: {int(bad.sum())} flat-area disagreements ===")
    G = env._b.grid
    sc = env.size / float(env._b.native)
    for k, (i, y, x) in enumerate(zip(*np.nonzero(bad))):
        if k > 25:
            print("  ...")
            break
        cx, cy = (x + 0.5) / sc, (y + 0.5) / sc
        col = round(cx / G) - 1
        row = round(cy / G) - 1
        u, v = cx - (col + 1) * G, cy - (row + 1) * G
        cell = row * env._spec["ncols"] + col
        val = st[i][cell] if 0 <= cell < env.state_dim else None
        print(f"  state{i:3d} px=({y},{x}) canvas=({cy:.1f},{cx:.1f}) cell=({row},{col}) "
              f"val={val} local=({v:+.2f},{u:+.2f}) |local|={np.hypot(u, v):.3f} "
              f"ours={ours[i, y, x]} ref={off[i, y, x]}")
    return bad


def rim_profile(name, n=8):
    """How wrong are we as a function of distance from a mark boundary?"""
    cls = se.REGISTRY[name]
    env = se.make(name, batch=8, size=None, device="cuda")
    st = env.sample_states(n, rng=np.random.default_rng(0))
    ref = cls.reference_env(size=env.size)
    off = cls.reference_render(ref, st)
    ours = env.render(st).cpu().numpy()
    d = np.abs(ours.astype(np.int16) - off.astype(np.int16)).max(-1)
    nz = d[d > 0]
    print(f"=== {name}: {nz.size} differing pixels, mean {nz.mean():.2f}, "
          f"median {np.median(nz):.1f}, p90 {np.percentile(nz, 90):.1f}, max {nz.max()} ===")
    hist, edges = np.histogram(nz, bins=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    for h, lo, hi in zip(hist, edges[:-1], edges[1:]):
        print(f"    |diff| in [{int(lo):3d},{int(hi):3d}) : {h:8d}  ({100*h/nz.size:5.2f}%)")


if __name__ == "__main__":
    flat_bad("PgxTicTacToe")
    print()
    for g in ("PgxTicTacToe", "PgxGo19x19", "PgxOthello"):
        rim_profile(g)
        print()
