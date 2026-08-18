"""Probe 10: WHERE does RubiksCube disagree with the official renderer, and why?

probe 6 verified the per-pixel LUT at 0/255 on uniform-random and reset-distribution grids; probe 8,
going through the shipped env and ``sample_states``, reports all-pixel 0.997500 with max 255. Both
cannot be describing the same thing, so this localises the disagreement: which pixels, how many, and
whether they are pixels the atlas OWNS or pixels it left in the plate.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

warnings.filterwarnings("ignore")

import numpy as np


def main() -> None:
    import stable_envs as se
    from stable_envs.envs.jumanji import _load_atlas
    from stable_envs.testing import conformance as C

    name = "JumanjiRubiksCube"
    cls = se.REGISTRY[name]
    env = se.make(name, batch=8, device="cuda")
    spec = cls.SPEC
    a = _load_atlas(spec, env.size, env.device)
    ref = C.reference_env(cls, env.size)

    states = env.sample_states(16, rng=np.random.default_rng(0))
    official = cls.reference_render(ref, states)
    ours = env.render(states).cpu().numpy()
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)     # (N,H,W)
    print(f"frames differing: {(d.reshape(len(d), -1).max(1) > 0).sum()}/{len(d)}")
    print(f"differing pixels per frame: {(d > 0).sum(axis=(1, 2))}")

    owned = np.zeros(env.size * env.size, bool)
    owned[a["pix"].cpu().numpy()] = True
    owned = owned.reshape(env.size, env.size)
    bad = d > 0
    print(f"of the differing pixels: owned={int((bad & owned).sum())} "
          f"unowned(plate)={int((bad & ~owned).sum())}")

    ys, xs = np.nonzero(bad.any(0))
    print(f"bounding box of disagreement: rows {ys.min()}-{ys.max()}, cols {xs.min()}-{xs.max()}")
    print(f"frame is {env.size}x{env.size}")

    # what do the two sides actually paint there?
    i = int(np.argmax((d > 0).sum(axis=(1, 2))))
    yy, xx = np.nonzero(d[i] > 0)
    for k in range(min(6, len(yy))):
        y, x = int(yy[k]), int(xx[k])
        print(f"  ({y:3d},{x:3d}) ours={ours[i, y, x]} official={official[i, y, x]} "
              f"owned={bool(owned[y, x])}")

    # Does the DISAGREEMENT depend on the state, or is it the same pixels every time?
    sets = [frozenset(zip(*np.nonzero(d[j] > 0))) for j in range(len(d))]
    common = set.intersection(*[set(s) for s in sets]) if sets else set()
    print(f"pixels differing in EVERY frame: {len(common)} of "
          f"{[len(s) for s in sets]}")


if __name__ == "__main__":
    main()
