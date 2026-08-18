"""Is MujocoPusher's interior parity draw-dependent, and is the seeded parity test's draw an outlier?

`test_render_parity_with_official_renderer` now draws from a fixed seed instead of from the env's own
generator at whatever point in its sequence the test order left it. MujocoPusher passed under the old
unreproducible draw and fails under the fixed one, and there are three candidate explanations that a
single run cannot separate:

  * the env's interior parity is draw-dependent -- the same class of flake this change exists to
    surface, and the seed found it rather than caused it;
  * the fixed seed is an outlier;
  * something else changed in the renderer between the two runs (raycast3d.py and
    mujoco_locomotion.py are being edited concurrently by other work).

So: measure the RATE over many seeded draws on the code as it stands, and report the parity seed
alongside them. `mujoco_locomotion` declares parity_interior_min = 1.0 with "0 disagreeing interior
pixels out of 12,112,519 over 256 states", so any nonzero rate here contradicts a declared number.

    python scratch/audit/pusher_parity_rate.py --envs MujocoPusher,MujocoReacher --draws 30
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

PARITY_STATE_SEED = 20260811


def run(name: str, draws: int) -> None:
    cls = se.REGISTRY[name]
    env = se.make(name, batch=16, device="cuda")
    ref = C.reference_env(cls, env.size)
    print(f"\n=== {name} @{env.size} ss={env.ss}  floor={getattr(env, 'parity_interior_min', 1.0)}",
          flush=True)
    below = 0
    seeds = list(range(draws)) + [PARITY_STATE_SEED]
    for sd in seeds:
        states = env.sample_states(64, rng=np.random.default_rng(sd))
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        f = r["interior_exact_frac"]
        m = C.interior_mask(official)
        n_int = int(m.sum())
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        n_bad = int(((d[:, :-1, :-1]) * m > 0).sum())
        tag = "  <-- the parity test's seed" if sd == PARITY_STATE_SEED else ""
        if f is not None and f < 1.0:
            below += 1
        print(f"  seed {sd:<9} interior_exact {f!r}  bad {n_bad}/{n_int}  "
              f"all_exact {r['exact_frac']:.5f}  mean {r['mean_abs']:.3f}  max {r['max_abs']}{tag}",
              flush=True)
    print(f"  -> {below}/{len(seeds)} draws below 1.0", flush=True)
    del env
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="MujocoPusher")
    ap.add_argument("--draws", type=int, default=30)
    a = ap.parse_args()
    for n in a.envs.split(","):
        if n not in se.REGISTRY:
            print(f"{n}: not registered")
            continue
        run(n, a.draws)
