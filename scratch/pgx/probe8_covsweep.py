"""How much sub-sampling does the coverage need? Sweep COV_SUB and watch all-pixel agreement.

The worst case for midpoint sub-sampling is an axis-aligned edge, where the error is 1/(2S) rather
than 1/S**2. This measures where that stops mattering against a rasterizer that computes coverage
analytically, and what it costs -- the maps are cached per (resolution, scale), so the cost lands on
the first render and not on the throughput number.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se
from stable_envs.testing import conformance as C


def sweep(names, subs, n=16):
    for nm in names:
        cls = se.REGISTRY[nm]
        for S in subs:
            cls.COV_SUB = S
            env = se.make(nm, batch=8, device="cuda")
            env._alpha_cache = {}
            st = env.sample_states(n, rng=np.random.default_rng(0))
            off = cls.reference_render(cls.reference_env(size=env.size), st)
            t0 = time.perf_counter()
            ours = env.render(st).cpu().numpy()
            t_first = time.perf_counter() - t0
            t0 = time.perf_counter()
            env.render(st)
            t_warm = time.perf_counter() - t0
            r = C.pixel_agreement(ours, off)
            print(f"{nm:16s} S={S:3d} interior={r['interior_exact_frac']} "
                  f"all={r['exact_frac']:.6f} mean={r['mean_abs']:.4f} max={r['max_abs']:3d} "
                  f"| build={t_first*1e3:8.1f}ms warm={t_warm*1e3:6.2f}ms")
            sys.stdout.flush()
        print()


if __name__ == "__main__":
    sweep(["PgxGo19x19", "PgxGo9x9", "PgxOthello"], [8, 16, 32, 64, 128])
