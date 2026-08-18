"""Sweep the number of sub-scanlines used for a disc's coverage, against resvg."""
import sys
import numpy as np
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se
from stable_envs.testing import conformance as C

for nm in ("PgxGo19x19", "PgxGo9x9", "PgxOthello", "PgxConnectFour"):
    cls = se.REGISTRY[nm]
    for N in (1, 2, 4, 8, 16, 32, 64):
        cls.SCAN_ROWS = N
        env = se.make(nm, batch=8, device="cuda")
        env._alpha_cache = {}
        st = env.sample_states(16, rng=np.random.default_rng(0))
        off = cls.reference_render(cls.reference_env(size=env.size), st)
        ours = env.render(st).cpu().numpy()
        r = C.pixel_agreement(ours, off)
        print(f"{nm:16s} N={N:3d} interior={r['interior_exact_frac']} all={r['exact_frac']:.6f} "
              f"mean={r['mean_abs']:.4f} max={r['max_abs']:3d}")
        sys.stdout.flush()
    print()
