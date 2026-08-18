"""Probe 8: the parity numbers, printed rather than only asserted.

``test_render_parity_with_official_renderer`` prints ``pixel_agreement`` but pytest captures it on a
pass, so the numbers the module docstring quotes have to come from somewhere runnable. This is that
somewhere: same states (``sample_states(64)``), same comparison, one line per registered class.
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

warnings.filterwarnings("ignore")


def main() -> None:
    import stable_envs as se
    from stable_envs.testing import conformance as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()

    names = [n for n in sorted(se.REGISTRY)
             if n.startswith("Jumanji") and (args.envs is None or n in args.envs)]
    for name in names:
        cls = se.REGISTRY[name]
        env = se.make(name, batch=8, device="cuda")
        ref = C.reference_env(cls, env.size)
        states = env.sample_states(args.n)
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        print(f"{name:34s} n={args.n} size={env.size} state_dim={env.state_dim:4d}  "
              f"exact={r['exact_frac']:.6f} interior={r['interior_exact_frac']} "
              f"mean_abs={r['mean_abs']:.4f} max_abs={r['max_abs']}", flush=True)


if __name__ == "__main__":
    main()
