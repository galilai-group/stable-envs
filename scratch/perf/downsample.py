"""Measure ``check_resolution_consistency`` for envs that have no bound in DOWNSAMPLE_EXACT_FRAC.

The four RoboCasa cells fail ``test_resolution_downsample_agrees_within_the_measured_bound`` with "has
no measured downsample bound recorded" -- a missing table entry, not a renderer fault, and invisible
in the main suite because that runs in an interpreter where RoboCasa does not import. The test's own
message says how to close it, and this is that: the same states the fixture draws, the same helper,
printed in the form the table takes.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import stable_envs as se
from stable_envs.testing import conformance as C
from stable_envs.testing import render_worker as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    for name in a.envs.split(","):
        try:
            env = se.make(name, batch=16, device="cuda")
            states, style = W.sample_inputs(env, a.n)
            style = style if env.style_dim else None
            rows = [C.check_resolution_consistency(env, states, style) for _ in range(a.repeats)]
            ef = [r["exact_frac"] for r in rows]
            print(f'    "{name}":{" " * max(1, 34 - len(name))}{min(ef):.6f},   '
                  f'# mean {rows[0]["mean_abs"]:.4f}/255, max {rows[0]["max_abs"]}/255, '
                  f'spread {min(ef):.6f}-{max(ef):.6f} over {a.repeats}', flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{name}: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
