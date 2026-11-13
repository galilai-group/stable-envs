"""Render seeded inputs in a fresh interpreter and write comparison artifacts.

Run with python -m tests._runner.render_worker --help. Per-environment
errors are recorded in the report; callers must inspect it, not just exit status.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from stable_envs._core.mujoco_gl import configure_mujoco_gl
from stable_envs.testing._sampling import (  # noqa: F401
    STATE_SEED,
    STYLE_SEED,
    digest,
    has_render_rows,
    render_any,
    sample_inputs,
    state_arrays,
    styled_inputs,
)

configure_mujoco_gl()
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np

# Protocol seeds are shared with the parent test.
ROLLOUT_SEED = 7
ROLLOUT_STEPS = 4


def _digest(a) -> str:
    return digest(a)


def rollout_digests(env, n: int) -> dict:
    """Hash a short seeded rollout when the environment supplies private batched dynamics."""
    from stable_envs._core.dynamics import has_dynamics

    if not has_dynamics(env):
        return {}
    from tests.helpers.rollout import sample

    if has_render_rows(env):
        tr = sample(env, steps=ROLLOUT_STEPS, batch=n, seed=ROLLOUT_SEED)
    else:
        # A replay State keeps what earlier episodes left after a reset, so the rollout uses a new env.
        with type(env)(num_envs=n, size=env.size, device=str(env.device)) as other:
            tr = sample(other, steps=ROLLOUT_STEPS, batch=n, seed=ROLLOUT_SEED)
    out = {"rollout_states_sha": _digest(tr["states"])}
    if tr.get("actions") is not None:
        out["rollout_actions_sha"] = _digest(tr["actions"])
    return out


def run(names, n: int, device: str = "cuda") -> tuple[dict, dict]:
    import stable_envs as se

    report, arrays = {}, {}
    for name in names:
        try:
            env = se.make(name, num_envs=n, device=device)
            states, style = sample_inputs(env, n)
            if has_render_rows(env):
                styled = None
                frames = env._render(states)
                frames_styled = env._render(states, style=style if env.style_dim else None)
            else:
                # A full State carries its Variation, so the styled frames come from States drawn under it.
                styled = styled_inputs(env, n, style) if env.style_dim else states
                frames = render_any(env, states)
                frames_styled = render_any(env, styled)
            rec = {
                "states_sha": _digest(states),
                "style_sha": _digest(style),
                "frames_sha": _digest(frames),
                "frames_styled_sha": _digest(frames_styled),
                "shape": list(frames.shape),
                "size": int(env.size),
                "state_dim": int(env.state_dim),
                "style_dim": int(env.style_dim),
            }
            rec.update(rollout_digests(env, n))
            report[name] = rec
            if styled is None:
                arrays[f"{name}::states"] = states
            else:
                arrays.update({f"{name}::states::{k}": v for k, v in state_arrays(states).items()})
                arrays.update({f"{name}::styled::{k}": v for k, v in state_arrays(styled).items()})
            arrays[f"{name}::style"] = style
        except Exception:  # noqa: BLE001
            # Record the traceback and continue so one unavailable family does not hide other results.
            report[name] = {"error": traceback.format_exc(limit=6)}
        finally:
            # Flush progress for the parent watchdog while a large registry is still rendering.
            print(f"render_worker: done {name}", flush=True)
    return report, arrays


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-npz", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--envs", default="", help="comma-separated; default every registered env")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)

    import stable_envs as se

    names = [s for s in a.envs.split(",") if s] or sorted(se.registry.REGISTRY)
    report, arrays = run(names, a.n, a.device)
    np.savez(a.out_npz, **arrays)
    with open(a.out_json, "w") as f:
        json.dump(
            {"envs": report, "n": a.n, "device": a.device, "state_seed": STATE_SEED, "style_seed": STYLE_SEED}, f
        )
    n_err = sum("error" in v for v in report.values())
    print(f"render_worker: {len(report) - n_err}/{len(report)} envs rendered, {n_err} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
