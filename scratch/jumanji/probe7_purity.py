"""Probe 7: G1 purity for the SHIPPED specs -- does any State leaf outside the grid reach a pixel?

The LUT verification (probe 6) proves the frame is a function of the grid GIVEN everything else held
where the adapter puts it. That is not the same claim as "the grid determines the frame", and the
difference is exactly where a hidden dependency lives: Minesweeper's ``flat_mine_locations``, Sokoban's
``agent_location``, Connector's ``agents``, RubiksCube's ``step_count``.

So: take a real reset State, render it; then substitute every non-grid leaf from a DIFFERENT episode's
State, keeping the grid fixed, and render again. Threshold 0/255. A leaf that moves a pixel belongs in
the state and its absence would make ``test_state_is_sufficient`` a vacuous check.
"""
from __future__ import annotations

import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

warnings.filterwarnings("ignore")


def main() -> None:
    import jax

    import jumanji
    import stable_envs as se
    from stable_envs.envs.jumanji import SPECS

    for spec in SPECS:
        env = jumanji.make(spec.env_id)
        viewer = spec.viewer(200)
        a, _ = env.reset(jax.random.PRNGKey(0))
        b, _ = env.reset(jax.random.PRNGKey(7))
        grid = spec.grid_of(a)
        f0 = spec.render(viewer, grid)

        worst, offenders = 0, []
        fields = [f for f in a.__dict__] if hasattr(a, "__dict__") else []
        for f in fields:
            try:
                mixed = a.replace(**{f: getattr(b, f)})
            except Exception:                                     # noqa: BLE001
                continue
            g2 = spec.grid_of(mixed)
            if not np.array_equal(g2, grid):
                continue                       # this leaf IS the grid (or part of it); not a hidden one
            d = int(np.abs(spec.render(viewer, spec.grid_of(mixed)).astype(np.int16)
                           - f0.astype(np.int16)).max())
            # the render above goes through the grid, so it can only differ if grid_of is unstable;
            # the real question is whether the OFFICIAL env-level render differs
            try:
                official_a = _official(env, spec, a)
                official_m = _official(env, spec, mixed)
                d = int(np.abs(official_a.astype(np.int16) - official_m.astype(np.int16)).max())
            except Exception as e:                                # noqa: BLE001
                d = -1
                offenders.append(f"{f}: {type(e).__name__}")
            worst = max(worst, d)
            if d > 0:
                offenders.append(f"{f} -> {d}/255")
        print({"id": spec.env_id, "non_grid_leaf_max_diff": worst, "offenders": offenders},
              flush=True)


def _official(env, spec, state) -> np.ndarray:
    """The env-level official render, through whichever viewer the env owns."""
    v = None
    for attr in ("_viewer", "viewer", "_env_viewer"):
        v = getattr(env, attr, None)
        if v is not None:
            break
    v._display = v._display_rgb_array
    out = v.render(state)
    a = np.asarray(out)
    return a[..., :3] if a.shape[-1] == 4 else a


if __name__ == "__main__":
    main()
