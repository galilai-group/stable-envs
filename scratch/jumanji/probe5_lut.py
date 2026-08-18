"""Probe 5: THE decisive experiment. Is a Jumanji frame a per-pixel lookup indexed by one cell?

Construction (see probe3 for the reasoning):
  * ``P*V`` official renders, one per (cell, value), against a fixed base grid;
  * ``owner[y,x]``  -- the single cell whose value moves this pixel; two owners is a model failure and
    is COUNTED rather than tolerated;
  * ``LUT[k,v]``    -- owned pixel k's colour when its owner holds v;
  * plate           -- everything unowned.

Then the part that actually decides it: compose random grids with EVERY cell different at once and diff
against the official renderer. Steps 1-3 only prove each cell is separable on its own; a shared border,
a colour bar or a title that reads several cells passes them and fails here.

Also answers the two questions a matplotlib reference raises, both by measurement:
  * which output SIZE (figure dpi) keeps the cell regions disjoint;
  * whether a SECOND live viewer in the same process changes the first one's frames (the trap that
    killed MiniWorld: a process-global renderer resource clobbered by the last env constructed).
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- official access

def maze_viewer(size: int):
    """``MazeEnvViewer`` in rgb_array mode at ``size`` px, plus a grid->frame callable.

    ``MazeEnvViewer.render(state)`` is exactly ``MazeViewer.render(_overlay_agent_and_target(state))``,
    so rendering a raw grid through the base class IS the official path with the (state -> grid)
    reduction already applied -- and that reduction is a pure function that discards nothing the
    renderer reads.
    """
    from jumanji.environments.routing.maze.viewer import MazeEnvViewer

    v = MazeEnvViewer("Maze", render_mode="rgb_array")
    _size_the_figure(v, size)
    return v


def cleaner_viewer(size: int):
    from jumanji.environments.routing.cleaner.viewer import CleanerViewer

    v = CleanerViewer("Cleaner", render_mode="rgb_array")
    _size_the_figure(v, size)
    return v


def _size_the_figure(v, size: int) -> None:
    """Pin the figure to ``size`` px square. figsize is 10x10 inches, so dpi = size/10."""
    import matplotlib.pyplot as plt
    plt.close(v._name)
    v.figure_size = (10.0, 10.0)
    v._se_dpi = size / 10.0


def render_grid(v, grid) -> np.ndarray:
    """The official ``MazeViewer.render`` on a raw integer grid, at the pinned dpi."""
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from jumanji.environments.commons.maze_utils.maze_rendering import MazeViewer

    fig_exists = plt.fignum_exists(v._name)
    out = MazeViewer.render(v, jnp.asarray(grid))
    if not fig_exists:                       # first render: fix the dpi, then redraw at it
        plt.figure(v._name).set_dpi(v._se_dpi)
        out = MazeViewer.render(v, jnp.asarray(grid))
    a = np.asarray(out)
    return a[..., :3] if a.shape[-1] == 4 else a


# ----------------------------------------------------------------- the construction

def build(render_fn, P: int, V: int, base: np.ndarray, verbose=True):
    t0 = time.perf_counter()
    f_base = render_fn(base)
    H, W, _ = f_base.shape
    per_cell = np.empty((P, V, H, W, 3), np.uint8)
    for p in range(P):
        for value in range(V):
            g = base.copy()
            g[p] = value
            per_cell[p, value] = render_fn(g)
    changed = np.zeros((P, H, W), bool)
    for p in range(P):
        ref = per_cell[p, base[p]].astype(np.int16)
        changed[p] = (per_cell[p].astype(np.int16) - ref != 0).any(axis=(0, 3))
    n_own = changed.sum(0)
    conflicts = int((n_own > 1).sum())
    owner = np.where(n_own == 1, changed.argmax(0), -1).astype(np.int32)
    ys, xs = np.nonzero(owner >= 0)
    cells = owner[ys, xs]
    lut = per_cell[cells, :, ys, xs]                       # (K, V, 3)
    stats = dict(H=H, W=W, P=P, V=V, K=int(len(ys)), owned_frac=len(ys) / (H * W),
                 conflict_px=conflicts, build_s=round(time.perf_counter() - t0, 1),
                 lut_MB=round(lut.nbytes / 1e6, 2))
    return f_base.copy(), (ys, xs, cells, lut), stats


def compose(plate, packed, grid):
    ys, xs, cells, lut = packed
    out = plate.copy()
    out[ys, xs] = lut[np.arange(len(ys)), grid[cells]]
    return out


def verify(render_fn, plate, packed, P, V, n=16, seed=0, sampler=None):
    rng = np.random.default_rng(seed)
    worst, bad = 0, 0
    for _ in range(n):
        g = sampler(rng) if sampler else rng.integers(0, V, P)
        d = int(np.abs(compose(plate, packed, g).astype(np.int16)
                       - render_fn(g).astype(np.int16)).max())
        worst = max(worst, d)
        bad += int(d != 0)
    return dict(states=n, max_abs_diff=worst, frames_differing=bad)


# ----------------------------------------------------------------- contamination

def contamination(size: int) -> dict:
    """Does constructing a SECOND viewer change the FIRST one's frames? Threshold 0/255."""
    v1 = maze_viewer(size)
    g = np.zeros(100, np.int32)
    g[3] = 2
    g[97] = 3
    before = render_grid(v1, g.reshape(10, 10))
    v2 = cleaner_viewer(size)
    from jumanji.environments.routing.cleaner.viewer import CleanerViewer  # noqa: F401
    import jax
    import jumanji
    env = jumanji.make("Cleaner-v0")
    s, _ = env.reset(jax.random.PRNGKey(0))
    v2._display = v2._display_rgb_array
    v2.render(s)
    after = render_grid(v1, g.reshape(10, 10))
    return {"max_abs_diff_after_second_env": int(np.abs(before.astype(np.int16)
                                                       - after.astype(np.int16)).max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", type=int, default=[200, 500, 1000])
    ap.add_argument("--verify", type=int, default=16)
    args = ap.parse_args()

    for size in args.sizes:
        print(f"\n############ Maze-v0 at size={size}")
        v = maze_viewer(size)
        R = C = 10
        P, V = R * C, 4
        base = np.zeros(P, np.int32)
        rf = lambda g: render_grid(v, np.asarray(g).reshape(R, C))
        plate, packed, stats = build(rf, P, V, base)
        print("  ", stats)
        print("   verify(uniform random):", verify(rf, plate, packed, P, V, args.verify))

        def realistic(rng):
            g = (rng.random(P) < 0.35).astype(np.int32)
            g[rng.integers(0, P)] = 2
            g[rng.integers(0, P)] = 3
            return g

        print("   verify(maze-like)     :",
              verify(rf, plate, packed, P, V, args.verify, sampler=realistic))

    print("\n############ cross-env contamination")
    print("  ", contamination(args.sizes[-1]))


if __name__ == "__main__":
    main()
