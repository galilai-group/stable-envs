"""SlidingTilePuzzle: the recorded refusal names the right mechanism and the wrong scope.

`docs/research/jumanji.md` refuses `SlidingTilePuzzle-v0` on 28,235 conflict pixels, because
`ax.imshow(state.puzzle, cmap=...)` passes no `vmin`/`vmax` and therefore normalises the colormap to
the board's own min and max — "every tile's colour is a function of the whole board". That is true of
the probe boards `_build_atlas` uses, which are `base()` with ONE cell overwritten. It is not
obviously true of the boards the env reaches: a sliding-tile puzzle's state is a **permutation** of
`0 .. N*N-1`, so its min is 0 and its max is `N*N-1` in every reachable state, and the normalisation
is then a CONSTANT rather than a function of the board.

If that holds, the per-cell model is sound on the reachable set and the fix is not to pin `vmin`/
`vmax` — which would be matching a modified reference, and is what the note ruled out — but to probe
on boards whose normalisation matches the reachable set's. An anchored base does that: a base holding
**two** 0s and **two** `N*N-1`s keeps min and max fixed no matter which single cell the probe
overwrites, so every LUT entry is the colour a real permutation would give.

Three measurements, in order, each of which can refute the next:

  1. is `state.puzzle` a permutation of `0..N*N-1` in every reachable state?
  2. with an anchored base, does the per-cell construction produce **0** conflict pixels?
  3. does the resulting atlas reproduce the official frame on states from the env's OWN reset
     distribution — the only states a parity number is allowed to be taken on?

Run: `PYTHONPATH=. <se-jumanji venv>/bin/python scratch/jumanji/probe_slidingtile_reachable.py`
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

OUT = os.path.join(HERE, "slidingtile_reachable.json")
SIZE = 200
N = 5
P = N * N
V = N * N


def viewer():
    from jumanji.environments.logic.sliding_tile_puzzle.viewer import SlidingTilePuzzleViewer
    v = SlidingTilePuzzleViewer("SlidingTilePuzzle", render_mode="rgb_array")
    v._display = v._display_rgb_array
    return v


def main():
    import jax
    import jax.numpy as jnp
    import jumanji
    from stable_envs.envs.jumanji import _official_frame, _Spec

    rep = {"size": SIZE, "n": N}

    # ------------------------------------------------------------ 1. the reachable set
    env = jumanji.make("SlidingTilePuzzle-v0")
    key = jax.random.PRNGKey(0)
    perm_ok, n_states, minmax = 0, 0, set()
    boards = []
    for ep in range(24):
        key, k = jax.random.split(key)
        state, ts = env.reset(k)
        for t in range(24):
            b = np.asarray(state.puzzle).reshape(-1)
            n_states += 1
            perm_ok += int(np.array_equal(np.sort(b), np.arange(P)))
            minmax.add((int(b.min()), int(b.max())))
            boards.append(b.copy())
            mask = np.asarray(ts.observation.action_mask).reshape(-1)
            if not mask.any():
                break
            key, k = jax.random.split(key)
            a = int(jax.random.choice(k, jnp.arange(mask.size), p=jnp.asarray(mask / mask.sum())))
            state, ts = env.step(state, jnp.int32(a))
    rep["reachable"] = {"states": n_states, "permutations": perm_ok,
                        "distinct_(min,max)": sorted(minmax)}
    print(f"1. reachable set: {perm_ok}/{n_states} states are permutations of 0..{P-1}; "
          f"(min, max) observed = {sorted(minmax)}", flush=True)
    if perm_ok != n_states:
        print("   -> NOT always a permutation; the refusal stands as recorded.", flush=True)

    # ------------------------------------------------------------ 2. the anchored construction
    class _STP(_Spec):
        env_id, cls_name, ds_name = ("SlidingTilePuzzle-v0", "JumanjiSlidingTilePuzzle",
                                     "jumanji-slidingtilepuzzle")
        rows = cols = N
        n_values = V
        verified_sizes = (SIZE,)

        def viewer(self):
            return viewer()

        def render(self, vw, grid):
            from stable_envs.envs.jumanji import _rgb
            from jumanji.environments.logic.sliding_tile_puzzle.types import State
            g = jnp.asarray(np.asarray(grid, np.int32).reshape(N, N))
            st = State(puzzle=g, empty_tile_position=jnp.zeros(2, jnp.int32),
                       key=jnp.zeros(2, jnp.uint32), step_count=jnp.int32(0))
            return _rgb(vw.render(st))

        #: `_official_frame` warms the figure with `spec.base()`, so the base has to be a real
        #: board and it changes per construction below.
        _base = np.arange(P, dtype=np.int32)

        def base(self):
            return self._base.copy()

    spec = _STP()

    def build(base):
        spec._base = np.asarray(base, np.int32)
        f0 = _official_frame(spec, SIZE, base)
        frames = np.empty((P, V, SIZE, SIZE, 3), np.uint8)
        for p in range(P):
            for val in range(V):
                g = base.copy()
                g[p] = val
                frames[p, val] = _official_frame(spec, SIZE, g)
        changed = np.zeros((P, SIZE, SIZE), bool)
        for p in range(P):
            ref = frames[p, base[p]].astype(np.int16)
            changed[p] = (frames[p].astype(np.int16) - ref != 0).any(axis=(0, 3))
        n_own = changed.sum(0)
        owner = np.where(n_own == 1, changed.argmax(0), -1).astype(np.int32)
        return f0, frames, int((n_own > 1).sum()), owner, float((owner >= 0).mean()), n_own

    # (a) the recorded construction: base = zeros, so min/max move with the probe
    _, _, conf_zero, _, own_zero, _ = build(np.zeros(P, np.int32))
    # (b) the recorded construction on the SOLVED board: still one 0 and one 24, so probing the cell
    #     that holds either extreme still moves the normalisation
    _, _, conf_solved, _, own_solved, _ = build(np.arange(P, dtype=np.int32))
    # (c) anchored: two 0s and two (P-1)s, so no single-cell probe can move min or max
    anchored = np.arange(P, dtype=np.int32).copy()
    anchored[1] = 0
    anchored[P - 2] = P - 1
    f0, frames, conf_anchor, owner, own_anchor, n_own = build(anchored)
    rep["construction"] = {
        "conflicts_base_zeros": conf_zero, "owned_frac_base_zeros": own_zero,
        "conflicts_base_solved": conf_solved, "owned_frac_base_solved": own_solved,
        "conflicts_base_anchored": conf_anchor, "owned_frac_base_anchored": own_anchor,
        "anchored_base": anchored.tolist()}
    print(f"2. conflict pixels: base=zeros {conf_zero}, base=solved {conf_solved}, "
          f"base=ANCHORED {conf_anchor}  (owned {100*own_anchor:.1f}% of the frame)", flush=True)

    # ---- WHERE are the residual conflicts? Two candidate mechanisms, and they look different:
    #      a leftover normalisation coupling touches whole tile interiors; a shared cell BORDER (the
    #      mechanism that restricts Connector to 200 px) is a thin line on the cell grid.
    ys_c, xs_c = np.nonzero(n_own > 1)
    if len(ys_c):
        # the frame is a 10x10-inch axes; recover the cell pitch from the OWNED pixels instead of
        # assuming a margin, then ask how far each conflict is from the nearest cell edge
        oy, ox = np.nonzero(owner >= 0)
        y0, y1, x0, x1 = oy.min(), oy.max(), ox.min(), ox.max()
        pitch_y, pitch_x = (y1 - y0 + 1) / N, (x1 - x0 + 1) / N
        dy = np.minimum((ys_c - y0) % pitch_y, pitch_y - ((ys_c - y0) % pitch_y))
        dx = np.minimum((xs_c - x0) % pitch_x, pitch_x - ((xs_c - x0) % pitch_x))
        near = np.minimum(dy, dx)
        rep["conflict_localisation"] = {
            "conflicts": int(len(ys_c)),
            "owners_per_conflict": {int(k): int(v) for k, v in
                                    zip(*np.unique(n_own[ys_c, xs_c], return_counts=True))},
            "frac_within_1px_of_a_cell_edge": float((near <= 1.0).mean()),
            "frac_within_2px_of_a_cell_edge": float((near <= 2.0).mean()),
            "frac_within_4px_of_a_cell_edge": float((near <= 4.0).mean()),
            "bbox_of_owned": [int(y0), int(y1), int(x0), int(x1)],
            "cell_pitch": [float(pitch_y), float(pitch_x)]}
        print("   residual conflicts:", json.dumps(rep["conflict_localisation"]), flush=True)

    # ------------------------------------------------------------ 3. verify on REAL boards
    if conf_anchor == 0:
        ys, xs = np.nonzero(owner >= 0)
        lut = frames[owner[ys, xs], :, ys, xs]                    # (K, V, 3)
        plate = f0.copy()
        worst, exact, tested = 0, 0, 0
        rng = np.random.default_rng(0)
        pick = rng.choice(len(boards), size=min(48, len(boards)), replace=False)
        for i in pick:
            g = boards[i]
            ref = _official_frame(spec, SIZE, g)
            out = plate.copy()
            out[ys, xs] = lut[np.arange(len(ys)), g[owner[ys, xs]]]
            d = int(np.abs(out.astype(np.int16) - ref.astype(np.int16)).max())
            worst = max(worst, d)
            exact += int(d == 0)
            tested += 1
        rep["verify_real_boards"] = {"boards": tested, "frames_exact": exact, "max_abs_diff": worst}
        print(f"3. verification on {tested} boards from the env's own rollout: "
              f"{exact}/{tested} frames bit-exact, max |diff| {worst}", flush=True)

        # and the control: a board that is NOT a permutation must go WRONG, which is what makes the
        # scope a real restriction rather than a formality
        bad = boards[int(pick[0])].copy()
        bad[bad == P - 1] = 0                                     # remove the max -> renormalise
        ref = _official_frame(spec, SIZE, bad)
        out = plate.copy()
        out[ys, xs] = lut[np.arange(len(ys)), bad[owner[ys, xs]]]
        d = int(np.abs(out.astype(np.int16) - ref.astype(np.int16)).max())
        rep["control_non_permutation_max_abs"] = d
        print(f"   control, a NON-permutation board (max removed): max |diff| {d} "
              f"-- the scope restriction is real", flush=True)

    with open(OUT, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
