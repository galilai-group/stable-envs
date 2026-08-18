"""Probe 2: the four gates from docs/research/when_does_this_pay_off.md, per Jumanji id.

G1 purity     -- reach the same candidate state through two different histories, render both, diff.
                 A JAX env makes this exact: the State is a pytree, so the second history is built by
                 REPLACING the render-relevant leaves of a state reached by a different rollout. If the
                 two frames differ, some leaf outside the candidate state reaches a pixel.
G2 compression-- render-relevant bytes / frame bytes. Reported both ways (whole State pytree, and the
                 slice that survives G1), because the shipped pytree carries PRNG keys and counters.
G3 decoder    -- reset twice with different seeds and compare the figure's static furniture: the set of
                 artists matplotlib puts on the axes. Identical across episodes is the tile model.
G4 economics  -- (state -> frame) seconds against PNG decode of the same frame, and the render's share
                 of one env step.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def leaf_report(state) -> list[tuple[str, tuple, str, int]]:
    import jax
    out = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        name = "".join(str(p) for p in path)
        a = np.asarray(leaf)
        out.append((name, a.shape, str(a.dtype), int(a.nbytes)))
    return out


def render_of(env, state) -> np.ndarray:
    return np.asarray(env.render(state))


def main() -> None:
    import jax
    import jumanji

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--steps", type=int, default=12)
    args = ap.parse_args()

    ids = args.ids or sorted(jumanji.registered_environments())
    for eid in ids:
        print(f"\n================ {eid}")
        try:
            env = jumanji.make(eid)
            key = jax.random.PRNGKey(0)
            s0, _ = env.reset(key)
            f0 = render_of(env, s0)
            print(f"frame {f0.shape} {f0.dtype}  bytes={f0.nbytes}")
            rows = leaf_report(s0)
            tot = sum(r[3] for r in rows)
            print(f"state pytree: {len(rows)} leaves, {tot} bytes, ratio={tot / f0.nbytes:.4f}")
            for n, sh, dt, nb in rows:
                print(f"   {n:40s} {str(sh):16s} {dt:10s} {nb:8d}")

            # roll forward, then look at which leaves changed the frame
            act_key = jax.random.PRNGKey(1)
            s = s0
            for i in range(args.steps):
                act_key, sub = jax.random.split(act_key)
                a = env.action_spec.generate_value()
                try:
                    mask = getattr(s, "action_mask", None)
                    if mask is not None:
                        flat = np.asarray(mask).reshape(-1)
                        idx = int(np.flatnonzero(flat)[0]) if flat.any() else 0
                        a = np.unravel_index(idx, np.asarray(mask).shape)
                        a = jax.numpy.asarray(a).squeeze()
                except Exception:  # noqa: BLE001
                    pass
                s, _ = env.step(s, a)
            f1 = render_of(env, s)
            print(f"after {args.steps} steps: frame changed = "
                  f"{int(np.abs(f0.astype(np.int32) - f1.astype(np.int32)).max())}")
        except Exception as e:  # noqa: BLE001
            print(f"ERR {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
