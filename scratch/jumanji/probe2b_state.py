"""Probe 2b: G1 purity and G2 compression, per Jumanji id, measured leaf by leaf.

The official renderer is the VIEWER's ``render``, not the env's: ``Snake.render`` and ``Sokoban.render``
call the viewer and discard its return value, so ``env.render(state)`` is None for them however the
display branch is bound. Everything here therefore goes through ``viewer.render(state)``.

G1 is asked in the form that is decisive rather than the form that is easy. For every leaf of the State
pytree, substitute the value that leaf holds in a DIFFERENT state (reached by a real rollout) and
re-render. A leaf that changes no pixel is not part of the render state; a leaf that does is, and the
sum of those leaves is the candidate state. This is the "two histories, same candidate state" test done
leaf-wise, which also tells you WHICH leaf is responsible instead of only that something is.

G2 is then the byte count of that surviving slice against the frame.
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

warnings.filterwarnings("ignore")


def get_viewer(env):
    for a in ("_viewer", "viewer", "_env_viewer"):
        v = getattr(env, a, None)
        if v is not None and hasattr(v, "_display_rgb_array"):
            v._display = v._display_rgb_array
            return v
    return None


def frame(v, state) -> np.ndarray:
    a = np.asarray(v.render(state))
    return a[..., :3] if a.ndim == 3 and a.shape[-1] == 4 else a


def rollout(env, key, n: int):
    import jax
    state, _ = env.reset(key)
    states = [state]
    for i in range(n):
        key, sub = jax.random.split(key)
        try:
            spec = env.action_spec
            mask = getattr(state, "action_mask", None)
            if mask is not None:
                m = np.asarray(mask)
                idx = np.flatnonzero(m.reshape(-1))
                if len(idx) == 0:
                    break
                pick = int(jax.random.choice(sub, jax.numpy.asarray(idx)))
                a = jax.numpy.asarray(np.unravel_index(pick, m.shape)).squeeze()
            else:
                a = spec.generate_value()
                if hasattr(spec, "num_values"):
                    nv = np.asarray(spec.num_values)
                    a = jax.numpy.asarray(
                        jax.random.randint(sub, nv.shape, 0, jax.numpy.asarray(nv))).squeeze()
        except Exception:  # noqa: BLE001
            break
        try:
            state, _ = env.step(state, a)
        except Exception:  # noqa: BLE001
            break
        states.append(state)
    return states


def main() -> None:
    import jax

    import jumanji

    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()
    ids = args.ids or sorted(jumanji.registered_environments())

    for eid in ids:
        print(f"\n================ {eid}")
        try:
            env = jumanji.make(eid)
            v = get_viewer(env)
            if v is None:
                print("  no matplotlib viewer")
                continue
            sts = rollout(env, jax.random.PRNGKey(0), 15)
            s0, s1 = sts[0], sts[-1]
            f0 = frame(v, s0)
            fb = int(f0.nbytes)
            print(f"  frame {f0.shape} bytes={fb}")

            paths = jax.tree_util.tree_flatten_with_path(s0)[0]
            paths1 = dict(("".join(str(p) for p in k), leaf)
                          for k, leaf in jax.tree_util.tree_flatten_with_path(s1)[0])
            live, dead, live_bytes = [], [], 0
            for k, leaf in paths:
                name = "".join(str(p) for p in k)
                other = paths1.get(name)
                a0, a1 = np.asarray(leaf), np.asarray(other)
                if other is None or (a0.shape == a1.shape and np.array_equal(a0, a1)):
                    dead.append((name, a0.shape, str(a0.dtype), int(a0.nbytes), "unchanged"))
                    continue
                mutated = s0
                try:
                    mutated = jax.tree_util.tree_map(
                        lambda x, y, _n=name: y if x is leaf else x, s0, s1)
                except Exception:  # noqa: BLE001
                    pass
                # direct field replacement is more reliable than tree_map identity matching
                top = name.lstrip(".").split(".")[0].split("[")[0]
                try:
                    mutated = s0._replace(**{top: getattr(s1, top)}) if hasattr(s0, "_replace") \
                        else s0.replace(**{top: getattr(s1, top)})
                except Exception:  # noqa: BLE001
                    pass
                d = int(np.abs(frame(v, mutated).astype(np.int32) - f0.astype(np.int32)).max())
                rec = (name, a0.shape, str(a0.dtype), int(a0.nbytes), f"diff={d}")
                (live if d else dead).append(rec)
                if d:
                    live_bytes += int(a0.nbytes)
            print(f"  LIVE (reach pixels): {live_bytes} bytes  ratio={live_bytes / fb:.5f}")
            for r in live:
                print(f"     LIVE {r[0]:34s} {str(r[1]):14s} {r[2]:8s} {r[3]:7d}  {r[4]}")
            for r in dead:
                print(f"     dead {r[0]:34s} {str(r[1]):14s} {r[2]:8s} {r[3]:7d}  {r[4]}")
            env.close()
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ERR {type(e).__name__}: {str(e)[:200]}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
