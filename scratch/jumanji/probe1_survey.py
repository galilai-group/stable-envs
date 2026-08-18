"""Probe 1: what IS the official Jumanji renderer?

Answers, per registered id, by measurement rather than by reading source:
  * does ``env.render(state)`` return an ndarray at all, and of what shape/dtype;
  * is it DETERMINISTIC across two calls in one process (the cheapest question, asked first);
  * is the frame uniform (a blank surface would make every later parity number meaningless).

Nothing here builds a renderer. This is the "is there a reference to match" question.
"""
from __future__ import annotations

import os
import sys
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np


def main() -> None:
    import jax
    import jumanji

    print(f"jax {jax.__version__} devices={jax.devices()}")
    print(f"jumanji {jumanji.__version__}")
    ids = sorted(jumanji.registered_environments())
    print(f"{len(ids)} registered ids\n")

    for eid in ids:
        row = {"id": eid}
        try:
            env = jumanji.make(eid)
            key = jax.random.PRNGKey(0)
            state, _ = env.reset(key)
            a = env.render(state)
            b = env.render(state)
            if a is None:
                row["render"] = "returns None"
            else:
                a = np.asarray(a)
                b = np.asarray(b)
                row["render"] = f"{a.shape} {a.dtype}"
                row["det"] = int(np.abs(a.astype(np.int32) - b.astype(np.int32)).max())
                row["uniq_colors"] = len(np.unique(a.reshape(-1, a.shape[-1]), axis=0))
        except Exception as e:  # noqa: BLE001
            row["render"] = f"ERR {type(e).__name__}: {str(e)[:120]}"
        print(row, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
