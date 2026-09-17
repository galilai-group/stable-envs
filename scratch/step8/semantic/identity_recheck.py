"""Re-establish, at Step-8 HEAD, the fact Workstream H's verdict rests on.

The Step-7 Craftax cells recorded ``digest_ours == digest_official`` on their frozen corpora.
That identity is the whole reason a paired Gate-H run would be uninformative, so it is
re-measured here rather than quoted: both frozen corpora are re-rendered by the stable-envs
renderer at the current HEAD **and** by the official craftax renderer, and the two byte streams
are compared.

Needs a jax-bearing interpreter (locally ``/Users/samib/se7-craftax-venv``). Writes
``examples/results/step8/semantic/identity_recheck.json``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch

OUT = "examples/results/step8/semantic/identity_recheck.json"

CELLS = {
    "CraftaxClassic": {
        "module": "stable_envs.envs.craftax",
        "cls": "CraftaxClassic",
        "swm_id": "swm/CraftaxClassicPixels-v1",
        "corpus": "examples/results/step7/craftax-classic-full/corpus",
        "sizes": [7, 16],
    },
    "CraftaxFull": {
        "module": "stable_envs.envs.craftax_full",
        "cls": "CraftaxFull",
        "swm_id": "swm/CraftaxPixels-v1",
        "corpus": "examples/results/step7/craftax-full/corpus",
        "sizes": [10, 16],
    },
}


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    import importlib

    import jax

    res = {
        "what": (
            "the Step-7 frozen Craftax corpora re-rendered at Step-8 HEAD by both renderers; "
            "the byte-identity claim Workstream H's verdict rests on, re-measured"
        ),
        "host": os.uname().nodename,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "jax": jax.__version__,
        "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable),
        "cells": {},
    }
    try:
        import craftax.craftax.constants  # noqa: F401
        from importlib.metadata import version

        res["craftax"] = version("craftax")
    except Exception as e:  # noqa: BLE001
        res["craftax"] = f"unknown: {type(e).__name__}"

    ok = True
    for name, spec in CELLS.items():
        cls = getattr(importlib.import_module(spec["module"]), spec["cls"])
        states = np.load(os.path.join(spec["corpus"], "states.npz"))["states"]
        frozen = json.load(open(os.path.join(spec["corpus"], "corpus.json")))
        row = {
            "swm_id": spec["swm_id"],
            "corpus": spec["corpus"],
            "n_states": int(len(states)),
            "state_dim": int(states.shape[1]),
            "corpus_sha256": sha(states),
            "corpus_sha256_frozen": frozen["sha256"],
            "corpus_unchanged": sha(states) == frozen["sha256"],
            "sizes": {},
        }
        ok &= row["corpus_unchanged"]
        env = cls(device="cpu")
        for size in spec["sizes"]:
            ours = env.render(states, size=size).cpu().numpy()
            refenv = cls.reference_env(size=size)
            off = cls.reference_render(refenv, states)
            d = np.abs(off.astype(np.int16) - ours.astype(np.int16))
            e = {
                "shape": list(ours.shape),
                "digest_ours": sha(ours),
                "digest_official": sha(off),
                "byte_identical": sha(ours) == sha(off),
                "exact_frac": float((d.max(-1) == 0).mean()),
                "max_abs": int(d.max()),
                "digest_frozen_ours": frozen["frames"][str(size)]["sha256"],
                "matches_frozen": sha(ours) == frozen["frames"][str(size)]["sha256"],
            }
            ok &= e["byte_identical"] and e["matches_frozen"]
            row["sizes"][str(size)] = e
        res["cells"][name] = row

    res["all_byte_identical"] = bool(ok)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", OUT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
