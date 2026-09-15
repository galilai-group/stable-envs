"""Are the 15 pre-existing DM Control cells BYTE-IDENTICAL after the phase-1 edits?

`_episode_slots`, `_hideable`, `_pristine` and `_apply_variation` are family-shared, so the new cell
cannot be argued to be free: it has to be measured. This renders each cell at its own default size on
CPU from a seeded state draw and prints a sha256 of the frames, at the default style AND at a style
that moves every leaf (which is what exercises `_apply_variation`'s restore and the visibility path).

Run it once with PYTHONPATH pointing at the baseline checkout and once at this worktree, then diff
the two JSON files. The state draw is seeded and `sample_states` is deterministic given the seed, so
the two runs see the same states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    import stable_envs as se
    names = sorted(n for n, c in se.REGISTRY.items()
                   if n.startswith("DMC") and getattr(c, "DOMAIN", ""))
    out = {"n": a.n, "device": a.device, "cells": {}}
    for nm in names:
        cls = se.REGISTRY[nm]
        try:
            env = cls(batch=a.n, device=a.device, seed=0)
        except Exception as e:                                        # noqa: BLE001
            out["cells"][nm] = {"SKIP": f"{type(e).__name__}: {e}"}
            print(nm, "SKIP", type(e).__name__, e)
            continue
        rec = {"size": int(env.size), "state_dim": int(env.state_dim),
               "state_names": list(env.state_names),
               "leaves": [p for p, _, _ in env.variation_space.flat_layout()]}
        st = env.sample_states(a.n, np.random.default_rng(4242))
        rec["states_sha"] = sha(st)
        rec["default_sha"] = sha(env.render(torch.as_tensor(st)).cpu().numpy())
        # a style that moves every leaf: colours to a fixed non-default triple, scales to an end of
        # their range, every Discrete to 0 (which HIDES a visibility leaf and switches a shape leaf)
        over = {}
        for p, _, _ in env.variation_space.flat_layout():
            head = p.split(".")[0]
            if head == "material":
                over[p] = [0.2, 0.9, 0.4]
            elif p.endswith("fovy_scale"):
                over[p] = [1.3]
            elif p.endswith("diffuse_scale"):
                over[p] = [0.6]
            else:
                over[p] = [0]
        rec["moved_style_sha"] = sha(env.render(torch.as_tensor(st), variation=over).cpu().numpy())
        out["cells"][nm] = rec
        print(nm, rec["size"], rec["states_sha"], rec["default_sha"], rec["moved_style_sha"])

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
