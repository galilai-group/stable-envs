"""How much of the observation the Step-7 additions actually are.

Question (2) of the Workstream-H brief is whether a downstream model consumes the newly added
information channels -- the inventory strip, night, and the sleeping state. That question has two
halves. This probe answers the half that does not need a model: **are those channels
information-bearing in the pixels a model would be handed, and how large are they?** If flipping a
channel moved no pixel, no model could consume it whatever its architecture.

Measured on the frozen Step-7 corpora, at every certified tile size, with the stable-envs renderer
only -- no reference needed, so it runs in the plain CPU venv. Writes
``examples/results/step8/semantic/channel_probe.json``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

OUT = "examples/results/step8/semantic/channel_probe.json"

CELLS = {
    "CraftaxClassic": {
        "module": "stable_envs.envs.craftax",
        "cls": "CraftaxClassic",
        "swm_id": "swm/CraftaxClassicPixels-v1",
        "corpus": "examples/results/step7/craftax-classic-full/corpus/states.npz",
        "sizes": [7, 16],
        "world_tile_rows": 7,
    },
    "CraftaxFull": {
        "module": "stable_envs.envs.craftax_full",
        "cls": "CraftaxFull",
        "swm_id": "swm/CraftaxPixels-v1",
        "corpus": "examples/results/step7/craftax-full/corpus/states.npz",
        "sizes": [10, 16],
        "world_tile_rows": 9,
    },
}


def compare(a: np.ndarray, b: np.ndarray, split: int) -> dict:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16))
    def part(x):
        return {"px_changed_frac": float((x.max(-1) != 0).mean()),
                "mean_abs": float(x.mean()), "max_abs": int(x.max())}
    return {"frame": part(d), "world_rows": part(d[:, :split]), "strip_rows": part(d[:, split:])}


def main() -> int:
    import importlib

    res = {
        "what": ("whether the Step-7 additions (inventory strip, night, sleeping) are "
                 "information-bearing in the rendered observation, and how large they are"),
        "corpora": "the frozen Step-7 corpora, unmodified except for the one channel under test",
        "cells": {},
    }
    for name, spec in CELLS.items():
        cls = getattr(importlib.import_module(spec["module"]), spec["cls"])
        env = cls(device="cpu")
        names = list(env.state_names)
        states = np.load(spec["corpus"])["states"]
        light = [i for i, n in enumerate(names) if n == "light_level" or n.startswith("light_r")]
        sleep = names.index("is_sleeping")
        inv = [i for i, n in enumerate(names) if n.startswith("inv_")]
        row = {"swm_id": spec["swm_id"], "n_states": int(len(states)),
               "n_light_dims": len(light), "n_inventory_dims": len(inv), "sizes": {}}
        for size in spec["sizes"]:
            split = spec["world_tile_rows"] * size
            base = env.render(states, size=size).cpu().numpy()
            H = base.shape[1]
            day, night = np.array(states, copy=True), np.array(states, copy=True)
            day[:, light], night[:, light] = 1.0, 0.1
            day[:, sleep], night[:, sleep] = 0.0, 0.0
            awake, asleep = np.array(states, copy=True), np.array(states, copy=True)
            awake[:, sleep], asleep[:, sleep] = 0.0, 1.0
            empty = np.array(states, copy=True)
            empty[:, inv] = 0.0
            e = {
                "frame_shape": list(base.shape[1:]),
                "world_rows": [0, int(split)],
                "strip_rows": [int(split), int(H)],
                "strip_frac_of_frame": float((H - split) / H),
                "day_vs_night": compare(env.render(day, size=size).cpu().numpy(),
                                        env.render(night, size=size).cpu().numpy(), split),
                "awake_vs_sleeping": compare(env.render(awake, size=size).cpu().numpy(),
                                             env.render(asleep, size=size).cpu().numpy(), split),
                "inventory_actual_vs_empty": compare(
                    base, env.render(empty, size=size).cpu().numpy(), split),
            }
            row["sizes"][str(size)] = e
        res["cells"][name] = row

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
