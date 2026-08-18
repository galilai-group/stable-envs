"""What the registry actually carries after the supersample change, printed rather than assumed.

A class attribute edited in a module docstring's neighbourhood is easy to get right in the source and
wrong in the registry -- ``_make_classes`` builds the DM Control cells with ``type()`` and the gym
cells from a table, so the declared floors reach a class through two different paths. This asks the
registry.
"""
from __future__ import annotations

import json
import os

import common as K  # noqa: E402

import stable_envs as se  # noqa: E402

PARITY_CELLS = ["DMCAcrobotSwingup", "DMCBallInCupCatch", "DMCCartpoleBalance", "DMCCartpoleSwingup",
                "DMCFingerSpin", "DMCFingerTurnEasy", "DMCFingerTurnHard", "DMCPendulumSwingup",
                "DMCPointMassEasy", "DMCReacherEasy", "DMCReacherHard",
                "MujocoReacher", "MujocoPusher", "MujocoInvertedPendulum",
                "MujocoInvertedDoublePendulum"]
#: Cells in the same two modules that must NOT have moved: the DM Control track-com domains and the
#: gym locomotion domains are VISUAL tier and were already at one ray per pixel.
UNCHANGED = ["DMCCheetahRun", "DMCHopperHop", "DMCHumanoidStand", "DMCSwimmerSwimmer6",
             "DMCWalkerWalk", "MujocoAnt", "MujocoHalfCheetah", "MujocoHopper", "MujocoHumanoid",
             "MujocoSwimmer", "MujocoWalker2d"]


def main():
    errs = se.envs.import_errors()
    if errs:
        print("env modules that did not import:", errs, flush=True)
    out = {"import_errors": {k: str(v) for k, v in errs.items()}, "cells": {}}
    bad = []
    for name in PARITY_CELLS + UNCHANGED:
        if name not in se.REGISTRY:
            bad.append(f"{name}: not registered")
            continue
        c = se.REGISTRY[name]
        env = c(batch=1, size=int(c.default_size), device="cuda")
        row = {"default_supersample": int(c.default_supersample),
               "instance_ss": int(env.ss),
               "fidelity_tier": env.fidelity_tier,
               "parity_interior_min": getattr(c, "parity_interior_min", None),
               "parity_all_exact_min": getattr(c, "parity_all_exact_min", None),
               "has_parity_note": bool(getattr(c, "parity_note", ""))}
        out["cells"][name] = row
        if int(env.ss) != 1:
            bad.append(f"{name}: instance ss is {env.ss}, expected 1")
        if row["parity_interior_min"] not in (None, 1.0) and not row["has_parity_note"]:
            bad.append(f"{name}: relaxed interior floor {row['parity_interior_min']} with no note")
        print(f"{name:30s} tier={row['fidelity_tier']:6s} ss={row['instance_ss']} "
              f"interior_min={row['parity_interior_min']} "
              f"all_exact_min={row['parity_all_exact_min']} note={row['has_parity_note']}", flush=True)
        del env
    with open(os.path.join(K.LOGS, "verify_landed.json"), "w") as f:
        json.dump(out, f, indent=1)
    if bad:
        raise SystemExit("LANDED WRONG:\n  " + "\n  ".join(bad))
    print(f"\nall {len(PARITY_CELLS) + len(UNCHANGED)} cells render at one ray per pixel and every "
          f"relaxed floor names its mechanism", flush=True)


if __name__ == "__main__":
    main()
