"""Byte-identity of the nvdiff backend's existing users before and after the Step-7 extensions.

nvdiff's only registered user is the six-cell locomotion visual tier (Fetch renders through
mujoco_warp, and the frozen Step-4 corpus holds no nvdiff-backed config), so the replay is: the same 64
states per cell, rendered under PYTHONPATH=baseline and PYTHONPATH=branch IN THE SAME JOB on the same
card, digests compared. Run twice with ``--phase baseline`` / ``--phase branch`` (the sbatch does), then
``--phase compare``.

Usage: python nvdiff_identity.py --phase {baseline,branch,compare} --out DIR [--n 64]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

CELLS = ("MujocoHalfCheetah", "MujocoHopper", "MujocoWalker2d", "MujocoAnt", "MujocoHumanoid", "MujocoSwimmer")


def render_phase(out: str, n: int, phase: str) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch
    import stable_envs as se
    res = {"phase": phase, "pythonpath": os.environ.get("PYTHONPATH"), "stable_envs_file": se.__file__,
           "gpu": torch.cuda.get_device_name(0), "cells": {}}
    for name in CELLS:
        if name not in se.REGISTRY:
            res["cells"][name] = {"missing": se.envs.import_errors().get("mujoco_locomotion", "not registered")}
            continue
        env = se.make(name, batch=16, device="cuda")
        st = env.sample_states(n, rng=np.random.default_rng(0))
        frames = []
        for lo in range(0, n, 16):
            frames.append(env.render(st[lo:lo + 16]).cpu().numpy())
        frames = np.concatenate(frames)
        np.savez_compressed(os.path.join(out, f"{phase}_{name}.npz"), frames=frames, states=st)
        res["cells"][name] = {"sha256": hashlib.sha256(frames.tobytes()).hexdigest(), "shape": list(frames.shape)}
        print(phase, name, res["cells"][name], flush=True)
    with open(os.path.join(out, f"{phase}.json"), "w") as fh:
        json.dump(res, fh, indent=1)


def compare(out: str) -> None:
    a = json.load(open(os.path.join(out, "baseline.json")))
    b = json.load(open(os.path.join(out, "branch.json")))
    rep = {"same_gpu": a["gpu"] == b["gpu"], "gpu": a["gpu"], "cells": {}}
    for name in CELLS:
        ca, cb = a["cells"].get(name, {}), b["cells"].get(name, {})
        if "sha256" in ca and "sha256" in cb:
            same = ca["sha256"] == cb["sha256"]
            row = {"identical": same}
            if not same:
                fa = np.load(os.path.join(out, f"baseline_{name}.npz"))["frames"].astype(np.int16)
                fb = np.load(os.path.join(out, f"branch_{name}.npz"))["frames"].astype(np.int16)
                d = np.abs(fa - fb)
                row.update({"differing_px": int((d.max(-1) > 0).sum()), "max_abs": int(d.max())})
            rep["cells"][name] = row
        else:
            rep["cells"][name] = {"missing": ca.get("missing") or cb.get("missing")}
    rep["all_identical"] = all(r.get("identical") for r in rep["cells"].values() if "identical" in r)
    print(json.dumps(rep, indent=1))
    with open(os.path.join(out, "compare.json"), "w") as fh:
        json.dump(rep, fh, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("baseline", "branch", "compare"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.phase == "compare":
        compare(a.out)
    else:
        render_phase(a.out, a.n, a.phase)
