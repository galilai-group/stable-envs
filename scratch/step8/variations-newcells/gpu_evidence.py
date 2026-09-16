#!/usr/bin/env python
"""GPU evidence for Workstream C: the registered default did not move, and mapped leaves isolate.

Two claims, measured on the card rather than argued from the diff:

1. **Default reproduction.** This workstream adds a map module and one import line and touches no
   renderer, so the registered default of every cell must be BYTE-identical to the baseline. The
   honest way to show that is to render the same states at the default style under BOTH trees in the
   SAME process-pair on the SAME card and compare digests -- `PYTHONPATH` selects which tree.
2. **Isolation.** For the one family with mapped style dims that reach a GPU renderer
   (PFRocketLanding), each mapped leaf is moved alone and must change pixels while leaving every
   other leaf's style dims untouched.

    python gpu_evidence.py --mode default --tag {baseline,branch} --out OUT.json
    python gpu_evidence.py --mode isolation --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

import numpy as np

CELLS_CUDA = ["OGBCube", "OGBScene", "PFRocketLanding"]
N_STATES = 8


def provenance():
    p = {"host": platform.node(), "python": sys.version.split()[0],
         "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
         "slurm_nodelist": os.environ.get("SLURM_NODELIST"),
         "pythonpath": os.environ.get("PYTHONPATH")}
    try:
        p["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                      cwd=os.path.dirname(__file__) or ".").stdout.strip() or None
    except Exception:
        p["git_sha"] = None
    try:
        out = subprocess.run(["nvidia-smi",
                              "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
        p["gpu"] = out
    except Exception:
        p["gpu"] = None
    for mod in ("torch", "mujoco", "mujoco_warp", "warp", "stable_worldmodel", "ogbench", "PyFlyt",
                "nvdiffrast"):
        try:
            p[mod] = __import__(mod).__version__
        except Exception as exc:
            p[mod] = f"unavailable: {type(exc).__name__}"
    try:
        import torch
        p["cuda"] = {"available": torch.cuda.is_available(),
                     "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                     "capability": list(torch.cuda.get_device_capability(0))
                     if torch.cuda.is_available() else None}
    except Exception as exc:
        p["cuda"] = f"unavailable: {type(exc).__name__}"
    return p


def digest(a):
    return hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=16).hexdigest()


def render_default(name):
    """Registered default: default size, default style, a fixed state draw."""
    import stable_envs as se
    env = se.make(name, batch=1, device="cuda")
    states = env.sample_states(N_STATES, rng=np.random.default_rng(0))
    frames = env.render(states).detach().cpu().numpy()
    return {"cell": name, "state_dim": int(env.state_dim), "size": int(env.size),
            "style_dim": int(np.asarray(env.default_style()).size),
            "n_states": int(N_STATES), "frames_shape": list(frames.shape),
            "frames_digest": digest(frames), "states_digest": digest(states),
            "default_style_digest": digest(np.asarray(env.default_style(), np.float32)),
            "default_style": np.asarray(env.default_style(), np.float64).tolist()}


def mode_default(args):
    rows, errs = [], {}
    for name in CELLS_CUDA:
        try:
            rows.append(render_default(name))
        except Exception as exc:
            errs[name] = f"{type(exc).__name__}: {exc}"
    return {"mode": "default", "tag": args.tag, "cells": rows, "errors": errs}


def mode_isolation(args):
    """PFRocketLanding: move one mapped leaf at a time and confirm it isolates."""
    import stable_envs as se
    from stable_envs.compat.worldmodel import VARIATION_MAPS
    from stable_envs.envs.rocket_landing import style_layout

    m = VARIATION_MAPS["PFRocketLanding"]
    layout = style_layout()
    env = se.make("PFRocketLanding", batch=1, device="cuda")
    states = env.sample_states(N_STATES, rng=np.random.default_rng(0))
    base_style = np.asarray(env.default_style(), np.float64).reshape(-1)
    base = env.render(states, style=np.tile(base_style, (N_STATES, 1))).detach().cpu().numpy()

    out = {"mode": "isolation", "cell": "PFRocketLanding", "base_digest": digest(base),
           "total_pixels": int(base[..., 0].size), "leaves": {}}
    for ours in sorted(m.pairs()):
        o, w = layout[ours]
        row = base_style.copy()
        # move this leaf to the far corner of its own range; every other dim is untouched
        row[o:o + w] = 255.0 - row[o:o + w]
        f = env.render(states, style=np.tile(row, (N_STATES, 1))).detach().cpu().numpy()
        d = np.abs(f.astype(np.int32) - base.astype(np.int32))
        untouched = [i for i in range(len(row)) if not (o <= i < o + w)]
        out["leaves"][ours] = {
            "style_slice": [int(o), int(w)],
            "maxdiff": int(d.max()), "npix": int((d.max(-1) > 0).sum()),
            "moves_pixels": bool(d.max() > 0),
            "other_dims_unchanged": bool(np.array_equal(row[untouched], base_style[untouched])),
            "digest": digest(f)}
    out["all_mapped_leaves_move_pixels"] = all(v["moves_pixels"] for v in out["leaves"].values())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["default", "isolation"], required=True)
    ap.add_argument("--tag", default="branch", help="which tree PYTHONPATH selects")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    doc = {"provenance": provenance()}
    doc.update(mode_default(a) if a.mode == "default" else mode_isolation(a))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
    print(json.dumps(doc, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
