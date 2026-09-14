"""Render the four trackcom cells on cuda (MJWarp) at the frozen Step-4 states and save the frames.

The cross-backend oracle for the CPU path, produced by the SAME tree the CPU frames come from, on the
certification card. ``verify_pixels.py`` (reacher-correctness) keeps digests only; this keeps pixels.
"""
import argparse, hashlib, json, os, socket
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
ap.add_argument("--cells", default="DMCCheetahRun,DMCHopperHop,DMCWalkerWalk,DMCHumanoidStand")
a = ap.parse_args()
import torch, stable_envs as se
frames, meta = {}, {"host": socket.gethostname(), "job": os.environ.get("SLURM_JOB_ID"), "device": a.device,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "cells": {}}
for c in a.cells.split(","):
    st = np.load(ROOT / f"examples/results/step4/states/{c}.npz")["states"]
    env = se.make(c, batch=64, device=a.device)
    f = env.render(st).cpu().numpy()
    frames[c] = f
    meta["cells"][c] = {"sha256": hashlib.sha256(np.ascontiguousarray(f).tobytes()).hexdigest(),
                        "mean_pixel": float(f.mean()), "n": int(len(st))}
    print(c, meta["cells"][c], flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out, **frames)
Path(str(a.out).replace(".npz", ".json")).write_text(json.dumps(meta, indent=1))
