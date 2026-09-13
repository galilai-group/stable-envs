"""CPU-side pixel identity check for the six raster2d configurations.

The frozen Step 4 oracle digests were taken on cuda, so they answer "did the GPU picture change".
They do not answer "did the CPU picture change", and the CPU path is the one this workstream edits.
This renders the SAME frozen states on cpu and digests them with the SAME function the Step 4 oracle
used, so a before/after pair on one machine is a byte-exact statement about the CPU picture.

    python scratch/step6/raster2d/cpu_digest.py --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import stable_envs as se                                   # noqa: E402
from stable_envs.testing import conformance as C           # noqa: E402

CONFIGS = ("PushT", "SimplePointMaze", "TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--states-dir", default=str(ROOT / "examples" / "results" / "step4" / "states"))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    out = {"host": socket.gethostname(), "cpu": platform.machine(), "device": a.device,
           "torch": torch.__version__, "configs": {}}
    for cfg in a.configs.split(","):
        states = np.load(Path(a.states_dir) / f"{cfg}.npz")["states"]
        env = se.make(cfg, batch=len(states), device=a.device)
        row = {"n_states": int(len(states)), "size": int(env.size)}
        frames = env.render(states)
        row["digest"] = C.frames_digest(frames)
        row["frames_shape"] = list(np.asarray(frames.cpu()).shape)
        row["mean_pixel"] = float(frames.float().mean())
        # batch invariance: the same state rendered alone must give the same pixels
        solo = se.make(cfg, batch=1, device=a.device)
        row["digest_b1"] = C.frames_digest(solo.render(states[:1]))
        row["b1_matches"] = row["digest_b1"] == C.frames_digest(frames[:1])
        # order invariance: a permuted batch must give permuted pixels
        perm = np.argsort(np.arange(len(states)) * 7 % len(states))
        pf = env.render(np.ascontiguousarray(states[perm]))
        row["order_invariant"] = C.frames_digest(pf[np.argsort(perm)]) == row["digest"]
        # render purity: calling twice must give the same thing
        row["pure"] = C.frames_digest(env.render(states)) == row["digest"]
        if env.style_dim:
            sty = np.tile(np.asarray(env.default_style(), np.float32), (len(states), 1))
            row["digest_default_style"] = C.frames_digest(env.render(states, style=sty))
        out["configs"][cfg] = row
        print(cfg, row["digest"][:16], "b1", row["b1_matches"], "order", row["order_invariant"],
              "pure", row["pure"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
