"""Does the frame depend on ``MJW_CHUNK``? It must not, or the chunk size is part of the contract.

``MJW_CHUNK`` sets the world count of the render context, and ``mujoco_warp`` compiles one megakernel
per context; HumanoidBench's ``render`` records that a batch of 1 and a batch of 3 do NOT agree there,
which is why that family pins its world count. Meta-World, Fetch and RoboCasa instead size the context
to the chunk, so raising the constant is only free if the pixels do not move. Same states, same env,
two chunk sizes, compared bit for bit.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se

MODS = ("metaworld", "humanoidbench", "fetch", "robocasa")


def set_chunk(v):
    for m in MODS:
        try:
            mod = importlib.import_module("stable_envs.envs." + m)
            mod.MJW_CHUNK = v
            mod._CTX_CACHE.clear()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--chunks", default="64,128,256,512")
    a = ap.parse_args()
    out = {}
    for name in a.envs.split(","):
        try:
            ref = None
            row = {}
            for cv in [int(x) for x in a.chunks.split(",")]:
                set_chunk(cv)
                torch.cuda.empty_cache()
                env = se.make(name, batch=a.batch, size=a.size, device="cuda")
                st = env.sample_states(a.batch, rng=np.random.default_rng(0))
                f = env.render(st).cpu().numpy()
                if ref is None:
                    ref = (cv, f)
                    row[cv] = 0
                else:
                    d = int(np.abs(f.astype(np.int16) - ref[1].astype(np.int16)).max())
                    nz = int((np.abs(f.astype(np.int16) - ref[1].astype(np.int16)).max(-1) > 0).sum())
                    row[cv] = d
                    print(f"{name:30s} chunk {cv:4d} vs {ref[0]}: max {d}/255 on {nz} px", flush=True)
                del env, f
                torch.cuda.empty_cache()
            out[name] = row
        except Exception as e:
            import traceback
            traceback.print_exc()
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
