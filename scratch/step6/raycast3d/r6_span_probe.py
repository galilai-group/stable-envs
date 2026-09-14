"""RAYCAST-6 — what window each ``composite`` caller actually needs.

``composite``'s tiled path casts ``(tile*ss)^2`` samples per environment; ``ogbench`` and ``maze2d``
both pass ``tile = 32`` and the profile says **72 % of a PointmazeLargeNavigate composite is
proportional to that number** while only 4.7 % of those samples can be covered. The ratio between
"the window the caller asked for" and "the window the bound says is needed" is therefore the design's
central number, and it is measured here rather than assumed: ``screen_span`` is intercepted and its
``(w, h)`` recorded per call, across every config that enters ``composite``.

Writes under ``examples/results/step6/raycast3d/experiments/``. No renderer code is modified.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))


def _root_tag(p: str) -> str:
    return re.sub(r"^/oscar/scratch/([^/]+)", r"$SCR", p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--batches", default="16,64,256")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import numpy as np
    import torch
    import stable_envs as se
    from stable_envs.render import raycast3d as R3

    torch.set_num_threads(a.threads)
    orig = R3.screen_span
    seen: list = []

    def probe(cam, prims, xpos, H, W, device, bound_scale=1.0):
        r = orig(cam, prims, xpos, H, W, device, bound_scale)
        seen.append((None, None) if r is None else (int(r[2]), int(r[3])))
        return r

    res = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "device": a.device, "stable_envs_from": _root_tag(se.__file__), "configs": {}}
    for cfg in [c for c in a.configs.split(",") if c]:
        rec = {}
        for B in [int(x) for x in a.batches.split(",") if x]:
            env = se.make(cfg, batch=B, device=a.device)
            npz = ROOT / "examples" / "results" / "step4" / "states" / f"{cfg}.npz"
            if npz.is_file():
                pool = np.asarray(np.load(npz)["states"], np.float32)
                st = pool[np.arange(B) % len(pool)]
            else:
                st = env.sample_states(B, np.random.default_rng(a.seed))
            seen.clear()
            R3.screen_span = probe
            try:
                env.render(st)
            finally:
                R3.screen_span = orig
            ok = [s for s in seen if s[0] is not None]
            rec[str(B)] = {
                "calls": len(seen), "none": len(seen) - len(ok),
                "w_max": max((s[0] for s in ok), default=None),
                "h_max": max((s[1] for s in ok), default=None),
                "size": int(env.size), "ss": int(env.ss),
                "caller_tile": int(getattr(type(env), "TILE", 0)),
            }
            print(cfg, B, rec[str(B)], flush=True)
            del env
        res["configs"][cfg] = rec

    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(res, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
