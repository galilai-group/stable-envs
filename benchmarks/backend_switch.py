"""Time the two CUDA simulation backends of each class at batches 1, 2, 4, ..., 4096, as stable-envs tune times them.

usage: python -m benchmarks.backend_switch ENV [ENV ...]

Prints both per-call step times at each batch and the smallest batch from which the large-batch backend is faster at
every larger measured batch: the value a class ships as its CUDA switch point.
"""
import json
import sys

import torch

import stable_envs as snv
from stable_envs._execution import backends as B


def main():
    out = {}
    for name in sys.argv[1:]:
        B._REGISTERED.clear()
        try:
            snv.make(name, num_envs=1, device="cuda").close()
        except Exception as e:
            print(name, "skip", e, flush=True); continue
        for path, (decl, measure) in B._REGISTERED.items():
            rungs = B._rungs(decl, torch.device("cuda"))
            if rungs.small == rungs.large:
                continue
            rows, small_lost = [], 0
            for b in [1 << k for k in range(13)]:
                large = B._time(measure, rungs.large, b, torch.device("cuda"))
                small = B._time(measure, rungs.small, b, torch.device("cuda")) if small_lost < 2 else float("inf")
                small_lost = small_lost + 1 if small > large else 0
                rows.append((b, small, large))
                times = f"{rungs.small}={small * 1e6:.0f}us {rungs.large}={large * 1e6:.0f}us"
                print(f"{name} {path} B={b} {times}", flush=True)
            switch = next((b for i, (b, s, l) in enumerate(rows) if all(l2 <= s2 for _, s2, l2 in rows[i:])), None)
            out[path] = {"switch": switch, "rows": rows}
            print(f"SWITCH {name} {path} {switch}", flush=True)
    print(json.dumps({k: v["switch"] for k, v in out.items()}))
if __name__ == "__main__":
    main()
