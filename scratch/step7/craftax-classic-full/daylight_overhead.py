"""Is the daylight path still the Step-6 gather, or did Step 7 quietly tax it?

Run once against the baseline tree and once against the branch, on the same machine, at one thread.
The daylight/awake/unit-gain path is supposed to be untouched: same uint8 gather, no float buffer,
no PRNG. It is NOT automatically untouched, because the frame grew from 7 to 9 tile rows and because
anything ``render`` does per call before the gather is paid by every frame including a daylight one.

Reports ms/render and the row-growth share, so the two effects are separated instead of summed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from stable_envs.envs.craftax import CraftaxClassic, OBS_H, OBS_W


def bench(fn, n=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    env = CraftaxClassic(device="cpu")
    D = env.state_dim
    res = {"tag": a.tag, "state_dim": D, "torch": torch.__version__,
           "machine": os.uname().machine, "threads": 1}
    for B in (1, 16, 64, 256, 1024):
        st = env.sample_states(B, np.random.default_rng(1))
        if D > 127:                       # branch: pin daylight + awake, the Step-6-comparable state
            from stable_envs.envs.craftax import N_STRIP
            st[:, OBS_H * OBS_W * 2 + 1 + N_STRIP] = 0.0
            st[:, OBS_H * OBS_W * 2 + 2 + N_STRIP] = 1.0
        for t in (7, 16):
            ms = bench(lambda: env.render(st, size=t)) * 1e3
            f = env.render(st, size=t)
            res[f"t{t}_B{B}_ms"] = round(ms, 4)
            res[f"t{t}_B{B}_fps"] = round(B / (ms / 1e3))
            res[f"t{t}_B{B}_rows"] = int(f.shape[1])
    p = a.out or f"examples/results/step7/craftax-classic-full/sweep/daylight_{a.tag}.json"
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
