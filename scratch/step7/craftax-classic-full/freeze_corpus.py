"""Freeze the Step-7 CraftaxClassic corpus: 64 states over the WHOLE observation and day cycle.

The Step-4 corpus stays exactly as it is -- this is a new artifact beside it, proposed for the
Step-8 freeze, and the frozen Step-4 record's replay obligation is asserted separately (the world
rows of a 127-slot row zero-extended to 149 with light_level 1.0).

Selection follows Step 4's: a typical state, per-dimension extremes, and a farthest-point spread,
drawn from a pool large enough that the spread has something to choose from. Stratification over the
lighting bands is added on top, because a corpus with no night rows cannot exercise the PRNG key and
a corpus with no daylight rows cannot exercise the uint8 fast path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from stable_envs.envs.craftax import CraftaxClassic, N_STRIP, OBS_H, OBS_W

OUT = "examples/results/step7/craftax-classic-full/corpus"
SLEEP = OBS_H * OBS_W * 2 + 1 + N_STRIP
LIGHT = SLEEP + 1


def farthest_point(x, k, first):
    chosen = [first]
    d = np.linalg.norm(x - x[first], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(x - x[i], axis=1))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    env = CraftaxClassic(device="cpu")
    rng = np.random.default_rng(a.seed)
    pool = env.sample_states(a.pool, rng)

    z = (pool - pool.mean(0)) / (pool.std(0) + 1e-9)
    typical = int(np.argmin(np.linalg.norm(z, axis=1)))
    extremes = sorted({int(np.argmax(pool[:, d])) for d in range(pool.shape[1])}
                      | {int(np.argmin(pool[:, d])) for d in range(pool.shape[1])})
    idx = [typical] + [i for i in extremes if i != typical]
    if len(idx) < a.keep:
        for i in farthest_point(z, a.keep, typical):
            if i not in idx:
                idx.append(i)
            if len(idx) == a.keep:
                break
    idx = sorted(idx[:a.keep])

    # make sure every lighting band and both sleep states survived the selection
    def band(st):
        L = st[:, LIGHT]
        return {"night": int((L < 0.5).sum()), "dusk": int(((L >= 0.5) & (L < 1.0)).sum()),
                "day": int((L == 1.0).sum())}

    kept = pool[idx]
    counts = band(kept)
    assert min(counts.values()) > 0, f"a lighting band did not survive selection: {counts}"
    assert 0 < int(kept[:, SLEEP].sum()) < len(kept), "sleeping rows did not survive selection"

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "states.npz"), states=kept)
    rec = {
        "env": "CraftaxClassic", "step": 7, "pool": a.pool, "kept": len(idx), "seed": a.seed,
        "selection": "typical + per-dim extremes + farthest-point spread, band-checked",
        "state_dim": int(env.state_dim), "state_names": list(env.state_names),
        "indices": idx, "sha256": hashlib.sha256(kept.tobytes()).hexdigest(),
        "band_counts": counts, "n_sleeping": int(kept[:, SLEEP].sum()),
        "count_coverage": {env.state_names[127 + i]:
                           sorted(set(np.round(kept[:, 127 + i]).astype(int).tolist()))
                           for i in range(N_STRIP)},
        "frames": {}}
    for size in (7, 16):
        f = env.render(kept, size=size).numpy()
        rec["frames"][str(size)] = {
            "shape": list(f.shape), "sha256": hashlib.sha256(f.tobytes()).hexdigest(),
            "mean_pixel": float(f.mean()),
            "distinct_colors": int(len(np.unique(f.reshape(-1, 3), axis=0)))}
    with open(os.path.join(OUT, "corpus.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("state_names", "indices", "count_coverage")}, indent=1))
    print("count coverage:", json.dumps(rec["count_coverage"])[:400])


if __name__ == "__main__":
    main()
