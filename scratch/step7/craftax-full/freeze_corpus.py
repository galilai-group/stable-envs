"""Freeze the Step-7 CraftaxFull corpus: 64 states over nine levels and all five cost bands.

A new artifact beside the frozen Step-4 corpora, in Step-4's own schema and proposed for the Step-8
freeze. Selection follows Step 4's -- a typical state, per-dimension extremes, and a farthest-point
spread over a larger pool -- with the band and coverage checks this env needs **asserted after
selection** rather than hoped for, because a corpus with no level-0 night row cannot exercise the
PRNG key, one with no fully-lit row cannot exercise the uint8 gather, and one with no dark tile
cannot exercise the light map.

The pool is written beside the selection so a later run can show which 64 of which 512 were kept.

    PYTHONPATH=. python scratch/step7/craftax-full/freeze_corpus.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))

from stable_envs.envs import craftax_full as CF                       # noqa: E402
from stable_envs.envs.craftax_full import CraftaxFull                 # noqa: E402

OUT = "examples/results/step7/craftax-full/corpus"
J = 5 * CF.N_TILE


def farthest_point(x, k, first):
    chosen = [first]
    d = np.linalg.norm(x - x[first], axis=1)
    while len(chosen) < k:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(x - x[i], axis=1))
    return chosen


def describe(st):
    lvl = np.round(st[:, J + 2]).astype(int) % CF.N_LEVEL
    light = CF._tent_np(st[:, J + 3])
    day = np.where(lvl == 0, light, 1.0)
    lit = (CF._tent_np(st[:, 4 * CF.N_TILE:5 * CF.N_TILE]) == 1.0).all(1)
    sleeping = st[:, J + 1] > 0.5
    return {
        "levels": sorted(set(int(v) for v in lvl)),
        "n_per_level": {str(v): int((lvl == v).sum()) for v in sorted(set(lvl.tolist()))},
        "level0_night": int(((lvl == 0) & (light < CF.NIGHT_THRESHOLD)).sum()),
        "level0_dusk": int(((lvl == 0) & (light >= CF.NIGHT_THRESHOLD) & (light < 1.0)).sum()),
        "level0_day": int(((lvl == 0) & (light == 1.0)).sum()),
        "underground_lit": int(((lvl > 0) & lit).sum()),
        "underground_dark": int(((lvl > 0) & ~lit).sum()),
        "uint8_fast_path": int(((day == 1.0) & lit & ~sleeping).sum()),
        "prng_drawn": int((day < CF.NIGHT_THRESHOLD).sum()),
        "sleeping": int(sleeping.sum()),
        "tiles_with_a_mob": int((np.round(st[:, 2 * CF.N_TILE:3 * CF.N_TILE]) != 0).sum()),
        "tiles_with_a_projectile": int((np.round(st[:, 3 * CF.N_TILE:4 * CF.N_TILE]) != 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    env = CraftaxFull(batch=a.keep, device="cpu")
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

    # Step 4's selection optimises for spread, not for coverage, and spread does not guarantee that
    # all nine levels survive -- level 4 did not, on this seed. Rather than widen the pool until the
    # draw happens to work, a repair pass swaps a missing level in against the most over-represented
    # one, and the swap is recorded so the corpus says what was done to it.
    lvl_of = lambda st: int(round(float(st[J + 2]))) % CF.N_LEVEL          # noqa: E731
    repairs = []
    for want in range(CF.N_LEVEL):
        have = [i for i in idx if lvl_of(pool[i]) == want]
        if have:
            continue
        cand = next((i for i in range(len(pool)) if lvl_of(pool[i]) == want and i not in idx), None)
        if cand is None:
            raise AssertionError(f"the pool itself contains no level-{want} state")
        counts = {}
        for i in idx:
            counts.setdefault(lvl_of(pool[i]), []).append(i)
        drop = max(counts.values(), key=len)[-1]
        idx = sorted([i for i in idx if i != drop] + [cand])
        repairs.append({"added_level": want, "added_index": int(cand), "dropped_index": int(drop)})
    kept = pool[idx]

    cov = describe(kept)
    for need in ("level0_night", "level0_dusk", "level0_day", "underground_lit",
                 "underground_dark", "sleeping", "uint8_fast_path"):
        assert cov[need] > 0, f"{need} did not survive selection: {cov}"
    assert len(cov["levels"]) == CF.N_LEVEL, f"not every level survived: {cov['levels']}"
    assert cov["sleeping"] < len(kept), "every row is asleep"

    # every mob class and every projectile orientation must be somewhere in the corpus
    mob = np.round(kept[:, 2 * CF.N_TILE:3 * CF.N_TILE]).astype(int)
    prj = np.round(kept[:, 3 * CF.N_TILE:4 * CF.N_TILE]).astype(int)
    classes = {name: int(((mob >= b) & (mob < b + n)).sum()) for name, b, n in
               (("melee", CF.MOB_BASE["melee"], CF.N_MELEE),
                ("passive", CF.MOB_BASE["passive"], CF.N_PASSIVE),
                ("ranged", CF.MOB_BASE["ranged"], CF.N_RANGED))}
    orients = sorted(set(((prj[prj != 0] - 1) % 4).tolist()))
    assert min(classes.values()) > 0, classes
    assert orients == [0, 1, 2, 3], orients

    os.makedirs(OUT, exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "states.npz"), states=kept)
    np.savez_compressed(os.path.join(OUT, "pool.npz"), states=pool)
    rec = {
        "env": "CraftaxFull", "step": 7, "pool": a.pool, "kept": len(idx), "seed": a.seed,
        "selection": "typical + per-dim extremes + farthest-point spread, coverage-checked",
        "state_dim": int(env.state_dim), "indices": idx,
        "sha256": hashlib.sha256(np.ascontiguousarray(kept).tobytes()).hexdigest(),
        "pool_sha256": hashlib.sha256(np.ascontiguousarray(pool).tobytes()).hexdigest(),
        "coverage": cov, "level_repairs": repairs, "mob_classes": classes, "projectile_orientations": orients,
        "strip_coverage": {CF.STRIP_SLOTS[i]:
                           [int(np.round(kept[:, J + 4 + CF.N_KEY_HALVES + i]).min()),
                            int(np.round(kept[:, J + 4 + CF.N_KEY_HALVES + i]).max())]
                           for i in range(CF.N_STRIP)},
        "frames": {}}
    for size in (10, 16):
        f = env.render(kept, size=size).cpu().numpy()
        rec["frames"][str(size)] = {
            "shape": list(f.shape), "sha256": hashlib.sha256(f.tobytes()).hexdigest(),
            "mean_pixel": float(f.mean()),
            "distinct_colors": int(len(np.unique(f.reshape(-1, 3), axis=0)))}
    with open(os.path.join(OUT, "corpus.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2)[:4000])
    print("wrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
