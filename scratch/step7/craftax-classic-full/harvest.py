"""Phase 0 — harvest 64 REAL Craftax-Classic states covering day, dusk, night and sleep.

Real rollouts only: the excluded components (strip, night, sleeping) are functions of fields that a
synthetic `sample_states` never populates, so a synthetic pool would report every one of them as a
dead dimension. The policy is random with a SLEEP bias once `player_energy < 9`, which is the only
way `is_sleeping` ever becomes True (`game_logic.update_player_intrinsics`).

Writes `states.pkl` (a list of EnvState pytrees) plus a JSON census under phase0/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.constants import Action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260915)
    ap.add_argument("--keep", type=int, default=64)
    ap.add_argument("--out", default="scratch/step7/craftax-classic-full/states.pkl")
    ap.add_argument("--census", default="examples/results/step7/craftax-classic-full/phase0/state_census.json")
    a = ap.parse_args()

    env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
    params = env.default_params
    rng = jax.random.PRNGKey(a.seed)
    rng, r0 = jax.random.split(rng)
    _obs, state = env.reset(r0, params)
    step = jax.jit(env.step, static_argnums=())

    pool, rows = [], []
    npr = np.random.default_rng(a.seed)
    for t in range(a.steps):
        energy = float(state.player_energy)
        sleeping = bool(state.is_sleeping)
        light = float(state.light_level)
        # SLEEP is the only route into is_sleeping, and only fires below energy 9
        if light < 0.5 and energy < 9 and npr.random() < 0.6:
            act = Action.SLEEP.value
        else:
            act = int(npr.integers(0, 17))
        pool.append(state)
        rows.append(dict(t=t, light_level=light, is_sleeping=sleeping, energy=energy,
                         health=float(state.player_health), food=float(state.player_food),
                         drink=float(state.player_drink),
                         inv={k: float(getattr(state.inventory, k))
                              for k in ("wood", "stone", "coal", "iron", "diamond", "sapling",
                                        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
                                        "wood_sword", "stone_sword", "iron_sword")}))
        rng, rs = jax.random.split(rng)
        _obs, state, _r, done, _i = step(rs, state, act, params)
        if bool(done):
            rng, r0 = jax.random.split(rng)
            _obs, state = env.reset(r0, params)

    # stratified keep: night+sleeping, night-awake, dusk (0.5<=l<1), bright day
    def idx_where(f):
        return [i for i, r in enumerate(rows) if f(r)]

    strata = {
        "night_sleeping": idx_where(lambda r: r["light_level"] < 0.5 and r["is_sleeping"]),
        "night_awake": idx_where(lambda r: r["light_level"] < 0.5 and not r["is_sleeping"]),
        "dusk": idx_where(lambda r: 0.5 <= r["light_level"] < 0.999),
        "day": idx_where(lambda r: r["light_level"] >= 0.999),
        "sleeping_any": idx_where(lambda r: r["is_sleeping"]),
    }
    want = {"night_sleeping": 16, "night_awake": 16, "dusk": 16, "day": 16}
    chosen = []
    for k, n in want.items():
        c = strata[k]
        if not c:
            continue
        take = npr.choice(len(c), size=min(n, len(c)), replace=False)
        chosen += [c[int(i)] for i in take]
    # top up from anywhere
    i = 0
    while len(chosen) < a.keep and i < len(rows):
        if i not in chosen:
            chosen.append(i)
        i += 1
    chosen = sorted(chosen[: a.keep])

    kept = [pool[i] for i in chosen]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(kept, f)

    census = {
        "steps_rolled": a.steps, "seed": a.seed, "kept": len(kept),
        "stratum_sizes": {k: len(v) for k, v in strata.items()},
        "kept_indices": chosen,
        "kept_rows": [rows[i] for i in chosen],
        "light_level_range": [min(r["light_level"] for r in rows),
                              max(r["light_level"] for r in rows)],
        "kept_light_below_half": sum(1 for i in chosen if rows[i]["light_level"] < 0.5),
        "kept_sleeping": sum(1 for i in chosen if rows[i]["is_sleeping"]),
        "inventory_max_seen": {k: max(r["inv"][k] for r in rows) for k in rows[0]["inv"]},
        "intrinsic_max_seen": {k: max(r[k] for r in rows) for k in ("health", "food", "drink", "energy")},
    }
    os.makedirs(os.path.dirname(a.census), exist_ok=True)
    with open(a.census, "w") as f:
        json.dump(census, f, indent=2)
    print(json.dumps({k: v for k, v in census.items() if k not in ("kept_rows", "kept_indices")}, indent=2))


if __name__ == "__main__":
    main()
