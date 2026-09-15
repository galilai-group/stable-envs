"""Harvest real full-Craftax states, stratified over levels, lighting bands, sleeping and mobs.

A random policy never descends, so a rollout alone gives level 0 and nothing else. It does not have
to: ``generate_world`` builds **all nine levels at reset** (``state.map`` is ``(9, 48, 48)``), so a
level-k frame is reached by writing ``player_level = k`` and standing the player on level k's own
up-ladder -- a state the simulator itself produces on a DESCEND. Both kinds are kept and labelled,
because the rollout states are the ones that certify "this is a state the env reaches" and the
level-shifted ones are the ones that cover the nine backgrounds.

Writes ``examples/results/step7/craftax-full/phase0/state_census.json`` and a pickle of the states
under ``scratch/step7/craftax-full/_pool.pkl`` (not committed; regenerate with this script).
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")

N_STEPS = 4000
SEED = 20260915


def rollout(n_steps=N_STEPS, seed=SEED):
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
    params = env.default_params
    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    _obs, state = env.reset(rk, params)
    step = jax.jit(env.step)

    states = [state]
    for i in range(n_steps):
        key, ak, sk = jax.random.split(key, 3)
        a = jax.random.randint(ak, (), 0, env.num_actions)
        _obs, state, _r, done, _info = step(sk, state, a, params)
        states.append(state)
        if bool(done):
            key, rk = jax.random.split(key)
            _obs, state = env.reset(rk, params)
            states.append(state)
    return env, params, states


def main():
    import jax
    import jax.numpy as jnp
    from craftax.craftax import constants as C

    env, params, states = rollout()
    rep = {"n_rollout_states": len(states), "seed": SEED, "n_steps": N_STEPS}

    def g(s, path):
        o = s
        for p in path.split("."):
            o = getattr(o, p)
        return np.asarray(o)

    # --- what a random policy actually visits
    levels = np.array([int(g(s, "player_level")) for s in states])
    light = np.array([float(g(s, "light_level")) for s in states])
    sleeping = np.array([bool(g(s, "is_sleeping")) for s in states])
    rep["rollout_levels_visited"] = sorted(set(int(x) for x in levels))
    rep["rollout_light_bands"] = {
        "day_1.0": int((light == 1.0).sum()),
        "dusk_0.5_1.0": int(((light >= 0.5) & (light < 1.0)).sum()),
        "night_lt_0.5": int((light < 0.5).sum()),
    }
    rep["rollout_sleeping"] = int(sleeping.sum())

    # --- inventory / attribute coverage: does a random policy move the strip at all?
    fields = ["inventory.wood", "inventory.stone", "inventory.coal", "inventory.iron",
              "inventory.diamond", "inventory.sapling", "inventory.pickaxe", "inventory.sword",
              "inventory.bow", "inventory.arrows", "inventory.torches", "inventory.ruby",
              "inventory.sapphire", "inventory.books",
              "player_health", "player_food", "player_drink", "player_energy", "player_mana",
              "player_xp", "player_dexterity", "player_strength", "player_intelligence",
              "sword_enchantment", "bow_enchantment"]
    seen = {}
    for f in fields:
        v = np.array([float(g(s, f)) for s in states])
        seen[f] = [float(v.min()), float(v.max())]
    arm = np.stack([g(s, "inventory.armour") for s in states])
    seen["inventory.armour"] = [float(arm.min()), float(arm.max())]
    pot = np.stack([g(s, "inventory.potions") for s in states])
    seen["inventory.potions"] = [float(pot.min()), float(pot.max())]
    ae = np.stack([g(s, "armour_enchantments") for s in states])
    seen["armour_enchantments"] = [float(ae.min()), float(ae.max())]
    ls = np.stack([g(s, "learned_spells") for s in states])
    seen["learned_spells"] = [float(ls.min()), float(ls.max())]
    rep["rollout_field_ranges"] = seen

    # --- how many entities land on screen, and how often two share a tile
    obs_h, obs_w = C.OBS_DIM
    counts = {"melee": [], "passive": [], "ranged": [], "mob_proj": [], "player_proj": []}
    overlaps = {"mob_on_mob": 0, "proj_on_proj": 0, "proj_on_mob": 0, "any_tile_ge3": 0}
    max_per_tile = 0
    for s in states[::5]:
        lvl = int(g(s, "player_level"))
        pp = g(s, "player_position")
        occ_mob = {}
        occ_proj = {}
        for name, attr in (("melee", "melee_mobs"), ("passive", "passive_mobs"),
                           ("ranged", "ranged_mobs")):
            m = getattr(s, attr)
            pos = np.asarray(m.position)[lvl]
            msk = np.asarray(m.mask)[lvl]
            k = 0
            for i in range(pos.shape[0]):
                lp = pos[i] - pp + np.array([obs_h, obs_w]) // 2
                if msk[i] and 0 <= lp[0] < obs_h and 0 <= lp[1] < obs_w:
                    k += 1
                    occ_mob[tuple(lp)] = occ_mob.get(tuple(lp), 0) + 1
            counts[name].append(k)
        for name, attr in (("mob_proj", "mob_projectiles"),
                           ("player_proj", "player_projectiles")):
            m = getattr(s, attr)
            pos = np.asarray(m.position)[lvl]
            msk = np.asarray(m.mask)[lvl]
            k = 0
            for i in range(pos.shape[0]):
                lp = pos[i] - pp + np.array([obs_h, obs_w]) // 2
                if msk[i] and 0 <= lp[0] < obs_h and 0 <= lp[1] < obs_w:
                    k += 1
                    occ_proj[tuple(lp)] = occ_proj.get(tuple(lp), 0) + 1
            counts[name].append(k)
        overlaps["mob_on_mob"] += sum(1 for v in occ_mob.values() if v > 1)
        overlaps["proj_on_proj"] += sum(1 for v in occ_proj.values() if v > 1)
        overlaps["proj_on_mob"] += sum(1 for t in occ_proj if t in occ_mob)
        for t in set(occ_mob) | set(occ_proj):
            n = occ_mob.get(t, 0) + occ_proj.get(t, 0)
            max_per_tile = max(max_per_tile, n)
            if n >= 3:
                overlaps["any_tile_ge3"] += 1
    rep["onscreen_entity_counts"] = {k: {"max": int(max(v)), "mean": float(np.mean(v))}
                                     for k, v in counts.items()}
    rep["tile_overlaps_over_800_frames"] = overlaps
    rep["max_entities_on_one_tile"] = int(max_per_tile)

    # --- the level-shifted pool: nine backgrounds from one reset
    base = states[0]
    up = np.asarray(base.up_ladders)
    shifted = []
    for lvl in range(int(np.asarray(base.map).shape[0])):
        pos = up[lvl]
        shifted.append(base.replace(player_level=jnp.int32(lvl),
                                    player_position=jnp.asarray(pos, jnp.int32)))
    rep["level_shifted"] = {"n": len(shifted), "up_ladders": up.tolist()}

    pool = {"rollout": states[::40], "level_shifted": shifted, "all_rollout_len": len(states)}
    with open(POOL, "wb") as f:
        pickle.dump(pool, f)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "state_census.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
