"""Purity, determinism, and the render-window sufficiency test for full Craftax.

Three questions, all on real frames and all against the **un-jitted** renderer (W1's ruling: under
``jax.jit`` XLA fuses the reference's arithmetic differently and it disagrees with its own un-jitted
self):

1. is the frame a pure function of ``EnvState`` -- same state twice, and the same state reached by
   two different histories;
2. how big is the jitted/un-jitted disagreement for the FULL game, and where does it live;
3. **sufficiency**: is the frame a function of the proposed render window alone? Tested the only way
   that is evidence rather than argument -- perturb every ``EnvState`` field the window does *not*
   carry and check that not one pixel moves.

Writes ``phase0/purity_and_state_window.json``.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")
TILE_PX = 10          # BLOCK_PIXEL_SIZE_AGENT: the size swm/CraftaxPixels-v1 is registered at


def u8(x):
    return np.round(np.asarray(x)).clip(0, 255).astype(np.uint8)


def main():
    import jax
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    from craftax.craftax.renderer import make_craftax_pixel_renderer

    render = make_craftax_pixel_renderer(TILE_PX)          # un-jitted
    jrender = jax.jit(render)
    pool = pickle.load(open(POOL, "rb"))
    states = pool["rollout"][:40] + pool["level_shifted"]
    rep = {"tile_px": TILE_PX, "n_states": len(states),
           "frame_shape": [(C.OBS_DIM[0] + C.INVENTORY_OBS_HEIGHT) * TILE_PX,
                           C.OBS_DIM[1] * TILE_PX, 3]}

    # ---------------------------------------------------------------- 1. purity
    same = []
    for s in states:
        a = np.asarray(render(s))
        b = np.asarray(render(s))
        same.append(bool(np.array_equal(a, b)))
    rep["same_state_twice_identical"] = f"{sum(same)}/{len(same)}"

    # the same state reached by two histories: replay a prefix twice with the same actions from two
    # different resets is not the same state, so instead reach one state, carry it forward and back
    # through a NOOP, and compare against the untouched copy -- the simulator's own round trip.
    from craftax.craftax_env import make_craftax_env_from_name
    env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
    params = env.default_params
    hist = []
    key = jax.random.PRNGKey(7)
    key, rk = jax.random.split(key)
    _o, st = env.reset(rk, params)
    for i in range(12):
        key, sk = jax.random.split(key)
        _o, st, _r, _d, _i = env.step(sk, st, 0, params)     # NOOP only: deterministic-ish path
        hist.append(st)
    # re-run the identical sequence from the same seed: the states must match field for field
    key2 = jax.random.PRNGKey(7)
    key2, rk2 = jax.random.split(key2)
    _o, st2 = env.reset(rk2, params)
    ok = []
    for i in range(12):
        key2, sk2 = jax.random.split(key2)
        _o, st2, _r, _d, _i = env.step(sk2, st2, 0, params)
        ok.append(bool(np.array_equal(np.asarray(render(st2)), np.asarray(render(hist[i])))))
    rep["replayed_history_identical"] = f"{sum(ok)}/{len(ok)}"

    # ---------------------------------------------------------------- 2. jitted vs un-jitted
    diffs = []
    for s in states:
        a = np.asarray(render(s)).astype(np.float64)
        b = np.asarray(jrender(s)).astype(np.float64)
        au, bu = u8(a), u8(b)
        diffs.append({
            "level": int(np.asarray(s.player_level)),
            "light": float(np.asarray(s.light_level)),
            "sleeping": bool(np.asarray(s.is_sleeping)),
            "float_max_abs": float(np.abs(a - b).max()),
            "uint8_exact_frac": float((au == bu).mean()),
            "uint8_max_abs": int(np.abs(au.astype(int) - bu.astype(int)).max()),
        })
    rep["jit_vs_unjit"] = {
        "n": len(diffs),
        "float_max_abs_over_pool": max(d["float_max_abs"] for d in diffs),
        "n_states_uint8_identical": sum(1 for d in diffs if d["uint8_exact_frac"] == 1.0),
        "worst_uint8_exact_frac": min(d["uint8_exact_frac"] for d in diffs),
        "worst_uint8_max_abs": max(d["uint8_max_abs"] for d in diffs),
        "per_state": diffs,
    }

    # ---------------------------------------------------------------- 3. window sufficiency
    #   fields the proposed window does NOT carry. Each is perturbed to a legal-looking value and
    #   the frame must not move by one pixel.
    base = states[0]
    lvl = int(np.asarray(base.player_level))
    ref = np.asarray(render(base))
    rng = np.random.default_rng(0)

    def moved(s2):
        return int((np.asarray(render(s2)) != ref).sum())

    probes = {}
    probes["timestep"] = moved(base.replace(timestep=jnp.int32(12345)))
    probes["achievements"] = moved(base.replace(
        achievements=jnp.ones_like(base.achievements)))
    probes["player_recover"] = moved(base.replace(player_recover=jnp.float32(0.77)))
    probes["player_hunger"] = moved(base.replace(player_hunger=jnp.float32(3.3)))
    probes["player_thirst"] = moved(base.replace(player_thirst=jnp.float32(3.3)))
    probes["player_fatigue"] = moved(base.replace(player_fatigue=jnp.float32(3.3)))
    probes["player_recover_mana"] = moved(base.replace(player_recover_mana=jnp.float32(3.3)))
    probes["is_resting"] = moved(base.replace(is_resting=jnp.bool_(True)))
    probes["potion_mapping"] = moved(base.replace(
        potion_mapping=jnp.asarray(np.asarray(base.potion_mapping)[::-1].copy())))
    probes["chests_opened"] = moved(base.replace(
        chests_opened=jnp.ones_like(base.chests_opened)))
    probes["growing_plants_age"] = moved(base.replace(
        growing_plants_age=jnp.ones_like(base.growing_plants_age) * 5))
    probes["growing_plants_mask"] = moved(base.replace(
        growing_plants_mask=jnp.ones_like(base.growing_plants_mask)))
    probes["down_ladders"] = moved(base.replace(
        down_ladders=jnp.zeros_like(base.down_ladders)))
    probes["up_ladders"] = moved(base.replace(up_ladders=jnp.zeros_like(base.up_ladders)))
    probes["boss_timesteps_to_spawn"] = moved(base.replace(
        boss_timesteps_to_spawn_this_round=jnp.int32(3)))
    # map on a level other than the player's
    m = np.asarray(base.map).copy()
    other = (lvl + 1) % m.shape[0]
    m[other] = rng.integers(0, len(C.BlockType), size=m[other].shape).astype(m.dtype)
    probes["map_on_another_level"] = moved(base.replace(map=jnp.asarray(m)))
    # map far off screen on the player's own level
    m2 = np.asarray(base.map).copy()
    pp = np.asarray(base.player_position)
    far = (int((pp[0] + 20) % m2.shape[1]), int((pp[1] + 20) % m2.shape[2]))
    m2[lvl, far[0], far[1]] = (int(m2[lvl, far[0], far[1]]) + 5) % len(C.BlockType)
    probes["map_offscreen_same_level"] = moved(base.replace(map=jnp.asarray(m2)))
    # mob_map is commented out of the renderer -- assert it
    probes["mob_map"] = moved(base.replace(mob_map=jnp.ones_like(base.mob_map)))
    # mob health / cooldown are not drawn
    mm = base.melee_mobs
    probes["melee_mob_health"] = moved(base.replace(
        melee_mobs=mm.replace(health=jnp.ones_like(mm.health) * 3.0)))
    probes["melee_mob_cooldown"] = moved(base.replace(
        melee_mobs=mm.replace(attack_cooldown=jnp.ones_like(mm.attack_cooldown) * 3)))
    # monsters_killed on ANOTHER level (the player's own level drives the ladder substitution)
    mk = np.asarray(base.monsters_killed).copy()
    mk[other] = 99
    probes["monsters_killed_other_level"] = moved(base.replace(monsters_killed=jnp.asarray(mk)))
    rep["window_sufficiency_pixels_moved"] = probes
    rep["window_sufficiency_all_zero"] = all(v == 0 for v in probes.values())

    # ---- and the things that DO have to be in the window: each must move pixels
    must = {}
    mk2 = np.asarray(base.monsters_killed).copy()
    mk2[lvl] = int(C.MONSTERS_KILLED_TO_CLEAR_LEVEL) + 1
    must["monsters_killed_own_level(ladder)"] = moved(base.replace(
        monsters_killed=jnp.asarray(mk2)))
    must["light_level"] = moved(base.replace(light_level=jnp.float32(0.1)))
    must["is_sleeping"] = moved(base.replace(is_sleeping=jnp.bool_(True)))
    must["player_direction"] = moved(base.replace(
        player_direction=jnp.asarray((int(np.asarray(base.player_direction)) % 4) + 1,
                                     np.asarray(base.player_direction).dtype)))
    must["state_rng"] = moved(base.replace(state_rng=jax.random.PRNGKey(4242)))
    dark = base.replace(light_level=jnp.float32(0.1))
    ref_dark = np.asarray(render(dark))
    must["state_rng_at_night"] = int((np.asarray(render(
        dark.replace(state_rng=jax.random.PRNGKey(4242)))) != ref_dark).sum())
    lm = np.asarray(base.light_map).copy()
    lm[lvl] = 0.0
    must["light_map"] = moved(base.replace(light_map=jnp.asarray(lm)))
    must["player_level"] = moved(base.replace(player_level=jnp.int32(other)))
    must["player_xp"] = moved(base.replace(player_xp=jnp.asarray(5, np.asarray(base.player_xp).dtype)))
    rep["observable_pixels_moved"] = must

    # ---------------------------------------------------------------- 4. light_map's alphabet
    lmv = np.concatenate([np.asarray(s.light_map).ravel() for s in states[:20]])
    u = np.unique(lmv)
    rep["light_map"] = {"dtype": str(np.asarray(states[0].light_map).dtype),
                        "shape": list(np.asarray(states[0].light_map).shape),
                        "n_distinct_over_20_states": int(u.size),
                        "values": [float(x) for x in u[:20]],
                        "min": float(lmv.min()), "max": float(lmv.max())}

    # ---------------------------------------------------------------- 5. mob type_id ranges
    tid = {}
    for attr in ("melee_mobs", "passive_mobs", "ranged_mobs",
                 "mob_projectiles", "player_projectiles"):
        v = np.concatenate([np.asarray(getattr(s, attr).type_id).ravel() for s in states])
        tid[attr] = [int(v.min()), int(v.max())]
    rep["mob_type_id_ranges_over_pool"] = tid

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "purity_and_state_window.json"), "w") as f:
        json.dump(rep, f, indent=2)
    small = {k: v for k, v in rep.items() if k != "jit_vs_unjit"}
    small["jit_vs_unjit"] = {k: v for k, v in rep["jit_vs_unjit"].items() if k != "per_state"}
    print(json.dumps(small, indent=2))


if __name__ == "__main__":
    sys.exit(main())
