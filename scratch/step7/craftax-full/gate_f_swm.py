"""Gate F for ``swm/CraftaxPixels-v1``: the frames a live stable_worldmodel workflow consumes.

``examples/swm_compat/run_swm_workflow.py`` drives ``swm.World`` over TwoRoom and PushT and is CUDA
shaped; it has no Craftax arm, and this cell has no batched simulator, so its Gate F path is
``swap_renderer`` — the official env's dynamics, our pixels. The statement worth making is therefore
the end-to-end one rather than a method-signature one:

    step the OFFICIAL swm env, read the render window out of the EnvState it produces, render that
    window with CraftaxFull, and compare against the observation the wrapper itself hands the
    workflow.

**What the wrapper hands the workflow is the JITTED frame.** ``CraftaxWrapper`` wraps ``env.step`` in
``jax.jit``, and under jit XLA fuses craftax's arithmetic differently from the un-jitted renderer this
cell is certified against — measured at up to 1.53e-05 in float32, which straddles a uint8 rounding
boundary wherever the picture is compressed towards one value. Any disagreement here is therefore
expected to be that, not a renderer defect, and is reported per-state with its band so the two can be
told apart.

Runs with or without ``stable_worldmodel``: without it the probe drives craftax's own env directly,
which is what the wrapper wraps, so the window reader can be validated before the Oscar job runs.

    PYTHONPATH=. python scratch/step7/craftax-full/gate_f_swm.py --out ... [--steps 60]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
sys.path.insert(0, ROOT)

from stable_envs.envs import craftax_full as CF                       # noqa: E402
from stable_envs.envs.craftax_full import CraftaxFull                 # noqa: E402

J = 5 * CF.N_TILE
S0 = J + 4 + CF.N_KEY_HALVES
#: ``EnvState`` field -> strip slot, for the slots that are a plain scalar read.
SCALARS = {
    "player_food": "food", "player_drink": "drink", "player_energy": "energy",
    "player_mana": "mana", "player_xp": "xp", "player_dexterity": "dexterity",
    "player_strength": "strength", "player_intelligence": "intelligence",
    "sword_enchantment": "ench_sword", "bow_enchantment": "ench_bow",
}
INVENTORY = ("wood", "stone", "coal", "iron", "diamond", "sapphire", "ruby", "sapling",
             "torches", "books", "arrows", "pickaxe", "sword", "bow")


def window_from_envstate(s) -> tuple[np.ndarray, dict]:
    """``EnvState`` -> the 544-slot render window, plus what had to be dropped to fit it."""
    from craftax.craftax import constants as C
    from craftax.craftax.util.game_logic_utils import is_boss_vulnerable

    k = CF.N_TILE
    lvl = int(np.asarray(s.player_level))
    pp = np.asarray(s.player_position)
    pad = C.MAX_OBS_DIM + 2
    tl = pp - np.array(C.OBS_DIM) // 2 + pad

    def view(arr, fill):
        a = np.pad(np.asarray(arr)[lvl], pad, constant_values=fill)
        return a[tl[0]:tl[0] + CF.OBS_H, tl[1]:tl[1] + CF.OBS_W]

    blk = view(s.map, C.BlockType.OUT_OF_BOUNDS.value).astype(np.int64).copy()
    itm = view(s.item_map, C.ItemType.NONE.value).astype(np.int64).copy()
    lit = view(s.light_map, 0.0).astype(np.float32)
    # the two global substitutions, resolved into the stored id -- which is what makes them cost no
    # state dimension (findings.md section 2.1)
    blk[blk == C.BlockType.NECROMANCER.value] = (
        C.BlockType.NECROMANCER_VULNERABLE.value if bool(is_boss_vulnerable(s))
        else C.BlockType.NECROMANCER.value)
    itm[itm == C.ItemType.LADDER_DOWN.value] = (
        C.ItemType.LADDER_DOWN.value
        if int(np.asarray(s.monsters_killed)[lvl]) >= C.MONSTERS_KILLED_TO_CLEAR_LEVEL
        else C.ItemType.LADDER_DOWN_BLOCKED.value)

    mob = np.zeros((CF.OBS_H, CF.OBS_W), np.int64)
    prj = np.zeros((CF.OBS_H, CF.OBS_W), np.int64)
    dropped = {"shared_mob_tiles": 0, "shared_projectile_tiles": 0}
    for name, attr, n_type in (("melee", "melee_mobs", CF.N_MELEE),
                               ("passive", "passive_mobs", CF.N_PASSIVE),
                               ("ranged", "ranged_mobs", CF.N_RANGED)):
        mo = getattr(s, attr)
        pos, msk, tid = (np.asarray(mo.position)[lvl], np.asarray(mo.mask)[lvl],
                         np.asarray(mo.type_id)[lvl])
        for i in range(pos.shape[0]):
            lp = pos[i] - pp + np.array([CF.OBS_H, CF.OBS_W]) // 2
            if msk[i] and 0 <= lp[0] < CF.OBS_H and 0 <= lp[1] < CF.OBS_W:
                if mob[lp[0], lp[1]]:
                    dropped["shared_mob_tiles"] += 1
                mob[lp[0], lp[1]] = CF.MOB_BASE[name] + int(tid[i]) % n_type
    for attr, dattr in (("mob_projectiles", "mob_projectile_directions"),
                        ("player_projectiles", "player_projectile_directions")):
        mo = getattr(s, attr)
        dr = np.asarray(getattr(s, dattr))[lvl]
        pos, msk, tid = (np.asarray(mo.position)[lvl], np.asarray(mo.mask)[lvl],
                         np.asarray(mo.type_id)[lvl])
        for i in range(pos.shape[0]):
            lp = pos[i] - pp + np.array([CF.OBS_H, CF.OBS_W]) // 2
            if msk[i] and 0 <= lp[0] < CF.OBS_H and 0 <= lp[1] < CF.OBS_W:
                o = (2 if (dr[i][0] > 0 or dr[i][1] > 0) else 0) + (1 if dr[i][1] != 0 else 0)
                if prj[lp[0], lp[1]]:
                    dropped["shared_projectile_tiles"] += 1
                prj[lp[0], lp[1]] = 1 + int(tid[i]) * 4 + o

    st = np.zeros(CraftaxFull.state_dim, np.float32)
    st[0:k], st[k:2 * k] = blk.reshape(-1), itm.reshape(-1)
    st[2 * k:3 * k], st[3 * k:4 * k] = mob.reshape(-1), prj.reshape(-1)
    st[4 * k:5 * k] = lit.reshape(-1)
    st[J] = int(np.asarray(s.player_direction)) - 1
    st[J + 1] = float(bool(np.asarray(s.is_sleeping)))
    st[J + 2] = lvl
    st[J + 3] = float(np.asarray(s.light_level))
    key = np.asarray(s.state_rng).astype(np.uint32).astype(np.int64)
    st[J + 4:J + 8] = [(key[0] >> 16) & 0xFFFF, key[0] & 0xFFFF,
                       (key[1] >> 16) & 0xFFFF, key[1] & 0xFFFF]

    def put(slot, v):
        st[S0 + CF._SLOT[slot]] = v

    put("health", max(int(np.floor(float(np.asarray(s.player_health)))), 1))
    for field, slot in SCALARS.items():
        put(slot, min(int(np.asarray(getattr(s, field))), CF.STRIP_MOD[CF._SLOT[slot]] - 1))
    inv = s.inventory
    for n in INVENTORY:
        put(n, min(int(np.asarray(getattr(inv, n))), CF.STRIP_MOD[CF._SLOT[n]] - 1))
    for i, v in enumerate(np.asarray(inv.armour)):
        put(f"armour{i}", int(v))
    for i, v in enumerate(np.asarray(inv.potions)):
        put(f"potion{i}", min(int(v), CF.N_COUNT - 1))
    for i, v in enumerate(np.asarray(s.armour_enchantments)):
        put(f"ench_armour{i}", int(v))
    ls = np.asarray(s.learned_spells)
    put("spell_fireball", int(bool(ls[0])))
    put("spell_iceball", int(bool(ls[1])))
    return st, dropped


def band_of(st):
    lvl = int(round(float(st[J + 2])))
    light = float(st[J + 3])
    day = light if lvl == 0 else 1.0
    lit = bool(np.all(st[4 * CF.N_TILE:5 * CF.N_TILE] == 1.0))
    if day == 1.0 and lit and st[J + 1] <= 0.5:
        return "fast_uint8"
    return "lit_no_prng" if day >= CF.NIGHT_THRESHOLD else "night_prng"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(
        ROOT, "examples", "results", "step7", "craftax-full", "gate_f"))
    a = ap.parse_args()
    import jax

    rec = {"steps": a.steps, "seed": a.seed, "jax": jax.__version__,
           "jax_threefry_partitionable": bool(jax.config.jax_threefry_partitionable)}
    try:
        from importlib import metadata
        rec["craftax"] = metadata.version("craftax")
    except Exception:                                                  # noqa: BLE001
        pass

    swm_ok = True
    try:
        import gymnasium
        import stable_worldmodel                                        # noqa: F401
        from importlib import metadata
        rec["stable_worldmodel"] = {
            "version": metadata.version("stable_worldmodel"),
            "file": stable_worldmodel.__file__,
        }
        try:
            env = gymnasium.make("swm/CraftaxPixels-v1")
            rec["source"] = "gymnasium.make('swm/CraftaxPixels-v1')"
        except Exception as e:                                         # noqa: BLE001
            # the id needs stable_worldmodel.envs to have registered, which drags in the whole
            # package. The class itself is what Gate F is about, so fall back to constructing it.
            from stable_worldmodel.envs.craftax.craftax import CraftaxPixelsWrapper
            env = CraftaxPixelsWrapper()
            rec["source"] = (f"CraftaxPixelsWrapper() constructed directly -- gymnasium.make on the "
                             f"registered id raised {type(e).__name__}: {str(e)[:120]}")
        u = env.unwrapped
        rec["observation_space"] = str(env.observation_space)
        rec["action_space"] = str(env.action_space)
        rec["num_actions"] = int(u.craftax_env.num_actions)
    except Exception as e:                                             # noqa: BLE001
        swm_ok = False
        rec["stable_worldmodel"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        rec["source"] = ("craftax.craftax_env.make_craftax_env_from_name('Craftax-Pixels-v1') -- "
                         "what the swm wrapper wraps; the wrapper layer itself is NOT exercised")
        from craftax.craftax_env import make_craftax_env_from_name
        raw = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)

    obs_list, states, infos_keys = [], [], None
    if swm_ok:
        obs, info = env.reset(seed=a.seed)
        rec["reset_info_keys"] = sorted(str(x) for x in info)
        obs_list.append(np.asarray(obs, np.float32))
        states.append(u._state)
        rng = np.random.default_rng(a.seed)
        for _ in range(a.steps):
            obs, _r, term, trunc, info = env.step(int(rng.integers(0, u.craftax_env.num_actions)))
            infos_keys = sorted(str(x) for x in info)
            obs_list.append(np.asarray(obs, np.float32))
            states.append(u._state)
            if term or trunc:
                obs, info = env.reset()
                obs_list.append(np.asarray(obs, np.float32))
                states.append(u._state)
        rec["step_info_keys"] = infos_keys
        env.close()
    else:
        params = raw.default_params
        key = jax.random.PRNGKey(a.seed)
        key, rk = jax.random.split(key)
        obs, st = raw.reset(rk, params)
        step = jax.jit(raw.step)                     # the wrapper jits step; match it
        obs_list.append(np.asarray(obs, np.float32))
        states.append(st)
        for _ in range(a.steps):
            key, ak, sk = jax.random.split(key, 3)
            act = jax.random.randint(ak, (), 0, raw.num_actions)
            obs, st, _r, _d, _i = step(sk, st, act, params)
            obs_list.append(np.asarray(obs, np.float32))
            states.append(st)

    official = np.stack(obs_list)
    rec["obs_shape"] = list(official.shape[1:])
    rec["obs_dtype"] = str(official.dtype)
    rec["obs_min"], rec["obs_max"] = float(official.min()), float(official.max())
    rec["obs_shape_matches_registration"] = list(official.shape[1:]) == [130, 110, 3]

    win = []
    dropped = {"shared_mob_tiles": 0, "shared_projectile_tiles": 0}
    for s in states:
        w, d = window_from_envstate(s)
        win.append(w)
        for kk in dropped:
            dropped[kk] += d[kk]
    win = np.stack(win)
    rec["window"] = {"state_dim": int(win.shape[1]), "bytes": int(win.shape[1] * 4),
                     "frame_bytes": int(np.prod(official.shape[1:])),
                     "compression": float(np.prod(official.shape[1:]) / (win.shape[1] * 4)),
                     "tiles_not_representable": dropped}

    r = CraftaxFull(batch=len(win), device="cpu", size=10)
    ours = r.render(win).cpu().numpy()
    # the wrapper's observation IS craftax's float frame divided by 255; multiply back and round the
    # way this repo rounds, which is the comparison the parity claim is stated in
    off_u8 = np.round(official.astype(np.float64) * 255.0).clip(0, 255).astype(np.uint8)
    d = np.abs(ours.astype(np.int16) - off_u8.astype(np.int16))
    bands = {}
    for i in range(len(win)):
        bands.setdefault(band_of(win[i]), []).append(i)
    rec["pixel_equality"] = {
        "protocol": "round(swm_obs * 255) vs CraftaxFull.render(window_from_envstate(state))",
        "exact_frac": float((d.max(-1) == 0).mean()), "max_abs": int(d.max()),
        "mean_abs": float(d.mean()),
        "frames": int(len(win)),
        "frames_identical": int(sum(1 for i in range(len(win)) if d[i].max() == 0)),
        "per_band": {b: {"n": len(ix), "exact_frac": float((d[ix].max(-1) == 0).mean()),
                         "max_abs": int(d[ix].max()),
                         "frames_identical": int(sum(1 for i in ix if d[i].max() == 0))}
                     for b, ix in sorted(bands.items())},
        "note": ("the wrapper jits env.step, so what a live workflow sees is the JITTED reference. "
                 "This cell is certified against the UN-JITTED one, which disagrees with its own "
                 "jitted self by up to 1.53e-05 in float32 -- enough to move a uint8 pixel wherever "
                 "the picture is compressed towards one value. Any residual here is that."),
    }
    # the same states through the un-jitted reference, which is what parity is claimed against
    ref = CraftaxFull.reference_env(size=10)
    unj = CraftaxFull.reference_render(ref, win[:min(8, len(win))])
    du = np.abs(ours[:len(unj)].astype(np.int16) - unj.astype(np.int16))
    rec["against_the_unjitted_reference"] = {
        "n": int(len(unj)), "exact_frac": float((du.max(-1) == 0).mean()),
        "max_abs": int(du.max())}

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "gate_f_swm.json" if swm_ok else "gate_f_swm_nowrapper.json")
    with open(p, "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2)[:4500])
    print("wrote", p)


if __name__ == "__main__":
    sys.exit(main())
