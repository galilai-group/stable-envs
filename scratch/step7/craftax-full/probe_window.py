"""The render window for full Craftax: what it must carry, and what it costs against a frame.

Also measures one thing that is not this workstream's to fix and must not be discovered later: what
W1's (correct) decision to return an **un-jitted** ``reference_env`` does to the Step-3 performance
denominator, which goes through the same hook.

Writes ``phase0/window.json``.
"""
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")


def main():
    import jax
    from craftax.craftax import constants as C
    from craftax.craftax.renderer import make_craftax_pixel_renderer

    obs_h, obs_w = C.OBS_DIM
    n_tiles = obs_h * obs_w
    rep = {}

    # ---------------------------------------------------------------- the window, field by field
    fields = [
        ("block id (post NECROMANCER/NECROMANCER_VULNERABLE substitution)", n_tiles, "0..36"),
        ("item id (post LADDER_DOWN/LADDER_DOWN_BLOCKED substitution)", n_tiles, "0..4"),
        ("mob slot (none | 8 melee | 3 passive | 8 ranged)", n_tiles, "0..19"),
        ("projectile slot (none | 8 types x 4 orientations)", n_tiles, "0..32"),
        ("light_map (per tile, float in [0,1])", n_tiles, "float32"),
        ("player slot (4 facings | sleeping | none)", 1, "0..5"),
        ("player_level", 1, "0..8"),
        ("light_level", 1, "float32"),
        ("is_sleeping", 1, "0/1"),
        ("state_rng, as four 16-bit halves", 4, "0..65535 each"),
    ]
    strip_fields = [
        ("player_health/food/drink/energy/mana", 5),
        ("player_xp/dexterity/strength/intelligence", 4),
        ("inventory wood/stone/coal/iron/diamond/sapphire/ruby/sapling/torches/books/arrows", 11),
        ("inventory.potions[6]", 6),
        ("inventory pickaxe/sword/bow", 3),
        ("inventory.armour[4]", 4),
        ("sword_enchantment/bow_enchantment", 2),
        ("armour_enchantments[4]", 4),
        ("learned_spells[2]", 2),
    ]
    n_world = sum(n for _, n, _ in fields)
    n_strip = sum(n for _, n in strip_fields)
    # player_level is already counted in `fields`; the strip's level digit reads the same slot
    total = n_world + n_strip
    frame_10 = (obs_h + C.INVENTORY_OBS_HEIGHT) * 10 * obs_w * 10 * 3
    frame_16 = (obs_h + C.INVENTORY_OBS_HEIGHT) * 16 * obs_w * 16 * 3
    rep["render_window"] = {
        "per_tile_fields": [{"name": n, "slots": k, "range": r} for n, k, r in fields if k == n_tiles],
        "per_frame_world_fields": [{"name": n, "slots": k, "range": r}
                                   for n, k, r in fields if k != n_tiles],
        "strip_fields": [{"name": n, "slots": k} for n, k in strip_fields],
        "n_tiles": n_tiles,
        "slots_world": n_world,
        "slots_strip": n_strip,
        "slots_total": total,
        "bytes_float32": total * 4,
        "frame_bytes_at_10px": frame_10,
        "frame_bytes_at_16px": frame_16,
        "compression_at_10px": frame_10 / (total * 4),
        "compression_at_16px": frame_16 / (total * 4),
        "classic_for_comparison": {"slots": 149, "bytes": 596},
        "full_EnvState_bytes": None,
    }

    pool = pickle.load(open(POOL, "rb"))
    st = pool["rollout"][0]
    nb = sum(int(np.asarray(x).nbytes) for x in jax.tree_util.tree_leaves(st))
    rep["render_window"]["full_EnvState_bytes"] = int(nb)
    rep["render_window"]["EnvState_over_frame_at_10px"] = nb / frame_10

    # ---------------------------------------------------------------- the reference_env hook
    t = 10
    render_un = make_craftax_pixel_renderer(t)
    render_j = jax.jit(render_un)
    states = pool["rollout"][:8]
    render_un(states[0]).block_until_ready()
    t0 = time.perf_counter()
    for s in states[:4]:
        render_un(s).block_until_ready()
    dt_un = (time.perf_counter() - t0) / 4
    render_j(states[0]).block_until_ready()
    t0 = time.perf_counter()
    for i in range(200):
        render_j(states[i % len(states)]).block_until_ready()
    dt_j = (time.perf_counter() - t0) / 200

    # Classic, through the repo's own bridge, before and after W1 removed the jit
    from stable_envs.envs.craftax import CraftaxClassic
    cc = CraftaxClassic(batch=8, device="cpu")
    sts = cc.sample_states(8)
    env = CraftaxClassic.reference_env(size=16)
    CraftaxClassic.reference_render(env, sts[:2])
    t0 = time.perf_counter()
    CraftaxClassic.reference_render(env, sts)
    dt_classic_bridge = (time.perf_counter() - t0) / 8
    jr = jax.jit(env[0])
    jr(env[1]).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(100):
        jr(env[1]).block_until_ready()
    dt_classic_jit = (time.perf_counter() - t0) / 100

    rep["reference_hook_denominator"] = {
        "what": "examples/bench_gpu_vs_cpu.py:cpu_throughput forms the Step-3 'official f/s' by "
                "calling cls.reference_render, the SAME hook the parity claim uses.",
        "full_craftax_unjitted_fps_at_10px": 1.0 / dt_un,
        "full_craftax_jitted_fps_at_10px": 1.0 / dt_j,
        "ratio": dt_un / dt_j,
        "classic_reference_render_fps_at_16px_on_this_mac": 1.0 / dt_classic_bridge,
        "classic_jitted_renderer_fps_at_16px_on_this_mac": 1.0 / dt_classic_jit,
        "step3_recorded_classic_official_fps": 1046,
        "baseline_76dc511_reference_env": "returned jax.jit(make_craftax_pixel_renderer(size))",
        "branch_reference_env": "returns the un-jitted renderer (W1, correctly, for parity)",
        "consequence": "re-running Step 3 against the branch divides the CraftaxClassic official "
                       "denominator by the jit speedup and multiplies S_cpu by the same factor. "
                       "This is a certification-integrity question for the coordinator, not a "
                       "renderer defect, and it is inherited by this workstream.",
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "window.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
