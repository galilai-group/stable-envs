"""The layer census: what a baked table would have to hold, and what the two global substitutions cost.

Three measurements that decide the shape of the renderer:

1. **are the overlay alphas binary at every reference tile size?** Classic's are ``{0, 128/255, 1}``
   and its composite is a real blend. If full Craftax's are ``{0, 1}`` the composite is a *select*
   and a layer is a stencil, which changes what can be baked and what must be blended.
2. **the two global substitutions** -- ``NECROMANCER -> NECROMANCER_VULNERABLE`` (a function of
   every mob on the level, on-screen or not) and ``LADDER_DOWN -> LADDER_DOWN_BLOCKED`` (a function
   of ``monsters_killed[level]``). Both are resolved *before* the gather, so a window that stores
   the **resolved** id absorbs them; this measures that they move pixels only on the tiles that hold
   those ids.
3. **table sizes** for each candidate bake, in bytes, at both reference tile sizes.

Writes ``phase0/layers.json``.
"""
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")


def main():
    import jax
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    from craftax.craftax.renderer import make_craftax_pixel_renderer

    rep = {}

    # ------------------------------------------------ 1. alpha alphabets at every reference size
    alpha = {}
    for t in (C.BLOCK_PIXEL_SIZE_AGENT, C.BLOCK_PIXEL_SIZE_IMG, C.BLOCK_PIXEL_SIZE_HUMAN):
        tex = C.load_all_textures_given_size(int(t))
        d = {}
        for k in ("full_map_player_textures_alpha", "melee_mob_texture_alphas",
                  "passive_mob_texture_alphas", "ranged_mob_texture_alphas",
                  "projectile_texture_alphas"):
            u = np.unique(np.asarray(tex[k]))
            d[k] = {"n": int(u.size), "values": [float(x) for x in u[:8]]}
        u = np.unique(np.asarray(tex["full_map_item_textures"])[..., 3])
        d["item_alpha"] = {"n": int(u.size), "values": [float(x) for x in u[:8]]}
        for k in ("sword_enchantment_textures", "arrow_enchantment_textures",
                  "armour_enchantment_textures"):
            u = np.unique(np.asarray(tex[k])[..., 3])
            d[k + "[a]"] = {"n": int(u.size), "values": [float(x) for x in u[:8]]}
        u = np.unique(np.asarray(tex["number_textures_alpha"]))
        d["number_textures_alpha"] = {"n": int(u.size), "values": [float(x) for x in u[:8]]}
        alpha[int(t)] = d
    rep["alpha_alphabets_by_tile_px"] = alpha
    rep["all_world_overlay_alphas_binary"] = {
        str(t): all(v["n"] <= 2 and set(v["values"]) <= {0.0, 1.0}
                    for k, v in d.items()
                    if k in ("full_map_player_textures_alpha", "melee_mob_texture_alphas",
                             "passive_mob_texture_alphas", "ranged_mob_texture_alphas",
                             "projectile_texture_alphas", "item_alpha"))
        for t, d in alpha.items()}

    # ------------------------------------------------ 2. the two global substitutions
    t = int(C.BLOCK_PIXEL_SIZE_AGENT)
    render = make_craftax_pixel_renderer(t)
    pool = pickle.load(open(POOL, "rb"))
    base = pool["rollout"][0]
    obs_h, obs_w = C.OBS_DIM

    def put_window(state, level, block_id=None, item_id=None, cells=((0, 0),)):
        """Write ``block_id``/``item_id`` into the given on-screen tiles of ``level``."""
        m = np.asarray(state.map).copy()
        it = np.asarray(state.item_map).copy()
        pp = np.asarray(state.player_position)
        r0, c0 = pp[0] - obs_h // 2, pp[1] - obs_w // 2
        for (r, c) in cells:
            if block_id is not None:
                m[level, r0 + r, c0 + c] = block_id
            if item_id is not None:
                it[level, r0 + r, c0 + c] = item_id
        return state.replace(map=jnp.asarray(m), item_map=jnp.asarray(it))

    lvl = int(np.asarray(base.player_level))
    # make the level fully lit so the substitution is not hidden by darkness
    lm = np.asarray(base.light_map).copy()
    lm[lvl] = 1.0
    lit = base.replace(light_map=jnp.asarray(lm), light_level=jnp.float32(1.0))

    necro = put_window(lit, lvl, block_id=C.BlockType.NECROMANCER.value, cells=((0, 0), (0, 1)))
    # boss vulnerable iff no melee and no ranged mobs on the level and the spawn timer has expired
    mm = necro.melee_mobs.replace(mask=jnp.zeros_like(necro.melee_mobs.mask))
    rm = necro.ranged_mobs.replace(mask=jnp.zeros_like(necro.ranged_mobs.mask))
    vuln = necro.replace(melee_mobs=mm, ranged_mobs=rm,
                         boss_timesteps_to_spawn_this_round=jnp.int32(0))
    notv = necro.replace(melee_mobs=mm, ranged_mobs=rm,
                         boss_timesteps_to_spawn_this_round=jnp.int32(5))
    a, b = np.asarray(render(vuln)), np.asarray(render(notv))
    diff = (a != b).any(-1)
    rep["boss_substitution"] = {
        "pixels_moved": int((a != b).sum()),
        "rows_moved": [int(diff.any(1).nonzero()[0].min()), int(diff.any(1).nonzero()[0].max())]
        if diff.any() else [],
        "cols_moved": [int(diff.any(0).nonzero()[0].min()), int(diff.any(0).nonzero()[0].max())]
        if diff.any() else [],
        "confined_to_the_two_necromancer_tiles":
            bool(diff.any()) and bool(diff[:, :].sum() > 0)
            and bool(not diff[2 * t:, :].any()) and bool(not diff[:, 2 * t:].any()),
        "reads": "melee_mobs.mask[level].sum(), ranged_mobs.mask[level].sum(), "
                 "boss_timesteps_to_spawn_this_round -- ALL of them off-screen-inclusive",
    }

    lad = put_window(lit, lvl, item_id=C.ItemType.LADDER_DOWN.value, cells=((0, 0),))
    mk = np.asarray(lad.monsters_killed).copy()
    mk[lvl] = int(C.MONSTERS_KILLED_TO_CLEAR_LEVEL)
    open_ = lad.replace(monsters_killed=jnp.asarray(mk))
    mk2 = mk.copy()
    mk2[lvl] = 0
    shut = lad.replace(monsters_killed=jnp.asarray(mk2))
    a, b = np.asarray(render(open_)), np.asarray(render(shut))
    d2 = (a != b).any(-1)
    rep["ladder_substitution"] = {
        "pixels_moved": int((a != b).sum()),
        "confined_to_tile_0_0": bool(d2.any()) and not bool(d2[t:, :].any())
        and not bool(d2[:, t:].any()),
        "reads": "monsters_killed[player_level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL",
    }

    # ------------------------------------------------ 3. candidate table sizes
    nb, ni = len(C.BlockType), len(C.ItemType)
    n_player = 5 + 1                       # 4 facings + sleep + "no player"
    n_mob = 1 + 8 + 3 + 8                  # none + melee + passive + ranged
    n_proj = 1 + 8 * 4                     # none + 8 projectile types x 4 orientations
    tabs = {
        "block only": nb,
        "block x item": nb * ni,
        "block x item x player": nb * ni * n_player,
        "block x item x mob": nb * ni * n_mob,
        "block x item x player x mob": nb * ni * n_player * n_mob,
        "block x item x player x mob x projectile": nb * ni * n_player * n_mob * n_proj,
        "Classic's own table (player x block x mob)": 6 * 17 * 8,
    }
    sizes = {}
    for name, n in tabs.items():
        sizes[name] = {"entries": int(n),
                       "uint8_bytes_at_10px": int(n * 10 * 10 * 3),
                       "uint8_bytes_at_16px": int(n * 16 * 16 * 3),
                       "float32_bytes_at_16px": int(n * 16 * 16 * 3 * 4)}
    rep["candidate_tables"] = sizes
    rep["overlay_alphabet"] = {"n_player_slots": n_player, "n_mob_slots": n_mob,
                               "n_projectile_slots": n_proj,
                               "n_block": nb, "n_item": ni}

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "layers.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
