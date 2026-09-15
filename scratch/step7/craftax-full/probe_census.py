"""Phase-0 census of craftax's FULL renderer: atlas, layers, and the strip's cell alphabet.

Everything here is read off the installed craftax 1.6.1 (``/Users/samib/se7-craftax-venv``) rather
than off the paper: the texture dict's own keys and shapes, the enum sizes, and the per-cell
alphabets that a baked ``(cell, value)`` table would have to hold.

Writes ``examples/results/step7/craftax-full/phase0/census.json``.
"""
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "examples", "results", "step7", "craftax-full", "phase0")


def main():
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    from craftax.craftax.craftax_state import StaticEnvParams

    rep = {"craftax_version": _ver(), "notes": []}

    sp = StaticEnvParams()
    rep["static_env_params"] = {
        "map_size": list(sp.map_size), "num_levels": int(sp.num_levels),
        "max_melee_mobs": int(sp.max_melee_mobs), "max_passive_mobs": int(sp.max_passive_mobs),
        "max_ranged_mobs": int(sp.max_ranged_mobs),
        "max_mob_projectiles": int(sp.max_mob_projectiles),
        "max_player_projectiles": int(sp.max_player_projectiles),
        "max_growing_plants": int(sp.max_growing_plants),
    }
    rep["geometry"] = {
        "OBS_DIM": list(C.OBS_DIM), "INVENTORY_OBS_HEIGHT": int(C.INVENTORY_OBS_HEIGHT),
        "BLOCK_PIXEL_SIZE_AGENT": int(C.BLOCK_PIXEL_SIZE_AGENT),
        "BLOCK_PIXEL_SIZE_IMG": int(C.BLOCK_PIXEL_SIZE_IMG),
        "BLOCK_PIXEL_SIZE_HUMAN": int(C.BLOCK_PIXEL_SIZE_HUMAN),
        "frame_at_agent_px": [(C.OBS_DIM[0] + C.INVENTORY_OBS_HEIGHT) * C.BLOCK_PIXEL_SIZE_AGENT,
                              C.OBS_DIM[1] * C.BLOCK_PIXEL_SIZE_AGENT, 3],
        "frame_at_img_px": [(C.OBS_DIM[0] + C.INVENTORY_OBS_HEIGHT) * C.BLOCK_PIXEL_SIZE_IMG,
                            C.OBS_DIM[1] * C.BLOCK_PIXEL_SIZE_IMG, 3],
    }
    rep["enums"] = {
        "BlockType": len(C.BlockType), "ItemType": len(C.ItemType),
        "MobType": len(C.MobType), "ProjectileType": len(C.ProjectileType),
        "Action": len(C.Action), "Achievement": len(C.Achievement),
    }
    rep["asset_png_count"] = len([f for f in os.listdir(os.path.join(
        os.path.dirname(C.__file__), "assets")) if f.endswith(".png")])

    # ---- the texture dict, at the registered default tile size
    t = int(C.BLOCK_PIXEL_SIZE_AGENT)
    tex = C.load_all_textures_given_size(t)
    entries = {}
    total = 0
    for k, v in sorted(tex.items()):
        a = np.asarray(v)
        entries[k] = {"shape": list(a.shape), "dtype": str(a.dtype),
                      "bytes_f32": int(a.size * 4)}
        total += int(a.size * 4)
    rep["textures_at_agent_px"] = {"tile_px": t, "n_keys": len(entries),
                                   "total_bytes_f32": total, "entries": entries}

    # which texture families are per-tile (t,t,*) vs full-frame (H,W,*)
    H = C.OBS_DIM[0] * t
    W = C.OBS_DIM[1] * t
    fam = {"per_tile": [], "full_frame": [], "other": []}
    for k, e in entries.items():
        s = e["shape"]
        if len(s) >= 3 and s[-3] == H and s[-2] == W:
            fam["full_frame"].append(k)
        elif len(s) >= 3 and s[-3] == t and s[-2] == t:
            fam["per_tile"].append(k)
        else:
            fam["other"].append(k)
    rep["texture_families"] = fam

    # ---- the mob/projectile tables: how many (class, type_id) sprites exist
    rep["mob_tables"] = {
        k: list(np.asarray(tex[k]).shape) for k in
        ("melee_mob_textures", "melee_mob_texture_alphas",
         "passive_mob_textures", "passive_mob_texture_alphas",
         "ranged_mob_textures", "ranged_mob_texture_alphas",
         "projectile_textures", "projectile_texture_alphas")
    }

    # ---- alpha alphabets: is the composite a real blend or a binary stencil?
    alphas = {}
    for k in ("full_map_player_textures_alpha", "melee_mob_texture_alphas",
              "passive_mob_texture_alphas", "ranged_mob_texture_alphas",
              "projectile_texture_alphas"):
        u = np.unique(np.asarray(tex[k]))
        alphas[k] = {"n_distinct": int(u.size),
                     "values": [float(x) for x in u[:12]],
                     "truncated": bool(u.size > 12)}
    # the item layer's alpha lives in channel 3 of an RGBA table
    it = np.asarray(tex["full_map_item_textures"])
    u = np.unique(it[..., 3])
    alphas["full_map_item_textures[...,3]"] = {
        "n_distinct": int(u.size), "values": [float(x) for x in u[:12]],
        "truncated": bool(u.size > 12)}
    rep["alpha_alphabets"] = alphas

    # ---- the player table: is it zero away from the centre tile, as in Classic?
    p = np.asarray(tex["full_map_player_textures"])
    pa = np.asarray(tex["full_map_player_textures_alpha"])
    cr, cc = C.OBS_DIM[0] // 2, C.OBS_DIM[1] // 2
    mask = np.ones((H, W), bool)
    mask[cr * t:(cr + 1) * t, cc * t:(cc + 1) * t] = False
    rep["player_table"] = {
        "n_slots": int(p.shape[0]),
        "shape": list(p.shape),
        "zero_away_from_centre_tex": bool(np.all(p[:, mask] == 0)),
        "zero_away_from_centre_alpha": bool(np.all(pa[:, mask] == 0)),
    }

    # ---- the item layer: is it per-tile after the full_map tiling?
    #   full_map_item_textures[i] is item_textures[i] tiled over the whole frame, so tile (r,c) of
    #   item i is the same t x t picture for every (r, c). Assert it.
    same = True
    for i in range(it.shape[0]):
        ref = it[i, 0:t, 0:t]
        for r in (0, cr, C.OBS_DIM[0] - 1):
            for c in (0, cc, C.OBS_DIM[1] - 1):
                if not np.array_equal(it[i, r * t:(r + 1) * t, c * t:(c + 1) * t], ref):
                    same = False
    rep["item_layer_is_tiled"] = bool(same)

    blk = np.asarray(tex["full_map_block_textures"])
    same_b = True
    for i in range(blk.shape[0]):
        ref = blk[i, 0:t, 0:t]
        for r in (0, cr, C.OBS_DIM[0] - 1):
            for c in (0, cc, C.OBS_DIM[1] - 1):
                if not np.array_equal(blk[i, r * t:(r + 1) * t, c * t:(c + 1) * t], ref):
                    same_b = False
    rep["block_layer_is_tiled"] = bool(same_b)

    # ---- night constants: they are NOT the Classic ones
    ntx = np.asarray(tex["night_noise_intensity_texture"])
    rep["night"] = {
        "night_texture_shape": list(np.asarray(tex["night_texture"]).shape),
        "night_texture_value": [float(x) for x in np.asarray(tex["night_texture"])[0, 0]],
        "vignette_shape": list(ntx.shape),
        "vignette_min": float(ntx.min()), "vignette_max": float(ntx.max()),
        "vignette_centre": float(ntx[H // 2, W // 2, 0]),
        "vignette_rule": "normalised radial distance sqrt((x-H//2)^2+(y-W//2)^2)/max, in jnp float32",
        "classic_vignette_rule": "1 - exp(-0.5 (x^2+y^2)/0.5^2) on linspace(-1,1), numpy float64",
        "vignette_is_classic_gaussian": _vignette_is_classic(ntx, H, W),
    }

    # ---- the strip's cell alphabet
    rep["strip"] = strip_census(C, t)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "census.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ("textures_at_agent_px",)}, indent=2)[:6000])
    print("wrote", os.path.join(OUT, "census.json"))


def _vignette_is_classic(ntx, H, W):
    xs, ys = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W))
    g = (1 - np.exp(-0.5 * (xs ** 2 + ys ** 2) / 0.25).T).astype(np.float32)
    return bool(np.allclose(ntx[..., 0], g, atol=1e-6))


def strip_census(C, t):
    """The inventory strip's geometry and per-cell alphabet, read off the renderer's own numbers."""
    left = (t - int(0.8 * t)) // 2
    right = t - int(0.8 * t) - left
    number_size = int(t * 0.4)
    number_offset = t - number_size
    number_double_offset = t - 2 * number_size
    grid_h, grid_w = int(C.INVENTORY_OBS_HEIGHT), int(C.OBS_DIM[1])

    # (x, y) -> the state fields the cell reads, and the size of its alphabet
    cells = {
        # row 0 -- intrinsics, the level digit and xp
        "(0,0) health": ("player_health 1..9 (max(floor(h),1))", 10, "icon+2digit"),
        "(1,0) food": ("player_food 0..9", 10, "icon+2digit"),
        "(2,0) drink": ("player_drink 0..9", 10, "icon+2digit"),
        "(3,0) energy": ("player_energy 0..9", 10, "icon+2digit"),
        "(4,0) mana": ("player_mana 0..9", 10, "icon+2digit"),
        "(6,0) level": ("player_level 0..8", 9, "1digit, no icon"),
        "(7,0) armour0": ("inventory.armour[0] x armour_enchantments[0]", None, "icon+alpha icon"),
        "(9,0) xp": ("player_xp 0..9", 10, "icon+1digit"),
        # row 1
        "(0,1) coal": ("inventory.coal 0..9", 10, "icon+2digit"),
        "(1,1) iron": ("inventory.iron 0..9", 10, "icon+2digit"),
        "(2,1) diamond": ("inventory.diamond 0..9", 10, "icon+2digit"),
        "(3,1) sapphire": ("inventory.sapphire 0..9", 10, "icon+2digit"),
        "(4,1) ruby": ("inventory.ruby 0..9", 10, "icon+2digit"),
        "(5,1) sapling": ("inventory.sapling 0..9", 10, "icon+2digit"),
        "(6,1) bow": ("inventory.bow 0..1", 2, "icon"),
        "(7,1) armour1": ("inventory.armour[1] x armour_enchantments[1]", None, "icon+alpha icon"),
        "(8,1) sword": ("inventory.sword 0..4 x sword_enchantment", None, "icon+alpha icon"),
        "(9,1) dexterity": ("player_dexterity 0..9", 10, "icon+1digit"),
        # row 2
        "(0,2) wood": ("inventory.wood 0..9", 10, "icon+2digit"),
        "(1,2) stone": ("inventory.stone 0..9", 10, "icon+2digit"),
        "(2,2) torches": ("inventory.torches 0..9", 10, "icon+2digit"),
        "(3,2) books": ("inventory.books 0..9", 10, "icon+2digit"),
        "(4,2) fireball": ("learned_spells[0]", 2, "icon"),
        "(5,2) iceball": ("learned_spells[1]", 2, "icon"),
        "(6,2) arrows": ("inventory.arrows x bow_enchantment", None, "icon+2digit+alpha icon"),
        "(7,2) armour2": ("inventory.armour[2] x armour_enchantments[2]", None, "icon+alpha icon"),
        "(8,2) pickaxe": ("inventory.pickaxe 0..4", 5, "icon"),
        "(9,2) strength": ("player_strength 0..9", 10, "icon+1digit"),
        # row 3
        "(0,3) potion_red": ("inventory.potions[0] 0..9", 10, "icon+2digit"),
        "(1,3) potion_green": ("inventory.potions[1] 0..9", 10, "icon+2digit"),
        "(2,3) potion_blue": ("inventory.potions[2] 0..9", 10, "icon+2digit"),
        "(3,3) potion_pink": ("inventory.potions[3] 0..9", 10, "icon+2digit"),
        "(4,3) potion_cyan": ("inventory.potions[4] 0..9", 10, "icon+2digit"),
        "(5,3) potion_yellow": ("inventory.potions[5] 0..9", 10, "icon+2digit"),
        "(7,3) armour3": ("inventory.armour[3] x armour_enchantments[3]", None, "icon+alpha icon"),
        "(9,3) intelligence": ("player_intelligence 0..9", 10, "icon+1digit"),
    }
    written = set()
    for k in cells:
        x, y = k.split(")")[0][1:].split(",")
        written.add((int(x), int(y)))
    blanks = [(x, y) for y in range(grid_h) for x in range(grid_w) if (x, y) not in written]
    return {
        "grid": [grid_h, grid_w],
        "n_positions": grid_h * grid_w,
        "n_written_cells": len(cells),
        "n_blank_cells": len(blanks),
        "blank_positions_xy": [list(b) for b in blanks],
        "icon_span": [left, t - right],
        "ones_digit_span": [number_offset, t],
        "tens_digit_span": [number_double_offset, number_offset],
        "number_size": number_size,
        "all_writes_inside_cell": bool(left >= 0 and t - right <= t
                                       and number_double_offset >= 0 and number_offset <= t),
        "cells": {k: {"reads": v[0], "alphabet": v[1], "draws": v[2]} for k, v in cells.items()},
    }


def _ver():
    try:
        from importlib import metadata
        return metadata.version("craftax")
    except Exception:  # noqa: BLE001
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
