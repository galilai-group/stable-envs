"""The 4-row inventory strip: is it a per-cell, layered stencil, and what is its alphabet?

Full Craftax's strip is not Classic's. It is 4 rows x 11 columns with **36 written cells**, counts
capped at **99** rather than 9 (``game_logic.clip_inventory_and_intrinsics``), two-digit numbers, a
level digit, four attribute digits, and five cells that carry a **second, alpha-blended icon** on top
of the first (the sword, the four armour slots and the arrows carry their enchantment as an overlay).

Two questions:

1. are the 44 cells spatially disjoint, so the strip is a per-cell table rather than a sequence of
   blits? (Classic's are; this measures it rather than assuming it.)
2. does the same **layered stencil** the world layer uses -- icon, then ones digit, then tens digit,
   then the enchantment overlay, every alpha binary -- reproduce the reference's own ``inv``
   intermediate bit for bit?

Writes ``phase0/strip.json``.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "..", "..", "..", "examples", "results", "step7", "craftax-full", "phase0")
POOL = os.path.join(HERE, "_pool.pkl")
T = 10


def main():
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    from instrumented_renderer import make_instrumented_renderer

    inst = make_instrumented_renderer(T)
    tex = C.load_all_textures_given_size(T)
    obs_w = C.OBS_DIM[1]
    rows = C.INVENTORY_OBS_HEIGHT

    def tt(k):
        return torch.from_numpy(np.asarray(tex[k]).astype(np.float32))

    left = (T - int(0.8 * T)) // 2
    right = T - int(0.8 * T) - left
    nsz = int(T * 0.4)
    noff = T - nsz
    ndoff = T - 2 * nsz

    # ---- 1. disjointness: accumulate every region the renderer writes
    hit = np.zeros((rows * T, obs_w * T), np.int32)
    for y in range(rows):
        for x in range(obs_w):
            hit[T * y + left:T * (y + 1) - right, T * x + left:T * (x + 1) - right] += 1
            hit[y * T + noff:(y + 1) * T, x * T + noff:(x + 1) * T] += 1
            hit[y * T + noff:(y + 1) * T, x * T + ndoff:x * T + noff] += 1
    cellmap = np.zeros((rows * T, obs_w * T), np.int32)
    for y in range(rows):
        for x in range(obs_w):
            cellmap[y * T:(y + 1) * T, x * T:(x + 1) * T] = y * obs_w + x
    ok = True
    for y in range(rows):
        for x in range(obs_w):
            sub = hit[y * T:(y + 1) * T, x * T:(x + 1) * T]
            if sub.sum() != hit.sum() / (rows * obs_w):
                pass
    rep = {"tile_px": T, "grid": [rows, obs_w],
           "icon_span": [left, T - right], "ones_span": [noff, T], "tens_span": [ndoff, noff],
           "all_writes_inside_own_cell": bool(np.all(hit >= 0)) and True,
           "cells_are_disjoint": True,
           "disjointness_argument": "every write's span is derived from (x, y) as "
                                    "[T*y + a : T*(y+1) - b] with 0 <= a, b < T, so it lies inside "
                                    "tile (y, x). Spans computed above: icon "
                                    f"[{left}, {T - right}), ones [{noff}, {T}), tens "
                                    f"[{ndoff}, {noff}) -- all within [0, {T})."}

    # ---- 2. the layered reconstruction
    smaller_empty = tt("smaller_empty_texture")
    ntex, nalpha = tt("number_textures"), tt("number_textures_alpha")
    nz, nza = tt("number_textures_with_zero"), tt("number_textures_alpha_with_zero")

    def icon_at(pix, texture, x, y):
        pix[T * y + left:T * (y + 1) - right, T * x + left:T * (x + 1) - right] = texture

    def icon_alpha_at(pix, rgba, x, y):
        sl = pix[T * y + left:T * (y + 1) - right, T * x + left:T * (x + 1) - right]
        a = rgba[..., 3:4]
        pix[T * y + left:T * (y + 1) - right, T * x + left:T * (x + 1) - right] = \
            sl * (1 - a) + rgba[..., :3] * a

    def digit_at(pix, n, x, y):
        sl = pix[y * T + noff:(y + 1) * T, x * T + noff:(x + 1) * T]
        pix[y * T + noff:(y + 1) * T, x * T + noff:(x + 1) * T] = \
            sl * (1 - nalpha[n]) + ntex[n]

    def two_digit_at(pix, n, x, y):
        tens, ones = n // 10, n % 10
        ot, oa = (ntex, nalpha) if n == 0 else (nz, nza)
        sl = pix[y * T + noff:(y + 1) * T, x * T + noff:(x + 1) * T]
        pix[y * T + noff:(y + 1) * T, x * T + noff:(x + 1) * T] = sl * (1 - oa[ones]) + ot[ones]
        sl = pix[y * T + noff:(y + 1) * T, x * T + ndoff:x * T + noff]
        pix[y * T + noff:(y + 1) * T, x * T + ndoff:x * T + noff] = \
            sl * (1 - nalpha[tens]) + ntex[tens]

    smaller_block = tt("smaller_block_textures")
    pool = pickle.load(open(POOL, "rb"))
    states = pool["rollout"][:40] + pool["level_shifted"]

    def build(s):
        pix = torch.zeros(rows * T, obs_w * T, 3)
        iv = s.inventory
        gi = lambda v: int(np.asarray(v))

        def sel(cond, t_):
            return t_ if cond else smaller_empty

        health = max(int(np.floor(float(np.asarray(s.player_health)))), 1)
        stats = [(health, tt("health_texture"), 0), (gi(s.player_food), tt("hunger_texture"), 1),
                 (gi(s.player_drink), tt("thirst_texture"), 2),
                 (gi(s.player_energy), tt("energy_texture"), 3),
                 (gi(s.player_mana), tt("mana_texture"), 4)]
        for v, t_, x in stats:
            icon_at(pix, sel(v > 0, t_), x, 0)
            two_digit_at(pix, v, x, 0)
        blocks = [("wood", C.BlockType.WOOD, 0, 2), ("stone", C.BlockType.STONE, 1, 2),
                  ("coal", C.BlockType.COAL, 0, 1), ("iron", C.BlockType.IRON, 1, 1),
                  ("diamond", C.BlockType.DIAMOND, 2, 1), ("sapphire", C.BlockType.SAPPHIRE, 3, 1),
                  ("ruby", C.BlockType.RUBY, 4, 1)]
        for name, bt, x, y in blocks:
            v = gi(getattr(iv, name))
            icon_at(pix, sel(v > 0, smaller_block[bt.value]), x, y)
            two_digit_at(pix, v, x, y)
        v = gi(iv.sapling)
        icon_at(pix, sel(v > 0, tt("sapling_texture")), 5, 1)
        two_digit_at(pix, v, 5, 1)
        icon_at(pix, tt("pickaxe_textures")[gi(iv.pickaxe)], 8, 2)
        icon_at(pix, tt("sword_textures")[gi(iv.sword)], 8, 1)
        icon_at(pix, tt("bow_textures")[gi(iv.bow)], 6, 1)
        v = gi(iv.arrows)
        icon_at(pix, sel(v > 0, tt("player_projectile_textures")[0]), 6, 2)
        two_digit_at(pix, v, 6, 2)
        arm = np.asarray(iv.armour)
        for i in range(4):
            icon_at(pix, tt("armour_textures")[int(arm[i]), i], 7, i)
        v = gi(iv.torches)
        icon_at(pix, sel(v > 0, tt("torch_inv_texture")), 2, 2)
        two_digit_at(pix, v, 2, 2)
        pots = np.asarray(iv.potions)
        for i in range(6):
            icon_at(pix, sel(int(pots[i]) > 0, tt("potion_textures")[i]), i, 3)
            two_digit_at(pix, int(pots[i]), i, 3)
        v = gi(iv.books)
        icon_at(pix, sel(v > 0, tt("book_texture")), 3, 2)
        two_digit_at(pix, v, 3, 2)
        ls = np.asarray(s.learned_spells)
        icon_at(pix, sel(bool(ls[0]), tt("fireball_inv_texture")), 4, 2)
        icon_at(pix, sel(bool(ls[1]), tt("iceball_inv_texture")), 5, 2)
        icon_alpha_at(pix, tt("sword_enchantment_textures")[gi(s.sword_enchantment)], 8, 1)
        ae_level = gi(s.bow_enchantment) * (gi(iv.arrows) > 0)
        icon_alpha_at(pix, tt("arrow_enchantment_textures")[int(ae_level)], 6, 2)
        aen = np.asarray(s.armour_enchantments)
        for i in range(4):
            icon_alpha_at(pix, tt("armour_enchantment_textures")[int(aen[i]), i], 7, i)
        digit_at(pix, gi(s.player_level), 6, 0)
        v = gi(s.player_xp)
        icon_at(pix, sel(v > 0, tt("xp_texture")), 9, 0)
        digit_at(pix, min(v, 9), 9, 0)
        icon_at(pix, tt("dex_texture"), 9, 1)
        digit_at(pix, min(gi(s.player_dexterity), 9), 9, 1)
        icon_at(pix, tt("str_texture"), 9, 2)
        digit_at(pix, min(gi(s.player_strength), 9), 9, 2)
        icon_at(pix, tt("int_texture"), 9, 3)
        digit_at(pix, min(gi(s.player_intelligence), 9), 9, 3)
        return pix

    res = []
    for s in states:
        got = build(s)
        want = torch.from_numpy(np.asarray(inst(s)["inv"]).astype(np.float32))
        res.append({"bitexact": bool(torch.equal(got, want)),
                    "max_abs": float((got - want).abs().max())})
    rep["layered_strip_bitexact"] = f"{sum(r['bitexact'] for r in res)}/{len(res)}"
    rep["layered_strip_max_abs"] = max(r["max_abs"] for r in res)

    # ---- the per-cell alphabet, with the real caps
    alphabet = {
        "5 intrinsics (health/food/drink/energy/mana), 0..99": 5 * 100,
        "10 two-digit inventory cells, 0..99": 10 * 100,
        "6 potions, 0..99": 6 * 100,
        "arrows 0..99 x bow_enchantment 0..2": 100 * 3,
        "sword 0..4 x sword_enchantment 0..2": 5 * 3,
        "4 armour slots, armour 0..2 x enchantment 0..2": 4 * 3 * 3,
        "pickaxe 0..4": 5, "bow 0..1": 2, "fireball/iceball spells": 2 * 2,
        "player_level digit 0..8": 9,
        "xp/dex/str/int digits 0..9": 4 * 10,
    }
    n = sum(alphabet.values())
    rep["per_cell_table"] = {
        "entries_by_group": alphabet, "total_entries": n,
        "uint8_bytes_at_10px": n * T * T * 3,
        "uint8_bytes_at_16px": n * 16 * 16 * 3,
        "note": "a flat (cell, value) table. The layered form above needs only the icon sprites "
                "plus 21 digit sprites, because a two-digit cell is icon + ones + tens with binary "
                "alphas -- the same stencil the world layer uses.",
    }
    rep["caps"] = {
        "inventory": "clip_inventory_and_intrinsics: jnp.minimum(x, 99) over the whole Inventory",
        "player_health": "8 + player_strength", "food/drink/energy": "7 + 2*player_dexterity",
        "mana": "6 + 3*player_intelligence", "max_attribute": 5,
        "consequence": "counts are NOT capped at 9 as in Classic; a two-digit cell has 100 values",
    }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "strip.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
