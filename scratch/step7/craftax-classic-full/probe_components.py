"""Phase 0 (1)(3)(4) — the inventory strip, the night chain and the sleeping desaturation.

Every number here is measured against the *un-jitted* instrumented reference, because the jitted
one is only uint8-exact with the official renderer (XLA fuses the extra outputs differently); see
``instrumented_vs_official`` in the emitted JSON.

Nothing under ``stable_envs/`` is edited or imported for the prototypes: the point of phase 0 is to
find out what the smallest capability is, not to add it.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import threefry_torch as tf3                                  # noqa: E402
from instrumented_renderer import make_instrumented_renderer  # noqa: E402

from craftax.craftax_classic.constants import (                # noqa: E402
    BlockType, INVENTORY_OBS_HEIGHT, OBS_DIM, load_all_textures)
from craftax.craftax_classic.renderer import make_craftax_pixel_renderer  # noqa: E402

OUT = "examples/results/step7/craftax-classic-full/phase0"
STATES = "scratch/step7/craftax-classic-full/states.pkl"

#: (field on EnvState / Inventory, icon key) in the reference's own draw order, laid out on the
#: INVENTORY_OBS_HEIGHT x OBS_DIM[1] grid. ``None`` is a cell the reference never writes.
CELLS = [
    # (row, col, accessor, icon)
    (0, 0, ("player_health",), "health_texture"),
    (0, 1, ("player_food",), "hunger_texture"),
    (0, 2, ("player_drink",), "thirst_texture"),
    (0, 3, ("player_energy",), "energy_texture"),
    (0, 4, ("inventory", "sapling"), "sapling_texture"),
    (0, 5, ("inventory", "wood"), ("smaller_block_textures", BlockType.WOOD.value)),
    (0, 6, ("inventory", "stone"), ("smaller_block_textures", BlockType.STONE.value)),
    (0, 7, ("inventory", "coal"), ("smaller_block_textures", BlockType.COAL.value)),
    (0, 8, ("inventory", "iron"), ("smaller_block_textures", BlockType.IRON.value)),
    (1, 0, ("inventory", "diamond"), ("smaller_block_textures", BlockType.DIAMOND.value)),
    (1, 1, ("inventory", "wood_pickaxe"), "wood_pickaxe_texture"),
    (1, 2, ("inventory", "stone_pickaxe"), "stone_pickaxe_texture"),
    (1, 3, ("inventory", "iron_pickaxe"), "iron_pickaxe_texture"),
    (1, 4, ("inventory", "wood_sword"), "wood_sword_texture"),
    (1, 5, ("inventory", "stone_sword"), "stone_sword_texture"),
    (1, 6, ("inventory", "iron_sword"), "iron_sword_texture"),
]
N_COUNT = 10        #: counts are capped at 9 (game_logic.py:1633, :1293, :1319)


def _get(state, path):
    o = state
    for p in path:
        o = getattr(o, p)
    return int(np.asarray(jax.device_get(o)))


def strip_geometry(bps):
    ls = (bps - int(0.8 * bps)) // 2 - 1
    rs = bps - int(0.8 * bps) - ls
    nsz = int(bps * 0.6)
    noff = bps - nsz
    return dict(icon_lo=ls, icon_hi=bps - rs, number_lo=noff - 1, number_hi=bps - 1,
                number_size=nsz, small=int(0.8 * bps))


def build_strip_atlas(tex, bps, device="cpu"):
    """``(16, 10, bps, bps, 3)`` float32: cell ``c`` holding count ``v``.

    Exactly the reference's per-cell arithmetic: zeros, then ``.set(icon or empty)``, then
    ``.mul(1 - number_alpha).add(number_premultiplied)``.
    """
    g = strip_geometry(bps)
    numtex = torch.as_tensor(np.asarray(tex["number_textures"], np.float32), device=device)
    numalpha = torch.as_tensor(np.asarray(tex["number_textures_alpha"], np.float32), device=device)
    out = torch.zeros(len(CELLS), N_COUNT, bps, bps, 3, dtype=torch.float32, device=device)
    for ci, (_r, _c, _path, icon) in enumerate(CELLS):
        if isinstance(icon, tuple):
            t = np.asarray(tex[icon[0]][icon[1]], np.float32)
        else:
            t = np.asarray(tex[icon], np.float32)
        t = torch.as_tensor(t, device=device)
        for v in range(N_COUNT):
            tile = torch.zeros(bps, bps, 3, dtype=torch.float32, device=device)
            if v > 0:
                tile[g["icon_lo"]:g["icon_hi"], g["icon_lo"]:g["icon_hi"], :] = t
            sl = (slice(g["number_lo"], g["number_hi"]), slice(g["number_lo"], g["number_hi"]))
            tile[sl] = tile[sl] * (1 - numalpha[v]) + numtex[v]
            out[ci, v] = tile
    return out


def strip_from_atlas(atlas, counts, bps, device="cpu"):
    """``counts`` (B, 16) int -> ``(B, 2*bps, 9*bps, 3)`` float32. A pure gather + scatter."""
    B = counts.shape[0]
    H, W = INVENTORY_OBS_HEIGHT * bps, OBS_DIM[1] * bps
    out = torch.zeros(B, H, W, 3, dtype=torch.float32, device=device)
    for ci, (r, c, _p, _i) in enumerate(CELLS):
        out[:, r * bps:(r + 1) * bps, c * bps:(c + 1) * bps, :] = atlas[ci, counts[:, ci]]
    return out


def night_torch(world, light, key, tex, device="cpu"):
    """The reference's night chain, in torch float32, in the reference's order."""
    H, W = world.shape[:2]
    night_tex = torch.as_tensor(np.asarray(tex["night_texture"], np.float32), device=device)
    noise_int = torch.as_tensor(np.asarray(tex["night_noise_intensity_texture"], np.float32),
                                device=device)
    inten = max(2.0 * (0.5 - float(light)), 0.0)
    noise = tf3.uniform(key, (H, W)) * 95 + 32
    mask = inten * noise_int
    nws = (1 - mask) * world + mask * noise[:, :, None]
    npx = nws if float(light) < 0.5 else world
    lum = 0.299 * npx[:, :, 0] + 0.587 * npx[:, :, 1] + 0.114 * npx[:, :, 2]
    lum = lum[:, :, None].expand(H, W, 3)
    npx = npx * 0.4 + (1 - 0.4) * lum
    npx = 0.5 * npx + 0.5 * night_tex
    return float(light) * world + (1 - float(light)) * npx


def sleep_torch(px):
    lum = 0.299 * px[:, :, 0] + 0.587 * px[:, :, 1] + 0.114 * px[:, :, 2]
    return (0.5 * lum)[:, :, None] + torch.as_tensor([0.0, 0.0, 8.0], device=px.device)


def bit(a, b):
    a = np.ascontiguousarray(np.asarray(a, np.float32))
    b = np.ascontiguousarray(np.asarray(b, np.float32))
    return bool(np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def u8(a):
    return np.round(np.asarray(a, np.float64)).clip(0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    states = pickle.load(open(STATES, "rb"))
    allt = load_all_textures()
    res = {"torch": torch.__version__, "jax": jax.__version__, "n_states": len(states)}

    for bps in (7, 16):
        tex = allt[bps]
        inst = make_instrumented_renderer(bps)          # un-jitted: bit-exact with official
        off = make_craftax_pixel_renderer(bps)
        Hw = OBS_DIM[0] * bps
        Hs = INVENTORY_OBS_HEIGHT * bps
        g = strip_geometry(bps)
        atlas = build_strip_atlas(tex, bps)

        # -- does any cell's number region leave its own tile, or overlap a neighbour?
        occ = np.zeros((Hs, OBS_DIM[1] * bps), np.int32)
        for (r, c, _p, _i) in CELLS:
            occ[r * bps + g["icon_lo"]:r * bps + g["icon_hi"],
                c * bps + g["icon_lo"]:c * bps + g["icon_hi"]] += 1
            occ[r * bps + g["number_lo"]:r * bps + g["number_hi"],
                c * bps + g["number_lo"]:c * bps + g["number_hi"]] += 1
        cell_disjoint = bool(occ.max() <= 2)   # a cell's own icon+number may overlap; cells may not

        n_strip_bit = n_strip_u8 = n_frame_rows = 0
        n_night_bit = n_night_u8 = n_sleep_bit = n_sleep_u8 = 0
        n_inst = 0
        night_states = sleep_states = 0
        strip_light_invariant = True
        counts_seen = set()
        worst_night = worst_sleep = 0.0

        for st in states:
            cap = jax.device_get(inst(st))
            frame = np.asarray(jax.device_get(off(st)))
            n_inst += int(bit(cap["frame"], frame))
            world = np.asarray(cap["world"])
            light = float(np.asarray(jax.device_get(st.light_level)))
            sleeping = bool(np.asarray(jax.device_get(st.is_sleeping)))

            # (1) strip
            cnt = np.array([[_get(st, p) for (_r, _c, p, _i) in CELLS]], np.int64)
            counts_seen.update(int(v) for v in cnt[0])
            mine = strip_from_atlas(atlas, torch.as_tensor(cnt), bps)[0].numpy()
            n_strip_bit += int(bit(mine, np.asarray(cap["inv"])))
            n_strip_u8 += int(np.array_equal(u8(mine), u8(np.asarray(cap["inv"]))))
            n_frame_rows += int(np.array_equal(u8(mine), u8(frame[Hw:Hw + Hs])))

            # (2) night chain, given the reference's own world layer
            key = torch.as_tensor(np.asarray(jax.device_get(st.state_rng), np.int64))
            got = night_torch(torch.as_tensor(world), light, key, tex).numpy()
            an = np.asarray(cap["after_night"])
            n_night_bit += int(bit(got, an))
            n_night_u8 += int(np.array_equal(u8(got), u8(an)))
            worst_night = max(worst_night, float(np.abs(got - an).max()))
            night_states += int(light < 0.5)

            # (3) sleeping
            sp = sleep_torch(torch.as_tensor(an)).numpy() if sleeping else an
            asl = np.asarray(cap["after_sleep"])
            n_sleep_bit += int(bit(sp, asl))
            n_sleep_u8 += int(np.array_equal(u8(sp), u8(asl)))
            worst_sleep = max(worst_sleep, float(np.abs(sp - asl).max()))
            sleep_states += int(sleeping)

            # strip must not depend on light_level or is_sleeping
            for l2, s2 in ((1.0, False), (0.0, True), (0.3, False)):
                f2 = np.asarray(jax.device_get(off(st.replace(
                    light_level=jnp.float32(l2), is_sleeping=jnp.bool_(s2)))))
                strip_light_invariant &= bit(f2[Hw:Hw + Hs], frame[Hw:Hw + Hs])

        res[f"bps{bps}"] = dict(
            block_pixel_size=bps,
            frame_shape=[(OBS_DIM[0] + INVENTORY_OBS_HEIGHT) * bps, OBS_DIM[1] * bps, 3],
            world_rows=[0, Hw], strip_rows=[Hw, Hw + Hs],
            strip_geometry=g,
            strip_atlas_entries=len(CELLS) * N_COUNT,
            strip_atlas_bytes_uint8=len(CELLS) * N_COUNT * bps * bps * 3,
            cells_are_disjoint=cell_disjoint,
            instrumented_vs_official_bitexact=f"{n_inst}/{len(states)}",
            strip_bitexact=f"{n_strip_bit}/{len(states)}",
            strip_uint8_exact=f"{n_strip_u8}/{len(states)}",
            strip_matches_official_frame_rows_uint8=f"{n_frame_rows}/{len(states)}",
            strip_invariant_to_light_and_sleep=strip_light_invariant,
            counts_observed=sorted(counts_seen),
            night_bitexact=f"{n_night_bit}/{len(states)}",
            night_uint8_exact=f"{n_night_u8}/{len(states)}",
            night_max_abs_float_diff=worst_night,
            n_states_light_below_half=night_states,
            sleep_bitexact=f"{n_sleep_bit}/{len(states)}",
            sleep_uint8_exact=f"{n_sleep_u8}/{len(states)}",
            sleep_max_abs_float_diff=worst_sleep,
            n_states_sleeping=sleep_states,
        )
        print(json.dumps(res[f"bps{bps}"], indent=2))

    with open(os.path.join(OUT, "components.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
