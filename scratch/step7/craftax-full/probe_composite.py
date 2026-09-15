"""Is the world layer a per-tile, N-layer stencil composite? Proved in torch against the reference.

The phase-0 question phase 1 turns on: can the 130x110 world window be written as

    tile = where(proj_a, proj, where(mob_a, mob, where(player_a, player, bg[block, item])))

with **five small integers per tile** and nothing else -- and is that bit-identical to craftax's own
scan of up to fourteen ``dynamic_update_slice`` blits? Every overlay alpha in the full game is binary
(``phase0/layers.json``), so each layer is a *select* rather than a blend, which is what makes the
formulation possible at all; Classic's alphas are ``{0, 128/255, 1}`` and its composite is a real
blend, so this is a different capability from W1's, not a resizing of it.

This builds the per-tile ids from ``EnvState`` the way the renderer computes them, composites them,
and compares against the instrumented renderer's ``world`` intermediate -- the last state before any
lighting. It also counts the states where two entities share a tile, which is the only case the
formulation cannot represent.

Writes ``phase0/composite.json``.
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

# overlay slot layout: 0 = nothing, then the three mob classes end to end
MOB_BASE = {"melee": 1, "passive": 9, "ranged": 12}      # 8, 3, 8 types
N_MOB_SLOT = 1 + 8 + 3 + 8
N_PROJ_SLOT = 1 + 8 * 4


def main():
    import jax.numpy as jnp
    from craftax.craftax import constants as C
    from instrumented_renderer import make_instrumented_renderer

    inst = make_instrumented_renderer(T)
    tex = C.load_all_textures_given_size(T)
    obs_h, obs_w = C.OBS_DIM

    def tt(x):
        return torch.from_numpy(np.asarray(x).astype(np.float32))

    block = tt(tex["block_textures"])                       # (37, T, T, 3)
    item = tt(tex["full_map_item_textures"])[:, :T, :T].clone()   # tiled -> one tile: (5, T, T, 4)
    # ItemType.NONE is slot 0 and the reference's scan starts at 1: it never draws it. Its texture
    # is ``debug.png`` and is NOT transparent, so an implementation that gathers slot 0 like any
    # other draws a debug tile over every empty square. Zero its alpha instead of special-casing it.
    item[C.ItemType.NONE.value, ..., 3] = 0.0
    player = tt(tex["player_textures"])                     # (5, T, T, 4)
    mob_rgb = torch.zeros(N_MOB_SLOT, T, T, 3)
    mob_a = torch.zeros(N_MOB_SLOT, T, T, 3)
    for name, key in (("melee", "melee"), ("passive", "passive"), ("ranged", "ranged")):
        r = tt(tex[f"{key}_mob_textures"])
        a = tt(tex[f"{key}_mob_texture_alphas"])
        b = MOB_BASE[name]
        mob_rgb[b:b + r.shape[0]] = r
        mob_a[b:b + a.shape[0]] = a
    pr = tt(tex["projectile_textures"])
    pa = tt(tex["projectile_texture_alphas"])
    proj_rgb = torch.zeros(N_PROJ_SLOT, T, T, 3)
    proj_a = torch.zeros(N_PROJ_SLOT, T, T, 3)
    for ty in range(pr.shape[0]):
        for o, (flip, trans) in enumerate(((False, False), (False, True),
                                           (True, False), (True, True))):
            r, a = pr[ty], pa[ty]
            if flip:
                r, a = torch.flip(r, (0,)), torch.flip(a, (0,))
            if trans:
                r, a = r.permute(1, 0, 2).contiguous(), a.permute(1, 0, 2).contiguous()
            proj_rgb[1 + ty * 4 + o] = r
            proj_a[1 + ty * 4 + o] = a

    pool = pickle.load(open(POOL, "rb"))
    states = pool["rollout"][:40] + pool["level_shifted"]

    def ids_for(s):
        """(block, item, player_slot, mob_slot, proj_slot) per tile, and the sharing count."""
        lvl = int(np.asarray(s.player_level))
        pp = np.asarray(s.player_position)
        pad = C.MAX_OBS_DIM + 2
        m = np.pad(np.asarray(s.map)[lvl], pad,
                   constant_values=C.BlockType.OUT_OF_BOUNDS.value)
        it = np.pad(np.asarray(s.item_map)[lvl], pad, constant_values=C.ItemType.NONE.value)
        tl = pp - np.array(C.OBS_DIM) // 2 + pad
        blk = m[tl[0]:tl[0] + obs_h, tl[1]:tl[1] + obs_w].astype(np.int64).copy()
        itm = it[tl[0]:tl[0] + obs_h, tl[1]:tl[1] + obs_w].astype(np.int64).copy()

        # the two global substitutions, resolved into the stored ids
        from craftax.craftax.util.game_logic_utils import is_boss_vulnerable
        boss = (C.BlockType.NECROMANCER_VULNERABLE.value if bool(is_boss_vulnerable(s))
                else C.BlockType.NECROMANCER.value)
        blk[blk == C.BlockType.NECROMANCER.value] = boss
        lad = (C.ItemType.LADDER_DOWN.value
               if int(np.asarray(s.monsters_killed)[lvl]) >= C.MONSTERS_KILLED_TO_CLEAR_LEVEL
               else C.ItemType.LADDER_DOWN_BLOCKED.value)
        itm[itm == C.ItemType.LADDER_DOWN.value] = lad

        pslot = np.zeros((obs_h, obs_w), np.int64)          # 0 = none; 1..5 = sprite+1
        idx = 4 if bool(np.asarray(s.is_sleeping)) else int(np.asarray(s.player_direction)) - 1
        pslot[obs_h // 2, obs_w // 2] = idx + 1

        mslot = np.zeros((obs_h, obs_w), np.int64)
        pjslot = np.zeros((obs_h, obs_w), np.int64)
        shared = 0
        for name, attr in (("melee", "melee_mobs"), ("passive", "passive_mobs"),
                           ("ranged", "ranged_mobs")):
            mo = getattr(s, attr)
            pos = np.asarray(mo.position)[lvl]
            msk = np.asarray(mo.mask)[lvl]
            tid = np.asarray(mo.type_id)[lvl]
            for i in range(pos.shape[0]):
                lp = pos[i] - pp + np.array([obs_h, obs_w]) // 2
                if msk[i] and 0 <= lp[0] < obs_h and 0 <= lp[1] < obs_w:
                    if mslot[lp[0], lp[1]] != 0:
                        shared += 1
                    mslot[lp[0], lp[1]] = MOB_BASE[name] + int(tid[i])
        for attr, dattr in (("mob_projectiles", "mob_projectile_directions"),
                            ("player_projectiles", "player_projectile_directions")):
            mo = getattr(s, attr)
            dr = np.asarray(getattr(s, dattr))[lvl]
            pos = np.asarray(mo.position)[lvl]
            msk = np.asarray(mo.mask)[lvl]
            tid = np.asarray(mo.type_id)[lvl]
            for i in range(pos.shape[0]):
                lp = pos[i] - pp + np.array([obs_h, obs_w]) // 2
                if msk[i] and 0 <= lp[0] < obs_h and 0 <= lp[1] < obs_w:
                    flip = bool(dr[i][0] > 0 or dr[i][1] > 0)
                    trans = bool(dr[i][1] != 0)
                    o = (2 if flip else 0) + (1 if trans else 0)
                    if pjslot[lp[0], lp[1]] != 0:
                        shared += 1
                    pjslot[lp[0], lp[1]] = 1 + int(tid[i]) * 4 + o
        return blk, itm, pslot, mslot, pjslot, shared

    def composite(blk, itm, pslot, mslot, pjslot):
        b = block[torch.from_numpy(blk)]                       # (H, W, T, T, 3)
        ia = item[torch.from_numpy(itm)]
        a = ia[..., 3:4].expand_as(ia[..., :3])
        px = b * (1 - a) + ia[..., :3] * a
        pa_ = player[torch.from_numpy(pslot.clip(1) - 1)][..., 3:4]
        prgb = player[torch.from_numpy(pslot.clip(1) - 1)][..., :3]
        on = torch.from_numpy((pslot > 0).astype(np.float32))[..., None, None, None]
        pa_ = pa_ * on
        px = px * (1 - pa_) + prgb * pa_
        ma = mob_a[torch.from_numpy(mslot)]
        px = px * (1 - ma) + mob_rgb[torch.from_numpy(mslot)] * ma
        ja = proj_a[torch.from_numpy(pjslot)]
        px = px * (1 - ja) + proj_rgb[torch.from_numpy(pjslot)] * ja
        return px.permute(0, 2, 1, 3, 4).reshape(obs_h * T, obs_w * T, 3)

    res = []
    n_shared = 0
    for s in states:
        blk, itm, pslot, mslot, pjslot, shared = ids_for(s)
        n_shared += shared
        got = composite(blk, itm, pslot, mslot, pjslot)
        want = torch.from_numpy(np.asarray(inst(s)["world"]).astype(np.float32))
        res.append({"bitexact": bool(torch.equal(got, want)),
                    "max_abs": float((got - want).abs().max()),
                    "shared_tiles": int(shared)})
    rep = {
        "tile_px": T, "n_states": len(states),
        "per_tile_composite_bitexact": f"{sum(r['bitexact'] for r in res)}/{len(res)}",
        "max_abs_over_pool": max(r["max_abs"] for r in res),
        "states_with_two_entities_on_one_tile": sum(1 for r in res if r["shared_tiles"]),
        "total_shared_tiles": n_shared,
        "per_tile_id_alphabet": {
            "block": len(C.BlockType), "item": len(C.ItemType),
            "player_slot": 6, "mob_slot": N_MOB_SLOT, "proj_slot": N_PROJ_SLOT},
        "tables_uint8_bytes_at_10px": {
            "block": len(C.BlockType) * T * T * 3,
            "item_rgba": len(C.ItemType) * T * T * 4,
            "player_rgba": 5 * T * T * 4,
            "mob_rgb+alpha": N_MOB_SLOT * T * T * 3 * 2,
            "proj_rgb+alpha": N_PROJ_SLOT * T * T * 3 * 2,
        },
    }
    rep["tables_total_uint8_bytes_at_10px"] = sum(rep["tables_uint8_bytes_at_10px"].values())
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "composite.json"), "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    sys.exit(main())
