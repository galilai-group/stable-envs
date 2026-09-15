"""Phase 0 — the world layer under ``is_sleeping``: is a sixth ``_Composite`` slot all it needs?

Reconstructs the CENTRE tile of the reference's pre-night world layer in torch, exactly the way
``_Composite._build`` does it — ``block * (1 - alpha) + player_tex * alpha`` — for the awake slot
(``player_direction - 1``) and for the sleeping slot (4), and compares both against the
instrumented reference's own ``world`` capture on real states.

If the awake reconstruction is exact and the sleeping one is exact, then extending the Step-6 table
from 5 player slots to 6 is the whole of the world-layer change, and nothing else about the gather
moves.
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
from instrumented_renderer import make_instrumented_renderer  # noqa: E402

from craftax.craftax_classic.constants import OBS_DIM, load_all_textures  # noqa: E402

OUT = "examples/results/step7/craftax-classic-full/phase0"
STATES = "scratch/step7/craftax-classic-full/states.pkl"


def main():
    os.makedirs(OUT, exist_ok=True)
    states = pickle.load(open(STATES, "rb"))
    allt = load_all_textures()
    res = {}

    for bps in (7, 16):
        tex = allt[bps]
        blk = np.asarray(tex["block_textures"], np.float32)              # (17, b, b, 3)
        pa = np.asarray(tex["full_map_player_textures_alpha"], np.float32)
        pt = np.asarray(tex["full_map_player_textures"], np.float32)
        cr, cc = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        sl = (slice(cr * bps, (cr + 1) * bps), slice(cc * bps, (cc + 1) * bps))
        pa_c = pa[:, sl[0], sl[1], :]
        pt_c = pt[:, sl[0], sl[1], :]

        inst = make_instrumented_renderer(bps)     # un-jitted == bit-exact with the official one
        n = n_awake = n_sleep = n_confined = 0
        skipped_mob_on_player = 0
        for st in states:
            pos = np.asarray(jax.device_get(st.player_position))
            b = int(np.asarray(jax.device_get(st.map))[pos[0], pos[1]])
            d = int(np.asarray(jax.device_get(st.player_direction))) - 1
            # a mob standing on the player's own tile would make block_textures the wrong base
            on_player = False
            for m in (st.zombies, st.cows, st.skeletons, st.arrows):
                p = np.asarray(jax.device_get(m.position))
                k = np.asarray(jax.device_get(m.mask))
                on_player |= bool(((p == pos).all(axis=1) & k).any())
            if on_player:
                skipped_mob_on_player += 1
                continue
            n += 1

            w_awake = np.asarray(jax.device_get(
                inst(st.replace(is_sleeping=jnp.bool_(False)))["world"]))
            w_sleep = np.asarray(jax.device_get(
                inst(st.replace(is_sleeping=jnp.bool_(True)))["world"]))

            def comp(slot):
                a = torch.as_tensor(pa_c[slot])
                return (torch.as_tensor(blk[b]) * (1 - a) + torch.as_tensor(pt_c[slot]) * a).numpy()

            n_awake += int(np.array_equal(comp(d).view(np.uint32),
                                          w_awake[sl].view(np.uint32)))
            n_sleep += int(np.array_equal(comp(4).view(np.uint32),
                                          w_sleep[sl].view(np.uint32)))
            # outside the centre tile the two must be identical
            m = np.ones(w_awake.shape[:2], bool)
            m[sl] = False
            n_confined += int(np.array_equal(w_awake[m].view(np.uint32),
                                             w_sleep[m].view(np.uint32)))

        res[f"bps{bps}"] = dict(
            n_states_used=n, skipped_mob_on_player_tile=skipped_mob_on_player,
            centre_tile_awake_bitexact=f"{n_awake}/{n}",
            centre_tile_sleeping_bitexact=f"{n_sleep}/{n}",
            is_sleeping_world_change_confined_to_centre_tile=f"{n_confined}/{n}",
        )
        print(json.dumps(res[f"bps{bps}"], indent=2))

    with open(os.path.join(OUT, "world_layer_sleeping_slot.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
