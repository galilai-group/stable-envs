"""Phase 0 — the sleeping player SPRITE (a world-layer change), and the day-cycle arithmetic.

Three things the strip/night/sleep split does not cover:

1. ``is_sleeping`` does not only desaturate. ``renderer.py`` selects
   ``player_texture_index = select(is_sleeping, 4, player_direction - 1)`` — a FIFTH player
   texture that the Step-6 ``_Composite`` table does not have. So the world layer changes too.
2. The existing table's load-bearing assumption ("the player tables are zero away from the centre
   tile") has to hold for texture 4 as well, or compositing the player into one tile stops being
   the same arithmetic as compositing it over the frame.
3. What fraction of the day cycle is noise-free (``light_level >= 0.5``), computed from
   ``calculate_light_level`` rather than estimated from one rollout.
"""
from __future__ import annotations

import json
import os

import numpy as np

from craftax.craftax_classic.constants import OBS_DIM, load_all_textures

OUT = "examples/results/step7/craftax-classic-full/phase0"


def main():
    os.makedirs(OUT, exist_ok=True)
    allt = load_all_textures()
    res = {}

    for bps in (7, 16):
        tex = allt[bps]
        pa = np.asarray(tex["full_map_player_textures_alpha"], np.float32)   # (5, H, W, 3)
        pt = np.asarray(tex["full_map_player_textures"], np.float32)
        H, W = OBS_DIM[0] * bps, OBS_DIM[1] * bps
        cr, cc = OBS_DIM[0] // 2, OBS_DIM[1] // 2
        mask = np.ones((H, W), bool)
        mask[cr * bps:(cr + 1) * bps, cc * bps:(cc + 1) * bps] = False
        off_centre = {i: dict(alpha_max=float(np.abs(pa[i][mask]).max()),
                              tex_max=float(np.abs(pt[i][mask]).max()))
                      for i in range(pa.shape[0])}
        alphas = sorted(set(np.unique(pa).tolist()))
        res[f"bps{bps}"] = dict(
            n_player_textures=int(pa.shape[0]),
            sleeping_slot_index=4,
            player_alpha_zero_off_centre=off_centre,
            distinct_player_alphas=alphas if len(alphas) <= 12 else
            [alphas[0], alphas[1], "...", alphas[-1]],
            n_distinct_player_alphas=len(alphas),
            composite_table_today=(4 + 1) * 17 * 8,
            composite_table_with_sleeping_slot=(5 + 1) * 17 * 8,
            composite_bytes_today_uint8=(4 + 1) * 17 * 8 * bps * bps * 3,
            composite_bytes_with_sleep_uint8=(5 + 1) * 17 * 8 * bps * bps * 3,
        )

        # the sleeping sprite genuinely differs from all four facing sprites
        res[f"bps{bps}"]["sleeping_sprite_differs_from_facing"] = {
            f"vs_slot_{i}": int((pt[4] != pt[i]).sum() + (pa[4] != pa[i]).sum())
            for i in range(4)}

    # day cycle: light_level(t) = 1 - |cos(pi * ((t/300) % 1 + 0.3))|**3
    day_length = 300
    t = np.arange(day_length)
    progress = (t / day_length) % 1 + 0.3
    light = 1 - np.abs(np.cos(np.pi * progress)) ** 3
    res["day_cycle"] = dict(
        day_length=day_length,
        steps_light_ge_half=int((light >= 0.5).sum()),
        frac_light_ge_half=float((light >= 0.5).mean()),
        frac_light_eq_one=float((light >= 0.9999).mean()),
        frac_needs_noise=float((light < 0.5).mean()),
        min_light=float(light.min()), max_light=float(light.max()),
        note=("light_level >= 0.5 makes night_static_intensity exactly 0, so the noise field "
              "cannot contribute: those frames are pure f(state) with NO PRNG. Below 0.5 the "
              "threefry field is required."))
    print(json.dumps(res, indent=2))
    with open(os.path.join(OUT, "player_sleep_and_daycycle.json"), "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
