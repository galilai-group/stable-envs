"""Crafted states: exercise the paths a fresh reset does not reach.

The blob parser makes procgen a programmable probe rather than a black box. Three cases the
level-0 frames never reached:

  * heist with `has_keys` all true, so the three ROTATED, `use_abs_coords` key-on-ring sprites draw;
  * miner with most of the grid set to SPACE, so the platform background is visible;
  * grids and entity positions pushed to the edges, where `clipLine`-style clipping decides the
    answer.

Also re-codes the platform backgrounds, which level 0 of miner never showed.
"""
from __future__ import annotations

import json
import os

import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_io as S  # noqa: E402
import probe02_coded_assets as P2  # noqa: E402

SRC_ROOT = P2.SRC_ROOT
CODED = P2.CODED

PLATFORM_BG = [
    "platform_backgrounds/alien_bg.png", "platform_backgrounds/another_world_bg.png",
    "platform_backgrounds/back_cave.png", "platform_backgrounds/caverns.png",
    "platform_backgrounds/cyberpunk_bg.png", "platform_backgrounds/parallax_forest.png",
    "platform_backgrounds/scifi_bg.png", "platform_backgrounds/scifi2_bg.png",
    "platform_backgrounds/living_tissue_bg.png", "platform_backgrounds/airadventurelevel1.png",
    "platform_backgrounds/airadventurelevel2.png", "platform_backgrounds/airadventurelevel3.png",
    "platform_backgrounds/airadventurelevel4.png", "platform_backgrounds/cave_background.png",
    "platform_backgrounds/blue_desert.png", "platform_backgrounds/blue_grass.png",
    "platform_backgrounds/blue_land.png", "platform_backgrounds/blue_shroom.png",
    "platform_backgrounds/colored_desert.png", "platform_backgrounds/colored_grass.png",
    "platform_backgrounds/colored_land.png", "platform_backgrounds/colored_shroom.png",
    "platform_backgrounds/landscape1.png", "platform_backgrounds/landscape2.png",
    "platform_backgrounds/landscape3.png", "platform_backgrounds/landscape4.png",
    "platform_backgrounds/battleback1.png", "platform_backgrounds/battleback2.png",
    "platform_backgrounds/battleback3.png", "platform_backgrounds/battleback4.png",
    "platform_backgrounds/battleback5.png", "platform_backgrounds/battleback6.png",
    "platform_backgrounds/battleback7.png", "platform_backgrounds/battleback8.png",
    "platform_backgrounds/battleback9.png", "platform_backgrounds/battleback10.png",
    "platform_backgrounds/sunrise.png",
    "platform_backgrounds_2/beach1.png", "platform_backgrounds_2/beach2.png",
    "platform_backgrounds_2/beach3.png", "platform_backgrounds_2/beach4.png",
    "platform_backgrounds_2/fantasy1.png", "platform_backgrounds_2/fantasy2.png",
    "platform_backgrounds_2/fantasy3.png", "platform_backgrounds_2/fantasy4.png",
    "platform_backgrounds_2/candy1.png", "platform_backgrounds_2/candy2.png",
    "platform_backgrounds_2/candy3.png", "platform_backgrounds_2/candy4.png",
]
SPACE_BG = [
    "space_backgrounds/deep_space_01.png", "space_backgrounds/spacegen_01.png",
    "space_backgrounds/milky_way_01.png", "space_backgrounds/ez_space_lite_01.png",
    "space_backgrounds/meyespace_v1_01.png", "space_backgrounds/eye_nebula_01.png",
    "space_backgrounds/deep_sky_01.png", "space_backgrounds/space_nebula_01.png",
    "space_backgrounds/Background-1.png", "space_backgrounds/Background-2.png",
    "space_backgrounds/Background-3.png", "space_backgrounds/Background-4.png",
    "space_backgrounds/parallax-space-backgound.png",
]

def build_tree():
    P2.BACKGROUNDS = P2.BACKGROUNDS + PLATFORM_BG + SPACE_BG
    P2.build_tree()

def craft(game, st, rng):
    """Push the state somewhere a reset never goes."""
    g = st["grid"].copy()
    if game == "miner":
        keep = rng.random(g.shape) < 0.25
        g = np.where(keep, g, np.int32(100))                     # SPACE -> background visible
        st["grid"] = g
    elif game == "heist":
        st["has_keys"] = [True] * len(st["has_keys"])
    elif game == "maze":
        # a few cells flipped, and the agent pushed hard against the top-right corner
        idx = rng.integers(0, g.size, size=40)
        flat = g.reshape(-1)
        flat[idx] = np.where(flat[idx] == 51, np.int32(100), np.int32(51))
        st["grid"] = flat.reshape(g.shape)
    ent = st["entities"]
    for e in ent:
        if e["type"] == 0:                                        # the agent
            e["x"] = float(rng.uniform(0.0, st["main_width"]))
            e["y"] = float(rng.uniform(0.0, st["main_height"]))
            e["is_reflected"] = int(rng.integers(0, 2))
    return st

def main():
    build_tree()
    from procgen import ProcgenGym3Env
    game = sys.argv[1]
    rng = np.random.default_rng(int(os.environ.get("SEED", "1")))
    env = ProcgenGym3Env(num=1, env_name=game, distribution_mode="hard", rand_seed=0,
                         start_level=int(os.environ.get("LEVEL", "3")), num_levels=1,
                         use_backgrounds=True, resource_root=CODED + os.sep)
    env.act(np.array([4], np.int32))
    st = S.parse(env.get_state()[0])
    st = craft(game, st, rng)
    env.set_state([S.dump(st)])
    st2 = S.parse(env.get_state()[0])
    assert S.dump(st2) == S.dump(st), "set_state/get_state did not round-trip the crafted state"
    _, obs, _ = env.observe()
    frame = np.asarray(obs["rgb"][0])
    rec = {
        "game": game, "use_backgrounds": True,
        "grid_w": st["grid_w"], "grid_h": st["grid_h"],
        "main": [st["main_width"], st["main_height"]],
        "unit": st["unit"], "view_dim": st["view_dim"],
        "x_off": st["x_off"], "y_off": st["y_off"], "visibility": st["visibility"],
        "background_index": st["background_index"], "bg_pct_x": st["bg_pct_x"],
        "bg_tile_ratio": st["bg_tile_ratio"],
        "grid": st["grid"].tolist(),
        "entities": [{k: e[k] for k in ("x", "y", "rx", "ry", "type", "image_type",
                                        "image_theme", "render_z", "rotation", "is_reflected",
                                        "use_abs_coords", "alpha", "will_erase")}
                     for e in st["entities"]],
        "frame": frame.tolist(),
    }
    if game == "heist":
        rec["has_keys"] = st["has_keys"]
    with open(os.path.join(HERE, f"crafted_{game}.json"), "w") as fh:
        json.dump(rec, fh)
    ids, cnt = np.unique(frame[..., 2], return_counts=True)
    print(f"crafted {game}: B histogram {dict(zip(ids.tolist(), cnt.tolist()))}")

if __name__ == "__main__":
    main()
