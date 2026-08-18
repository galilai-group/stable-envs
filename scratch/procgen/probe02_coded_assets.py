"""Measure Qt's blit rule directly: replace every asset with a coordinate-coded image.

`resource_root` is a procgen option, so the shipped renderer can be pointed at a tree in which each
sprite pixel encodes its own (x, y) and an id. Rendering a real state then reports, for every one of
the 4096 output pixels, WHICH sprite painted it and WHICH source pixel it sampled -- the sampling
rule, read off rather than guessed.

Coding (opaque, so SourceOver is the identity and nothing blends):
    sprite      R = sx,        G = sy,        B = sprite_id   (all sprites are <= 158 px)
    background  R = sx & 255,  G = sy & 255,  B = (sx>>8) | ((sy>>8)<<4)
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import state_io as S  # noqa: E402

SRC_ROOT = ("/mnt/vast/home/randall.balestriero/miniconda3/envs/test/lib/python3.11/"
            "site-packages/procgen/data/assets")
CODED = "/mnt/vast/home/randall.balestriero/se_logs/procgen/coded_assets"

#: sprite path -> id. Ids are deliberately >= 1 so 0 can mean "background / nothing".
SPRITES = {
    "kenney/Ground/Sand/sandCenter.png": 1,
    "misc_assets/cheese.png": 2,
    "kenney/Enemies/mouse_move.png": 3,
    "kenney/Ground/Dirt/dirtCenter.png": 4,
    "misc_assets/gemYellow.png": 5,
    "misc_assets/spaceAstronauts_008.png": 6,
    "misc_assets/keyBlue.png": 7,
    "misc_assets/keyGreen.png": 8,
    "misc_assets/keyRed.png": 9,
    "misc_assets/lock_blue.png": 10,
    "misc_assets/lock_green.png": 11,
    "misc_assets/lock_red.png": 12,
    "misc_assets/robot_greenDrive1.png": 13,
    "misc_assets/elementStone007.png": 14,
    "misc_assets/gemBlue.png": 15,
    "misc_assets/window.png": 16,
    "misc_assets/dirt.png": 17,
    "misc_assets/tile_bricksGrey.png": 18,
}
BACKGROUNDS = [f"topdown_backgrounds/{n}" for n in
               ["floortiles.png", "backgrounddetailed1.png", "backgrounddetailed2.png",
                "backgrounddetailed3.png", "backgrounddetailed4.png", "backgrounddetailed5.png",
                "backgrounddetailed6.png", "backgrounddetailed7.png", "backgrounddetailed8.png"]]


def build_tree():
    if os.path.exists(CODED):
        shutil.rmtree(CODED)
    for root, _dirs, files in os.walk(SRC_ROOT):
        rel = os.path.relpath(root, SRC_ROOT)
        dst = CODED if rel == "." else os.path.join(CODED, rel)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            os.symlink(os.path.join(root, f), os.path.join(dst, f))
    for path, sid in SPRITES.items():
        w, h = Image.open(os.path.join(SRC_ROOT, path)).size
        a = np.zeros((h, w, 4), np.uint8)
        a[..., 0] = np.arange(w)[None, :]
        a[..., 1] = np.arange(h)[:, None]
        a[..., 2] = sid
        a[..., 3] = 255
        p = os.path.join(CODED, path)
        os.remove(p)
        Image.fromarray(a, "RGBA").save(p)
    for path in BACKGROUNDS:
        w, h = Image.open(os.path.join(SRC_ROOT, path)).size
        x = np.arange(w)[None, :] * np.ones((h, 1), np.int64)
        y = np.arange(h)[:, None] * np.ones((1, w), np.int64)
        a = np.stack([x & 255, y & 255, (x >> 8) | ((y >> 8) << 4)], -1).astype(np.uint8)
        p = os.path.join(CODED, path)
        os.remove(p)
        Image.fromarray(a, "RGB").save(p)
    print(f"coded tree at {CODED}")


def main():
    build_tree()
    from procgen import ProcgenGym3Env
    game = sys.argv[1] if len(sys.argv) > 1 else "maze"
    use_bg = "--nobg" not in sys.argv
    env = ProcgenGym3Env(num=1, env_name=game, distribution_mode="hard", rand_seed=0,
                         start_level=int(os.environ.get("LEVEL", "0")), num_levels=1,
                         use_backgrounds=use_bg, resource_root=CODED + os.sep)
    env.act(np.array([4], np.int32))
    blob = env.get_state()[0]
    st = S.parse(blob)
    _, obs, _ = env.observe()
    frame = np.asarray(obs["rgb"][0])                                    # (64,64,3) uint8

    rec = {
        "game": game, "use_backgrounds": use_bg,
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
    tag = f"{game}{'_bg' if use_bg else '_nobg'}_L{os.environ.get('LEVEL','0')}"
    with open(os.path.join(HERE, f"coded_{tag}.json"), "w") as fh:
        json.dump(rec, fh)
    ids, cnt = np.unique(frame[..., 2], return_counts=True)
    print(f"{game} bg={use_bg}: sprite-id histogram (B channel) "
          f"{dict(zip(ids.tolist(), cnt.tolist()))}")
    print(f"  unit={st['unit']} view_dim={st['view_dim']} x_off={st['x_off']} y_off={st['y_off']}")


if __name__ == "__main__":
    main()
