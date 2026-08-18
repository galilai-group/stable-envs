"""Decide the blit hypothesis against the coordinate-coded frames.

Rebuilds each coded frame from the parsed state alone and reports how many of the 4096 pixels carry
the predicted (source x, source y, sprite id). A wrong fixed-point rule shows up as a whole column or
row of pixels sampling the neighbouring source pixel, not as a subtle colour shift, so this is a
sharp test.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtblit as Q  # noqa: E402

SRC_ROOT = ("/mnt/vast/home/randall.balestriero/miniconda3/envs/test/lib/python3.11/"
            "site-packages/procgen/data/assets")

SPRITE_ID = {
    "kenney/Ground/Sand/sandCenter.png": 1, "misc_assets/cheese.png": 2,
    "kenney/Enemies/mouse_move.png": 3, "kenney/Ground/Dirt/dirtCenter.png": 4,
    "misc_assets/gemYellow.png": 5, "misc_assets/spaceAstronauts_008.png": 6,
    "misc_assets/keyBlue.png": 7, "misc_assets/keyGreen.png": 8, "misc_assets/keyRed.png": 9,
    "misc_assets/lock_blue.png": 10, "misc_assets/lock_green.png": 11,
    "misc_assets/lock_red.png": 12, "misc_assets/robot_greenDrive1.png": 13,
    "misc_assets/elementStone007.png": 14, "misc_assets/gemBlue.png": 15,
    "misc_assets/window.png": 16, "misc_assets/dirt.png": 17,
    "misc_assets/tile_bricksGrey.png": 18,
}

#: `asset_for_type` per game, transcribed from games/<name>.cpp. type -> list of paths (themes).
ASSETS = {
    "maze": {51: ["kenney/Ground/Sand/sandCenter.png"], 2: ["misc_assets/cheese.png"],
             0: ["kenney/Enemies/mouse_move.png"]},
    "heist": {51: ["kenney/Ground/Dirt/dirtCenter.png"], 9: ["misc_assets/gemYellow.png"],
              0: ["misc_assets/spaceAstronauts_008.png"],
              2: ["misc_assets/keyBlue.png", "misc_assets/keyGreen.png",
                  "misc_assets/keyRed.png"],
              1: ["misc_assets/lock_blue.png", "misc_assets/lock_green.png",
                  "misc_assets/lock_red.png"]},
    "miner": {0: ["misc_assets/robot_greenDrive1.png"], 1: ["misc_assets/elementStone007.png"],
              2: ["misc_assets/gemBlue.png"], 6: ["misc_assets/window.png"],
              9: ["misc_assets/dirt.png"], 10: ["misc_assets/tile_bricksGrey.png"]},
}
#: `image_for_type` overrides
IMG_FOR_TYPE = {"miner": {3: 1, 4: 2}}
TOPDOWN_BG = [f"topdown_backgrounds/{n}" for n in
              ["floortiles.png", "backgrounddetailed1.png", "backgrounddetailed2.png",
               "backgrounddetailed3.png", "backgrounddetailed4.png", "backgrounddetailed5.png",
               "backgrounddetailed6.png", "backgrounddetailed7.png", "backgrounddetailed8.png"]]


def _platform_bg():
    import probe04_crafted as P4
    return P4.PLATFORM_BG + P4.SPACE_BG


BG_LIST = {"maze": lambda: TOPDOWN_BG, "heist": lambda: TOPDOWN_BG, "miner": _platform_bg}

_SIZE: dict = {}


def size(path):
    if path not in _SIZE:
        _SIZE[path] = Image.open(os.path.join(SRC_ROOT, path)).size
    return _SIZE[path]


def coded_sprite(path):
    w, h = size(path)
    a = np.zeros((h, w, 3), np.uint8)
    a[..., 0] = np.arange(w)[None, :]
    a[..., 1] = np.arange(h)[:, None]
    a[..., 2] = SPRITE_ID[path]
    return a


def coded_bg(path):
    w, h = size(path)
    x = np.arange(w)[None, :] * np.ones((h, 1), np.int64)
    y = np.arange(h)[:, None] * np.ones((1, w), np.int64)
    return np.stack([x & 255, y & 255, (x >> 8) | ((y >> 8) << 4)], -1).astype(np.uint8)


def paint(canvas, img, rect):
    ys, xs, sy, sx = Q.blit_index(*rect, img.shape[1], img.shape[0], canvas.shape[1],
                                  canvas.shape[0])
    if ys.size == 0:
        return 0
    canvas[np.ix_(ys, xs)] = img[np.ix_(sy, sx)]
    return ys.size * xs.size


def predict(rec):
    game = rec["game"]
    W = H = 64
    mw, mh = rec["main"]
    unit, view_dim, x_off, y_off, _ = Q.prepare(mw, mh, False, (0, 0), 0.0, H)
    assert abs(unit - rec["unit"]) < 1e-9 and abs(view_dim - rec["view_dim"]) < 1e-9, (
        f"prepare_for_drawing mismatch: {unit} vs {rec['unit']}, {view_dim} vs {rec['view_dim']}")
    canvas = np.zeros((H, W, 3), np.uint8)
    skipped = []

    if rec["use_backgrounds"]:
        bgs = BG_LIST[game]()
        bgp = bgs[rec["background_index"]] if rec["background_index"] < len(bgs) else None
        if bgp is None:
            skipped.append(f"background_index {rec['background_index']} out of table")
        else:
            bw, bh = size(bgp)
            mr = Q.screen_rect(0, mh, mw, mh, unit, view_dim, x_off, y_off)
            bg_ar = np.float32(np.float32(bw) / np.float32(bh))
            world_ar = np.float32(mw * 1.0 / mh)
            off_x = np.float32(np.float32(rec["bg_pct_x"]) * np.float32(bg_ar - world_ar))
            ax, ay, aw, ah = -float(off_x), 0.0, float(np.float32(bg_ar / world_ar)), 1.0
            rect = (mr[0] + mr[2] * ax, mr[1] + mr[3] * ay, mr[2] * aw, mr[3] * ah)
            paint(canvas, coded_bg(bgp), rect)

    grid = np.asarray(rec["grid"], np.int32)
    ents = rec["entities"]

    def sprite_for(t, theme):
        t = IMG_FOR_TYPE.get(game, {}).get(t, t)
        if t >= 100:
            return None
        names = ASSETS[game].get(t)
        if names is None:
            return None
        return names[theme] if theme < len(names) else names[0]

    def draw_ents(z):
        n = 0
        for e in ents:
            if e["render_z"] != z:
                continue
            if game == "heist" and e["type"] == 11 and not rec["has_keys"][e["image_theme"]]:
                continue
            p = sprite_for(e["image_type"], e["image_theme"])
            if p is None:
                continue
            if e["rotation"] != 0.0:
                skipped.append(f"rotated entity type={e['type']} rot={e['rotation']}")
                continue
            if e["use_abs_coords"]:
                rect = Q.abs_rect(view_dim * (e["x"] - e["rx"]), view_dim * (e["y"] + e["ry"]),
                                  2 * view_dim * e["rx"], 2 * view_dim * e["ry"], unit)
            else:
                rect = Q.screen_rect(e["x"] - e["rx"], e["y"] + e["ry"], 2 * e["rx"], 2 * e["ry"],
                                     unit, view_dim, x_off, y_off)
            img = coded_sprite(p)
            if e["is_reflected"]:
                img = img[:, ::-1]
            n += paint(canvas, img, rect)
        return n

    draw_ents(-1)
    for x in range(mw):
        for y in range(mh):
            t = int(grid[y, x])
            p = sprite_for(t, 0)
            if p is None:
                continue
            rect = Q.screen_rect(x, y + 1, 1, 1, unit, view_dim, x_off, y_off, eps=0.02)
            paint(canvas, coded_sprite(p), rect)
    draw_ents(0)
    draw_ents(1)
    return canvas, skipped


def main():
    pats = sys.argv[1:] or ["coded_*.json", "crafted_*.json"]
    files = sorted(f for p in pats for f in glob.glob(os.path.join(HERE, p)))
    for path in files:
        rec = json.load(open(path))
        got = np.asarray(rec["frame"], np.uint8)
        pred, skipped = predict(rec)
        eq = (pred == got).all(-1)
        # a pixel is "explained" if id+sx+sy all match
        bad = np.argwhere(~eq)
        print(f"{os.path.basename(path):28s} exact {eq.mean()*100:6.2f}%  "
              f"({(~eq).sum()} bad)  skipped={skipped}")
        if bad.size:
            ids_bad = sorted({(int(got[r, c, 2]), int(pred[r, c, 2])) for r, c in bad[:400]})
            print(f"    (got_id, pred_id) pairs: {ids_bad[:12]}")
            for r, c in bad[:6]:
                print(f"    px({r},{c}) got={got[r,c].tolist()} pred={pred[r,c].tolist()}")


if __name__ == "__main__":
    main()
