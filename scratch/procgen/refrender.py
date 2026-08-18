"""Numpy reference renderer for procgen maze / heist / miner. The readable definition.

Every rule here was measured against the shipped binary, not read off the source:

  * the axis-aligned blit (`qtblit.blit_index`) -- settled by replacing every asset with a
    coordinate-coded image and reading which source pixel each device pixel sampled
    (`probe02`/`probe03`, 100% of 4096 px on 7 scenes across the 3 games);
  * the rotated blit -- coverage is "device pixel CENTRE inside the rotated quad" (539/539 pixels,
    0 false positives) and sampling is Qt's `fetchTransformed` fixed-point walk with NO half-point
    bias (539/539; the half-point variant scores 28.9%), `probe07`;
  * `qPremultiply` + SourceOver on the RGB32 buffer -- settled by the full-frame diff in `probe05`,
    which is bit-exact on maze and miner.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

import qtblit as Q

F = np.float32
PI_F = F(3.14159265358979323846264338327950288)

SRC_ROOT = ("/mnt/vast/home/randall.balestriero/miniconda3/envs/test/lib/python3.11/"
            "site-packages/procgen/data/assets")

#: `asset_for_type`, transcribed per game. type -> theme list.
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
IMG_FOR_TYPE = {"miner": {3: 1, 4: 2}}

TOPDOWN_BG = [f"topdown_backgrounds/{n}" for n in
              ["floortiles.png", "backgrounddetailed1.png", "backgrounddetailed2.png",
               "backgrounddetailed3.png", "backgrounddetailed4.png", "backgrounddetailed5.png",
               "backgrounddetailed6.png", "backgrounddetailed7.png", "backgrounddetailed8.png"]]
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
BG = {"maze": TOPDOWN_BG, "heist": TOPDOWN_BG, "miner": PLATFORM_BG + SPACE_BG}

_C: dict = {}


def sprite(path):
    """Premultiplied RGBA, exactly as `convertToFormat(ARGB32_Premultiplied)` leaves it."""
    k = ("s", path)
    if k not in _C:
        a = np.asarray(Image.open(os.path.join(SRC_ROOT, path)).convert("RGBA"), np.uint8)
        _C[k] = Q.premultiply(a.copy())
    return _C[k]


def background(path):
    """`convertToFormat(RGB32)`: RGB kept, alpha forced opaque."""
    k = ("b", path)
    if k not in _C:
        _C[k] = np.asarray(Image.open(os.path.join(SRC_ROOT, path)).convert("RGB"), np.uint8)
    return _C[k]


# ------------------------------------------------------------------ the rotated path

def _rot(deg):
    if deg in (90.0, -270.0):
        return 1.0, 0.0
    if deg in (270.0, -90.0):
        return -1.0, 0.0
    if deg in (180.0, -180.0):
        return 0.0, -1.0
    b = np.deg2rad(np.float64(deg))
    return float(np.sin(b)), float(np.cos(b))


def src_to_dev(rect, deg, sw, sh):
    rx, ry, rw, rh = rect
    sina, cosa = _rot(deg)
    m11, m12, m21, m22 = cosa, sina, -sina, cosa
    dx, dy = rx + rw / 2.0, ry + rh / 2.0
    tx, ty = -rw / 2.0, -rh / 2.0
    dx += tx * m11 + ty * m21
    dy += tx * m12 + ty * m22
    m11, m12 = m11 * (rw / sw), m12 * (rw / sw)
    m21, m22 = m21 * (rh / sh), m22 * (rh / sh)
    return np.array([[m11, m12], [m21, m22]]), np.array([dx, dy])


def rotated_blit(canvas, img, rect, deg):
    H, W = canvas.shape[:2]
    sh, sw = img.shape[:2]
    M, T = src_to_dev(rect, deg, sw, sh)
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64) @ M + T
    s = sum((corners[(i + 1) % 4, 0] - corners[i, 0]) * (corners[(i + 1) % 4, 1] + corners[i, 1])
            for i in range(4))
    sgn = 1.0 if s < 0 else -1.0
    ys, xs = np.mgrid[0:H, 0:W]
    px, py = xs + 0.5, ys + 0.5
    inside = np.ones((H, W), bool)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        inside &= (((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) * sgn) >= 0
    if not inside.any():
        return
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    inv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
    iT = -T @ inv
    fx = ((inv[1, 0] * py + inv[0, 0] * px + iT[0]) * 65536).astype(np.int64) >> 16
    fy = ((inv[1, 1] * py + inv[0, 1] * px + iT[1]) * 65536).astype(np.int64) >> 16
    ok = inside & (fx >= 0) & (fx < sw) & (fy >= 0) & (fy < sh)
    src = img[np.clip(fy, 0, sh - 1), np.clip(fx, 0, sw - 1)]
    a = np.where(ok, src[..., 3], 0)
    canvas[...] = Q.source_over(canvas, np.where(ok[..., None], src[..., :3], 0), a)


def blit(canvas, img, rect, opaque=False):
    ys, xs, sy, sx = Q.blit_index(*rect, img.shape[1], img.shape[0], canvas.shape[1],
                                  canvas.shape[0])
    if ys.size == 0:
        return
    src = img[np.ix_(sy, sx)]
    sl = np.ix_(ys, xs)
    canvas[sl] = src if opaque else Q.source_over(canvas[sl], src[..., :3], src[..., 3])


# ------------------------------------------------------------------ the whole frame

def render(game, grid, entities, background_index, bg_pct_x, has_keys=(), size=64,
           use_backgrounds=True):
    """`game_draw` for the three scoped games. `grid` is (h, w) int32; `entities` a list of dicts."""
    H = W = size
    mh, mw = grid.shape
    unit, view_dim, x_off, y_off, _ = Q.prepare(mw, mh, False, (0, 0), 0.0, H)
    canvas = np.zeros((H, W, 3), np.uint8)

    if use_backgrounds:
        img = background(BG[game][int(background_index)])
        bh, bw = img.shape[:2]
        mr = Q.screen_rect(0, mh, mw, mh, unit, view_dim, x_off, y_off)
        bg_ar = F(F(bw) / F(bh))
        world_ar = F(mw * 1.0 / mh)
        off_x = F(F(bg_pct_x) * F(bg_ar - world_ar))
        blit(canvas, img, (mr[0] + mr[2] * -float(off_x), mr[1],
                           mr[2] * float(F(bg_ar / world_ar)), mr[3]), opaque=True)

    def path_for(t, theme):
        t = IMG_FOR_TYPE.get(game, {}).get(int(t), int(t))
        if t >= 100:
            return None
        names = ASSETS[game].get(t)
        return None if names is None else names[min(int(theme), len(names) - 1)]

    def ents(z):
        for e in entities:
            if e["render_z"] != z:
                continue
            if game == "heist" and e["type"] == 11 and not has_keys[e["image_theme"]]:
                continue
            p = path_for(e["image_type"], e["image_theme"])
            if p is None:
                continue
            if e["use_abs_coords"]:
                rect = Q.abs_rect(view_dim * (e["x"] - e["rx"]), view_dim * (e["y"] + e["ry"]),
                                  2 * view_dim * e["rx"], 2 * view_dim * e["ry"], unit)
            else:
                rect = Q.screen_rect(e["x"] - e["rx"], e["y"] + e["ry"], 2 * e["rx"], 2 * e["ry"],
                                     unit, view_dim, x_off, y_off)
            img = sprite(p)
            if e["is_reflected"]:
                img = img[:, ::-1]
            if e["rotation"] != 0.0:
                rotated_blit(canvas, img, rect, float(F(F(e["rotation"]) * F(180)) / PI_F))
            else:
                blit(canvas, img, rect)

    ents(-1)
    for x in range(mw):
        for y in range(mh):
            p = path_for(grid[y, x], 0)
            if p is not None:
                blit(canvas, sprite(p),
                     Q.screen_rect(x, y + 1, 1, 1, unit, view_dim, x_off, y_off, eps=0.02))
    ents(0)
    ents(1)
    return canvas
