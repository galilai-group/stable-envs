"""The decisive test: predict the REAL 64x64 observation, byte for byte, from the parsed state.

The coordinate-coded probe settled the sampling rule. What is left is the colour path -- how Qt
decodes each PNG, how `convertToFormat(ARGB32_Premultiplied)` premultiplies it, and how SourceOver
composites it onto the RGB32 buffer -- and this asks that question against the shipped renderer with
the shipped assets.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import qtblit as Q  # noqa: E402
import state_io as S  # noqa: E402
import probe03_fit_blit as P3  # noqa: E402

SRC_ROOT = P3.SRC_ROOT
_RGBA: dict = {}


def rgba(path):
    """The sprite as Qt sees it: RGBA8888, then `qPremultiply`."""
    if path not in _RGBA:
        im = Image.open(os.path.join(SRC_ROOT, path))
        a = np.asarray(im.convert("RGBA"), np.uint8)
        _RGBA[path] = Q.premultiply(a.copy())
    return _RGBA[path]


def bg_rgb(path):
    """A background as Qt sees it: `convertToFormat(RGB32)`, i.e. RGB kept, alpha forced opaque."""
    im = Image.open(os.path.join(SRC_ROOT, path))
    return np.asarray(im.convert("RGB"), np.uint8)


def paint_rgba(canvas, img, rect):
    ys, xs, sy, sx = Q.blit_index(*rect, img.shape[1], img.shape[0], canvas.shape[1],
                                  canvas.shape[0])
    if ys.size == 0:
        return
    src = img[np.ix_(sy, sx)]
    dst = canvas[np.ix_(ys, xs)]
    canvas[np.ix_(ys, xs)] = Q.source_over(dst, src[..., :3], src[..., 3])


def paint_opaque(canvas, img, rect):
    ys, xs, sy, sx = Q.blit_index(*rect, img.shape[1], img.shape[0], canvas.shape[1],
                                  canvas.shape[0])
    if ys.size == 0:
        return
    canvas[np.ix_(ys, xs)] = img[np.ix_(sy, sx)]


def render(rec, size=64):
    game = rec["game"]
    W = H = size
    mw, mh = rec["main"]
    unit, view_dim, x_off, y_off, _ = Q.prepare(mw, mh, False, (0, 0), 0.0, H)
    canvas = np.zeros((H, W, 3), np.uint8)
    notes = []

    if rec["use_backgrounds"]:
        bgs = P3.BG_LIST[game]()
        bgp = bgs[rec["background_index"]]
        img = bg_rgb(bgp)
        bh, bw = img.shape[:2]
        mr = Q.screen_rect(0, mh, mw, mh, unit, view_dim, x_off, y_off)
        bg_ar = np.float32(np.float32(bw) / np.float32(bh))
        world_ar = np.float32(mw * 1.0 / mh)
        off_x = np.float32(np.float32(rec["bg_pct_x"]) * np.float32(bg_ar - world_ar))
        ax, aw = -float(off_x), float(np.float32(bg_ar / world_ar))
        paint_opaque(canvas, img, (mr[0] + mr[2] * ax, mr[1], mr[2] * aw, mr[3]))

    grid = np.asarray(rec["grid"], np.int32)

    def sprite_for(t, theme):
        t = P3.IMG_FOR_TYPE.get(game, {}).get(t, t)
        if t >= 100:
            return None
        names = P3.ASSETS[game].get(t)
        return None if names is None else (names[theme] if theme < len(names) else names[0])

    def draw_ents(z):
        for e in rec["entities"]:
            if e["render_z"] != z:
                continue
            if game == "heist" and e["type"] == 11 and not rec["has_keys"][e["image_theme"]]:
                continue
            p = sprite_for(e["image_type"], e["image_theme"])
            if p is None:
                continue
            if e["rotation"] != 0.0:
                notes.append(f"rotated entity type={e['type']}")
                continue
            if e["use_abs_coords"]:
                rect = Q.abs_rect(view_dim * (e["x"] - e["rx"]), view_dim * (e["y"] + e["ry"]),
                                  2 * view_dim * e["rx"], 2 * view_dim * e["ry"], unit)
            else:
                rect = Q.screen_rect(e["x"] - e["rx"], e["y"] + e["ry"], 2 * e["rx"], 2 * e["ry"],
                                     unit, view_dim, x_off, y_off)
            img = rgba(p)
            if e["is_reflected"]:
                img = img[:, ::-1]
            paint_rgba(canvas, img, rect)

    draw_ents(-1)
    for x in range(mw):
        for y in range(mh):
            p = sprite_for(int(grid[y, x]), 0)
            if p is None:
                continue
            paint_rgba(canvas, rgba(p),
                       Q.screen_rect(x, y + 1, 1, 1, unit, view_dim, x_off, y_off, eps=0.02))
    draw_ents(0)
    draw_ents(1)
    return canvas, notes


def main():
    from procgen import ProcgenGym3Env
    games = sys.argv[1:] or ["maze", "heist", "miner"]
    rng = np.random.default_rng(0)
    for game in games:
        tot = worst = nbad = 0
        nframes = 0
        per_level = []
        for level in range(8):
            env = ProcgenGym3Env(num=1, env_name=game, distribution_mode="hard", rand_seed=0,
                                 start_level=level, num_levels=1, use_backgrounds=True)
            for step in range(6):
                env.act(rng.integers(0, 15, size=1).astype(np.int32))
                st = S.parse(env.get_state()[0])
                _, obs, _ = env.observe()
                got = np.asarray(obs["rgb"][0])
                rec = {
                    "game": game, "use_backgrounds": bool(st["use_backgrounds"]),
                    "main": [st["main_width"], st["main_height"]],
                    "background_index": st["background_index"], "bg_pct_x": st["bg_pct_x"],
                    "grid": st["grid"],
                    "entities": st["entities"],
                }
                if game == "heist":
                    rec["has_keys"] = st["has_keys"]
                pred, notes = render(rec)
                d = np.abs(pred.astype(np.int16) - got.astype(np.int16))
                nbad += int((d.max(-1) > 0).sum())
                tot += d.size // 3
                worst = max(worst, int(d.max()))
                nframes += 1
                if step == 0:
                    per_level.append((level, int((d.max(-1) > 0).sum()), int(d.max()),
                                      sorted(set(notes))))
        print(f"{game}: {nframes} frames, exact {(1 - nbad/tot)*100:.4f}% of pixels, "
              f"max|diff| {worst}, bad px {nbad}/{tot}")
        for lv, b, m, n in per_level:
            if b:
                print(f"    level {lv}: {b} bad px, max {m}, notes={n}")


if __name__ == "__main__":
    main()
