"""Extract crafter's art and its resampler into the constants stable_envs/envs/crafter.py embeds.

Two blobs and one verification:

  _TEX_B64     the 56 assets at their native 16x16, RGBA, in the order of ``TEX_NAMES``.
  _NEAREST_B64 (64, 64) uint8: row ``d-1`` holds the source index each of the ``d`` output pixels of
               ``PIL.Image.resize((d,d), NEAREST)`` takes from a 16-pixel source. MEASURED off PIL
               rather than derived: the obvious rule ``floor((i+0.5)*16/d)`` disagrees at 34 of the
               first 64 sizes because Pillow divides before it multiplies and accumulates.

The verification is the point of the script: every tile of the composed atlas is rebuilt here with
crafter's OWN ``_draw`` / ``_draw_alpha`` / ``ItemView`` and compared against the pure-numpy
composition the env ships, at every unit size from 3 to 40.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np
from PIL import Image

import crafter
from crafter import constants, engine

env = crafter.Env(seed=0)
tex = env._textures
names = sorted(tex._originals)
N = len(names)

rgba = np.zeros((N, 16, 16, 4), np.uint8)
for i, n in enumerate(names):
    a = tex._originals[n]
    assert a.shape[:2] == (16, 16), (n, a.shape)
    rgba[i, ..., :a.shape[-1]] = a
    if a.shape[-1] == 3:
        rgba[i, ..., 3] = 255
CH = np.array([tex._originals[n].shape[-1] for n in names], np.uint8)

MAXD = 64
nearest = np.zeros((MAXD, MAXD), np.uint8)
row = (np.arange(16, dtype=np.uint8)[:, None] * np.ones(16, np.uint8)).astype(np.uint8)
col = row.T
for d in range(1, MAXD + 1):
    r = np.array(Image.fromarray(row).resize((d, d), resample=Image.NEAREST))[:, 0]
    c = np.array(Image.fromarray(col).resize((d, d), resample=Image.NEAREST))[0, :]
    assert np.array_equal(r, c), d
    nearest[d - 1, :d] = r


def enc(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def wrap(s: str, per: int = 116) -> str:
    return "\n".join(f'    "{s[i:i + per]}"' for i in range(0, len(s), per))


# ------------------------------------------------------------------ the composition, in pure numpy

def resample(i: int, d: int) -> np.ndarray:
    """asset ``i`` at ``d`` x ``d``, RGBA, and whether it carries a real alpha channel."""
    ix = nearest[d - 1, :d].astype(np.int64)
    return rgba[i][ix][:, ix]


def draw(canvas, t, has_alpha):
    canvas[: t.shape[0], : t.shape[1]] = t[..., :3]


def draw_alpha(canvas, off, t, has_alpha):
    x, y = off
    w, h = t.shape[:2]
    if not has_alpha:
        canvas[x:x + w, y:y + h] = t[..., :3]
        return
    a = t[..., 3:].astype(np.float32) / 255
    c = t[..., :3].astype(np.float32) / 255
    cur = canvas[x:x + w, y:y + h].astype(np.float32) / 255
    canvas[x:x + w, y:y + h] = (255 * (a * c + (1 - a) * cur)).astype(np.uint8)


MATS = [None] + list(constants.materials)
OBJTEX = [None, "player-left", "player-right", "player-up", "player-down", "player-sleep",
          "cow", "zombie", "skeleton", "arrow-left", "arrow-right", "arrow-up", "arrow-down",
          "plant", "plant-ripe", "fence"]
ITEMS = list(constants.items)
IDX = {n: i for i, n in enumerate(names)}


def world_atlas(unit):
    out = np.zeros((len(MATS) * len(OBJTEX) + 1, unit, unit, 3), np.uint8)
    for m, mat in enumerate(MATS):
        mt = resample(IDX["unknown" if mat is None else mat], unit)
        for o, ob in enumerate(OBJTEX):
            c = np.zeros((unit, unit, 3), np.uint8) + 127
            draw(c, mt, True)
            if o:
                draw_alpha(c, (0, 0), resample(IDX[ob], unit), CH[IDX[ob]] == 4)
            out[m * len(OBJTEX) + o] = c
    out[len(MATS) * len(OBJTEX)] = 127
    return out


def item_atlas(unit):
    ui, ua = int(0.8 * unit), int(0.6 * unit)
    oi, oa = int(0.1 * unit), int(0.4 * unit)
    out = np.zeros((len(ITEMS) * 10 + 1, unit, unit, 3), np.uint8)
    for i, name in enumerate(ITEMS):
        for a in range(1, 10):
            c = np.zeros((unit, unit, 3), np.uint8)
            if ui:
                draw_alpha(c, (oi, oi), resample(IDX[name], ui), CH[IDX[name]] == 4)
            if ua:
                draw_alpha(c, (oa, oa), resample(IDX[str(a)], ua), CH[IDX[str(a)]] == 4)
            out[i * 10 + a] = c
    return out


# ------------------------------------------------------------------ verify against crafter's own code

worst = 0
for unit in range(3, 41):
    u = np.array([unit, unit])
    ours_w = world_atlas(unit)
    ours_i = item_atlas(unit)
    view = engine.ItemView(tex, [1, 1])
    for m, mat in enumerate(MATS):
        for o, ob in enumerate(OBJTEX):
            c = np.zeros((unit, unit, 3), np.uint8) + 127
            engine._draw(c, np.array([0, 0]), tex.get(mat, u))
            if o:
                engine._draw_alpha(c, np.array([0, 0]), tex.get(ob, u))
            worst = max(worst, int(np.abs(c.astype(int) - ours_w[m * len(OBJTEX) + o].astype(int)).max()))
    for i, name in enumerate(ITEMS):
        for a in range(1, 10):
            c = view({name: a}, u)
            worst = max(worst, int(np.abs(c.astype(int) - ours_i[i * 10 + a].astype(int)).max()))
print(f"atlas composition vs crafter's own _draw/_draw_alpha/ItemView, units 3-40: max abs diff {worst}")

blob_t, blob_n = enc(rgba), enc(nearest)
print(f"texture blob {len(blob_t)} b64 chars, nearest blob {len(blob_n)} b64 chars")
with open("scratch/crafter/assets_gen.py", "w") as f:
    f.write(f"TEX_NAMES = {tuple(names)!r}\n\n")
    f.write(f"TEX_HAS_ALPHA = {tuple(int(c == 4) for c in CH)!r}\n\n")
    f.write(f"MATERIALS = {tuple(constants.materials)!r}\n\n")
    f.write(f"ITEMS = {tuple(ITEMS)!r}\n\n")
    f.write(f"_TEX_SHAPE = {rgba.shape!r}\n_TEX_B64 = (\n{wrap(blob_t)}\n)\n\n")
    f.write(f"_NEAREST_SHAPE = {nearest.shape!r}\n_NEAREST_B64 = (\n{wrap(blob_n)}\n)\n")
print("wrote scratch/crafter/assets_gen.py")
