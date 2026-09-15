"""Emit the Step-7 embedded textures for ``stable_envs/envs/craftax.py``.

Follows ``scratch/pixel_native/gen_craftax_consts.py`` — same ``zlib+base64`` float32 blobs, same
"assert the assumption before emitting the table" discipline. What is new:

  * the player tables gain a **fifth** entry, ``player-sleep.png``. The renderer selects
    ``player_texture_index = select(is_sleeping, 4, player_direction - 1)``, so a sleeping player is
    a different SPRITE and not only a post-process. Asserted zero away from the centre tile, like
    the four facing sprites, so the one-tile composite stays the same arithmetic;
  * three tables for the inventory strip — 16 icons, 10 premultiplied digits, 10 digit alphas —
    from which a ``(16, 10)`` composited cell table is built at import time.

Assertions that gate the emission (a wrong one fails here, not in a parity run):

  * the strip's per-cell writes stay inside their own tile and no two cells overlap;
  * the rebuilt strip is **bit-identical** to the reference's own ``inv_pixels`` on real states;
  * the alphas are still {0, 128/255, 1}.

Run in the reference venv: `/Users/samib/se7-craftax-venv/bin/python`.
"""
from __future__ import annotations

import base64
import os
import pickle
import zlib

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

OUT = "scratch/step7/craftax-classic-full/step7_consts.py.txt"
STATES = "scratch/step7/craftax-classic-full/states.pkl"

#: (row, col, accessor path, icon key) — the reference's own draw order and layout.
CELLS = [
    (0, 0, ("player_health",), "health_texture"),
    (0, 1, ("player_food",), "hunger_texture"),
    (0, 2, ("player_drink",), "thirst_texture"),
    (0, 3, ("player_energy",), "energy_texture"),
    (0, 4, ("inventory", "sapling"), "sapling_texture"),
    (0, 5, ("inventory", "wood"), ("smaller_block_textures", 6)),
    (0, 6, ("inventory", "stone"), ("smaller_block_textures", 4)),
    (0, 7, ("inventory", "coal"), ("smaller_block_textures", 8)),
    (0, 8, ("inventory", "iron"), ("smaller_block_textures", 9)),
    (1, 0, ("inventory", "diamond"), ("smaller_block_textures", 10)),
    (1, 1, ("inventory", "wood_pickaxe"), "wood_pickaxe_texture"),
    (1, 2, ("inventory", "stone_pickaxe"), "stone_pickaxe_texture"),
    (1, 3, ("inventory", "iron_pickaxe"), "iron_pickaxe_texture"),
    (1, 4, ("inventory", "wood_sword"), "wood_sword_texture"),
    (1, 5, ("inventory", "stone_sword"), "stone_sword_texture"),
    (1, 6, ("inventory", "iron_sword"), "iron_sword_texture"),
]


def blob(a: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a).tobytes(), 9)).decode()


def wrap(b64: str, width: int = 116, indent: str = "    ") -> str:
    return "\n".join(f'{indent}"{b64[i:i + width]}"' for i in range(0, len(b64), width))


def geometry(bps: int):
    ls = (bps - int(0.8 * bps)) // 2 - 1
    rs = bps - int(0.8 * bps) - ls
    nsz = int(bps * 0.6)
    return ls, bps - rs, bps - nsz - 1, bps - 1, nsz


def extract(T, bps, H, W):
    icon_lo, icon_hi, num_lo, num_hi, nsz = geometry(bps)
    small = int(0.8 * bps)
    assert icon_hi - icon_lo == small, (bps, icon_lo, icon_hi, small)
    assert num_hi - num_lo == nsz, (bps, num_lo, num_hi, nsz)
    assert num_lo >= 0 and num_hi <= bps, (bps, num_lo, num_hi)

    # --- player, now five sprites
    pf = T["full_map_player_textures"].astype(np.float32)
    pa = T["full_map_player_textures_alpha"].astype(np.float32)
    assert pf.shape[0] == 5, f"expected 5 player textures, got {pf.shape[0]}"
    cr, cc = H // 2, W // 2
    mask = np.ones((H * bps, W * bps), bool)
    mask[cr * bps:(cr + 1) * bps, cc * bps:(cc + 1) * bps] = False
    assert not pf[:, mask].any() and not pa[:, mask].any(), (
        f"player tables non-zero away from the centre tile at t={bps}; the per-tile blit is not "
        f"equivalent to the full-frame one and slot 4 cannot be added this way")
    sl = (slice(cr * bps, (cr + 1) * bps), slice(cc * bps, (cc + 1) * bps))
    ptex, palpha = pf[:, sl[0], sl[1]], pa[:, sl[0], sl[1]]

    # --- strip ingredients
    icons = []
    for (_r, _c, _p, key) in CELLS:
        a = np.asarray(T[key[0]][key[1]] if isinstance(key, tuple) else T[key], np.float32)
        assert a.shape == (small, small, 3), (key, a.shape, small)
        icons.append(a)
    icons = np.stack(icons)
    digits = np.asarray(T["number_textures"], np.float32)
    dalpha = np.asarray(T["number_textures_alpha"], np.float32)
    assert digits.shape == (10, nsz, nsz, 3) and dalpha.shape == digits.shape, digits.shape
    assert not digits[0].any() and not dalpha[0].any(), "digit 0 must be the empty texture"
    assert set(np.unique(dalpha).tolist()) <= {0.0, 1.0}, "digit alphas are meant to be binary"

    vals = sorted(set(np.unique(palpha).tolist()))
    assert set(vals) <= {0.0, float(np.float32(128 / 255)), 1.0}, (bps, vals)
    return ptex, palpha, icons, digits, dalpha


def build_strip(icons, digits, dalpha, bps):
    """``(16, 10, bps, bps, 3)`` — the reference's per-cell arithmetic, once, at build time."""
    icon_lo, icon_hi, num_lo, num_hi, _ = geometry(bps)
    out = np.zeros((len(CELLS), 10, bps, bps, 3), np.float32)
    for c in range(len(CELLS)):
        for v in range(10):
            tile = np.zeros((bps, bps, 3), np.float32)
            if v > 0:
                tile[icon_lo:icon_hi, icon_lo:icon_hi] = icons[c]
            sl = (slice(num_lo, num_hi), slice(num_lo, num_hi))
            tile[sl] = tile[sl] * (1 - dalpha[v]) + digits[v]
            out[c, v] = tile
    return out


def check_disjoint(bps, W):
    icon_lo, icon_hi, num_lo, num_hi, _ = geometry(bps)
    occ = np.zeros((2 * bps, W * bps), np.int32)
    for (r, c, _p, _i) in CELLS:
        occ[r * bps + icon_lo:r * bps + icon_hi, c * bps + icon_lo:c * bps + icon_hi] += 1
        occ[r * bps + num_lo:r * bps + num_hi, c * bps + num_lo:c * bps + num_hi] += 1
    assert occ.max() <= 2, (
        f"strip cells overlap at t={bps} (max {occ.max()} writers); the per-cell table is not a "
        f"valid factorisation of the reference's sequential .at[] writes")


def main():
    import jax
    from craftax.craftax_classic.constants import INVENTORY_OBS_HEIGHT, OBS_DIM, load_all_textures
    from craftax.craftax_classic.renderer import make_craftax_pixel_renderer

    ALL = load_all_textures()
    H, W = OBS_DIM
    assert INVENTORY_OBS_HEIGHT == 2, INVENTORY_OBS_HEIGHT
    states = pickle.load(open(STATES, "rb")) if os.path.isfile(STATES) else []

    out, total = [], 0
    for bps in (7, 16):
        T = {k: np.asarray(v) for k, v in ALL[bps].items()}
        ptex, palpha, icons, digits, dalpha = extract(T, bps, H, W)
        check_disjoint(bps, W)

        # the emission gate: the rebuilt strip must be the reference's own inv_pixels, bit for bit
        atlas = build_strip(icons, digits, dalpha, bps)
        render = make_craftax_pixel_renderer(bps)
        n_ok = 0
        for st in states:
            frame = np.asarray(jax.device_get(render(st)))
            want = frame[H * bps:]
            got = np.zeros_like(want)
            for ci, (r, c, path, _i) in enumerate(CELLS):
                o = st
                for p in path:
                    o = getattr(o, p)
                v = int(np.asarray(jax.device_get(o)))
                got[r * bps:(r + 1) * bps, c * bps:(c + 1) * bps] = atlas[ci, min(v, 9)]
            n_ok += int(np.array_equal(np.ascontiguousarray(got).view(np.uint32),
                                       np.ascontiguousarray(want).view(np.uint32)))
        assert not states or n_ok == len(states), (
            f"t={bps}: rebuilt strip differs from the reference on {len(states) - n_ok} of "
            f"{len(states)} real states")
        print(f"  t={bps}: strip bit-exact on {n_ok}/{len(states)} real states; "
              f"atlas {atlas.size * 4} B float32, ingredients "
              f"{(icons.size + digits.size + dalpha.size) * 4} B")

        for nm, arr in [("PLAYER_TEX", ptex), ("PLAYER_ALPHA", palpha),
                        ("STRIP_ICON", icons), ("STRIP_DIGIT", digits),
                        ("STRIP_DIGIT_ALPHA", dalpha)]:
            key = f"_{nm}_{bps}"
            b = blob(arr.astype(np.float32))
            total += len(b)
            out += [f"{key}_SHAPE = {tuple(arr.shape)}", f"{key}_B64 = (", wrap(b), ")", ""]

    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"embedded: {total} b64 chars -> {OUT}")


if __name__ == "__main__":
    main()
