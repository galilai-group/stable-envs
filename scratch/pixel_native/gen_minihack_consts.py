"""Emit the embedded constants for ``stable_envs/envs/minihack.py``.

The env has to render without NetHack installed -- that is the whole point of a state dataset -- so the
official tileset and the glyph->tile table are extracted once here and embedded in the module as
zlib+base64 blobs.  Nothing about them is reconstructed or guessed: both come straight out of
``minihack.tiles``.

Also emitted, and also measured rather than assumed:

  * the glyph vocabulary ``sample_states`` draws from.  The conformance suite perturbs each state
    dimension by +25 and calls a dimension dead if the frame does not change, so every glyph the sampler
    can emit must satisfy ``tile(g) != tile(g+25)``.  That is a property of NetHack's glyph->tile table
    (5976 glyphs onto 1023 tiles, so collisions are common), not something to hope for;
  * which of those glyphs render as pure black, since the ``background_color`` variation recolours
    exactly those pixels and would be inert if the sampler never emitted one.

Run under SLURM with the nle venv.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np


def wrap(b64: str, width: int = 116, indent: str = "    ") -> str:
    lines = [b64[i:i + width] for i in range(0, len(b64), width)]
    return "\n".join(f'{indent}"{ln}"' for ln in lines)


def main():
    from minihack.tiles import glyph2tile
    from minihack.tiles.glyph_mapper import GlyphMapper
    from nle import nethack

    gm = GlyphMapper()
    raw = gm.tiles
    items = list(raw.values()) if isinstance(raw, dict) else list(raw)
    atlas = np.stack([np.asarray(t, dtype=np.uint8) for t in items])
    g2t = np.asarray(glyph2tile, dtype=np.int16)
    n_glyph = int(nethack.MAX_GLYPH)
    assert g2t.shape == (n_glyph,), (g2t.shape, n_glyph)
    assert atlas.shape[1:] == (16, 16, 3), atlas.shape

    # glyph ids NetHack actually uses for the pieces of a MiniHack room, read out of nle rather than
    # written down: the glyph number for a monster/object/dungeon-feature is an offset into a class
    # block, and the block bases move between NetHack builds.
    S = nethack.GLYPH_CMAP_OFF
    named = {
        "stone": nethack.GLYPH_CMAP_OFF + 0,          # S_stone, the unlit rock outside the room
        "vwall": S + 1, "hwall": S + 2,
        "tlcorn": S + 3, "trcorn": S + 4, "blcorn": S + 5, "brcorn": S + 6,
        "room": S + 19, "corr": S + 21,
        "upstair": S + 22, "dnstair": S + 23,
        "fountain": S + 31, "door_vert": S + 15, "door_horiz": S + 16,
    }
    mons = {f"mon{i}": nethack.GLYPH_MON_OFF + i for i in (0, 3, 12, 20, 41, 63, 100, 200)}
    objs = {f"obj{i}": nethack.GLYPH_OBJ_OFF + i for i in (0, 5, 20, 60, 120, 200, 300)}
    pets = {"pet0": nethack.GLYPH_PET_OFF + 0}
    cand = {**named, **mons, **objs, **pets}

    def tile(g):
        return int(g2t[g])

    ok, rejected = {}, {}
    for k, g in cand.items():
        g = int(g)
        if g + 25 >= n_glyph or tile(g) == tile(g + 25):
            rejected[k] = (g, tile(g), tile(min(g + 25, n_glyph - 1)))
        else:
            ok[k] = g

    black = {k: bool((atlas[tile(g)] == 0).all()) for k, g in ok.items()}
    print("accepted glyphs (tile(g) != tile(g+25)):")
    for k, g in ok.items():
        print(f"  {k:<10} glyph {g:5d} tile {tile(g):5d} -> tile(g+25) {tile(g+25):5d}  "
              f"all-black={black[k]}")
    if rejected:
        print("rejected (a +25 perturbation would not change the tile):")
        for k, v in rejected.items():
            print(f"  {k:<10} glyph {v[0]:5d} tile {v[1]} == {v[2]}")

    # the background glyph must be accepted, because every batch element shares it in the corners and a
    # dead dimension there is reported as dead overall
    assert "stone" in ok, "the fill glyph must change tile under +25 or the dim-coverage test fails"
    assert any(black.values()), "no sampled glyph renders black; background_color would be inert"

    a_b64 = base64.b64encode(zlib.compress(atlas.tobytes(), 9)).decode()
    g_b64 = base64.b64encode(zlib.compress(g2t.tobytes(), 9)).decode()

    out = []
    out.append(f"N_TILE = {atlas.shape[0]}")
    out.append("TILE_PX = 16")
    out.append(f"N_GLYPH = {n_glyph}")
    out.append(f"GRID_H, GRID_W = 21, 79")
    out.append("")
    out.append("#: measured glyph ids, see scratch/pixel_native/gen_minihack_consts.py")
    for k, g in ok.items():
        out.append(f"G_{k.upper()} = {g}")
    out.append("")
    out.append("_ATLAS_B64 = (")
    out.append(wrap(a_b64))
    out.append(")")
    out.append("")
    out.append("_GLYPH2TILE_B64 = (")
    out.append(wrap(g_b64))
    out.append(")")
    p = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/pixel_native/minihack_consts.py.txt"
    with open(p, "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"\natlas b64 {len(a_b64)} chars, glyph2tile b64 {len(g_b64)} chars")
    print("wrote", p)


if __name__ == "__main__":
    main()
