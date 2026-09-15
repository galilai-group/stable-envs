"""Bake full Craftax's textures into ``stable_envs/envs/_craftax_full_tex.py``.

Every texture craftax loads is **integral in 0..255** — ``load_texture`` reads the PNG as int, forces
the alpha channel to ``alpha // 255`` (so every alpha is exactly 0 or 1) and resizes with
``Image.NEAREST``. Measured over all 47 keys at 10 px and 16 px: the only non-integral entry is
``night_noise_intensity_texture``, which this env builds at runtime because it is a function of the
frame size rather than an asset. So the bake is **uint8 and lossless**, a quarter the size of a
float32 one, and the loader widens to float32 at use.

The NEAREST resize is also why 20 px textures are not a 2x replication of the 10 px ones, which is the
mechanism behind the resolution-consistency oracle in ``findings.md`` §7.2.

    /Users/samib/se7-craftax-venv/bin/python scratch/step7/craftax-full/gen_consts.py
"""
import base64
import os
import sys
import zlib

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                   "stable_envs", "envs", "_craftax_full_tex.py")
SIZES = (10, 16)

#: 0 = nothing, then the three mob classes end to end: 8 melee, 3 passive, 8 ranged.
MOB_BASE = {"melee": 1, "passive": 9, "ranged": 12}
N_MOB_SLOT = 1 + 8 + 3 + 8
N_PROJ_SLOT = 1 + 8 * 4
#: ``(flip, transpose)`` in the order the renderer's two selects produce, for orientation o.
ORIENTS = ((False, False), (False, True), (True, False), (True, True))

ICONS = ("health_texture", "hunger_texture", "thirst_texture", "energy_texture", "mana_texture",
         "sapling_texture", "torch_inv_texture", "book_texture", "fireball_inv_texture",
         "iceball_inv_texture", "xp_texture", "dex_texture", "str_texture", "int_texture",
         "smaller_empty_texture")


def u8(a):
    a = np.asarray(a)
    if not np.array_equal(a, np.round(a)) or a.min() < 0 or a.max() > 255:
        raise ValueError(f"texture is not integral in 0..255: min {a.min()} max {a.max()}")
    return np.round(a).astype(np.uint8)


def blob(a):
    a = np.ascontiguousarray(u8(a))
    return base64.b64encode(zlib.compress(a.tobytes(), 9)).decode(), tuple(int(x) for x in a.shape)


def tables_for(t):
    from craftax.craftax import constants as C
    tex = C.load_all_textures_given_size(int(t))
    out = {}

    out["block"] = np.asarray(tex["block_textures"])                       # (37, t, t, 3)

    it = np.asarray(tex["full_map_item_textures"])[:, :t, :t]              # (5, t, t, 4), tiled
    ia = it[..., 3].copy()
    # ItemType.NONE is slot 0 and the reference's scan starts at 1, so it is NEVER drawn; its texture
    # is debug.png and is opaque. Zeroing its alpha here makes slot 0 a no-op in the gather instead
    # of a special case in the renderer -- see findings.md §3.1.
    ia[int(C.ItemType.NONE.value)] = 0
    out["item_rgb"], out["item_a"] = it[..., :3], ia

    pl = np.asarray(tex["player_textures"])                               # (5, t, t, 4)
    out["player_rgb"], out["player_a"] = pl[..., :3], pl[..., 3]

    mr = np.zeros((N_MOB_SLOT, t, t, 3), np.int64)
    ma = np.zeros((N_MOB_SLOT, t, t), np.int64)
    for name in ("melee", "passive", "ranged"):
        r = np.asarray(tex[f"{name}_mob_textures"])
        a = np.asarray(tex[f"{name}_mob_texture_alphas"])[..., 0]
        b = MOB_BASE[name]
        mr[b:b + r.shape[0]], ma[b:b + a.shape[0]] = r, a
    out["mob_rgb"], out["mob_a"] = mr, ma

    pr = np.asarray(tex["projectile_textures"])
    pa = np.asarray(tex["projectile_texture_alphas"])[..., 0]
    jr = np.zeros((N_PROJ_SLOT, t, t, 3), np.int64)
    ja = np.zeros((N_PROJ_SLOT, t, t), np.int64)
    for ty in range(pr.shape[0]):
        for o, (flip, trans) in enumerate(ORIENTS):
            r, a = pr[ty], pa[ty]
            if flip:
                r, a = r[::-1], a[::-1]
            if trans:
                r, a = r.transpose(1, 0, 2), a.T
            jr[1 + ty * 4 + o], ja[1 + ty * 4 + o] = r, a
    out["proj_rgb"], out["proj_a"] = jr, ja

    # ---- strip ingredients
    out["small_block"] = np.asarray(tex["smaller_block_textures"])
    for k in ("number_textures", "number_textures_with_zero"):
        out[k] = np.asarray(tex[k])
        out[k + "_a"] = np.asarray(tex[k.replace("_textures", "_textures_alpha")
                                   if k.endswith("_with_zero") else "number_textures_alpha"])[..., 0]
    out["number_textures_a"] = np.asarray(tex["number_textures_alpha"])[..., 0]
    out["number_textures_with_zero_a"] = np.asarray(tex["number_textures_alpha_with_zero"])[..., 0]
    out["icons"] = np.stack([np.asarray(tex[k]) for k in ICONS])
    out["arrow_icon"] = np.asarray(tex["player_projectile_textures"])[0]
    for k in ("pickaxe_textures", "sword_textures", "bow_textures", "potion_textures"):
        out[k] = np.asarray(tex[k])
    out["armour"] = np.asarray(tex["armour_textures"])                     # (3, 4, s, s, 3)
    ae = np.asarray(tex["armour_enchantment_textures"])                    # (3, 4, s, s, 4)
    out["armour_ench_rgb"], out["armour_ench_a"] = ae[..., :3], ae[..., 3]
    se = np.asarray(tex["sword_enchantment_textures"])
    out["sword_ench_rgb"], out["sword_ench_a"] = se[..., :3], se[..., 3]
    re_ = np.asarray(tex["arrow_enchantment_textures"])
    out["arrow_ench_rgb"], out["arrow_ench_a"] = re_[..., :3], re_[..., 3]
    return out


def main():
    from importlib import metadata
    import craftax  # noqa: F401
    ver = metadata.version("craftax")
    lines = [
        '"""Full-Craftax texture tables, baked so the renderer imports no jax and no craftax.',
        "",
        f"Generated by ``scratch/step7/craftax-full/gen_consts.py`` from **craftax {ver}** at tile",
        f"sizes {SIZES}. uint8 and LOSSLESS: craftax's ``load_texture`` reads each PNG as int, forces",
        "the alpha channel to ``alpha // 255`` so every alpha is exactly 0 or 1, and resizes with",
        "``Image.NEAREST``; measured over all 47 texture keys at both sizes, the only non-integral",
        "entry is ``night_noise_intensity_texture``, which is a function of the frame size rather than",
        "an asset and is built at runtime by :func:`stable_envs.envs.craftax_full.night_vignette`.",
        "",
        "Do not hand-edit. Re-run the generator against a new craftax and re-measure parity.",
        '"""',
        "from __future__ import annotations",
        "",
        f'CRAFTAX_VERSION = "{ver}"',
        f"SIZES = {SIZES!r}",
        f"MOB_BASE = {MOB_BASE!r}",
        f"N_MOB_SLOT = {N_MOB_SLOT}",
        f"N_PROJ_SLOT = {N_PROJ_SLOT}",
        "",
        "#: ``{tile_px: {name: (base64(zlib(uint8 bytes)), shape)}}``",
        "TABLES = {",
    ]
    total = 0
    for t in SIZES:
        tabs = tables_for(t)
        lines.append(f"    {t}: {{")
        for name in sorted(tabs):
            b, shape = blob(tabs[name])
            total += len(b)
            lines.append(f'        "{name}": (')
            for i in range(0, len(b), 110):
                lines.append(f'            "{b[i:i + 110]}"')
            lines.append(f"        , {shape!r}),")
        lines.append("    },")
    lines.append("}")
    lines.append("")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT} ({os.path.getsize(OUT):,} bytes, {total:,} base64 chars)")


if __name__ == "__main__":
    sys.exit(main())
