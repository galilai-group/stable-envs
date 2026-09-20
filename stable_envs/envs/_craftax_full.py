"""Full Craftax: nine levels, a 130x110 window, and a composite that is a stencil rather than a blend.

Reference: ``Craftax-Pixels-v1`` (JAX), which ``swm/CraftaxPixels-v1`` registers at
``BLOCK_PIXEL_SIZE_AGENT = 10``. The observation is
``craftax.craftax.renderer.make_craftax_pixel_renderer(10)(state) / 255`` — a 9x11 window of one of
nine 48x48 levels, an item layer over it, the player, up to eight mobs and six projectiles, a
per-tile ``light_map``, the day/night chain, a sleep dim, and a **4-row** inventory strip below.
130x110x3 at the registered size; 208x176x3 at craftax's 16 px size.

**This is a different renderer from Craftax Classic, and the difference is structural rather than a
matter of scale.** Every overlay alpha in the full game is exactly ``{0, 1}`` — craftax's
``load_texture`` forces ``alpha // 255``, measured over every alpha table at 10, 16 and 64 px — so a
layer is a **stencil select**, not a blend. Three things follow, and all three are why this file
exists instead of a generalisation of :mod:`stable_envs.envs.craftax`:

* **The product cannot be baked.** Classic's ``_Composite`` bakes ``(player, block, mob)`` into 816
  tiles because its alphas are ``{0, 128/255, 1}`` and a blend has no cheaper exact form. Here the
  product is ``block x item x player x mob x projectile`` = **732,600** tiles, 563 MB at 16 px. What
  is baked instead is ``block x item`` (185) and ``mob x projectile`` (660), and the player and the
  strip digits ride as two more thin stencil layers: **one background gather and three stencil
  layers**, which is seven ``index_select``s (a colour and a mask per layer) and three selects, four
  of them touching only part of the frame.
* **The lighting chain is not Classic's.** It has no luminance "enhance" step, it multiplies the
  per-tile ``light_map`` into the picture *before* anything else, it forces
  ``daylight = 1.0`` off level 0, and it sleeps towards black rather than towards a desaturated blue.
  Offered the reference's own inputs, ``craftax.night_and_sleep`` reproduces it on 4 of 49 real
  states with ``max |diff| 230``. The vignette is a different function too: a normalised radial
  distance, not Classic's Gaussian.
* **The strip is not Classic's.** 4 rows x 11 columns, 36 written cells, counts capped at **99**
  rather than 9 (``game_logic.clip_inventory_and_intrinsics``), two-digit numbers, a level digit,
  four attribute digits, and five cells carrying a second alpha-blended enchantment icon.

``stable_envs/render/prng.py`` **is** shared, and is used unchanged: the noise field is
``prng.uniform(key, (9t, 11t)) * 95 + 32``, which is character for character what craftax writes.

Parity: measured on 64 sampled states at 10 px and 16 px over the whole observation and over all
five cost bands. **100% of all pixels exact, max abs diff 0** against the **un-jitted** reference.
See ``tests/fixtures/regression/join/step7/craftax-full/findings.md``.

**One ULP, and it decides the parity number.** craftax normalises the night vignette as
``ntx / ntx.max()`` under jax, and XLA lowers a float32 divide by a scalar as a **reciprocal
multiply**. Against a true float32 divide that disagrees on 2,265 of the 9,900 vignette elements by
one ULP each, and one ULP in the vignette is one ULP in the night mask, which is enough to move a
uint8 pixel: the chain went from 35/49 bit-exact to **49/49** when the divide became
``v * (1 / v.max())``. :func:`night_vignette` writes it that way and
``tests/unit/test_craftax_full.py::test_the_vignette_is_a_reciprocal_multiply_and_not_a_divide``
pins it -- against a sha256 frozen from craftax, six exact probe values, and the fact that the naive
divide is still wrong.

**A second trap, silent rather than subtle.** ``ItemType.NONE`` is slot 0 of the item table and the
reference's scan starts at 1, so it is *never drawn* — and its texture is ``debug.png``, which is
opaque. An implementation that gathers slot 0 like any other slot paints a debug tile over every
empty square. ``_craftax_full_tex`` zeroes slot 0's alpha at bake time, so the gather needs no
special case.

**"The official frame" is not one object.** Under ``jax.jit`` XLA fuses the reference's arithmetic
differently and it disagrees with its own un-jitted self by up to 1.53e-05 in float32 **on the
49-state phase-0 pool** -- independent review measured 3.05e-05 over 402 states, so the figure is a
property of the pool rather than a bound. On that phase-0 pool 48 of 49 frames stayed
uint8-identical and one differed on 0.0023% of pixels by 1. The claim
certified here is therefore a **uint8** claim against the **un-jitted** renderer, exactly as for
Classic, and :meth:`CraftaxFull.reference_env` returns the un-jitted function.

What this carries and what it does not:

  * **one mob slot and one projectile slot per tile, and a six-projectile bridge budget.** A mob and
    a projectile on the same tile ARE representable -- the baked overlay table is the ``mob x
    projectile`` product and draws the pair, which an earlier version of this docstring wrongly said
    it could not. What is not representable is a **second entity of the same kind** on one tile: two
    mobs, or two projectiles. The window keeps the last one drawn, the reference composites
    both, and **the disagreement is confined to that tile** -- its size depends on which pair
    collides, so no single number describes it. Measured on an ``EnvState`` built with the pair
    on one tile: two melee mobs **156** at both sizes, two mob projectiles **237** at 10 px and
    **253** at 16 px; independent review measured 113-187 on its own pairs. craftax's
    ``mob_map`` keeps the three mob classes off each other's tiles, so the mob case does not
    arise in play; two projectiles on one tile can. Measured over 800 frames of a 4,000-step
    random-policy rollout, the most entities ever on one tile was **1**.

    Separately, the **reference bridge** carries at most
    ``MAX_MOB_PROJ + MAX_PLAYER_PROJ`` = 6 projectiles, because ``EnvState``'s projectile masks are
    ``(9, 3)`` and ``(9, 3)`` and a seventh cannot exist in a state at all. A window holding seven
    renders here and cannot be scored against the reference, which drops the extra. See
    :class:`CraftaxFull` and
    ``tests/unit/test_craftax_full.py::test_a_window_carrying_more_projectiles_than_the_simulator_can_hold_is_not_representable``.
  * ``player_xp`` above 9 is **clamped, not wrapped** — the reference gathers
    ``number_textures[player_xp]`` from a 10-entry table and jax clamps an out-of-range gather, so an
    xp of 12 draws a "9". The state slot carries the digit that is drawn.
  * the two global substitutions are **resolved into the stored tile id**:
    ``NECROMANCER -> NECROMANCER_VULNERABLE`` (a function of every mob on the level, on-screen or
    not) and ``LADDER_DOWN -> LADDER_DOWN_BLOCKED`` (a function of ``monsters_killed``). Measured:
    each moves pixels only on the tiles holding that id, so neither needs a state dimension.
  * ``do_night_noise`` is part of the reference the way ``jax_threefry_partitionable`` is: the
    registered env uses the default ``True`` and :meth:`reference_env` builds it that way.

The render window is **544 float32 numbers, 2,176 bytes**, against a 42,900-byte frame at the
registered size (**19.7x**) and 109,824 bytes at 16 px (**50.5x**). craftax's own ``EnvState`` is
273,276 bytes — **6.37x larger than the frame it draws** — because it carries all nine levels of
``map``, ``item_map`` and ``light_map`` while the frame shows one 9x11 window of one of them.
Twenty-one ``EnvState`` fields the window omits were each perturbed and moved **0** pixels
(``phase0/purity_and_state_window.json``).

Out-of-range ids and counts **wrap** rather than clamp, for the reason Classic's do: a clamping
decoder makes a slot that already holds the last class insensitive to the conformance suite's ``+25``
probe, and that is reported as a dead state dimension when it is an artifact of the probe. The two
exceptions are stated where they are made: ``light_level`` and the per-tile ``light_map`` fold on a
tent of period 2 (identity on ``[0, 1]``, so an in-range value passes through bit-exact), and the
health slot is lifted to at least 1 because the reference draws ``max(floor(health), 1)``.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import _resources as _RES
from ..render import prng
from ..spaces import Box, Dict, Discrete
from ._craftax_full_tex import MOB_BASE, N_MOB_SLOT, N_PROJ_SLOT, TABLES

#: tile edge in px. 10 is what ``swm/CraftaxPixels-v1`` is registered at (``BLOCK_PIXEL_SIZE_AGENT``);
#: craftax also ships 16 (``_IMG``) and 64 (``_HUMAN``). The baked tables carry 10 and 16, the two
#: sizes parity is certified at; any other size resamples the finished tile from the nearer of them.
TILE_PX_AGENT = 10
TILE_PX_IMG = 16
#: Sizes craftax itself draws at, and therefore the only sizes an official frame exists for.
REFERENCE_TILE_PX = (10, 16, 64)
#: Sizes this env has baked tables for; anything else is a resample of the finished tile.
BAKED_TILE_PX = tuple(sorted(TABLES))

OBS_H, OBS_W = 9, 11  #: ``craftax.craftax.constants.OBS_DIM``
STRIP_H = 4  #: ``INVENTORY_OBS_HEIGHT``
N_TILE = OBS_H * OBS_W  #: 99 world tiles
N_BLOCK = 37  #: ``len(BlockType)``
N_ITEM = 5  #: ``len(ItemType)``
N_LEVEL = 9  #: ``StaticEnvParams.num_levels``
N_FACING = 4  #: left, right, up, down -- the values ``player_direction - 1`` takes
PLAYER_ASLEEP = 4  #: ``player_texture_index = select(is_sleeping, 4, player_direction - 1)``
NO_PLAYER = 5  #: the sixth player slot: "nothing is drawn here". Mask is all false.
N_PLAYER_SLOT = 6
#: Counts the strip can show. ``clip_inventory_and_intrinsics`` does ``jnp.minimum(x, 99)`` over the
#: whole ``Inventory``, and the intrinsic ceilings are ``8 + strength`` / ``7 + 2*dexterity`` /
#: ``6 + 3*intelligence`` with ``max_attribute = 5``. So a strip cell renders a TWO-digit number and
#: the alphabet is 100, not Classic's 10.
N_COUNT = 100
N_DIGIT = 10
#: ``state_rng`` is two uint32 and a uint32 does NOT survive a float32 state row -- only 2.008% of
#: random uint32 round-trip through one. Split into 16-bit halves, 100% do. Four slots, not two.
#: See ``stable_envs/render/prng.py``.
N_KEY_HALVES = 4
#: ``night_texture``: the colour night is tinted towards, ``[[[0, 16, 64]]]`` in craftax's constants.
NIGHT_TINT = (0.0, 16.0, 64.0)
#: Below this ``light_level`` the reference mixes in a noise field; at or above it
#: ``night_intensity = max(2*(0.5 - daylight), 0)`` is exactly 0, so ``(1-0)*px + 0*noise`` is ``px``
#: in float32 whatever the noise was, and no PRNG is drawn at all.
NIGHT_THRESHOLD = 0.5
#: craftax's day length; used only by ``sample_states`` to give ``light_level`` the distribution the
#: simulator produces rather than a uniform one.
DAY_LENGTH = 300
#: ``monsters_killed[level] >= MONSTERS_KILLED_TO_CLEAR_LEVEL`` opens the down ladder.
MONSTERS_KILLED_TO_CLEAR_LEVEL = 8

#: ``StaticEnvParams`` mob budgets. A window with more than this many of a class cannot be written
#: into an ``EnvState``, so ``sample_states`` respects them and ``reference_render`` takes the first.
MAX_MELEE, MAX_PASSIVE, MAX_RANGED = 3, 3, 2
MAX_MOB_PROJ, MAX_PLAYER_PROJ = 3, 3
#: Sprites per class, as the baked mob table lays them out: slot 0 is "no mob".
N_MELEE, N_PASSIVE, N_RANGED = 8, 3, 8
N_PROJ_TYPE = 8
#: ``o -> (flip, transpose)``. The renderer classifies a direction as
#: ``flip = (d0 > 0) or (d1 > 0)`` and ``transpose = d1 != 0``, so the inverse is this table and NOT
#: the one pairing orientations with directions in the obvious order -- the same trap Classic's
#: ``ARROW_DIR`` documents, where getting it wrong swaps two ids and two whole tiles per frame.
PROJ_DIR = ((-1, 0), (0, -1), (1, 0), (0, 1))

#: ``BlockType`` ids the renderer or the state bridge needs by name.
B_OUT_OF_BOUNDS, B_NECROMANCER, B_NECROMANCER_VULNERABLE = 1, 32, 36
#: ``GRASS`` is the sampler's stand-in block; ``DARKNESS`` renders as an all-zero tile.
B_GRASS, B_DARKNESS = 2, 18
B_WOOD, B_STONE, B_COAL, B_IRON, B_DIAMOND, B_SAPPHIRE, B_RUBY = 6, 4, 8, 9, 10, 21, 22
#: ``ItemType`` ids.
I_NONE, I_TORCH, I_LADDER_DOWN, I_LADDER_DOWN_BLOCKED = 0, 1, 2, 4

#: Index into the baked ``icons`` table, in the order ``gen_consts.ICONS`` stacks them.
IC_HEALTH, IC_HUNGER, IC_THIRST, IC_ENERGY, IC_MANA = 0, 1, 2, 3, 4
IC_SAPLING, IC_TORCH, IC_BOOK, IC_FIREBALL, IC_ICEBALL = 5, 6, 7, 8, 9
IC_XP, IC_DEX, IC_STR, IC_INT, IC_EMPTY = 10, 11, 12, 13, 14


_TRANSPARENT_MOBS: dict = {}


def transparent_mob_slots(src: int) -> np.ndarray:
    """Mob slots whose sprite leaves at least one pixel of the tile beneath it visible.

    Exactly one of the twenty is fully opaque at every baked size (ranged type 5), and a tile under it
    has four dead state dimensions -- block, item, mob and projectile are all unobservable there,
    which is a true statement about the reference and an unhelpful one for a ``batch=1``
    ``check_state_dims_matter``. ``sample_states`` keeps it out of the one band a batch of 1 draws;
    every other band may use it.
    """
    if src not in _TRANSPARENT_MOBS:
        a = _unpack("mob_a", src).astype(bool)
        _TRANSPARENT_MOBS[src] = np.array([i for i in range(a.shape[0]) if not a[i].all()])
    return _TRANSPARENT_MOBS[src]


def _tent(x: torch.Tensor) -> torch.Tensor:
    """Fold a float back into ``[0, 1]`` on a tent of period 2, identity inside it.

    ``light_level`` and every per-tile ``light_map`` value live in ``[0, 1]``, and neither a clamp nor
    the obvious wrap will do. A clamp leaves a fully lit slot insensitive to the conformance suite's
    ``+25`` probe, which is reported as a dead state dimension; ``x - floor(x)`` sends 1.0 to 0.0 and
    turns every lit state into darkness. The tent does neither, and because it is applied only outside
    ``[0, 1]`` an in-range value passes through bit-exact instead of through two float32 roundings.
    """
    tent = 1.0 - (x.remainder(2.0) - 1.0).abs()
    return torch.where((x >= 0.0) & (x <= 1.0), x, tent)


def _tent_np(x):
    x = np.asarray(x, np.float32)
    tent = (1.0 - np.abs(np.mod(x, 2.0) - 1.0)).astype(np.float32)
    return np.where((x >= 0.0) & (x <= 1.0), x, tent).astype(np.float32)


#: The values craftax's own light maps take, measured over every level of four generated worlds
#: (``tests/fixtures/regression/join/step7/craftax-full/phase0/chain.json``: 23 distinct values, min 0.0, max 1.0).
#: Used only by ``sample_states``, so a drawn dark tile looks like a tile the simulator produces
#: rather than like a uniform draw.
LIGHT_MAP_ALPHABET = np.array(
    [
        0.0,
        0.10557276010513306,
        0.15147191286087036,
        0.17537885904312134,
        0.19999998807907104,
        0.20000000298023224,
        0.2788897156715393,
        0.3055727481842041,
        0.3675444722175598,
        0.3999999761581421,
        0.4000000059604645,
        0.4343145489692688,
        0.47888970375061035,
        0.5527863502502441,
        0.6000000238418579,
        0.6343145370483398,
        0.699999988079071,
        0.717157244682312,
        0.7527863383293152,
        0.800000011920929,
        0.8999999761581421,
        0.9788897037506104,
        1.0,
    ],
    np.float32,
)


def strip_geometry(t: int):
    """``(icon_lo, icon_hi, ones_lo, tens_lo, digit_hi)`` inside one ``t x t`` strip cell.

    The reference's own expressions: the icon occupies ``[left : t - right]`` in both axes with
    ``left = (t - int(0.8t)) // 2``; the ones digit ``[t - int(0.4t) : t]`` and the tens digit
    ``[t - 2*int(0.4t) : t - int(0.4t)]`` in the column axis, both at rows ``[t - int(0.4t) : t]``.
    Every span lies inside the cell, which is what makes the strip a per-cell table rather than a
    sequence of blits.
    """
    s = int(0.8 * t)
    left = (t - s) // 2
    d = int(0.4 * t)
    return left, left + s, t - d, t - 2 * d, d


def _unpack(name: str, src: int) -> np.ndarray:
    b64, shape = TABLES[src][name]
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), np.uint8).reshape(shape)


def _source_size(t: int) -> int:
    """Which baked size a ``t``-px render resamples from.

    **A baked size that DIVIDES ``t`` wins over a nearer one that does not**, and that is what makes
    this renderer resolution-consistent where the reference is not. ``check_resolution_consistency``
    renders at S and 2S; at the registered S=10 that is 20, and 16 is nearer to 20 than 10 is. Taking
    the nearer source would resample a 16 px tile up to 20 px and the 2S frame would be a different
    picture rather than a 2x replication of the S one -- measured: ``exact_frac`` 0.32 with the
    nearest rule against 1.000 on the world rows with this one. The largest such divisor is preferred
    so a 32 px render comes from the 16 px art rather than from the 10 px art doubled three times.
    """
    t = int(t)
    if t in TABLES:
        return t
    exact = [s for s in BAKED_TILE_PX if t % s == 0]
    if exact:
        return max(exact)
    return min(BAKED_TILE_PX, key=lambda s: abs(s - t))


def _resample(a: np.ndarray, src: int, t: int, axes=(-3, -2)) -> np.ndarray:
    """Nearest-resample a finished tile from ``src`` to ``t`` px, the repo's own rule.

    ``((i + 0.5) * src / t).long()`` — exact 2x pixel replication at ``t = 2*src``. The **finished
    cell** is resampled, never the ingredients: craftax re-lays-out the strip and re-loads every
    texture at each size with ``Image.NEAREST``, so re-deriving at 2S would give a picture that is not
    a 2x replication of the S one and would make a resolution-consistency bound unmeetable for a
    reason that is a resampling rule rather than a renderer change.
    """
    if int(t) == int(src):
        return a
    ix = ((np.arange(t) + 0.5) * src / t).astype(np.int64).clip(0, src - 1)
    for ax in axes:
        a = np.take(a, ix, axis=ax)
    return a


#: ``(x, y, kind, *params)`` for each of the 36 cells the reference writes. ``kind`` decides both the
#: icon alphabet and whether the cell carries digits; the 8 positions absent from this table are never
#: written and stay black.
#:
#: ``arrows`` is the one cell whose enchantment is drawn AFTER its digits (the enchantment covers the
#: icon square, which overlaps the digit squares), so it cannot fold its enchantment into the icon
#: layer the way the sword and armour cells do. It is baked whole instead -- 100 counts x 3
#: enchantments -- and takes the no-op digit entry.
STRIP_CELLS = (
    (0, 0, "count", "health", ("icon", IC_HEALTH)),
    (1, 0, "count", "food", ("icon", IC_HUNGER)),
    (2, 0, "count", "drink", ("icon", IC_THIRST)),
    (3, 0, "count", "energy", ("icon", IC_ENERGY)),
    (4, 0, "count", "mana", ("icon", IC_MANA)),
    (6, 0, "level", None, ()),
    (7, 0, "armour", ("armour0", "ench_armour0"), (0,)),
    (9, 0, "attr", "xp", ("icon", IC_XP, True)),
    (0, 1, "count", "coal", ("block", B_COAL)),
    (1, 1, "count", "iron", ("block", B_IRON)),
    (2, 1, "count", "diamond", ("block", B_DIAMOND)),
    (3, 1, "count", "sapphire", ("block", B_SAPPHIRE)),
    (4, 1, "count", "ruby", ("block", B_RUBY)),
    (5, 1, "count", "sapling", ("icon", IC_SAPLING)),
    (6, 1, "tool", "bow", ("bow_textures", 2)),
    (7, 1, "armour", ("armour1", "ench_armour1"), (1,)),
    (8, 1, "sword", ("sword", "ench_sword"), ()),
    (9, 1, "attr", "dexterity", ("icon", IC_DEX, False)),
    (0, 2, "count", "wood", ("block", B_WOOD)),
    (1, 2, "count", "stone", ("block", B_STONE)),
    (2, 2, "count", "torches", ("icon", IC_TORCH)),
    (3, 2, "count", "books", ("icon", IC_BOOK)),
    (4, 2, "flag", "spell_fireball", ("icon", IC_FIREBALL)),
    (5, 2, "flag", "spell_iceball", ("icon", IC_ICEBALL)),
    (6, 2, "arrows", ("arrows", "ench_bow"), ()),
    (7, 2, "armour", ("armour2", "ench_armour2"), (2,)),
    (8, 2, "tool", "pickaxe", ("pickaxe_textures", 5)),
    (9, 2, "attr", "strength", ("icon", IC_STR, False)),
    (0, 3, "count", "potion0", ("potion", 0)),
    (1, 3, "count", "potion1", ("potion", 1)),
    (2, 3, "count", "potion2", ("potion", 2)),
    (3, 3, "count", "potion3", ("potion", 3)),
    (4, 3, "count", "potion4", ("potion", 4)),
    (5, 3, "count", "potion5", ("potion", 5)),
    (7, 3, "armour", ("armour3", "ench_armour3"), (3,)),
    (9, 3, "attr", "intelligence", ("icon", IC_INT, False)),
)

#: How many icon-layer entries each ``kind`` needs.
_KIND_VARIANTS = {
    "count": 2,
    "level": 1,
    "armour": 9,
    "attr": None,
    "tool": None,
    "sword": 15,
    "flag": 2,
    "arrows": N_COUNT * 3,
}


def _cell_variants(kind, params) -> int:
    if kind == "attr":
        return 2 if params[2] else 1
    if kind == "tool":
        return int(params[1])
    return int(_KIND_VARIANTS[kind])


#: ``(cell index -> base offset in the icon block)``, and the block's total size.
STRIP_ICON_BASE = []
_acc = 0
for _c in STRIP_CELLS:
    STRIP_ICON_BASE.append(_acc)
    _acc += _cell_variants(_c[2], _c[4])
STRIP_ICON_BASE = tuple(STRIP_ICON_BASE)
N_STRIP_ICON = _acc
del _acc, _c

#: One table draws the whole frame's background: the 185 ``block x item`` world composites, then a
#: black tile for the 8 strip positions the reference never writes, then the 410 strip icon cells.
#: A strip cell's *icon* is its background, so the strip costs one gather like a world tile does.
N_WORLD_BG = N_BLOCK * N_ITEM
BG_BLACK = N_WORLD_BG
BG_STRIP_BASE = N_WORLD_BG + 1
N_BG = BG_STRIP_BASE + N_STRIP_ICON

#: ``mob x projectile``, the two stencil layers that can both land on any tile, baked into one so the
#: world overlay is a single gather. Index ``mob * N_PROJ_SLOT + proj``; entry 0 is empty.
N_OVERLAY = N_MOB_SLOT * N_PROJ_SLOT

#: The digit layer: ones sprite x tens sprite. ``ones`` 0..9 indexes ``number_textures`` (slot 0 is
#: the empty texture, which is how a value of 0 draws nothing) and 10..19 indexes
#: ``number_textures_with_zero``; ``tens`` indexes ``number_textures``, whose slot 0 suppresses a
#: leading zero. Entry 0 is (empty, empty) and is the no-op a digit-less cell takes.
N_ONES, N_TENS = 20, 10
N_DIGIT_ENTRY = N_ONES * N_TENS


# --------------------------------------------------------------------------- table construction


def _blend(px, rgb, a):
    """The reference's own ``px * (1 - a) + rgb`` on integer arrays. ``a`` is 0/1, so this is a select
    -- but it is written the reference's way so a non-premultiplied texture would show up here rather
    than silently, and :func:`_check_premultiplied` asserts the equivalence at build time."""
    return px * (1 - a) + rgb


def _check_premultiplied(rgb, a, what):
    if np.any(rgb * (1 - a) != 0):
        raise ValueError(
            f"_craftax_full_tex: {what} is not zero where its alpha is zero, so the reference's "
            f"'multiply by (1-alpha) then ADD' is not a select and this table cannot be gathered. "
            f"Re-derive the layer as an explicit blend rather than a stencil."
        )


def _strip_icon_tiles(src: int) -> np.ndarray:
    """``(N_STRIP_ICON, src, src, 3)`` — every strip cell's background picture.

    A cell's background is its **icon**, because ``_render_icon`` *sets* its square and leaves the
    rest of the cell black; the digits then blend over it. Two families of cell carry more than an
    icon here:

    * the sword and the four armour slots fold their **enchantment** in, because the reference draws
      it after an icon and those cells have no digits, so the composition is unambiguous;
    * the **arrows** cell is baked whole -- 100 counts x 3 enchantments -- because its enchantment is
      drawn *after* its digits and covers the icon square, which overlaps the digit squares. Folding
      it into the icon would put it underneath the digits it is meant to cover.
    """
    t = src
    lo, hi, oc, tc, d = strip_geometry(t)
    icons = _unpack("icons", src).astype(np.int64)
    small = _unpack("small_block", src).astype(np.int64)
    pot = _unpack("potion_textures", src).astype(np.int64)
    arrow = _unpack("arrow_icon", src).astype(np.int64)
    num, num_a = _unpack("number_textures", src).astype(np.int64), _unpack("number_textures_a", src).astype(np.int64)
    numz = _unpack("number_textures_with_zero", src).astype(np.int64)
    numz_a = _unpack("number_textures_with_zero_a", src).astype(np.int64)
    _check_premultiplied(num, num_a[..., None], "number_textures")
    _check_premultiplied(numz, numz_a[..., None], "number_textures_with_zero")
    arm = _unpack("armour", src).astype(np.int64)
    arm_e, arm_ea = _unpack("armour_ench_rgb", src).astype(np.int64), _unpack("armour_ench_a", src).astype(np.int64)
    sw = _unpack("sword_textures", src).astype(np.int64)
    sw_e, sw_ea = _unpack("sword_ench_rgb", src).astype(np.int64), _unpack("sword_ench_a", src).astype(np.int64)
    ar_e, ar_ea = _unpack("arrow_ench_rgb", src).astype(np.int64), _unpack("arrow_ench_a", src).astype(np.int64)
    empty = icons[IC_EMPTY]

    def blank():
        return np.zeros((t, t, 3), np.int64)

    def put(tile, sprite):
        tile[lo:hi, lo:hi] = sprite
        return tile

    def over(tile, rgb, a):
        sl = tile[lo:hi, lo:hi]
        tile[lo:hi, lo:hi] = sl * (1 - a[..., None]) + rgb * a[..., None]
        return tile

    def digits(tile, v):
        ones, tens = v % 10, v // 10
        ot, oa = (num, num_a) if v == 0 else (numz, numz_a)
        tile[oc:, oc:] = _blend(tile[oc:, oc:], ot[ones], oa[ones][..., None])
        tile[oc:, tc:oc] = _blend(tile[oc:, tc:oc], num[tens], num_a[tens][..., None])
        return tile

    out = []
    for x, y, kind, _slots, p in STRIP_CELLS:
        if kind == "count":
            src_tex = icons[p[1]] if p[0] == "icon" else small[p[1]] if p[0] == "block" else pot[p[1]]
            out += [put(blank(), empty), put(blank(), src_tex)]
        elif kind == "flag":
            out += [put(blank(), empty), put(blank(), icons[p[1]])]
        elif kind == "attr":
            if p[2]:
                out += [put(blank(), empty), put(blank(), icons[p[1]])]
            else:
                out += [put(blank(), icons[p[1]])]
        elif kind == "level":
            out += [blank()]
        elif kind == "tool":
            tab = _unpack(p[0], src).astype(np.int64)
            out += [put(blank(), tab[i]) for i in range(int(p[1]))]
        elif kind == "sword":
            for lvl in range(5):
                for e in range(3):
                    out.append(over(put(blank(), sw[lvl]), sw_e[e], sw_ea[e]))
        elif kind == "armour":
            i = int(p[0])
            for lvl in range(3):
                for e in range(3):
                    out.append(over(put(blank(), arm[lvl, i]), arm_e[e, i], arm_ea[e, i]))
        elif kind == "arrows":
            for v in range(N_COUNT):
                for e in range(3):
                    tile = digits(put(blank(), arrow if v > 0 else empty), v)
                    lv = e if v > 0 else 0
                    out.append(over(tile, ar_e[lv], ar_ea[lv]))
        else:  # pragma: no cover - guarded by spec
            raise AssertionError(kind)
    a = np.stack(out)
    if a.shape[0] != N_STRIP_ICON:
        raise AssertionError(f"strip icon block is {a.shape[0]}, spec says {N_STRIP_ICON}")
    return a


def _bg_tiles(src: int) -> np.ndarray:
    """``(N_BG, src, src, 3)`` — the whole frame's background in one table."""
    block = _unpack("block", src).astype(np.int64)  # (37, s, s, 3)
    irgb = _unpack("item_rgb", src).astype(np.int64)  # (5, s, s, 3)
    ia = _unpack("item_a", src).astype(np.int64)[..., None]  # (5, s, s, 1)
    _check_premultiplied(irgb * ia, ia, "full_map_item_textures")
    bg = block[:, None] * (1 - ia[None]) + (irgb * ia)[None]  # (37, 5, s, s, 3)
    bg = bg.reshape(N_WORLD_BG, *block.shape[1:])
    return np.concatenate([bg, np.zeros((1, *block.shape[1:]), np.int64), _strip_icon_tiles(src)])


def _overlay_tiles(src: int):
    """``(N_OVERLAY, s, s, 3)`` rgb and ``(N_OVERLAY, s, s)`` mask: ``mob`` then ``projectile``.

    Both alphas are binary, so ``where(proj_a, proj, where(mob_a, mob, bg))`` collapses to one
    lookup: the pair's colour is the projectile where it is opaque and the mob otherwise, and the
    pair's mask is the union. Projectile over mob is the reference's order.
    """
    mr, ma = _unpack("mob_rgb", src).astype(np.int64), _unpack("mob_a", src).astype(bool)
    jr, ja = _unpack("proj_rgb", src).astype(np.int64), _unpack("proj_a", src).astype(bool)
    rgb = np.where(ja[None, :, :, :, None], jr[None], mr[:, None])
    mask = ma[:, None] | ja[None, :]
    return rgb.reshape(N_OVERLAY, *mr.shape[1:]), mask.reshape(N_OVERLAY, *ma.shape[1:])


def _player_tiles(src: int):
    """``(6, s, s, 3)`` rgb and mask. Slot 5 is ``NO_PLAYER``: an all-false mask, exactly the identity."""
    pr, pa = _unpack("player_rgb", src).astype(np.int64), _unpack("player_a", src).astype(bool)
    rgb = np.concatenate([pr, np.zeros((1, *pr.shape[1:]), np.int64)])
    mask = np.concatenate([pa, np.zeros((1, *pa.shape[1:]), bool)])
    return rgb, mask


def _digit_tiles(src: int):
    """``(200, s, s, 3)`` rgb and mask for ``ones * 10 + tens``.

    ``ones`` 0..9 is ``number_textures`` (slot 0 empty, which is how a value of 0 draws nothing) and
    10..19 is ``number_textures_with_zero``; ``tens`` is ``number_textures``, whose slot 0 suppresses
    a leading zero. Entry 0 is (empty, empty) — the no-op a cell without digits takes.
    """
    t = src
    _, _, oc, tc, _ = strip_geometry(t)
    num, num_a = _unpack("number_textures", src).astype(np.int64), _unpack("number_textures_a", src).astype(bool)
    numz = _unpack("number_textures_with_zero", src).astype(np.int64)
    numz_a = _unpack("number_textures_with_zero_a", src).astype(bool)
    rgb = np.zeros((N_DIGIT_ENTRY, t, t, 3), np.int64)
    mask = np.zeros((N_DIGIT_ENTRY, t, t), bool)
    for o in range(N_ONES):
        osp, oa = (num[o], num_a[o]) if o < N_DIGIT else (numz[o - N_DIGIT], numz_a[o - N_DIGIT])
        for te in range(N_TENS):
            k = o * N_TENS + te
            rgb[k, oc:, oc:], mask[k, oc:, oc:] = osp, oa
            rgb[k, oc:, tc:oc], mask[k, oc:, tc:oc] = num[te], num_a[te]
    return rgb, mask


class _Tables:
    """Every table this renderer gathers from, decoded once per ``(device, tile size)``.

    Four tables and nothing else, all small enough to stay resident at any size craftax draws at:
    ``bg`` (596 tiles: 185 ``block x item`` composites, a black tile, 410 strip icon cells),
    ``ov`` (660 ``mob x projectile`` pairs), ``pl`` (6 player slots), ``dg`` (200 digit pairs).

    **uint8 only, and that is exact rather than a compromise.** Every craftax texture is integral in
    0..255 and a stencil composite of integers is an integer, so a float32 table would hold the same
    numbers; the lit path converts the finished uint8 composite instead of gathering four times the
    bytes, which is worth about 3x on a lit frame.
    """

    _MAX_ENTRIES = 6
    _cache = _RES.BoundedCache(_MAX_ENTRIES, name="craftax_full.tables")

    @staticmethod
    def _rows(a: np.ndarray, t: int, device) -> torch.Tensor:
        """``(N, t, t, C)`` -> ``(N*t, t*C)``: one output scanline of one tile per row, so a gather
        writes the final raster directly instead of building a tile-major buffer and transposing it."""
        n = a.shape[0]
        return torch.as_tensor(np.ascontiguousarray(a.reshape(n * t, -1)), device=device)

    @classmethod
    def get(cls, device: torch.device, t: int):
        key = (str(device), int(t))
        have = cls._cache.get(key)
        if have is not None:
            return have
        t = int(t)
        src = _source_size(t)
        bg = _resample(_bg_tiles(src), src, t)
        ov_rgb, ov_m = _overlay_tiles(src)
        pl_rgb, pl_m = _player_tiles(src)
        dg_rgb, dg_m = _digit_tiles(src)
        ov_rgb, ov_m = _resample(ov_rgb, src, t), _resample(ov_m, src, t, axes=(-2, -1))
        pl_rgb, pl_m = _resample(pl_rgb, src, t), _resample(pl_m, src, t, axes=(-2, -1))
        dg_rgb, dg_m = _resample(dg_rgb, src, t), _resample(dg_m, src, t, axes=(-2, -1))

        def pack(rgb, m=None):
            # uint8 only. The composite is a stencil select over integral textures, so it never needs
            # a float table: the float chain converts the finished uint8 composite, which is exact
            # because the float table would have held the same numbers.
            u8 = cls._rows(rgb.astype(np.uint8), t, device)
            if m is None:
                return {"u8": u8}
            # The mask is stored with all THREE channels, which looks wasteful and is not. A
            # one-channel mask moves a third of the bytes and lets ``torch.where`` broadcast -- and
            # measured on this CPU it is **1.7x slower** on a fully lit frame (18,928 -> 10,852 f/s
            # at 10 px, B=256), because the broadcast costs more than the bytes it saves. Tried,
            # measured, reverted; the comment is here so it is not tried again.
            return {"u8": u8, "m": cls._rows(np.repeat(m[..., None], 3, -1), t, device)}

        tab = {"bg": pack(bg), "ov": pack(ov_rgb, ov_m), "pl": pack(pl_rgb, pl_m), "dg": pack(dg_rgb, dg_m), "t": t}
        return cls._cache.put(key, tab)


_VIGNETTE_CACHE: dict = {}
_CONST_CACHE: dict = {}
_GRID_CACHE: dict = {}


def night_vignette(device: torch.device, h: int, w: int) -> torch.Tensor:
    """``(h, w, 1)`` float32 — full Craftax's ``night_noise_intensity_texture``.

    A **normalised radial distance** ``sqrt((x - h//2)^2 + (y - w//2)^2) / max``: zero at the frame
    centre, one at the farthest corner. Classic's is a Gaussian on a ``linspace(-1, 1)`` grid, which
    is why :func:`stable_envs.envs.craftax.night_vignette` cannot serve here.

    **The normalisation is a reciprocal multiply and not a divide.** craftax computes it under jax,
    and XLA lowers a float32 divide by a scalar as ``x * (1/s)``; against a true float32 divide that
    disagrees on 2,265 of the 9,900 elements at the registered size, by one ULP each, and one ULP
    here is one ULP in the night mask, which moves uint8 pixels. Measured: the lighting chain went
    from 35/49 bit-exact to 49/49 when this line changed.
    """
    key = (str(device), int(h), int(w))
    if key not in _VIGNETTE_CACHE:
        x = (np.arange(h, dtype=np.float32) - (h // 2))[:, None]
        y = (np.arange(w, dtype=np.float32) - (w // 2))[None, :]
        v = np.sqrt(x**2 + y**2).astype(np.float32)
        v = (v * np.float32(np.float32(1.0) / np.float32(v.max()))).astype(np.float32)
        _VIGNETTE_CACHE[key] = torch.as_tensor(v, device=device).unsqueeze(-1)
    return _VIGNETTE_CACHE[key]


def _night_tint(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _CONST_CACHE:
        _CONST_CACHE[key] = torch.tensor(NIGHT_TINT, device=device)
    return _CONST_CACHE[key]


def light_night_and_sleep(
    world: torch.Tensor, light_map: torch.Tensor, daylight: torch.Tensor, sleeping: torch.Tensor, keys: torch.Tensor
) -> torch.Tensor:
    """Full Craftax's lighting chain, in the reference's own order, op for op.

    ``world`` ``(n, h, w, 3)`` float32 pre-lighting pixels; ``light_map`` ``(n, h, w, 1)`` the
    per-tile multiplier expanded to pixels; ``daylight`` ``(n,)`` already carrying the
    ``select(player_level == 0, light_level, 1.0)`` override; ``sleeping`` ``(n,)`` bool; ``keys``
    ``(n, 2)`` int32 threefry keys.

    **Deliberately not algebraically collapsed.** The chain folds into two scalars and a per-pixel
    constant in real arithmetic and into a different rounding at every step in float32, and this
    renderer's claim is bit equality rather than closeness. The reductions taken are the ones that
    change no bits:

    * the noise field is drawn **only for rows that can use it**. ``night_intensity`` is exactly 0
      for ``daylight >= 0.5``, so ``(1 - 0)*px + 0*noise`` is ``px`` in float32 whatever the noise
      was, and those rows get no threefry evaluation at all. Eight of the nine levels are in that
      band unconditionally, because ``daylight`` is forced to 1.0 off level 0;
    * a batch that is entirely fully lit, awake and at unit gain never reaches here -- ``render``
      keeps it on the uint8 gather.
    """
    n, h, w, _ = world.shape
    dev = world.device
    dl = daylight.view(n, 1, 1, 1)
    px = world * light_map
    night = px.clone()
    dark = daylight < NIGHT_THRESHOLD
    if bool(dark.any()):
        idx = dark.nonzero(as_tuple=True)[0]
        sub = px.index_select(0, idx)
        noise = (prng.uniform(keys.index_select(0, idx), (h, w)) * 95 + 32).unsqueeze(-1)
        mask = night_vignette(dev, h, w) * torch.clamp_min(2.0 * (NIGHT_THRESHOLD - dl.index_select(0, idx)), 0.0)
        night.index_copy_(0, idx, (1.0 - mask) * sub + mask * noise)
    night.mul_(0.5).add_(_night_tint(dev) * 0.5)
    out = night.mul_(1 - dl).add_(px * dl)
    return out.mul_(1.0 - sleeping.view(n, 1, 1, 1).to(out.dtype) * 0.5)


#: The 41 per-frame numbers the strip reads, in state order.
STRIP_SLOTS = (
    "health",
    "food",
    "drink",
    "energy",
    "mana",
    "xp",
    "dexterity",
    "strength",
    "intelligence",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapphire",
    "ruby",
    "sapling",
    "torches",
    "books",
    "arrows",
    "potion0",
    "potion1",
    "potion2",
    "potion3",
    "potion4",
    "potion5",
    "pickaxe",
    "sword",
    "bow",
    "armour0",
    "armour1",
    "armour2",
    "armour3",
    "ench_sword",
    "ench_bow",
    "ench_armour0",
    "ench_armour1",
    "ench_armour2",
    "ench_armour3",
    "spell_fireball",
    "spell_iceball",
)
N_STRIP = len(STRIP_SLOTS)
_SLOT = {name: i for i, name in enumerate(STRIP_SLOTS)}
#: The ceiling each strip slot's value wraps at, so a decoded row always indexes a table entry that
#: exists. Counts wrap at 100 (the simulator's own cap), tools/armour/enchantments at their sprite
#: count, attribute digits at 10 (the reference gathers a 10-entry table and jax CLAMPS, so an xp of
#: 12 draws a "9"; the state slot carries the digit that is drawn).
_SLOT_MOD = {n: N_COUNT for n in STRIP_SLOTS}
_SLOT_MOD.update({n: N_DIGIT for n in ("xp", "dexterity", "strength", "intelligence")})
_SLOT_MOD.update(
    {"pickaxe": 5, "sword": 5, "bow": 2, "ench_sword": 3, "ench_bow": 3, "spell_fireball": 2, "spell_iceball": 2}
)
_SLOT_MOD.update({f"armour{i}": 3 for i in range(4)})
_SLOT_MOD.update({f"ench_armour{i}": 3 for i in range(4)})
STRIP_MOD = tuple(_SLOT_MOD[n] for n in STRIP_SLOTS)


def _cell_groups():
    """Cells grouped by how their icon variant and digit index are computed, so a render does a
    handful of vectorised ops over the whole strip instead of 36 per-cell ones."""
    g = {
        "count": [],
        "flag": [],
        "attr_cond": [],
        "attr_fixed": [],
        "tool": [],
        "armour": [],
        "sword": [],
        "arrows": [],
        "level": [],
    }
    for ci, (x, y, kind, slots, p) in enumerate(STRIP_CELLS):
        pos = y * OBS_W + x
        base = BG_STRIP_BASE + STRIP_ICON_BASE[ci]
        if kind == "count":
            g["count"].append((pos, base, _SLOT[slots]))
        elif kind == "flag":
            g["flag"].append((pos, base, _SLOT[slots]))
        elif kind == "attr":
            g["attr_cond" if p[2] else "attr_fixed"].append((pos, base, _SLOT[slots]))
        elif kind == "tool":
            g["tool"].append((pos, base, _SLOT[slots]))
        elif kind == "armour":
            g["armour"].append((pos, base, _SLOT[slots[0]], _SLOT[slots[1]]))
        elif kind == "sword":
            g["sword"].append((pos, base, _SLOT[slots[0]], _SLOT[slots[1]]))
        elif kind == "arrows":
            g["arrows"].append((pos, base, _SLOT[slots[0]], _SLOT[slots[1]]))
        elif kind == "level":
            g["level"].append((pos, base))
    return g


CELL_GROUPS = _cell_groups()


def _strip_index(device):
    """Per-device index tensors for :func:`_cell_groups`, plus the all-black strip defaults."""
    key = str(device)
    if key not in _GRID_CACHE:

        def T(rows, col):
            return torch.as_tensor([r[col] for r in rows], dtype=torch.long, device=device)

        idx = {k: tuple(T(v, i) for i in range(len(v[0]))) for k, v in CELL_GROUPS.items() if v}
        bg0 = torch.full((STRIP_H * OBS_W,), BG_BLACK, dtype=torch.long, device=device)
        dg0 = torch.zeros(STRIP_H * OBS_W, dtype=torch.long, device=device)
        mod = torch.as_tensor(STRIP_MOD, dtype=torch.float32, device=device)
        _GRID_CACHE[key] = (idx, bg0, dg0, mod)
    return _GRID_CACHE[key]


@register
class CraftaxFull(Env):
    """Full Craftax: nine levels, an item/mob/projectile stencil over a block window, and a 4-row strip.

    **The render window is wider than the reference bridge in one place, and it is stated here rather
    than discovered.** The window has room for a projectile on each of its 99 tiles; an ``EnvState``
    has room for ``MAX_MOB_PROJ + MAX_PLAYER_PROJ`` = **6**, because its two projectile masks are
    ``(9, 3)`` each and a seventh cannot exist in a state at all. A window carrying seven renders
    here and **cannot be scored against the reference**, which places six and drops the rest.
    :meth:`sample_states` never produces such a row and
    ``tests/unit/test_craftax_full.py::test_a_window_carrying_more_projectiles_than_the_simulator_can_hold_is_not_representable``
    asserts both halves. The boundary pool keeps one such row deliberately, scored apart
    (``tests/fixtures/regression/join/step7/craftax-full/certify/boundary_pool.json``).
    """

    family = "craftax"
    domain = "full"

    name = "craftax-full"
    #: 99 blocks, 99 items, 99 mob slots, 99 projectile slots, 99 light-map values, then the player
    #: slot, the sleeping flag, the level, the daylight, four PRNG-key halves and the 41 strip numbers.
    state_dim = 5 * N_TILE + 4 + N_KEY_HALVES + N_STRIP
    state_names = (
        tuple(f"block_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"item_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"mob_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"proj_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"light_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + ("player_facing", "is_sleeping", "player_level", "light_level")
        + tuple(f"rng_half{i}" for i in range(N_KEY_HALVES))
        + tuple(f"inv_{n}" for n in STRIP_SLOTS)
    )
    #: The frame is ``((9 + 4) * size, 11 * size, 3)``. 10 is what ``swm/CraftaxPixels-v1`` is
    #: registered at, and the size the parity claim is headlined on; 16 is craftax's other shipped
    #: size and is certified too.
    default_size = TILE_PX_AGENT
    reference_id = "Craftax-Pixels-v1"
    #: REFERENCE-side versions the parity numbers were measured against. ``craftax`` pins the atlas,
    #: the strip layout and the texture resampling; ``jax`` is load-bearing for the reason
    #: :meth:`reference_env` refuses to build a reference whose PRNG path differs.
    reference_versions = {"craftax": "1.6.1", "jax": "0.11.1"}
    #: ``stable_worldmodel``'s ``CraftaxWrapper`` (
    #: ``stable_worldmodel/envs/craftax/craftax.py``) builds its step function as
    #: ``self._step_fn = jax.jit(_step, backend=self.backend)`` (and reset likewise), so a LIVE
    #: ``gymnasium.make("swm/CraftaxPixels-v1")`` workflow always observes the JIT-produced frame,
    #: never the un-jitted one the parity claim above is measured against. See
    #: ``CraftaxClassic.SWM_WRAPPER_GATE_F`` for the mechanism (jit fuses float32 arithmetic
    #: differently and disagrees with its own un-jitted self by up to 1.5e-05, which a sleeping or
    #: lit row can turn into a uint8 rounding flip); the same identity-recheck evidence
    #: (``tests/fixtures/regression/join/step8/semantic/identity_recheck.json``) establishes the un-jitted
    #: byte-identity this class certifies, and the same bounded (<=1 uint8 step) gap applies to a
    #: live SWM caller of this cell.
    SWM_WRAPPER_GATE_F = {
        "swm_wrapper_jits": ["reset", "step"],
        "swm_wrapper_source": "stable_worldmodel/envs/craftax/craftax.py:CraftaxWrapper.__init__ "
        "(self._reset_fn = jax.jit(_reset, ...); self._step_fn = "
        "jax.jit(_step, ...))",
        "certified_reference": "un-jitted craftax 1.6.1 (see reference_env)",
        "live_swm_reference": "jitted craftax 1.6.1, via CraftaxWrapper",
        "certified_exact_frac": 1.0,
        "certified_max_abs_diff": 0,
        "live_swm_exact_frac_phase0_pool": "48/49 states uint8-identical jitted vs un-jitted; 1/49 "
        "differed on 0.0023% of pixels, max |diff| 1 (see "
        "parity_note above, phase-0 pool)",
        "max_abs_diff_swm_wrapper_vs_certified": 1,
        "note": "not hidden: a stated discrepancy between the certified claim and what a live "
        "stable_worldmodel workflow observes, not a renderer defect here.",
    }
    parity_all_exact_min = 1.0
    parity_note = (
        "Measured on the frozen Step-7 corpus at 10 px (the registered observation) and 16 px, over "
        "the WHOLE observation -- the 9x11 world window and the 4-row inventory strip -- and over all "
        "five cost bands: level 0 at full daylight, level 0 at dusk, level 0 at night (the noise "
        "field drawn), levels 1-8 fully lit, and levels 1-8 in darkness, with sleeping rows "
        "throughout. Against craftax 1.6.1 under jax 0.11.1 with jax_threefry_partitionable at its "
        "default True and do_night_noise at its default True: 100% of all pixels exact, max abs diff "
        "0. The claim is a UINT8 claim and deliberately not a float one, because the reference does "
        "not have one float value to be equal to: under jax.jit XLA fuses its arithmetic differently "
        "and it disagrees with its own un-jitted self by up to 1.53e-05, which is invisible in uint8 "
        "on a fully lit frame and not invisible wherever the picture is compressed towards one value "
        "-- which the light_map multiply, the night blend and the sleep halving all do. On a 49-state "
        "phase-0 pool 48 of 49 jitted frames were uint8-identical to the un-jitted ones and one "
        "differed on 0.0023% of pixels by 1. reference_env therefore returns the un-jitted function, "
        "which is the renderer as written. jax is a declared reference version because "
        "jax_threefry_partitionable selects between two threefry counter layouts that agree on 0 of "
        "448 draws, and the night noise field is drawn from one of them."
    )
    #: Where ``check_resolution_consistency`` applies, decided by running the check's own protocol on
    #: the OFFICIAL renderer rather than by argument from this one -- W1's ruling (Sami's decision A)
    #: applied by precedent. At S=10 / 2S=20 over the phase-0 pool, craftax itself scores
    #: ``exact_frac`` 0.4895 overall: **0.4817 on the world rows in full daylight**, 0.0991 at night,
    #: and 0.94 on the strip rows in every band, with ``max_abs`` 191. Two mechanisms, neither a
    #: geometry question:
    #:
    #: * ``load_texture`` re-reads and **resamples each PNG with ``Image.NEAREST`` at every size**, so
    #:   the 20 px textures are not a 2x replication of the 10 px ones. This is the daylight number,
    #:   measured with no noise and no lighting at all;
    #: * the noise field is drawn at ``(9t, 11t)`` -- the frame's own resolution -- so element ``i`` of
    #:   the 20 px field is a different threefry counter from element ``i`` of the 10 px one.
    #:
    #: The scope is therefore **all bands**, which is broader than Classic's band-scoped declaration
    #: (there, only ``light_level < 0.5`` was inapplicable): here the reference is resolution-
    #: inconsistent even in full daylight, so there is no band in which the check measures what it is
    #: defined to measure. This renderer resamples the FINISHED tile, which makes it exactly
    #: 2x-consistent where the reference is not; matching craftax's per-size resample instead would
    #: buy no parity at a size that has no official frame. The frozen Step-2 threshold is not edited,
    #: and exact parity at each supported size is unaffected and still required.
    resolution_consistency_applicability = {
        "all bands": "not_applicable: the official renderer is itself resolution-inconsistent in "
        "every band -- world rows 0.4817 in full daylight and 0.0991 at night, strip "
        "rows 0.94 throughout, max abs 191 at S=10/2S=20",
        "mechanisms": "craftax.load_texture resamples every PNG with Image.NEAREST at each size, so "
        "2S textures are not a 2x replication of the S ones; and the night noise field "
        "is drawn at the frame's own resolution",
        "scope_note": "broader than CraftaxClassic.resolution_consistency_applicability, which marks "
        "only light_level < 0.5 inapplicable and keeps the frozen bound above it",
        "evidence": "tests/fixtures/regression/join/step7/craftax-full/phase0/resolution_and_dead_dims.json",
        "measured_against": "craftax 1.6.1 un-jitted, jax 0.11.1, jax_threefry_partitionable=True",
        "still_required": "exact parity against the official renderer at 10 px and at 16 px",
    }
    #: The distribution the single registered performance tier is measured on. Step 3's harness draws
    #: ``probe.sample_states(64)`` (``examples/bench_step3.py``), so this env's own sampler IS the
    #: certification distribution and its shares are a declared property of the cell rather than a
    #: weighting invented for the benchmark. Bands are assigned by row index (``i % 6``), so the
    #: shares are exact rather than drawn.
    certification_state_distribution = {
        "draw": "CraftaxFull.sample_states(64) -- examples/bench_step3.py's own protocol",
        "band_assignment": "stratified by row index (i % 6); values inside a band from craftax's own "
        "day cycle and light-map alphabet",
        "level0_day_share": 1 / 6,
        "level0_dusk_share": 1 / 6,
        "level0_night_share": 2 / 6,
        "underground_lit_share": 1 / 6,
        "underground_dark_share": 1 / 6,
        #: **0.25, not the 2/6 the lighting bands alone suggest.** The two fully-lit bands are 2/6 of
        #: the draw, but ``is_sleeping`` is stratified independently on ``i % 4`` and a sleeping row
        #: takes the float chain whatever its lighting is. Over the twelve-row period of the two
        #: stratifications, rows 2, 8 and 9 are the only ones that stay on the uint8 gather: 3/12.
        #: The frozen corpus measures 14 of 64, which is the same number after selection. An earlier
        #: value of 2/6 survived because the test that checks this declaration allowed 0.1 of slack
        #: and the error was 0.083.
        "uint8_fast_path_share": 0.25,
        "sleeping_share": 0.25,
        "note": "a row stays on the uint8 gather only if it is fully lit AND awake AND at unit gain, "
        "so the fast-path share is NOT the 2/6 the lighting bands alone give: the sleeping "
        "stratification (i % 4 == 3) takes a quarter of them away and the answer is 0.25. "
        "The two level-0 night bands are the only ones that draw the threefry field. The "
        "LIGHTING shares are exact when the batch is a multiple of six and otherwise move by "
        "the remainder: at the certification batch of 64 the night share measures 0.328 "
        "against 1/3, while the fast-path share is exactly 0.25 at that batch",
    }

    #: **The measured performance record, at the labels Sami's reference ruling produces.** The rule
    #: applied literally: if the official renderer can be timed on the certification host under the
    #: required conditions -- one core for CPU, same card and same job for GPU, **jitted as craftax
    #: executes it**, warm, compile excluded -- then it IS the reference and the ratio is computed
    #: from it; ``NO_REFERENCE`` is what you retain when it cannot be timed there, never a number
    #: carried over from another machine. Parity is unaffected and stays against the **un-jitted**
    #: function as a uint8 claim (:attr:`parity_note`): the speed ratio and the parity claim use
    #: different references on purpose, because the jitted renderer disagrees with itself in uint8
    #: and the un-jitted one is not what anybody runs.
    #:
    #: Both rungs come from the repo's own classifier (``scripts/regression/semantic/certify.py:classify``) walked
    #: over the frozen ladder in ``tests/fixtures/regression/fidelity/certification.json``, not from reading the thresholds by
    #: hand.
    performance_record_cpu = {
        "tier": "CPU_FASTER",
        "qualifiers": ["BELOW_G4"],
        "mixed_default_fps": 1469.1,
        "mixed_default_batch": 64,
        "official_jitted_fps": 613.4,
        "s_cpu": 2.395,
        "decode_bar_fps": 4294.0,
        "cores_cpu": 0.342,
        "host": "node2339.oscar.ccv.brown.edu",
        "job": "6409963",
        #: The one-core condition, enforced by the cgroup and logged rather than asserted.
        #: ``taskset -c 0`` does not work inside a SLURM cpuset, so the allocation itself is one core.
        "condition": (
            "--cpus-per-task=1; Cpus_allowed_list 1; nproc 1; sched_getaffinity [1]; "
            "torch threads 1; jax default_backend cpu"
        ),
        "paired_verdict": (
            "USABLE -- baseline legs 101,598 and 105,684 f/s (spread 1.040), the "
            "baseline reproduces its known 115,166 f/s to 0.90x, load 0.92 per node "
            "core"
        ),
        "superseded": (
            "an earlier take on node2352 under --cpus-per-task=4 read S_cpu 3.810. It is "
            "NOT the number: torch.set_num_threads(1) binds torch and says nothing about "
            "jax, whose CPU backend has its own pool, so the reference could use four "
            "cores while ours used one. The two takes also differ by node, so the drop "
            "from 3.810 to 2.395 is not cleanly attributable to the core budget -- what "
            "is attributable is that only the second was taken under the condition the "
            "ladder is defined at"
        ),
    }
    #: R is a real ratio here rather than ``NO_REFERENCE``, because the official renderer **could** be
    #: timed on the same card in the same job -- and measuring it **demoted** this cell. On the
    #: cores-only ``NO_REFERENCE`` ladder 21.8 cores of decode reads ``GPU_FAST``; against the
    #: reference the class-R ladder wants ``R >= 500`` for that rung and R is 101, which is
    #: ``GPU_FASTER``. Retaining ``NO_REFERENCE`` would have been the more flattering label and was
    #: available by simply not measuring, which is the reason to measure.
    performance_record_gpu = {
        "tier": "GPU_FASTER",
        "backend_class": "R",
        "mixed_fps": 94343.9,
        "mixed_batch": 4096,
        "size": 10,
        "official_jitted_fps": 934.5,
        "R": 100.96,
        "decode_bar_fps": 4323.3,
        "cores": 21.82,
        "card": "NVIDIA GeForce RTX 3090",
        "host": "gpu2109.oscar.ccv.brown.edu",
        "job": "6410052",
        "condition": (
            "same card, same job; the official timed from $SCR/venvs/se7-craftax-gpu with "
            "jax.default_backend() == 'gpu' and devices ['cuda:0'] ASSERTED before timing, "
            "jitted, warm, 6.72 s of compile excluded; ours timed from "
            "$SCR/venvs/se7-craftax on the same allocation"
        ),
        "batch_scales": (
            "425x (fps(4096)/fps(1)) against the 20x bar, measured on the sibling "
            "gpu2106 run of the same code, job 6397699; the R run drew B=256 and 4096 "
            "only, so its own record carries no B=1 cell"
        ),
        "why_not_no_reference": (
            "$SCR/venvs/se7-craftax is a jax[cpu] build, so jax.default_backend() "
            "is 'cpu' inside a GPU job whatever JAX_PLATFORMS says; job 6409535 "
            "would have timed the 'same-card reference' on the CPU and was "
            "cancelled before it ran. W1's jax[cuda12] venv was reused rather "
            "than a second one built, and it carries no torch, so the two "
            "interpreters share one allocation"
        ),
    }
    #: What the two references are, and why they are allowed to differ.
    reference_labels = {
        "parity": "the UN-JITTED renderer -- a uint8 claim; see parity_note",
        "speed": "the JITTED renderer, as craftax executes it, timed on the certification host",
        "official_unjitted_fps_cpu": 0.43,
        "official_jitted_fps_cpu": 613.4,
        "ratio_between_the_two_references": 1426,
        "note": (
            "S_cpu against the un-jitted function would read 1,808 and certify nothing real. "
            "The ruling is applied literally: the speed ratio uses the renderer as a user runs "
            "it, and NO_REFERENCE is retained only where it cannot be timed on the host."
        ),
    }

    # ------------------------------------------------------------------ state
    #
    # The public :data:`State` groups the 544 packed numbers by what they describe. Five per-tile
    # planes (the block, the item drawn over it, the mob slot, the projectile slot and the light-map
    # value), the player's own four scalars, the day-cycle key, and the 41 inventory/attribute slots
    # the 4-row strip reads. Every offset is the Step-7 packed column it has always been, in the order
    # ``state_names`` gives, so ``pack`` writes the frozen row and the frozen corpus round-trips
    # unchanged.
    #
    # No field has offset ``None``: there is no hidden dynamical state because there is no simulator
    # here at all (``step`` raises the contract's refusal), so the packed row IS the whole state.
    state_layout = (
        (("world", "blocks"), N_TILE, 0),
        (("world", "items"), N_TILE, N_TILE),
        (("world", "mobs"), N_TILE, 2 * N_TILE),
        (("world", "projectiles"), N_TILE, 3 * N_TILE),
        (("world", "light_map"), N_TILE, 4 * N_TILE),
        (("player", "facing"), 1, 5 * N_TILE),
        (("player", "is_sleeping"), 1, 5 * N_TILE + 1),
        (("player", "level"), 1, 5 * N_TILE + 2),
        (("world", "light_level"), 1, 5 * N_TILE + 3),
        (("world", "rng_key"), N_KEY_HALVES, 5 * N_TILE + 4),
    ) + tuple((("inventory", nm), 1, 5 * N_TILE + 4 + N_KEY_HALVES + i) for i, nm in enumerate(STRIP_SLOTS))

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(H, W)`` of one frame. ``size`` is the TILE edge here, not the frame edge.

        ``Env.frame_spec`` assumes a square frame of ``size`` pixels, and this frame is neither
        ``size`` pixels nor square: at the registered 10 px tile it is ``(130, 110)``. A consumer
        allocating from the spec would get a buffer of the wrong shape AND the wrong size, so the cell
        answers for its own geometry, exactly as the classic-control family does for its 400x600.
        """
        return ((OBS_H + STRIP_H) * self.size, OBS_W * self.size)

    def make_variation_space(self) -> Dict:
        """Knobs that are the identity at their defaults and genuinely move pixels off them.

        No per-block colour knob, for Classic's reason: craftax's blocks are 16x16 pixel-art PNGs
        resampled to the tile size, not procedurally drawn shapes, so a "grass colour" would have to
        invent a recolouring rule the reference does not have. ``channel_gain`` is the one global
        recolouring that is exactly the identity at 1.0; the two visibility flags are ablations a
        world-model experiment actually wants.
        """
        return Dict(
            {
                "channel_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
                "player_visible": Discrete(2, init_value=1),
                "mobs_visible": Discrete(2, init_value=1),
            }
        )

    # ---------------------------------------------------------------- decode

    def _decode(self, s: torch.Tensor, n: int):
        """The state row -> the small integers and floats the tables are indexed and scaled by.

        Every class id is a **wrap**, never a clamp: ``check_state_dims_matter`` probes a dimension by
        adding 25 to it, and a clamping decoder leaves a slot that already holds the last class
        insensitive to the probe, which is reported as a dead state dimension when it is an artifact
        of the probe. Real states are always in range so the rule never fires in practice.

        The two floats fold instead of wrapping. ``light_level`` and each per-tile ``light_map`` value
        live in ``[0, 1]`` and a plain ``x - floor(x)`` would send 1.0 to 0.0, turning every fully lit
        state into darkness; a **tent of period 2** is the identity on ``[0, 1]`` and folds everything
        else back into it. It is applied only outside ``[0, 1]``, so an in-range value passes through
        bit-exact rather than through two float32 roundings.
        """
        k = N_TILE
        blk = s[:, 0:k].round().remainder(N_BLOCK).long().view(n, OBS_H, OBS_W)
        itm = s[:, k : 2 * k].round().remainder(N_ITEM).long().view(n, OBS_H, OBS_W)
        mob = s[:, 2 * k : 3 * k].round().remainder(N_MOB_SLOT).long().view(n, OBS_H, OBS_W)
        prj = s[:, 3 * k : 4 * k].round().remainder(N_PROJ_SLOT).long().view(n, OBS_H, OBS_W)
        lm = _tent(s[:, 4 * k : 5 * k]).view(n, OBS_H, OBS_W)
        j = 5 * k
        facing = s[:, j].round().remainder(N_FACING).long()
        sleeping = s[:, j + 1] > 0.5
        level = s[:, j + 2].round().remainder(N_LEVEL).long()
        light = _tent(s[:, j + 3])
        halves = s[:, j + 4 : j + 4 + N_KEY_HALVES].round().remainder(65536.0)
        cnt = s[:, j + 4 + N_KEY_HALVES :].round().remainder(_strip_index(s.device)[3].view(1, N_STRIP)).long()
        # the reference draws max(floor(player_health), 1), so a health slot of 0 is a 1
        cnt[:, _SLOT["health"]] = cnt[:, _SLOT["health"]].clamp_min(1)
        daylight = torch.where(level == 0, light, torch.ones_like(light))
        return blk, itm, mob, prj, lm, facing, sleeping, level, light, daylight, halves, cnt

    # ---------------------------------------------------------------- tile ids

    def _tile_ids(self, dec, sty):
        """``(bg (n,13,11), ov (n,9,11), dg (n,4,11), player_slot (n,))`` -- one table row per tile.

        Everything that could have been a branch is folded in here, so one batch may hold any mixture
        of levels, lighting bands, sleeping and awake players and appearances.
        """
        blk, itm, mob, prj, _lm, facing, sleeping, level, _l, _d, _h, cnt = dec
        n, dev = blk.shape[0], blk.device
        idx, bg0, dg0, _mod = _strip_index(dev)

        vis_m = (sty["mobs_visible"] > 0.5).view(n, 1, 1)
        m = torch.where(vis_m, mob, torch.zeros_like(mob))
        world_bg = blk * N_ITEM + itm
        ov = m * N_PROJ_SLOT + prj

        pslot = torch.where(sleeping, torch.full_like(facing, PLAYER_ASLEEP), facing)
        pslot = torch.where(sty["player_visible"].view(n) > 0.5, pslot, torch.full_like(pslot, NO_PLAYER))

        bg = bg0.view(1, -1).expand(n, -1).clone()
        dg = dg0.view(1, -1).expand(n, -1).clone()

        def two_digit(v):
            """``ones * 10 + tens`` for ``_render_two_digit_number``: 0 draws nothing, 1..9 draws one
            digit from the ``with_zero`` alphabet, 10..99 adds a tens digit."""
            ones = torch.where(v == 0, torch.zeros_like(v), N_DIGIT + v.remainder(N_DIGIT))
            return ones * N_TENS + v.div(N_DIGIT, rounding_mode="floor")

        if "count" in idx:
            pos, base, slot = idx["count"]
            v = cnt[:, slot]
            bg[:, pos] = base + (v > 0).long()
            dg[:, pos] = two_digit(v)
        if "flag" in idx:
            pos, base, slot = idx["flag"]
            bg[:, pos] = base + (cnt[:, slot] > 0).long()
        if "attr_cond" in idx:
            pos, base, slot = idx["attr_cond"]
            v = cnt[:, slot]
            bg[:, pos] = base + (v > 0).long()
            dg[:, pos] = v * N_TENS
        if "attr_fixed" in idx:
            pos, base, slot = idx["attr_fixed"]
            bg[:, pos] = base
            dg[:, pos] = cnt[:, slot] * N_TENS
        if "tool" in idx:
            pos, base, slot = idx["tool"]
            bg[:, pos] = base + cnt[:, slot]
        if "armour" in idx:
            pos, base, a, e = idx["armour"]
            bg[:, pos] = base + cnt[:, a] * 3 + cnt[:, e]
        if "sword" in idx:
            pos, base, a, e = idx["sword"]
            bg[:, pos] = base + cnt[:, a] * 3 + cnt[:, e]
        if "arrows" in idx:
            pos, base, a, e = idx["arrows"]
            bg[:, pos] = base + cnt[:, a] * 3 + cnt[:, e]
        if "level" in idx:
            pos, base = idx["level"]
            bg[:, pos] = base
            dg[:, pos] = level.view(n, 1) * N_TENS
        strip_bg = bg.view(n, STRIP_H, OBS_W)
        return (
            torch.cat([world_bg, strip_bg], dim=1).to(torch.int32),
            ov.to(torch.int32),
            dg.view(n, STRIP_H, OBS_W).to(torch.int32),
            pslot.to(torch.int32),
        )

    # ---------------------------------------------------------------- render

    @staticmethod
    def _rows(ids: torch.Tensor, t: int) -> torch.Tensor:
        """``(n, H, W)`` tile ids -> the flat row index of every output scanline, in output order."""
        a = torch.arange(t, dtype=torch.int32, device=ids.device).view(1, 1, t, 1)
        return (ids.unsqueeze(2) * t + a).reshape(-1)

    @torch.no_grad()
    def render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(..., 544)`` -> ``(..., 13*size, 11*size, 3)`` uint8, in the reference's own order.

        There is no ``scene()``: the frame is a gather from texture tables with three stencil selects
        over it, not a rasterisation of vector primitives. One gather draws the whole background --
        world tiles and the strip's icon cells come out of the same table -- then the player lands on
        one tile, the mob/projectile pair on the world rows and the digit pair on the strip rows.

        A batch that is **fully lit, awake and at unit gain** never leaves uint8: the composite is a
        select and a select of rounded values is the rounding of a select, so the uint8 tables give
        exactly what the float path would. Anything else takes the float tables, because the
        ``light_map`` multiply, the night chain, the sleep halving and ``channel_gain`` all happen
        before the rounding.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        n = s.shape[0]
        t = int(size or self.size)
        if supersample not in (None, 1):
            raise ValueError(
                f"CraftaxFull.render: supersample={supersample} is meaningless here -- the frame is a "
                f"gather from a bitmap atlas, so there is no sub-pixel coverage to estimate."
            )
        sty = self.resolve_style(style, variation, batch=n)
        dec = self._decode(s, n)
        img = self._render_atlas(dec, sty, t, out)
        H = (OBS_H + STRIP_H) * t
        if out is not None and img.data_ptr() != out.data_ptr():
            out.view(n, H, OBS_W * t, 3).copy_(img)
            img = out.view(n, H, OBS_W * t, 3)
        return img.view(*lead, *img.shape[1:]) if lead else img

    def _render_atlas(self, dec, sty, t, out) -> torch.Tensor:
        """The gather. **The composite never leaves uint8**, whatever the lighting.

        Every table entry is an integer -- craftax's textures are integral in 0..255 and a stencil
        composite of integers is an integer -- so the uint8 composite and the float one are the same
        numbers, and converting afterwards is exact rather than a rounding. That is worth about 3x on
        a lit frame, because the background gather and the three stencil layers -- seven
        ``index_select``s and three selects in all -- move one byte a channel instead of four, and the
        4 of 13 rows the strip occupies never become float at all at unit gain.
        """
        blk, itm, mob, prj, lm, facing, sleeping, level, light, daylight, halves, cnt = dec
        n, dev = blk.shape[0], blk.device
        H, W = (OBS_H + STRIP_H) * t, OBS_W * t
        wh = OBS_H * t
        tab = _Tables.get(dev, t)
        bg_id, ov_id, dg_id, pslot = self._tile_ids(dec, sty)

        # One host read, not four: each is a device->host synchronisation on CUDA, and the two
        # questions -- "is the appearance the default one" and "can this batch skip the float chain
        # entirely" -- are answered from one stack. The chain is the identity when every tile is fully
        # lit (``light_map == 1`` multiplies by one), the daylight override has left every row at 1.0
        # (``1*px + 0*night`` is exactly ``px``) and nothing is asleep.
        gain = sty["channel_gain"]
        flags = torch.stack(
            [torch.all(gain == 1.0), torch.all(daylight == 1.0) & torch.all(lm == 1.0) & ~sleeping.any()]
        )
        unit_gain, identity_chain = (bool(x) for x in flags.tolist())
        plain = unit_gain and identity_chain

        img = out.view(n, H, W, 3) if out is not None else torch.empty(n, H, W, 3, dtype=torch.uint8, device=dev)
        torch.index_select(tab["bg"]["u8"], 0, self._rows(bg_id, t), out=img.view(-1, t * 3))

        # the player, on the centre tile only -- 1/99 of the frame, so it never earns a full-frame pass
        pl_rows = (
            pslot.view(n, 1).to(torch.int32) * t + torch.arange(t, dtype=torch.int32, device=dev).view(1, t)
        ).reshape(-1)
        cr, cc = OBS_H // 2, OBS_W // 2
        k = tab["pl"]
        sl = img[:, cr * t : (cr + 1) * t, cc * t : (cc + 1) * t]
        sl.copy_(
            torch.where(
                torch.index_select(k["m"], 0, pl_rows).view(n, t, t, 3),
                torch.index_select(k["u8"], 0, pl_rows).view(n, t, t, 3),
                sl,
            )
        )

        def over(name, ids, tile_rows, lo, hi):
            rows = self._rows(ids, t)
            k = tab[name]
            v = img[:, lo:hi]
            v.copy_(
                torch.where(
                    torch.index_select(k["m"], 0, rows).view(n, tile_rows * t, W, 3),
                    torch.index_select(k["u8"], 0, rows).view(n, tile_rows * t, W, 3),
                    v,
                )
            )

        over("ov", ov_id, OBS_H, 0, wh)  # mobs and projectiles, on the world rows
        over("dg", dg_id, STRIP_H, wh, H)  # the strip's digits, on the strip rows
        if plain:
            return img

        def lit(world):
            lmp = lm.repeat_interleave(t, 1).repeat_interleave(t, 2).unsqueeze(-1)
            return light_night_and_sleep(world, lmp, daylight, sleeping, prng.key_from_halves(halves))

        if unit_gain:
            # ``plain`` already returned, so the chain is NOT the identity here. The strip is never
            # lit, so at unit gain it stays exactly as gathered and only the 9 of 13 world rows make
            # the float round trip.
            img[:, :wh] = lit(img[:, :wh].to(torch.float32)).round_().clamp_(0, 255).to(torch.uint8)
            return img
        # a non-default channel_gain multiplies before the rounding, so the WHOLE frame -- strip
        # included -- has to be float. This is off the certified appearance by construction.
        px = img.to(torch.float32)
        if not identity_chain:
            px[:, :wh] = lit(px[:, :wh])
        px = px * gain.to(px.dtype).view(n, 1, 1, 3)
        return px.round_().clamp_(0, 255).to(torch.uint8)

    # ---------------------------------------------------------------- sampling

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States stratified over (level, lighting band, light map, sleeping), with the strip drawn.

        **Stratified by row index, not by luck**, for the reason W1's Classic sampler is: three
        families of state dimension are unobservable outside a specific band, and every one of them is
        a property of the reference rather than of this renderer.

        ===============================  ==========================================================
        dimension                        reaches a pixel only when
        ===============================  ==========================================================
        the four ``state_rng`` halves    ``player_level == 0`` **and** ``light_level < 0.5``
        ``light_level``                  ``player_level == 0`` (``daylight`` is forced to 1.0 above)
        the 99 ``light_map`` slots        the tile is not fully lit
        ===============================  ==========================================================

        plus a fourth: ``ench_bow`` reaches no pixel on a row holding no arrows, because the reference
        draws ``arrow_enchantment_textures[bow_enchantment * (arrows > 0)]``.

        So **row 0 is always the richest band** -- level 0, night, dark tiles in the window, the
        player awake and at least one arrow in the inventory -- and a ``batch=1`` instance still has
        every dimension live. The five other
        bands follow on ``i % 6`` and cover the two uint8 fast paths, dusk, and underground darkness.

        **The strip is drawn directly rather than harvested**, and that is sound rather than
        convenient: each cell is a function of its own value alone (measured -- every write lies
        inside its own cell), and the simulator caps every count at 99. A 4,000-step random-policy
        rollout leaves **25 of the 41 strip slots at zero for every step** and never descends past
        level 0, so a pool built from rollouts alone would present ~25 dead dimensions -- the failure
        ``docs/research/pixel_native.md`` records for Procgen miner and heist.

        Mob budgets are the simulator's (``max_melee_mobs=3, max_passive_mobs=3, max_ranged_mobs=2``,
        three projectiles each way) and no entity is placed on the centre tile, so ``player_visible``
        is never inert on a sampled frame.

        Sub-1 ``light_map`` values on level 0 are what the renderer itself sees at the map border,
        where it pads the light map with 0; intermediate values there are drawn directly, which the
        renderer treats identically and which :meth:`reference_render` writes back verbatim.
        """
        rng = np.random.default_rng() if rng is None else rng
        b = int(batch)
        st = np.zeros((b, self.state_dim), np.float32)
        k = N_TILE
        st[:, 0:k] = rng.integers(2, N_BLOCK, size=(b, k))  # 0/1 are INVALID/OUT_OF_BOUNDS
        st[:, k : 2 * k] = rng.integers(1, N_ITEM, size=(b, k)) * (rng.random((b, k)) < 0.30)
        # The centre tile is under the player sprite, so most of it is hidden. TORCH is a small icon
        # that lands exactly where the player is opaque and is invisible there at 10 px; a ladder is
        # not. Every other row therefore carries a ladder at the centre, so the ``item_r4c5``
        # dimension is reachable at any batch size -- measured: without this it reported dead at batch
        # 1 through 8 while being demonstrably live, which is the shape of a wrong state contract.
        centre = (OBS_H // 2) * OBS_W + (OBS_W // 2)
        even = np.arange(b) % 2 == 0
        st[even, k + centre] = rng.integers(I_LADDER_DOWN, N_ITEM, size=int(even.sum()))
        j = 5 * k
        band = np.arange(b) % 6
        lvl = np.where(np.isin(band, (3, 4)), rng.integers(1, N_LEVEL, size=b), 0)
        light = np.empty(b, np.float32)
        lit = np.ones((b, k), np.float32)
        # the alphabet without its endpoints: a tile at light_map 0 is BLACK whatever block, item, mob
        # or projectile it holds, so its four id dimensions die with it, and a tile already at 1.0 is
        # insensitive to the conformance suite's +25 probe (the tent sends 25 to exactly 1.0).
        # Band 0 -- the richest band, and the one a batch of 1 gets -- therefore dims without
        # extinguishing, and the darker draws live in band 4.
        # ... and not too far from 1 either, for a reason that is arithmetic rather than aesthetic:
        # the frame is uint8, so a dim tile ROUNDS a small colour change away. At light_map 0.11 a
        # block/item swap worth four levels lands inside one uint8 step and the dimension reads as
        # dead. Band 0 dims to 0.6 and below, band 4 carries the deep values -- and the same argument
        # caps band 0's night depth at 0.3, since the night blend halves the range again.
        partial = LIGHT_MAP_ALPHABET[(LIGHT_MAP_ALPHABET >= 0.6) & (LIGHT_MAP_ALPHABET < 1.0)]
        for i in range(b):
            if band[i] == 5:
                light[i] = rng.random() * NIGHT_THRESHOLD
            elif band[i] == 0:
                light[i] = 0.3 + rng.random() * (NIGHT_THRESHOLD - 0.3)
            elif band[i] == 1:
                light[i] = NIGHT_THRESHOLD + rng.random() * (1.0 - NIGHT_THRESHOLD)
            else:
                light[i] = 1.0
            if band[i] == 0:
                lit[i] = rng.choice(partial, size=k)
                # DARKNESS renders as an all-zero tile, and zero times any light is still zero, so a
                # DARKNESS tile has a dead light dimension. Band 0 carries none.
                dark_block = st[i, 0:k] == B_DARKNESS
                st[i, 0:k] = np.where(dark_block, B_GRASS, st[i, 0:k])
                # **Every tile in band 0 carries a ladder**, and that is the one place this sampler
                # departs from a plain draw. The item alphabet is NONE, TORCH and three ladders, and
                # ``check_state_dims_matter``'s fallback nudge on a NONE slot is ``+1``, which lands
                # on TORCH -- a small icon that disappears entirely under the opaque part of a mob or
                # projectile sprite. Measured: with a plain draw, a ``batch=1`` instance reported one
                # or two ``item_r*c*`` dimensions dead on half of twelve seeds, every one of them a
                # tile under an overlay. With ladders it reports none, and a ladder still leaves the
                # block beneath it visible, so no other dimension is traded away for it.
                st[i, k : 2 * k] = rng.integers(I_LADDER_DOWN, N_ITEM, size=k)
            elif band[i] == 4:
                dark = rng.random(k) < 0.5
                lit[i] = np.where(dark, rng.choice(LIGHT_MAP_ALPHABET, size=k), 1.0)
        st[:, 4 * k : 5 * k] = lit
        st[:, j] = rng.integers(0, N_FACING, size=b)
        st[:, j + 1] = (np.arange(b) % 4 == 3).astype(np.float32)
        st[:, j + 2] = lvl
        st[:, j + 3] = light
        st[:, j + 4 : j + 4 + N_KEY_HALVES] = rng.integers(0, 65536, size=(b, N_KEY_HALVES))
        for c, name in enumerate(STRIP_SLOTS):
            st[:, j + 4 + N_KEY_HALVES + c] = rng.integers(0, STRIP_MOD[c], size=b)
        st[:, j + 4 + N_KEY_HALVES + _SLOT["health"]] = rng.integers(1, N_COUNT, size=b)
        # **Fifth guard, and the last one independent review found.** The reference draws the arrow
        # enchantment as ``arrow_enchantment_textures[bow_enchantment * (arrows > 0)]``, so on a row
        # holding no arrows the ``ench_bow`` slot reaches no pixel at all -- and a ``batch=1``
        # instance is one row. Measured before this line existed: ``inv_ench_bow`` read dead on 3 of
        # 200 seeds (100, 202, 248). Band 0 therefore always carries at least one arrow.
        #
        # It is a CLAMP and not a redraw, deliberately: a redraw would consume a further draw from
        # ``rng`` and shift every subsequent value, which would change the frozen corpus. No band-0
        # row in the frozen pool holds zero arrows, so this line moves nothing there --
        # ``test_the_arrow_guard_leaves_the_frozen_corpus_untouched`` asserts that against the
        # recorded sha256 rather than asserting it in prose.
        arrows_at = j + 4 + N_KEY_HALVES + _SLOT["arrows"]
        st[band == 0, arrows_at] = np.maximum(st[band == 0, arrows_at], 1.0)

        # mobs and projectiles: the simulator's budgets, never on the centre tile
        cr, cc = OBS_H // 2, OBS_W // 2
        centre = cr * OBS_W + cc
        for i in range(b):
            free = [p for p in range(N_TILE) if p != centre]
            rng.shuffle(free)
            it = iter(free)
            ok = transparent_mob_slots(_source_size(self.size))
            for base, n_type, cap in (
                (MOB_BASE["melee"], N_MELEE, MAX_MELEE),
                (MOB_BASE["passive"], N_PASSIVE, MAX_PASSIVE),
                (MOB_BASE["ranged"], N_RANGED, MAX_RANGED),
            ):
                for _ in range(int(rng.integers(0, cap + 1))):
                    v = base + int(rng.integers(0, n_type))
                    if band[i] == 0 and v not in ok:
                        v = base + (v - base + 1) % n_type
                    st[i, 2 * k + next(it)] = v
            free2 = [p for p in range(N_TILE) if p != centre and st[i, 2 * k + p] == 0]
            rng.shuffle(free2)
            it2 = iter(free2)
            for _ in range(int(rng.integers(0, MAX_MOB_PROJ + MAX_PLAYER_PROJ + 1))):
                st[i, 3 * k + next(it2)] = 1 + int(rng.integers(0, N_PROJ_TYPE)) * 4 + int(rng.integers(0, 4))
        return st

    # ---------------------------------------------------------------- reference bridge

    @staticmethod
    def reference_env(size: int = TILE_PX_AGENT):
        """The official full-Craftax renderer at ``size`` px per tile, plus a template ``EnvState``.

        A tuple rather than a gym env because Craftax is a functional JAX env: the renderer is a pure
        function of ``EnvState``. Only the sizes craftax ships textures for are accepted -- anything
        else has no official frame, and inventing one by resampling would compare this renderer
        against a second implementation of itself.

        **Not jitted, and that is a correctness decision rather than a performance one.** Under
        ``jax.jit`` XLA fuses the arithmetic differently and the reference disagrees with its own
        un-jitted self by up to 1.53e-05 in float32 -- invisible in uint8 on a fully lit frame, and
        not invisible wherever the picture is compressed towards one value, which the ``light_map``
        multiply, the night blend and the sleep halving all do. The un-jitted function is the renderer
        as written, so it is the one certified against. It costs about 0.34 s a frame instead of
        0.00066 s, which is the right trade for a check that runs on tens of states and is the whole
        basis of a parity claim -- and it is why the ``S_cpu`` denominator is reported against both.

        ``jax_threefry_partitionable`` is asserted rather than assumed: jax has two threefry code
        paths, they agree on 0 of 448 draws, and the flag's default flipped to ``True`` in jax 0.4.36.
        ``do_night_noise`` is left at its default ``True``, which is what the registered env uses.
        """
        if int(size) not in REFERENCE_TILE_PX:
            raise ValueError(
                f"craftax draws full Craftax at {REFERENCE_TILE_PX} px per tile; there is no official "
                f"frame to compare a {size} px render against"
            )
        import jax
        from craftax.craftax.renderer import make_craftax_pixel_renderer
        from craftax.craftax_env import make_craftax_env_from_name

        if not bool(jax.config.jax_threefry_partitionable):
            raise RuntimeError(
                "CraftaxFull.reference_env: jax_threefry_partitionable is off, so this jax draws the "
                "night noise field with the pre-0.4.36 threefry counter layout. That is a different "
                "field for the same state_rng (0 of 448 draws agree between the two paths), so a "
                "parity number taken against it would not be a parity number. Run with the flag at "
                "its default (True), or declare the other path and re-measure."
            )
        env = make_craftax_env_from_name("Craftax-Pixels-v1", auto_reset=False)
        _obs, template = env.reset(jax.random.PRNGKey(0), env.default_params)
        return (make_craftax_pixel_renderer(int(size), do_night_noise=True), template, int(size))

    @staticmethod
    def reference_render(env, states) -> np.ndarray:
        """Official frames for the given render states -- the WHOLE observation, no crop.

        The window is written back into a full ``EnvState``: the block and item windows go into the
        middle of the player's own level with the player at their centre, the per-tile ``light_map``
        goes in beside them, each occupied tile becomes one mob of the right class and type or one
        projectile whose direction is chosen to classify to the stored orientation, and the 41 strip
        numbers go into the intrinsics, the inventory, the enchantments and the learned spells.

        **The two global substitutions are pinned rather than reproduced.**
        ``boss_timesteps_to_spawn_this_round`` is set to 1, which makes ``is_boss_vulnerable`` false
        whatever the mobs are doing, so a stored ``NECROMANCER`` renders as ``NECROMANCER`` and the
        vulnerable sprite is reached by storing its own id (which the substitution does not touch).
        ``monsters_killed[level]`` is set to the clear threshold, so a stored ``LADDER_DOWN`` renders
        open and the blocked sprite is reached by storing ``LADDER_DOWN_BLOCKED`` directly. Both
        substitutions are thereby absorbed into the tile id, which is what makes them cost no state
        dimension.
        """
        import jax
        import jax.numpy as jnp

        render, tmpl, _tile_px = env

        def _i(field, v):
            return jnp.asarray(v, jnp.asarray(field).dtype)

        s = np.asarray(states, dtype=np.float32).reshape(-1, CraftaxFull.state_dim)
        k = N_TILE
        L, HH, WW = np.asarray(tmpl.map).shape
        py, px_ = HH // 2, WW // 2
        r0, c0 = py - OBS_H // 2, px_ - OBS_W // 2
        cr, cc = OBS_H // 2, OBS_W // 2
        j = 5 * k
        out = []
        for row in s:
            blk = np.round(row[0:k]).astype(np.int64) % N_BLOCK
            itm = np.round(row[k : 2 * k]).astype(np.int64) % N_ITEM
            mob = np.round(row[2 * k : 3 * k]).astype(np.int64) % N_MOB_SLOT
            prj = np.round(row[3 * k : 4 * k]).astype(np.int64) % N_PROJ_SLOT
            lit = _tent_np(row[4 * k : 5 * k])
            facing = int(np.round(row[j])) % N_FACING
            sleeping = bool(row[j + 1] > 0.5)
            lvl = int(np.round(row[j + 2])) % N_LEVEL
            light = _tent_np(np.asarray(row[j + 3]))
            halves = np.round(row[j + 4 : j + 4 + N_KEY_HALVES]).astype(np.int64) % 65536
            key = np.array([(halves[0] << 16) | halves[1], (halves[2] << 16) | halves[3]], np.uint32)
            cnt = np.round(row[j + 4 + N_KEY_HALVES :]).astype(np.int64) % np.asarray(STRIP_MOD)
            cnt[_SLOT["health"]] = max(int(cnt[_SLOT["health"]]), 1)

            def win(a, v, fill=None):
                a = np.asarray(a).copy()
                a[lvl, r0 : r0 + OBS_H, c0 : c0 + OBS_W] = v.reshape(OBS_H, OBS_W)
                return a

            m = win(tmpl.map, blk)
            im = win(tmpl.item_map, itm)
            lmv = win(tmpl.light_map, lit.astype(np.float32))
            mk = np.asarray(tmpl.monsters_killed).copy()
            mk[lvl] = MONSTERS_KILLED_TO_CLEAR_LEVEL

            def mobs_of(base, n_type, cap, template_mobs):
                pos = np.zeros((L,) + np.asarray(template_mobs.position).shape[1:], np.int32)
                msk = np.zeros(np.asarray(template_mobs.mask).shape, bool)
                tid = np.zeros(np.asarray(template_mobs.type_id).shape, np.int32)
                n = 0
                for p in range(k):
                    v = int(mob[p])
                    if base <= v < base + n_type and n < cap:
                        pos[lvl, n] = (r0 + p // OBS_W, c0 + p % OBS_W)
                        msk[lvl, n], tid[lvl, n] = True, v - base
                        n += 1
                return template_mobs.replace(
                    position=jnp.asarray(pos, jnp.int32), mask=jnp.asarray(msk), type_id=_i(template_mobs.type_id, tid)
                )

            occupied = [(p, int(prj[p])) for p in range(k) if prj[p] != 0]

            def proj_of(template_mobs, take, dirs_template):
                pos = np.zeros((L,) + np.asarray(template_mobs.position).shape[1:], np.int32)
                msk = np.zeros(np.asarray(template_mobs.mask).shape, bool)
                tid = np.zeros(np.asarray(template_mobs.type_id).shape, np.int32)
                dr = np.zeros(np.asarray(dirs_template).shape, np.int32)
                for n, (p, v) in enumerate(take):
                    pos[lvl, n] = (r0 + p // OBS_W, c0 + p % OBS_W)
                    msk[lvl, n] = True
                    tid[lvl, n] = (v - 1) // 4
                    dr[lvl, n] = PROJ_DIR[(v - 1) % 4]
                return (
                    template_mobs.replace(
                        position=jnp.asarray(pos, jnp.int32),
                        mask=jnp.asarray(msk),
                        type_id=_i(template_mobs.type_id, tid),
                    ),
                    _i(dirs_template, dr),
                )

            mp, mpd = proj_of(tmpl.mob_projectiles, occupied[:MAX_MOB_PROJ], tmpl.mob_projectile_directions)
            pp, ppd = proj_of(
                tmpl.player_projectiles,
                occupied[MAX_MOB_PROJ : MAX_MOB_PROJ + MAX_PLAYER_PROJ],
                tmpl.player_projectile_directions,
            )
            inv = tmpl.inventory
            st = tmpl.replace(
                map=_i(tmpl.map, m),
                item_map=_i(tmpl.item_map, im),
                light_map=jnp.asarray(lmv, jnp.float32),
                monsters_killed=_i(tmpl.monsters_killed, mk),
                boss_timesteps_to_spawn_this_round=_i(tmpl.boss_timesteps_to_spawn_this_round, 1),
                player_position=jnp.asarray([py, px_], jnp.int32),
                player_level=_i(tmpl.player_level, lvl),
                player_direction=_i(tmpl.player_direction, facing + 1),
                is_sleeping=jnp.bool_(sleeping),
                light_level=jnp.float32(light),
                state_rng=jnp.asarray(key, jnp.uint32),
                player_health=jnp.float32(cnt[_SLOT["health"]]),
                player_food=_i(tmpl.player_food, cnt[_SLOT["food"]]),
                player_drink=_i(tmpl.player_drink, cnt[_SLOT["drink"]]),
                player_energy=_i(tmpl.player_energy, cnt[_SLOT["energy"]]),
                player_mana=_i(tmpl.player_mana, cnt[_SLOT["mana"]]),
                player_xp=_i(tmpl.player_xp, cnt[_SLOT["xp"]]),
                player_dexterity=_i(tmpl.player_dexterity, cnt[_SLOT["dexterity"]]),
                player_strength=_i(tmpl.player_strength, cnt[_SLOT["strength"]]),
                player_intelligence=_i(tmpl.player_intelligence, cnt[_SLOT["intelligence"]]),
                sword_enchantment=_i(tmpl.sword_enchantment, cnt[_SLOT["ench_sword"]]),
                bow_enchantment=_i(tmpl.bow_enchantment, cnt[_SLOT["ench_bow"]]),
                armour_enchantments=_i(tmpl.armour_enchantments, [cnt[_SLOT[f"ench_armour{i}"]] for i in range(4)]),
                learned_spells=jnp.asarray([bool(cnt[_SLOT["spell_fireball"]]), bool(cnt[_SLOT["spell_iceball"]])]),
                inventory=inv.replace(
                    armour=_i(inv.armour, [cnt[_SLOT[f"armour{i}"]] for i in range(4)]),
                    potions=_i(inv.potions, [cnt[_SLOT[f"potion{i}"]] for i in range(6)]),
                    **{
                        n: _i(getattr(inv, n), cnt[_SLOT[n]])
                        for n in (
                            "wood",
                            "stone",
                            "coal",
                            "iron",
                            "diamond",
                            "sapphire",
                            "ruby",
                            "sapling",
                            "torches",
                            "books",
                            "arrows",
                            "pickaxe",
                            "sword",
                            "bow",
                        )
                    },
                ),
                melee_mobs=mobs_of(MOB_BASE["melee"], N_MELEE, MAX_MELEE, tmpl.melee_mobs),
                passive_mobs=mobs_of(MOB_BASE["passive"], N_PASSIVE, MAX_PASSIVE, tmpl.passive_mobs),
                ranged_mobs=mobs_of(MOB_BASE["ranged"], N_RANGED, MAX_RANGED, tmpl.ranged_mobs),
                mob_projectiles=mp,
                mob_projectile_directions=mpd,
                player_projectiles=pp,
                player_projectile_directions=ppd,
            )
            out.append(np.round(np.asarray(jax.device_get(render(st)))).clip(0, 255).astype(np.uint8))
        return np.stack(out)
