"""Crafter — the whole 64x64 observation, inventory strip included, from 80 numbers.

Reference: ``crafter`` 1.8.3 (``CrafterReward-v1``, Hafner 2021). One frame is
``crafter.env.Env.render``: a 9x7 window of the world map at 7 px per tile, each cell drawn as a
material texture with at most one object alpha-blended over it, then a daylight post-process, then a
2x9 inventory strip concatenated underneath, then the whole thing pasted into a 64x64 canvas. Every
one of those steps is a lookup into a fixed set of 56 16x16 PNGs plus per-pixel arithmetic, which is
why the family belongs here.

**The state Crafter hands you is bigger than the frame, and the render window is not.** Measured
(``scratch/crafter/probe.py``): the world's two arrays alone — ``_mat_map`` (64x64 uint8) and
``_obj_map`` (64x64 uint32) — are 20,480 bytes against a 12,288-byte frame, **1.67x worse**, before
counting the object list, the player's counters or the RNG. That is the same shape of result the
Craftax note records for a JAX reimplementation of this game (6.37x), and it has the same cause and
the same fix: the simulator's state carries a 64x64 world and the frame shows a 9x7 window of it.
What this env carries is the **render window**: 63 cell codes, 16 inventory counts and the daylight
level, **80 float32 = 320 bytes against 12,288 (38.4x)**, or 80 bytes packed to the one byte per
field the alphabet needs (**153.6x**).

Parity against the official renderer, measured twice and in both directions:

  * **599 of 599 frames bit-identical** along a live 690-step uniform-random rollout, at every daylight
    level in [0.5, 1] and including 84 frames with the player asleep (``scratch/crafter/probe.py``);
  * **512 of 512 frames bit-identical** over the whole embedded state pool this env samples from,
    driven through ``reference_render`` — i.e. the world rebuilt from the state and drawn by
    ``crafter.Env.render`` — including 223 windows that leave the world (``scratch/crafter/parity.py``:
    ``exact_frac 1.0, mean |diff| 0.000, max |diff| 0``).

The atlas is exact by construction rather than by fitting — every tile is composed with crafter's own
``_draw`` / ``_draw_alpha`` / ``ItemView``, checked at every tile size from 3 to 40 px
(``scratch/crafter/gen_assets.py``: max abs diff 0) — so ``parity_interior_min`` stays at 1.0 with
nothing relaxed.

Throughput on one H200: **701,594 frames/s at batch 16,384** (8.6 GB/s of frame output; 244,566 f/s at
batch 256) against **1,375 f/s** for ``crafter.Env.render`` on a live CPU env — **510x**.

One cell code, not a (material, object) pair
--------------------------------------------

Each of the 63 cells is ONE index into ``material x object-texture``, 13 x 16 = 208 combinations, plus
one code for a cell outside the world. Splitting it into two dimensions would be the same mistake
MiniGrid's docstring names: an opaque object hides its material, so on such a cell the material
dimension provably cannot change the picture, and a state with dimensions that cannot reach the frame
is either not minimal or not fully rendered. One code per cell makes every dimension load-bearing, and
it is also what makes the frame a single gather.

The player's facing is not a dimension either, and neither is "is the player asleep": both are already
in the centre cell's code (``player-left`` / ``player-up`` / ... / ``player-sleep``), because
``Player.texture`` is exactly that function. ``player-sleep`` in the centre is what switches the
sleeping post-process on — a biconditional in the reference, so carrying a separate flag could only
introduce a state that contradicts itself.

What is excluded, and why
-------------------------

  * **Night below half daylight.** ``LocalView._light`` mixes in a vignette of noise drawn from
    ``self._world.random`` when ``daylight < 0.5``, so those frames are not a function of any state —
    they are a function of the world's RNG cursor, and rendering the same state twice gives two
    pictures. This is a G1 failure in the reference, not a limitation here, and the honest response is
    to carry the half of the cycle that is pure: ``daylight >= 0.5`` is **49% of steps** and every
    state this env samples or renders against a reference is in it. Above the threshold the noise
    branch is dead and the rest of the post-process is exact arithmetic, reproduced here bit for bit.
  * **The world outside the window, and therefore ``step``.** This state is an OBSERVATION. The next
    observation depends on cells the window does not contain, on objects outside it, on the player's
    hidden hunger/thirst/fatigue counters and on the world RNG, so there is nothing to step and no
    batched simulator is offered. That is the same split MiniGrid draws between its top-down classes
    (which do ship a ``step``) and its POV class (which cannot).

Constants that were measured rather than read off the source
------------------------------------------------------------

  * **``PIL.Image.resize(..., NEAREST)`` is not ``floor((i+0.5)*src/dst)``.** That rule — the one
    Craftax's atlas uses and the one any reading of Pillow's affine transform suggests — disagrees with
    Pillow 12.1.1 on **34 of the first 64 output sizes**, always at an output pixel whose exact
    coordinate is an integer: at ``src=16, dst=6`` the rule says source row 12 and Pillow takes 11,
    because Pillow divides before it multiplies and 2.6666666666666665 x 4.5 is 11.999999999999998.
    Crafter resizes every texture through that call, so the atlas would have been wrong by a whole
    source row on 8 of the 14 tile sizes an ordinary render asks for. The index map is therefore
    **measured off Pillow and embedded** (``_NEAREST_B64``), not derived.
  * **The alpha blend is float32 and truncating.** ``engine._draw_alpha`` divides by 255, blends in
    float32 and multiplies back with ``.astype(np.uint8)``, which truncates. Composing in float64, or
    rounding, moves interior pixels of every sprite by one level.
  * **``_draw`` does not blend.** A material texture with an alpha channel is written with its alpha
    DISCARDED (``texture[..., :3]``), so a transparent material texel paints its RGB rather than the
    canvas's 127. Three of the twelve materials ship an alpha channel.
  * **The daylight blend is a truncation, not a rounding.** ``_light`` returns float64 and
    ``Env.render`` assigns it into a uint8 canvas, which is a C cast. Rounding instead costs about a
    third of all pixels one level.
  * **The inventory strip is never lit.** ``render`` concatenates the item view AFTER ``_light`` has
    run on the world view, so the strip is the same picture at midnight and at noon.

The atlas
---------

The decoder is 56 PNGs — **9,452 bytes of base64** at their native 16x16 — from which the 370-tile
atlas (208 material/object composites, one out-of-world tile, 160 inventory tiles, one blank) is
composed on device per tile size. It is a constant of the family: two envs built from different seeds
and reset independently produce byte-identical atlases, so the G3 gate is clean.

The atlas is composed rather than embedded pre-rendered, because crafter resamples its 16x16 source
art at whatever tile size is asked for. An atlas embedded at 7 px and resampled again for a 14 px
render would be a picture of a picture, and only the one embedded size would be comparable to a
reference.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..render import tilemap as tm
from ..spaces import Box, Dict, Discrete

VIEW = 9  #: ``Env(view=(9, 9))`` — the default, and what every paper uses
GRID_W = VIEW  #: world columns
N_ITEM = 16  #: ``len(constants.items)``
ITEM_ROWS = -(-N_ITEM // VIEW)  #: ``ceil(len(items) / view[0])`` — 2
GRID_H = VIEW - ITEM_ROWS  #: world rows — 7
OFF_X, OFF_Y = GRID_W // 2, GRID_H // 2  #: ``LocalView._offset``: the player's cell
AREA = 64  #: ``Env(area=(64, 64))``
N_CELL = GRID_W * GRID_H  #: 63

#: material id -> asset name; id 0 is ``World._mat_names[0] = None``, which ``Textures.get`` draws as
#: ``unknown``. Straight from ``crafter.constants.materials``, in its order.
MATERIALS = ("water", "grass", "stone", "path", "sand", "tree", "lava", "coal", "iron", "diamond", "table", "furnace")
#: object-texture id -> asset name; 0 is "no object". Every ``Object.texture`` an env can produce.
OBJTEX = (
    None,
    "player-left",
    "player-right",
    "player-up",
    "player-down",
    "player-sleep",
    "cow",
    "zombie",
    "skeleton",
    "arrow-left",
    "arrow-right",
    "arrow-up",
    "arrow-down",
    "plant",
    "plant-ripe",
    "fence",
)
ITEMS = (
    "health",
    "food",
    "drink",
    "energy",
    "sapling",
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)

N_MAT = len(MATERIALS) + 1  # 13, including the ``None`` material
N_OBJ = len(OBJTEX)  # 16
OUTSIDE = N_MAT * N_OBJ  # 208: the cell is off the 64x64 world; LocalView leaves it grey
N_CODE = OUTSIDE + 1  # 209
OBJ_SLEEP = OBJTEX.index("player-sleep")
#: object id -> the facing ``Player.texture`` was reading, for the reference bridge
FACING = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

ITEM_BASE = N_CODE  #: atlas slot of (item 0, amount 0)
BLACK_TILE = ITEM_BASE + N_ITEM * 10
N_TILE = BLACK_TILE + 1  # 370

#: ``LocalView`` fills its canvas with this before drawing, so an out-of-world cell stays flat grey.
DEFAULT_OUTSIDE = (127, 127, 127)
#: ``Env.render`` pastes the view into a ``np.zeros`` canvas, so any border left over when the frame
#: size is not a multiple of the view width is black.
DEFAULT_PAD = (0, 0, 0)
#: ``_light``'s two blend factors: ``ImageEnhance.Color(...).enhance(0.4)`` then
#: ``_tint(..., (0, 16, 64), 0.5)``. They are NOT style values, and that is the reason PushT's
#: docstring gives for making ``scale`` a multiplier: a style tensor is float32, ``float32(0.4)``
#: widens to 0.4000000059604645, and the reference's blend is ``a + alpha * (b - a)`` in C double —
#: so wherever ``b - a`` is negative and divisible by 5 the product crosses an integer from the other
#: side and the pixel comes out one level low. Measured before this change: an off-by-one on ~10% of
#: channels of every lit frame. Keeping the constants in float64 here and exposing SCALES that
#: default to exactly 1.0 makes "matches the official renderer" an identity rather than a rounding.
NIGHT_SATURATION, NIGHT_AMOUNT = 0.4, 0.5
DEFAULT_NIGHT = (0.0, 16.0, 64.0, 1.0, 1.0)
#: ``_sleep``: ``enhance(0.0)`` then ``_tint(..., (0, 0, 16), 0.5)``. Not a variation knob — a batch
#: cannot be relied on to contain a sleeping player, and a knob that is inert on most batches is what
#: ``check_variations_do_something`` exists to reject.
SLEEP_TINT, SLEEP_AMOUNT = (0.0, 0.0, 16.0), 0.5
#: below this the reference mixes in noise from the world RNG and the frame stops being a function of
#: any state; see the module docstring
DAYLIGHT_MIN = 0.5
#: PIL's ITU-R 601-2 luma, in the fixed point ``convert("L")`` actually uses
LUMA = (19595, 38470, 7471)

MIN_UNIT, MAX_UNIT = 3, 64  #: tile sizes the embedded resize table covers

TEX_NAMES = (
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "arrow-down",
    "arrow-left",
    "arrow-right",
    "arrow-up",
    "coal",
    "cow",
    "debug",
    "debug-2",
    "debug-3",
    "diamond",
    "drink",
    "energy",
    "fence",
    "food",
    "furnace",
    "grass",
    "health",
    "iron",
    "iron_pickaxe",
    "iron_sword",
    "lava",
    "leaves",
    "log",
    "path",
    "plant",
    "plant-ripe",
    "plant-young",
    "player",
    "player-down",
    "player-left",
    "player-right",
    "player-sleep",
    "player-up",
    "sand",
    "sapling",
    "skeleton",
    "stone",
    "stone_pickaxe",
    "stone_sword",
    "table",
    "tree",
    "unknown",
    "water",
    "wood",
    "wood_pickaxe",
    "wood_sword",
    "zombie",
)
#: which of them carry a real alpha channel — ``_draw_alpha`` blends only those, and writes the rest
#: over the background
TEX_HAS_ALPHA = (
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    1,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    1,
    1,
)

_TEX_SHAPE = (56, 16, 16, 4)
_TEX_B64 = (
    "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"
)

_NEAREST_SHAPE = (64, 64)
_NEAREST_B64 = (
    "eNqlktmWhCAMBSe40bbK///tNDthMwTfq07doPyb+9ZrjhfynuP385nhYZXfZ4rfPzMCWLbjvJ4p/vO9Ff+A6y5nBGLdDnnyBWLZfgHnxRWIxQZcj2Ie"
    "cDUBXAHoAYcW3CwB6AH7oQNYAsdLrsDzXIHmQwBDAAIFDAssbwPMgkEB+AFmwbjA8zqAI9C8HeAWDAogHMAEMATCDwgLfo+gRnkTYBYMCjRvBph/wC0Y"
    "ENgDhgCzYEQQ+RDgbqjoD+AGRMFNFjjeC0IAVZDxwwLMjws8XwQQBZpHA5IAigDxISAseBVAMcAFEAWYdwEDAsebAWlAvGFfAMkB4hPGAHNDRTlgHDAk"
    "cDwKQAteBBmfLKAJIo8D/II3QXyAWkDyCKr/AHhAKri7AsgH1AI6ggZPFrR4qiDwvYCOwPJ5QGVBQ5DzeQBaUBFAeoBGQE9Q42PAuwDyAxZPgG9YEwhq"
    "gLuhIvIkQcEnFywXlILIJwfIAnoCfMB2QHZD1ecrAQ0BNAfUBHchyPhcUAnAghe+K/gHJJU8/w=="
)

#: 512 states harvested from real ``crafter.Env`` rollouts; see ``sample_states``.
_POOL_SHAPE = (512, 79)
_POOL_B64 = (
    "eNrt3E1z4iAYAODt1I94c3Zn7+TD9+aMO+3ePXY82en+AH6yP2kjpEoSIELeJGDeHIrE+pSvECBUzqxHYXkPGOOb8vihHNybK7XYOKbz7udye9mxFlfz"
    "eNOzcVcvRA5urxzLrrBrkXIGNXVtKP4cI464CbkXTC6JidN59140j5tLkDgwcNWH9WpqqiZj6nw5mJDbP8wl9du2+HVu8HJm9wbmTE51LusaVGBy0JOD"
    "1giqwR0Oh+NBHMfyuAYfVVwc7/eXF91h4T4kdwyS+zgOwZ36ch/1qnDk1i7cn8k5W1VMz9ky++bNnXTc376pO+FyR+KoZq/ceiTuNEjqTq7cBrMLiJQ7"
    "ed16ulJ36sv1um27ld07btm94ZbdG27ZdXOrgbl+zTgmTgzBLxfGy6MMqtgtyJnmpBLgcWDiykRL7VLF7kGhOykCobU4k1MFmeYkKLGBOVOh2coOuqvC"
    "EGQt4LGatXPMpPbisDJr5la4mSVuZM7cjLFTt0zaz8nku+xSj8lnPUqsBGsx1uLqC1qt5aHc8iRFrPIMzNm9zrXPuLi601RTR86uWZ8eE4fLdVwVGXHe"
    "3I44M3cd+fFqHCgGjucyeo2dRey3HBVuayNG/h3jbLMkzp9bzIrbOHA/g+Z+EWfjGK9Gp/zqXDUutTK2LcR729oAln/HODs4cjlxxEXFadVPqTKTGjr3"
    "6sL9i5tbEEecB8ecty+DOsRv7+i1Hu1JFAfr7sDHOdDMPAbgXp6Q4/DIzspn4RjMKHVM4xHXyYFpx2GT227P57LbLX+K/vbM6kHW6oQbi+TE9eASD25v"
    "49R7hdQuUrtIRgnkA9U9Mz3VbHH21OWuqQuLS4gLh0uG5HpeFcThcWE3lOfm1sQFyvW9yIjz5ta4NTsplz0lZ1BTJaZTHbkdccQFya1mxW1QuwDiguUE"
    "dAuEt5VCGeRqTHjb+s7KOLmGUwWZyYmUs44Cis5RwAp1SWs1BmdS03tMqw7PsTvXUtNWCdRVdG5JnIErJuakZ+rj8/aFXLuCx+N0aqrrZhTVXHbaIix0"
    "/YJShHPjEuJi5HbEzZVr7DcmbkrOpTdO58QVrpx9FJB3jQIi4wSkBup09Es3UVOmozZOM63NTE414RuR06ifzpm1VkWBW7PEOXE19dOyTsCG4F6j4lpX"
    "hXpxfFmWbORFFjS3IM6fW0TEPbBXG/Rvcc6dd0MXxv9278s1VeKm5m7qjDi/snP+fmPiiAuPK8bmEuIC4Rr7jTnv1+6I68Mls+XOdi/H5XbEEUfc5Fy/"
    "fxZucmviXLtPHgJX4GaWOJVbY7U7OT0mjrgBuBS3GSNznf2d7kt1iRuCy0xO1fCI68UlxI3CdS5DI3OrWXBCmxu3JC5KzvWLsJnxca4Xd5+7wwipy9w5"
    "m+fDWTwvzid1Xg1lVtxucg4GSR2EyUHQVRESt8Qtu6VTf9fNLZC5DUrnPhbnd6+IlMMuO0uPQtwjHLBpOOl8r6P0zmwX59juOjhwG2rz2rMmc2a9uRfi"
    "PLlX4ny5hRu37+Z+TMblbG/vAh7m9s0epeaw5+K6Vsgkp99/x33X3IkjjjjicDiIlPPpjYEFyQHyvQKUhpIEx6k1i8YBUuqAoWZW5ZLQOGAhc9UHAJeT"
    "DaWxFbJvM0bkSi9IDpSaDY9jgMtVHwBHbjcqx4jz5ACJA3XPLG7q8DjRBYSaOkAuO2bcgus91G5w3D6+s+00Al3quPlPa1N3kKV2MGSWuA6u9Abi1iFx"
    "IDk2EsedZj3QWnBz3L4MmiTbMhs1Zyq0cbj/bNaeAA=="
)

_POOL_DAY_B64 = (
    "eNrtkt9LU3EYxk+0kdkPjGBkJIEQjsqYaCgu+VQQ0bIwkYqyUKrRSvuxyppu5xsIkrsoEG9MhYSoaw3xquZFOLyrKGJggeHsxpIYjEql9uyv8OJcHA7n"
    "nO953+f5PE9gbT3PDpzkcaiB+d5G7OEmfM9Pkxo+y/n4OQZDzXT7L2JZLbgmWuluuURk+TKz8SCJDVfxmBC/5q5xoa6N64/aqZi6QTJ7k+Pbb/PFF2ak"
    "9g6T1Xe54r3H5MYOXOkOgqP3KQ0/IFwWYeB9hNStTjrdXbx90oWrKEpvT5SXmSijZ2JYr2IsuG0+NNhs6bdZ/86mxG0YqzTsazb8sQ32kKFs3NA6bXia"
    "MhyaNyQXDYNZg2/J8HvFEPxn+JG7LOth/q5nvdd3ndN5/af/NUfzNFfztUf7tFf7pUN6pEv6pFN6pVv65UN+5Ev+5FN+5Vv+xSHPI8dFfMRJvMRN/MRR"
    "PMVVfMVZvMVd/JWD8lAuykc5KS/lpvyUo/JUroFcvonXx6hqO8q3zUcoeHGYdMVBEmN1zHj9xPtr8GT38/FEJduGfPhnyxko2UPfKS89sV3MjJTy981O"
    "Ap92UJsupmbRw0RmK7szRRT83MT0XCHfP6/ja9LF3vE1+X1On5w+OX1y+uT0afX36T86AVjQ"
)


def _unpack(b64: str, shape, dtype=np.uint8) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=dtype).reshape(shape)


def _resize(tex: np.ndarray, nearest: np.ndarray, d: int) -> np.ndarray:
    """One 16x16 RGBA asset at ``d`` x ``d``, through PIL's MEASURED nearest-neighbour index map."""
    ix = nearest[d - 1, :d].astype(np.int64)
    return tex[ix][:, ix]


def _draw_alpha(canvas: np.ndarray, off, t: np.ndarray, has_alpha: bool) -> None:
    """``crafter.engine._draw_alpha``, including its float32 arithmetic and its truncation."""
    x, y = off
    w, h = t.shape[:2]
    if not has_alpha:
        canvas[x : x + w, y : y + h] = t[..., :3]
        return
    a = t[..., 3:].astype(np.float32) / 255
    c = t[..., :3].astype(np.float32) / 255
    cur = canvas[x : x + w, y : y + h].astype(np.float32) / 255
    canvas[x : x + w, y : y + h] = (255 * (a * c + (1 - a) * cur)).astype(np.uint8)


class _Atlas:
    """``(N_TILE, unit, unit, 3)`` uint8, composed once per (device, tile size) and kept.

    Slots: ``m * N_OBJ + o`` for a material/object cell, ``OUTSIDE`` for a cell off the world,
    ``ITEM_BASE + i * 10 + amount`` for an inventory slot, ``BLACK_TILE`` for the two unused strip
    cells. Composing here rather than embedding a pre-rendered atlas is what makes every tile size
    exact: the source art is 16x16 and crafter resamples it per size, so an atlas embedded at one
    size and resampled again would be a picture of a picture.
    """

    _cache: dict = {}
    _tex: np.ndarray | None = None
    _nearest: np.ndarray | None = None

    @classmethod
    def _art(cls):
        if cls._tex is None:
            cls._tex = _unpack(_TEX_B64, _TEX_SHAPE)
            cls._nearest = _unpack(_NEAREST_B64, _NEAREST_SHAPE)
        return cls._tex, cls._nearest

    @classmethod
    def get(cls, device, unit: int) -> torch.Tensor:
        key = (str(device), int(unit))
        if key not in cls._cache:
            cls._cache[key] = torch.as_tensor(cls._build(int(unit)), device=device)
        return cls._cache[key]

    @classmethod
    def _build(cls, unit: int) -> np.ndarray:
        """Compose every tile in crafter's own ``(x, y)`` order, then store it transposed.

        The transpose is not bookkeeping. Crafter loads each PNG with ``image.transpose((1, 0) + ...)``
        and draws into a ``(x, y, 3)`` canvas, then ``Env.render`` returns ``canvas.transpose((1, 0,
        2))``. So the reference's final frame is the transpose of the composition, tile CONTENT
        included — and a renderer that lays the tiles out in row-major order without transposing each
        one produces a frame that is right everywhere the art is symmetric and wrong everywhere it is
        not. Measured before the swap: 1,980 of 4,096 pixels differed on frame 0 with interior
        agreement still reporting 1.000, because every tile was its own transpose.
        """
        tex, nearest = cls._art()
        idx = {n: i for i, n in enumerate(TEX_NAMES)}
        out = np.zeros((N_TILE, unit, unit, 3), np.uint8)
        names = ("unknown",) + MATERIALS
        for m, mat in enumerate(names):
            mt = _resize(tex[idx[mat]], nearest, unit)
            for o, ob in enumerate(OBJTEX):
                c = np.zeros((unit, unit, 3), np.uint8) + DEFAULT_OUTSIDE[0]
                c[:] = mt[..., :3]  # _draw: alpha DISCARDED, never blended
                if o:
                    _draw_alpha(c, (0, 0), _resize(tex[idx[ob]], nearest, unit), bool(TEX_HAS_ALPHA[idx[ob]]))
                out[m * N_OBJ + o] = c.swapaxes(0, 1)
        out[OUTSIDE] = np.asarray(DEFAULT_OUTSIDE, np.uint8)
        ui, ua = int(0.8 * unit), int(0.6 * unit)
        oi, oa = int(0.1 * unit), int(0.4 * unit)
        for i, name in enumerate(ITEMS):
            for a in range(1, 10):
                c = np.zeros((unit, unit, 3), np.uint8)
                if ui:
                    _draw_alpha(c, (oi, oi), _resize(tex[idx[name]], nearest, ui), bool(TEX_HAS_ALPHA[idx[name]]))
                if ua:
                    _draw_alpha(c, (oa, oa), _resize(tex[idx[str(a)]], nearest, ua), bool(TEX_HAS_ALPHA[idx[str(a)]]))
                out[ITEM_BASE + i * 10 + a] = c.swapaxes(0, 1)
        return out


def _luma(x: torch.Tensor) -> torch.Tensor:
    """``(...,3)`` -> ``(...,1)``: PIL's ``convert("L")``, in the fixed point it really uses.

    ``>> 16`` of a non-negative integer is ``floor(/65536)``, and the largest value reachable is
    255*65536 + 32768, so float64 carries it exactly.
    """
    return torch.floor((x[..., 0:1] * LUMA[0] + x[..., 1:2] * LUMA[1] + x[..., 2:3] * LUMA[2] + 32768.0) / 65536.0)


@register
class Crafter(Env):
    """Crafter's 64x64 observation: a 9x7 lit world window plus the 2x9 inventory strip."""

    name = "crafter"
    state_dim = N_CELL + N_ITEM + 1
    state_names = (
        tuple(f"cell[{x},{y}]" for x in range(GRID_W) for y in range(GRID_H))
        + tuple(f"inv_{n}" for n in ITEMS)
        + ("daylight",)
    )
    #: Step-9 state (contract §3). Three keys because the row is genuinely three things -- the lit
    #: world window, the inventory strip below it, and the scalar that decides how the world half is
    #: tinted -- drawn by three different parts of the renderer. The packed row and its column order
    #: are unchanged, every column is covered, so nothing is hidden and the frozen layout does not
    #: move. ``cells`` stays ONE key of ``N_CELL`` columns rather than a 9x7 nest: the public State
    #: describes the environment, and a per-cell key would be 63 names for one grid.
    state_layout = ((("cells",), N_CELL, 0), (("inventory",), N_ITEM, N_CELL), (("daylight",), 1, N_CELL + N_ITEM))
    default_size = 64
    reference_id = "CrafterReward-v1"

    # ---- geometry --------------------------------------------------------

    def _unit(self, size: int) -> int:
        """``Env.render``'s ``unit = size // view``. Not a divisor: crafter centres the leftover."""
        u = int(size) // VIEW
        if not MIN_UNIT <= u <= MAX_UNIT:
            raise ValueError(
                f"Crafter: size {size} gives {u} px per tile, outside the {MIN_UNIT}-{MAX_UNIT} px "
                f"range the embedded resize table covers; use a size in "
                f"[{MIN_UNIT * VIEW}, {MAX_UNIT * VIEW + VIEW - 1}]"
            )
        return u

    # ---- variation -------------------------------------------------------

    def make_variation_space(self) -> Dict:
        """Knobs that are the identity at their defaults and genuinely move pixels off them.

        There is deliberately **no RGB leaf**, and that is a claim about Crafter rather than
        packaging. Every world pixel leaves the renderer through ``daylight * c + (1 - daylight) *
        night(c)``, and ``daylight == 1`` happens at one step in 300, so a colour written into the
        world layer is essentially never painted literally — declaring one as an ``RGBBox`` would
        advertise "this colour appears in the frame", which ``check_colors_apply_exactly`` checks and
        which is not true here. ``chrome`` therefore carries the out-of-world fill and the canvas pad
        as one 2x3 leaf, exactly as MiniGrid's ``chrome`` carries its three palette entries.

        ``night`` is one 5-vector for the same reason: the tint colour, the saturation factor and the
        tint amount only mean anything together, and none of them is a colour that reaches a pixel.

        There is no knob for the sleeping post-process. A 16-state batch cannot be relied on to
        contain a sleeping player (measured: 14% of steps under a uniform policy), and a knob that is
        inert on most batches is what ``check_variations_do_something`` exists to reject.
        """
        return Dict(
            {
                #: rows: the fill LocalView leaves where the window is off the world, and the canvas
                #: colour Env.render pastes the view onto
                "chrome": Box(
                    low=0,
                    high=255,
                    shape=(2, 3),
                    dtype=np.uint8,
                    init_value=np.array([DEFAULT_OUTSIDE, DEFAULT_PAD], np.uint8),
                ),
                #: tint r, g, b, then SCALES on ``_light``'s saturation (0.4) and tint amount (0.5)
                "night": Box(
                    low=np.array([0, 0, 0, 0, 0], np.float32),
                    high=np.array([255, 255, 255, 2, 2], np.float32),
                    shape=(5,),
                    dtype=np.float32,
                    init_value=DEFAULT_NIGHT,
                ),
                "channel_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
                "objects_visible": Discrete(2, init_value=1),
                "inventory_visible": Discrete(2, init_value=1),
            }
        )

    # ---- rendering -------------------------------------------------------

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
        """``(...,80)`` -> ``(...,size,size,3)`` uint8 on device, in the reference's own order.

        There is no ``scene()``: the frame is a gather from a texture atlas followed by two per-pixel
        post-processes, not a rasterisation of vector primitives.
        """
        if supersample not in (None, 1):
            raise ValueError(
                f"Crafter.render: supersample={supersample} is meaningless here — the frame is a "
                f"gather from a bitmap atlas, so there is no sub-pixel coverage to estimate."
            )
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        n = s.shape[0]
        size = int(size or self.size)
        unit = self._unit(size)
        atlas = _Atlas.get(self.device, unit)
        sty = self.resolve_style(style, variation, batch=n)

        codes = torch.round(s[:, :N_CELL]).long().remainder(N_CODE).view(n, GRID_W, GRID_H)
        inv = torch.round(s[:, N_CELL : N_CELL + N_ITEM]).long().clamp_(0, 9)
        day = s[:, -1].double()

        world = codes.permute(0, 2, 1).contiguous()  # (n, GRID_H, GRID_W)
        # objects off: fall back to the same material with object id 0. OUTSIDE (=13*16) is congruent
        # to 0 modulo N_OBJ, so it survives this untouched and needs no special case.
        keep = (sty["objects_visible"] > 0.5).view(n, 1, 1)
        world = torch.where(keep, world, world - world % N_OBJ)

        slot = ITEM_BASE + torch.arange(N_ITEM, device=self.device) * 10 + inv
        slot = torch.where((sty["inventory_visible"] > 0.5).view(n, 1), slot, torch.full_like(slot, BLACK_TILE))
        strip = torch.full((n, ITEM_ROWS * GRID_W), BLACK_TILE, dtype=torch.long, device=self.device)
        strip[:, :N_ITEM] = slot  # ItemView: index i -> (i // VIEW, i % VIEW)
        idx = torch.cat([world, strip.view(n, ITEM_ROWS, GRID_W)], 1).to(torch.int32)
        tiles = tm.blit(atlas, idx, unit)  # (n, VIEW*u, VIEW*u, 3) uint8

        view = self._post(tiles, world, day, sty, unit)  # (n, VIEW*u, VIEW*u, 3) uint8
        if out is None:
            out = torch.empty(n, size, size, 3, dtype=torch.uint8, device=self.device)
        buf = out.view(n, size, size, 3)
        buf[:] = (
            torch.trunc(sty["chrome"].view(n, 2, 3)[:, 1].double()).clamp_(0, 255).to(torch.uint8).view(n, 1, 1, 3)
        )
        b = (size - unit * VIEW) // 2
        buf[:, b : b + unit * VIEW, b : b + unit * VIEW] = view
        return buf.view(*lead, size, size, 3) if lead else buf

    def _post(
        self, tiles: torch.Tensor, world: torch.Tensor, day: torch.Tensor, sty, unit: int, budget: int = 256 << 20
    ) -> torch.Tensor:
        """``_light``, then ``_sleep``, then the channel gain, then the truncation ``render`` does.

        Chunked over the batch because the arithmetic is float64 and must be: the reference's last
        step is a C cast of a float64 to uint8, so a value that lands 1e-6 below an integer must land
        below it here too, and float32 does not carry that. A 16,384-frame batch at the default size
        is 1.6 GB of float64, which is why this loops rather than materialising it.
        """
        n = tiles.shape[0]
        H = GRID_H * unit
        chrome = torch.trunc(sty["chrome"].view(n, 2, 3).double())
        night = sty["night"].double()
        gain = sty["channel_gain"].double()
        asleep = (world[:, OFF_Y, OFF_X] % N_OBJ == OBJ_SLEEP) & (world[:, OFF_Y, OFF_X] != OUTSIDE)
        outside = world == OUTSIDE
        step = max(1, int(budget // max(1, H * tiles.shape[2] * 3 * 8 * 4)))
        frame = torch.empty_like(tiles)
        for b0 in range(0, n, step):
            b1 = min(n, b0 + step)
            k = b1 - b0
            w = tiles[b0:b1, :H].double()
            om = outside[b0:b1].repeat_interleave(unit, 1).repeat_interleave(unit, 2)
            w = torch.where(om.unsqueeze(-1), chrome[b0:b1, 0].view(k, 1, 1, 3), w)

            sat = NIGHT_SATURATION * night[b0:b1, 3].view(k, 1, 1, 1)
            amt = NIGHT_AMOUNT * night[b0:b1, 4].view(k, 1, 1, 1)
            tint = torch.trunc(night[b0:b1, :3]).view(k, 1, 1, 3)
            lum = _luma(w)
            # PIL's Image.blend is `(UINT8)((int)a + alpha * ((int)b - (int)a))` — a truncation
            nite = torch.trunc(lum + sat * (w - lum))
            nite = (1 - amt) * nite + amt * tint
            d = day[b0:b1].view(k, 1, 1, 1)
            lit = d * w + (1 - d) * nite

            if bool(asleep[b0:b1].any()):
                sl = torch.trunc(lit)
                sl = (1 - SLEEP_AMOUNT) * _luma(sl).expand_as(sl) + SLEEP_AMOUNT * torch.tensor(
                    SLEEP_TINT, dtype=torch.float64, device=lit.device
                )
                lit = torch.where(asleep[b0:b1].view(k, 1, 1, 1), sl, lit)

            v = torch.cat([lit, tiles[b0:b1, H:].double()], 1) * gain[b0:b1].view(k, 1, 1, 3)
            frame[b0:b1] = torch.trunc(v).clamp_(0, 255).to(torch.uint8)
        return frame

    # ---- states ----------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Real states, harvested from ``crafter.Env`` rollouts and embedded, not synthesised.

        The pool is 512 windows taken from uniform-random rollouts across eight seeds, keeping only
        the ``daylight >= 0.5`` half of the cycle the reference renders purely (see the module
        docstring). It is EMBEDDED rather than generated on demand for the same reason the atlas is:
        ``crafter`` is an optional dependency, and an env whose states cannot be drawn without it
        would report every conformance check as an error rather than losing only its parity check.

        A quarter of the pool sits against the world edge, where ``LocalView`` leaves the window's
        outer rows or columns at its flat grey fill. Those are reached by walking the player to a
        border cell with the world's own ``move`` — which is what walking there does — rather than by
        stamping the code in, so the window contents on the inside of the border are a real
        neighbourhood and not an invention.
        """
        rng = rng or self.rng
        pool = _pool()
        if batch <= len(pool):
            return pool[rng.choice(len(pool), batch, replace=False)].copy()
        return pool[rng.integers(0, len(pool), batch)].copy()

    # ---- reference bridge -------------------------------------------------

    @classmethod
    def reference_env(cls, size: int | None = None):
        """The official ``crafter.Env`` at the requested frame size. Needs ``crafter`` installed."""
        import crafter

        n = int(size or cls.default_size)
        env = crafter.Env(area=(AREA, AREA), view=(VIEW, VIEW), size=(n, n), seed=0)
        assert tuple(env._local_view._grid) == (GRID_W, GRID_H), (
            f"crafter builds a {tuple(env._local_view._grid)} world view; this env assumes ({GRID_W}, {GRID_H})"
        )
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames, by rebuilding the world the state describes and calling ``Env.render``.

        The whole official path runs — ``LocalView`` over ``world.objects``, ``_light``, ``_sleep``,
        ``ItemView`` and the paste into the canvas — so nothing about the comparison is arranged in
        our favour. The player's world position is recovered from the OUT-OF-WORLD border the code map
        carries: a window whose outer ``k`` columns are outside is the window of a player standing at
        ``x = 4 - k``, and there is exactly one such position.
        """
        from crafter import objects

        names = (None,) + MATERIALS
        w = env._world
        out = []
        for st in np.asarray(states, np.float64):
            codes = np.round(st[:N_CELL]).astype(np.int64).reshape(GRID_W, GRID_H) % N_CODE
            inv = np.round(st[N_CELL : N_CELL + N_ITEM]).astype(np.int64).clip(0, 9)
            day = float(st[-1])
            if day < DAYLIGHT_MIN:
                raise NotImplementedError(
                    f"Crafter: daylight {day:.4f} is below {DAYLIGHT_MIN}, where LocalView._light "
                    f"mixes in noise drawn from the world RNG. That frame is not a function of any "
                    f"state, so there is nothing to compare against."
                )
            cx, cy = cls._player_pos(codes)
            centre = int(codes[OFF_X, OFF_Y])
            if centre == OUTSIDE or centre % N_OBJ not in FACING and centre % N_OBJ != OBJ_SLEEP:
                raise NotImplementedError(
                    f"Crafter: the centre cell holds {OBJTEX[centre % N_OBJ]!r} rather than the "
                    f"player. LocalView is centred on a player, so such a window has no reference."
                )

            w.reset(seed=0)
            p = objects.Player(w, (cx, cy))
            w.add(p)
            p.sleeping = centre % N_OBJ == OBJ_SLEEP
            if not p.sleeping:
                p.facing = FACING[centre % N_OBJ]
            p.inventory = {n_: int(v) for n_, v in zip(ITEMS, inv)}
            for gx in range(GRID_W):
                for gy in range(GRID_H):
                    c = int(codes[gx, gy])
                    if c == OUTSIDE:
                        continue
                    pos = (cx + gx - OFF_X, cy + gy - OFF_Y)
                    m, o = divmod(c, N_OBJ)
                    w[pos] = names[m]
                    if o and (gx, gy) != (OFF_X, OFF_Y):
                        w.add(cls._make_object(objects, w, pos, o, p))
            w.daylight = day
            env._player = p
            out.append(env.render())
        return np.stack(out)

    @staticmethod
    def _make_object(objects, world, pos, o: int, player):
        name = OBJTEX[o]
        if name == "cow":
            return objects.Cow(world, pos)
        if name == "zombie":
            return objects.Zombie(world, pos, player)
        if name == "skeleton":
            return objects.Skeleton(world, pos, player)
        if name == "fence":
            return objects.Fence(world, pos)
        if name.startswith("arrow-"):
            return objects.Arrow(world, pos, FACING[OBJTEX.index("player-" + name[6:])])
        if name.startswith("plant"):
            pl = objects.Plant(world, pos)
            pl.grown = 301 if name == "plant-ripe" else 0
            return pl
        raise NotImplementedError(
            f"Crafter: {name!r} is a player texture on a cell that is not the window's centre; "
            f"crafter's world holds exactly one player and it stands at the centre by construction."
        )

    @staticmethod
    def _player_pos(codes: np.ndarray) -> tuple[int, int]:
        """The unique world position whose 9x7 window is off the world exactly where ``codes`` says.

        Refuses rather than approximates. Any out-of-world pattern that is not "the first ``k`` or
        last ``k`` columns, unioned with the first or last few rows" is unreachable, and rendering a
        near-miss would produce a plausible parity number for a state the reference cannot hold.
        """
        m = codes == OUTSIDE
        cols = m.all(1)
        rows = m.all(0)
        if not np.array_equal(m, cols[:, None] | rows[None, :]):
            raise NotImplementedError(
                "Crafter: the out-of-world cells in this state are not a border of whole rows and "
                "columns, so no player position produces them."
            )

        def span(flags, off, n, limit):
            k0 = int(np.argmin(flags)) if flags[0] else 0
            k1 = int(np.argmin(flags[::-1])) if flags[-1] else 0
            if k0 and k1:
                raise NotImplementedError("Crafter: the window cannot leave the world on both sides")
            if flags[k0 : n - k1].any():
                raise NotImplementedError("Crafter: out-of-world cells are not a contiguous border")
            if max(k0, k1) > limit:
                raise NotImplementedError(
                    f"Crafter: {max(k0, k1)} out-of-world lines exceeds the {limit} a {n}-wide window can have"
                )
            if k0:
                return off - k0
            if k1:
                return AREA - 1 - (n - 1 - off) + k1
            return AREA // 2

        return span(cols, OFF_X, GRID_W, OFF_X), span(rows, OFF_Y, GRID_H, OFF_Y)


_POOL: np.ndarray | None = None


def _pool() -> np.ndarray:
    """``(512, 80)`` float32 — the embedded state pool, decoded once."""
    global _POOL
    if _POOL is None:
        codes = _unpack(_POOL_B64, _POOL_SHAPE).astype(np.float32)
        day = _unpack(_POOL_DAY_B64, (_POOL_SHAPE[0], 1), np.float32)
        _POOL = np.concatenate([codes, day], 1)
    return _POOL
