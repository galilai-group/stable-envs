"""Procgen — `maze`, `heist`, `miner`, `chaser`, bit-exact against the shipped Qt renderer.

The prior verdict on this family (``docs/research/pixel_native.md``, before this module) was
"conditional": the state decomposition held, but the renderer was Qt5 ``QPainter`` behind 10.5 kLOC of
per-game C++ and an "opaque" state blob, and the cost was judged prohibitive. **Both halves of that
were wrong, and the way they were wrong is the interesting part.**

**The blob is not opaque.** ``get_state()`` is ``Game::serialize`` — a flat little-endian stream of
4-byte ints, 4-byte floats and length-prefixed strings, in a fixed order. Transcribing that order gives
the parser at the bottom of this module, which re-serialises byte for byte on 24/24 states in **all 16
games** (the per-game tail is transcribed for these three and kept verbatim for the rest); 54-90% of
what it holds is three ``std::mt19937`` states written as decimal TEXT that never reaches
``game_draw``. Once parsed, procgen becomes a *programmable* probe rather than a black
box: any state can be written back with ``set_state``, which is how the measurements below were taken
and how ``reference_render`` places the official env.

**Qt's raster engine did not have to be re-derived; it had to be READ OFF.** ``resource_root`` is a
procgen option, so the shipped binary can be pointed at an asset tree in which every sprite pixel
encodes its own ``(x, y)`` and an id. Rendering then reports, for all 4096 output pixels, WHICH sprite
painted each one and WHICH source pixel it sampled. Three rules fell out, none of them guessed
(``scratch/procgen/probe0{2,3,6,7}*.py``):

| rule | what it is | how it was measured |
|---|---|---|
| axis-aligned blit | ``qt_scale_image``'s 16.16 walk: ``tr = QRectF::toRect() ∩ device``, ``ix = int(65536·sw/dstw)``, ``basex = uint32(ceil((x0+0.5-dstx)·sx·65536) - 1)``, ``sample = (basex + i·ix) >> 16`` | 4096/4096 px on 7 coded scenes across the 3 games, incl. reflected sprites and both background sets |
| rotated blit — coverage | a device pixel is written iff its **centre** lies inside the rotated quad | 539/539 measured pixels, **0** false positives, over 40 rotations |
| rotated blit — sampling | ``fetchTransformed``'s per-pixel fixed point, **with no half-point bias**: ``px = int((m21(y+.5) + m11(x+.5) + dx)·65536) >> 16`` on the inverted matrix | 539/539; the plausible ``- (1<<15)`` variant scores **28.9%** |

Geometry is **float32 all the way to the ``QRectF`` constructor**, because procgen computes every rect
in C ``float``. This is not a detail: heist's ``view_dim`` is 12.999999, not 13, and its ``x_off`` is
2.3475061e-06 rather than 0 — both readable in the state blob, and both worth a whole pixel row if the
arithmetic is done in float64.

Parity, measured against the shipped renderer over 20 levels x 12 steps per game with backgrounds on
(``scratch/procgen/probe08_parity.py``):

| game | frames | pixels exact | max abs diff |
|---|---|---|---|
| maze | 240 | **100.0000%** | 0 |
| heist | 240 | **100.0000%** | 0 |
| miner | 240 | **100.0000%** | 0 |
| chaser | 240 | **100.0000%** | 0 |

``parity_interior_min`` stays at 1.0 and needs no ``parity_note``. Note that maze reports
``interior_exact_frac = None`` rather than 1.0: at 2.56 output pixels per cell a 25x25 maze has no
flat 3x3 neighbourhood anywhere, so the interior statistic is undefined and the suite falls back to
all-pixel agreement, which is 1.0. Craftax's pixel art does the same thing for the same reason.

Four more facts the renderer's SHAPE depends on, each measured rather than assumed
(``scratch/procgen/probe09_facts.py``):

  * a device column or row is covered by **at most 2** grid cells (``RENDER_EPS = 0.02`` overlaps
    neighbours by 0.04 of a cell), at every size from 64 to 512 in all three games — so the grid is
    exactly 4 composited layers, not a loop over 625 cells;
  * heist's z=0 entities (keys, locks, exit) **never** share a device pixel — rendering them in
    reversed order changed 0 of 300 frames — so a fixed slot order is safe even though the reference's
    order is a maze-scan artefact;
  * every entity radius is a constant of its type (agent 0.5 / 0.375, key 0.375 x 0.232, lock 0.5,
    exit 0.375 x 0.257, ring key 0.03 x 0.0186), so only positions are state;
  * all 9 topdown backgrounds are SQUARE, so ``offset_x = bg_pct_x * (bg_ar - world_ar)`` is
    identically 0 for maze and heist and ``bg_pct_x`` is genuinely not part of their render state.
    52 of miner's 62 platform/space backgrounds are not square, so it is part of miner's.

**Scope, stated plainly.** ``distribution_mode="hard"`` and the default options. All four games hold
``options.center_agent`` false in hard mode — maze/heist/miner through
``(distribution_mode == MemoryMode)``, chaser by setting it outright — so the camera here is a fixed
full-world view and the scrolling-camera path is not implemented. Memory mode and the other 12 games
are out of scope, and the reason is NOT that the raster rules are missing (both belong to
``BasicAbstractGame`` and are shared by all 16). Two things decide a game: its own draw overrides,
and whether its entity cast is a constant of the game. chaser's is (3 orbs, 3 eggs, 3 enemies);
bigfish's is a Bernoulli(1/10) spawner reaching 24 fish at once, and a fixed-width slot state for
that is either lossy or full of dead dimensions. That renderer is written and bit-exact and lives in
``scratch/procgen/bigfish_env.py`` rather than here; ``docs/research/pixel_native.md`` 2.9 has the
numbers and the same argument applied to bossfight, dodgeball, plunder and starpilot.

**Assets are loaded from the installed ``procgen`` wheel, not embedded.** MiniHack and Craftax embed
theirs because they are 68 kB and 8.7 kB; procgen's are 25 MB of PNGs and miner alone reaches 62
distinct backgrounds totalling 176 MB decoded, so embedding is not a trade anyone would want. The
sprite atlas and the row-resampled background atlas are packed once and cached under
``~/.cache/stable-envs/procgen``. This follows MiniGrid, which likewise builds its atlas from the
installed upstream package.

**Throughput, with its crossover, because the peak ratio alone would flatter it.** procgen's own
renderer is C++ drawing 12.3 kB frames and it is fast; this one wins only once the batch pays for a
chain of serial composites. Both sides swept on one H200 with 8 threads for the reference
(``scratch/procgen/probe12_crossover.py``):

| batch | maze | heist | miner | chaser |
|---|---|---|---|---|
| 1 | 482 (0.01x) | 91 (0.00x) | 312 (0.00x) | 92 (0.00x) |
| 128 | 60,619 (**1.00x**) | 11,491 (0.11x) | 40,757 (0.60x) | 11,515 (0.08x) |
| 256 | 120,608 (2.00x) | 23,026 (0.22x) | 81,119 (**1.19x**) | 23,022 (0.17x) |
| 1024 | 476,390 (7.89x) | 91,305 (0.88x) | 330,954 (4.86x) | 91,487 (0.67x) |
| 2048 | 928,461 (15.38x) | 175,736 (**1.70x**) | 647,174 (9.51x) | 182,264 (**1.33x**) |
| 4096 | **1,688,605 (27.97x)** | **317,259 (3.06x)** | **1,213,165 (17.83x)** | **359,105 (2.63x)** |

Ratios are against each game's official PEAK over ``num`` (maze 60,374 at num=64; heist 103,661 at
num=256; miner 68,056 at num=256) — not against ``num=64``, which understates heist's reference by
10%. **The crossover is batch 128 for maze, 256 for miner, 2048 for heist and 2048 for chaser**, and below it this
renderer is slower than the thing it reproduces, by two orders of magnitude at batch 1. Anyone
rendering a handful of frames at a time should call procgen.

heist is the outlier for a structural reason: 12 entity slots and a rotated-blit path, so its serial
chain is roughly three times maze's over the same 4096 pixels, and the per-call floor (~11 ms) is
what the batch has to amortise.

Two things bought the rest, and both are worth knowing before touching this file: the torch
expression of one SourceOver blend is nine full-frame kernels and ran at 0.73 GB/s, about 100x off
the roofline for a 12.3 kB frame; and ``draw_background`` used to loop in Python over the distinct
background indices in the batch, which cost ~10 ms per call **regardless of batch size**. The
official figure is step+render because procgen exposes no render-only entry point —
``Game::observe`` is called from ``Game::step``, and ``gym3``'s ``observe()`` only copies buffers.

**What the state does and does not carry.** miner's grid distinguishes ``MOVING_BOULDER`` /
``MOVING_DIAMOND`` from their stationary forms, and ``image_for_type`` maps both pairs to the same
sprite — so the render state folds them together. The frame is unaffected by construction; the
*dynamics* are not, and this module claims only the frame.
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np
import torch

from ..env import Env
from ..registry import register
from ..spaces import Box, Dict, Discrete

_SPEC = importlib.util.find_spec("procgen")
if _SPEC is None or _SPEC.origin is None:  # optional dependency: lose this env only
    raise ImportError(
        "procgen is not installed; `pip install procgen` provides both the assets this "
        "renderer packs and the reference env parity is measured against"
    )
ASSET_ROOT = os.path.join(os.path.dirname(_SPEC.origin), "data", "assets")
if not os.path.isdir(ASSET_ROOT):
    raise ImportError(f"procgen is importable but its asset tree is missing at {ASSET_ROOT}")

try:  # triton is optional; the torch path is the readable one
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except Exception:  # noqa: BLE001
    HAVE_TRITON = False

F = np.float32
PI_F = F(3.14159265358979323846264338327950288)  # cpp-utils.h's `const float PI`
DEG2RAD = 0.017453292519943295769  # QTransform::rotate's own constant
RENDER_EPS = 0.02  # basic-abstract-game.cpp
P_SPACE, P_WALL = 100, 51  # object-ids.h


# ===================================================================== Qt, in closed form


def _qround(v):
    """``qRound(qreal)``: ``int(v + 0.5)`` for v >= 0, ``int(v - 0.5)`` below."""
    v = np.asarray(v, np.float64)
    return np.where(v >= 0, np.floor(v + 0.5), np.ceil(v - 0.5)).astype(np.int64)


def _to_rect(x, y, w, h):
    """``QRectF::toRect()`` -> ``(left, top, width, height)``."""
    l, t = _qround(x), _qround(y)
    return l, t, _qround(x + w) - l, _qround(y + h) - t


def _prepare(mw, mh, rect_h):
    """``BasicAbstractGame::prepare_for_drawing`` with ``center_agent`` false, in float32.

    Reproduced operand by operand because the C++ is ``float`` and the float64 answer differs: heist
    gets ``view_dim = 12.999999`` and ``x_off = 2.3475061e-06`` here, and both are in the blob.
    """
    cx, cy = F(mw) * F(0.5), F(mh) * F(0.5)
    vis = F(max(mw, mh))
    raw_unit = F(F(64) / vis)
    unit = F(np.float64(raw_unit) * (rect_h / 64.0))
    view_dim = F(64.0 / np.float64(raw_unit))
    x_off = F(unit * F(cx - F(view_dim / F(2))))
    y_off = F(unit * F(cy - F(view_dim / F(2))))
    return unit, view_dim, x_off, y_off


def _screen_rect(x, y, dx, dy, g, eps=0.0):
    """``get_screen_rect``. ``g`` is ``(unit, view_dim, x_off, y_off)``; every operand is float32."""
    unit, view_dim, x_off, y_off = g
    e = F(eps)
    return (
        float(F(F(F(x) - e) * unit - x_off)),
        float(F(F(F(view_dim) - F(y) - e) * unit + y_off)),
        float(F(F(F(dx) + F(2) * e) * unit)),
        float(F(F(F(dy) + F(2) * e) * unit)),
    )


def _premultiply(rgba: np.ndarray) -> np.ndarray:
    """``qPremultiply``, the integer form ``convertToFormat(ARGB32_Premultiplied)`` uses."""
    a = rgba[..., 3].astype(np.uint32)
    out = rgba.copy()
    for c in range(3):
        t = rgba[..., c].astype(np.uint32) * a
        out[..., c] = (((t + (t >> 8) + 0x80) >> 8) & 0xFF).astype(np.uint8)
    return out


def _over(dst: torch.Tensor, src: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Qt's ``PRIM_SOURCE_OVER`` on premultiplied ARGB: ``d = s + BYTE_MUL(d, 255 - sa)``.

    ``BYTE_MUL(x, a) = ((x*a + ((x*a) >> 8) + 0x80) >> 8) & 0xff`` — the rounding-by-reciprocal form,
    NOT ``x*a//255``, which disagrees on a large fraction of blended pixels.

    ``BYTE_MUL(d, 255) == d`` exactly for every d in 0..255, which is what makes the coverage mask
    free: a pixel outside a primitive is fed ``src = 0, a = 0`` and comes back untouched, so nothing
    in the render path needs a ``torch.where`` over the frame.
    """
    t = dst * (255 - a)
    return (src + (((t + (t >> 8) + 0x80) >> 8) & 0xFF)).clamp_(0, 255)


def _qt_sincos(deg: np.ndarray):
    """``QTransform::rotate``'s trig, branch for branch, evaluated on the HOST in float64.

    Two reasons it is not ``torch.sin`` on the device. The quadrant short circuits are exact
    assignments in Qt (``a == 90.`` gives ``sina = 1``, and note that ``-180`` is NOT short-circuited
    while ``180`` is), and reproducing them from a transcendental would be luck. And the remaining
    values go through glibc's ``sin``/``cos``; CUDA's are correctly rounded but not bit-identical, and
    a fixed-point truncation is a discontinuous function of its argument. heist's agent only ever
    takes 8 rotations (``face_direction`` of an 8-way action), so this costs one small transfer.
    """
    deg = np.asarray(deg, np.float64)
    rad = deg * DEG2RAD
    sina = np.where(
        deg == 90.0,
        1.0,
        np.where(
            deg == -270.0,
            1.0,
            np.where((deg == 270.0) | (deg == -90.0), -1.0, np.where(deg == 180.0, 0.0, np.sin(rad))),
        ),
    )
    cosa = np.where(
        (deg == 90.0) | (deg == -270.0) | (deg == 270.0) | (deg == -90.0),
        0.0,
        np.where(deg == 180.0, -1.0, np.cos(rad)),
    )
    return sina, cosa


# ===================================================================== per-game description


class _EntSlot:
    """One entity the frame can contain. Everything except position is a measured constant."""

    def __init__(
        self, key, sprite, rx, ry, z, rotated=False, reflect=False, abs_coords=False, fixed_xy=None, rotation=None
    ):
        self.key, self.sprite = key, sprite
        self.rx, self.ry, self.z = F(rx), F(ry), z
        self.rotated, self.reflect, self.abs_coords = rotated, reflect, abs_coords
        self.fixed_xy, self.rotation = fixed_xy, rotation


class _Spec:
    """Everything about one game the renderer needs, and nothing about its dynamics.

    ``fills`` is the second kind of grid primitive. maze/heist/miner draw every non-SPACE cell as a
    sprite blit, but ``draw_grid_obj`` is a virtual whose default is ``fillRect(rect,
    color_for_type(...))`` and which games override: chaser paints its ORB as a solid
    ``QColor(0, 255, 0)`` over the CENTRED ``ORB_DIM = 0.3`` sub-rect of the cell. An entry is
    ``(class_index, (r, g, b), fraction)`` and it is composited from a coverage table rather than a
    gather, because there is no source image to sample.
    """

    def __init__(self, name, dim, classes, sprites, backgrounds, ent_slots, grid_slot, fills=()):
        self.name, self.dim, self.classes = name, dim, classes
        self.sprites, self.backgrounds, self.ent_slots = sprites, backgrounds, ent_slots
        self.grid_slot = np.asarray(grid_slot, np.int64)
        self.fills = tuple(fills)


def _aspect_ry(rx, w, h):
    """``match_aspect_ratio``: ``ry = rx / (width*1.0/height)``, with both steps in float32."""
    return float(F(F(rx) / F(w * 1.0 / h)))


#: `topdown_simple_backgrounds` in `resources.cpp` -- ONE entry, which is why chaser's background
#: index is not a state dimension: it is 0 in every state a rollout reaches, and declaring it would
#: be a dead dimension rather than a knob.
TOPDOWN_SIMPLE_BG = ["topdown_backgrounds/floortiles.png"]
#: `water_backgrounds`. NONE of the seven is square (342x192 x4, 980x581, 2100x1500, 2048x1536), so
#: `offset_x = bg_pct_x * (bg_ar - world_ar)` is not identically 0 and `bg_pct_x` would be render
#: state for any game that uses them — the same call miner's non-square platform backgrounds force.
#: Kept here rather than in `scratch/procgen/bigfish_env.py` (its only consumer today) because it is
#: a transcription of `resources.cpp`, not a property of that experiment.
WATER_BG = [
    f"water_backgrounds/{n}"
    for n in [
        "water1.png",
        "water2.png",
        "water3.png",
        "water4.png",
        "underwater1.png",
        "underwater2.png",
        "underwater3.png",
    ]
]
TOPDOWN_BG = [
    f"topdown_backgrounds/{n}"
    for n in [
        "floortiles.png",
        "backgrounddetailed1.png",
        "backgrounddetailed2.png",
        "backgrounddetailed3.png",
        "backgrounddetailed4.png",
        "backgrounddetailed5.png",
        "backgrounddetailed6.png",
        "backgrounddetailed7.png",
        "backgrounddetailed8.png",
    ]
]
#: `platform_backgrounds` in `resources.cpp` order, then every space background appended -- which is
#: what `images_load` does at the bottom of the file, and what makes miner's index reach 61.
PLATFORM_BG = [
    "platform_backgrounds/alien_bg.png",
    "platform_backgrounds/another_world_bg.png",
    "platform_backgrounds/back_cave.png",
    "platform_backgrounds/caverns.png",
    "platform_backgrounds/cyberpunk_bg.png",
    "platform_backgrounds/parallax_forest.png",
    "platform_backgrounds/scifi_bg.png",
    "platform_backgrounds/scifi2_bg.png",
    "platform_backgrounds/living_tissue_bg.png",
    "platform_backgrounds/airadventurelevel1.png",
    "platform_backgrounds/airadventurelevel2.png",
    "platform_backgrounds/airadventurelevel3.png",
    "platform_backgrounds/airadventurelevel4.png",
    "platform_backgrounds/cave_background.png",
    "platform_backgrounds/blue_desert.png",
    "platform_backgrounds/blue_grass.png",
    "platform_backgrounds/blue_land.png",
    "platform_backgrounds/blue_shroom.png",
    "platform_backgrounds/colored_desert.png",
    "platform_backgrounds/colored_grass.png",
    "platform_backgrounds/colored_land.png",
    "platform_backgrounds/colored_shroom.png",
    "platform_backgrounds/landscape1.png",
    "platform_backgrounds/landscape2.png",
    "platform_backgrounds/landscape3.png",
    "platform_backgrounds/landscape4.png",
    "platform_backgrounds/battleback1.png",
    "platform_backgrounds/battleback2.png",
    "platform_backgrounds/battleback3.png",
    "platform_backgrounds/battleback4.png",
    "platform_backgrounds/battleback5.png",
    "platform_backgrounds/battleback6.png",
    "platform_backgrounds/battleback7.png",
    "platform_backgrounds/battleback8.png",
    "platform_backgrounds/battleback9.png",
    "platform_backgrounds/battleback10.png",
    "platform_backgrounds/sunrise.png",
    "platform_backgrounds_2/beach1.png",
    "platform_backgrounds_2/beach2.png",
    "platform_backgrounds_2/beach3.png",
    "platform_backgrounds_2/beach4.png",
    "platform_backgrounds_2/fantasy1.png",
    "platform_backgrounds_2/fantasy2.png",
    "platform_backgrounds_2/fantasy3.png",
    "platform_backgrounds_2/fantasy4.png",
    "platform_backgrounds_2/candy1.png",
    "platform_backgrounds_2/candy2.png",
    "platform_backgrounds_2/candy3.png",
    "platform_backgrounds_2/candy4.png",
    "space_backgrounds/deep_space_01.png",
    "space_backgrounds/spacegen_01.png",
    "space_backgrounds/milky_way_01.png",
    "space_backgrounds/ez_space_lite_01.png",
    "space_backgrounds/meyespace_v1_01.png",
    "space_backgrounds/eye_nebula_01.png",
    "space_backgrounds/deep_sky_01.png",
    "space_backgrounds/space_nebula_01.png",
    "space_backgrounds/Background-1.png",
    "space_backgrounds/Background-2.png",
    "space_backgrounds/Background-3.png",
    "space_backgrounds/Background-4.png",
    "space_backgrounds/parallax-space-backgound.png",
]

_KEY_RY, _RING_RY, _EXIT_RY = _aspect_ry(0.375, 84, 52), _aspect_ry(0.03, 84, 52), _aspect_ry(0.375, 70, 48)
#: `add_entity(1 - r*(2i + 1.25), r*.75, ...)` with `ring_key_r = 0.03f`, evaluated in float32
_RING_XY = [(float(F(F(1) - F(0.03) * F(2 * i + 1.25))), float(F(F(0.03) * F(0.75)))) for i in range(3)]
_RING_ROT = float(PI_F / F(2))

SPECS = {
    "maze": _Spec(
        "maze",
        25,
        ("space", "wall", "goal"),
        ["kenney/Ground/Sand/sandCenter.png", "misc_assets/cheese.png", "kenney/Enemies/mouse_move.png"],
        TOPDOWN_BG,
        [_EntSlot("agent", 2, 0.5, 0.5, z=1, reflect=True)],
        [-1, 0, 1],
    ),
    "heist": _Spec(
        "heist",
        13,
        ("space", "wall"),
        [
            "kenney/Ground/Dirt/dirtCenter.png",
            "misc_assets/gemYellow.png",
            "misc_assets/spaceAstronauts_008.png",
            "misc_assets/keyBlue.png",
            "misc_assets/keyGreen.png",
            "misc_assets/keyRed.png",
            "misc_assets/lock_blue.png",
            "misc_assets/lock_green.png",
            "misc_assets/lock_red.png",
        ],
        TOPDOWN_BG,
        [_EntSlot(f"lock{i}", 6 + i, 0.5, 0.5, z=0) for i in range(3)]
        + [_EntSlot(f"key{i}", 3 + i, 0.375, _KEY_RY, z=0) for i in range(3)]
        + [_EntSlot("exit", 1, 0.375, _EXIT_RY, z=0), _EntSlot("agent", 2, 0.375, 0.375, z=1, rotated=True)]
        + [
            _EntSlot(
                f"ring{i}",
                3 + i,
                0.03,
                _RING_RY,
                z=1,
                rotated=True,
                abs_coords=True,
                fixed_xy=_RING_XY[i],
                rotation=_RING_ROT,
            )
            for i in range(3)
        ],
        [-1, 0],
    ),
    "miner": _Spec(
        "miner",
        20,
        ("space", "boulder", "diamond", "dirt"),
        [
            "misc_assets/robot_greenDrive1.png",
            "misc_assets/elementStone007.png",
            "misc_assets/gemBlue.png",
            "misc_assets/window.png",
            "misc_assets/dirt.png",
        ],
        PLATFORM_BG,
        [_EntSlot("exit", 3, 0.5, 0.5, z=-1), _EntSlot("agent", 0, 0.5, 0.5, z=1, reflect=True)],
        [-1, 1, 2, 4],
    ),
    #: chaser: 13x13 maze (`maze_dim = 13` in HardMode), 3 enemies, 3 large orbs (one per quadrant
    #: with one quadrant losing one, `extra_orb_sign = -1`). The ORB grid value is a fillRect, not a
    #: sprite, so it is a `fills` entry and its class maps to slot -1 in the gather.
    "chaser": _Spec(
        "chaser",
        13,
        ("space", "wall", "orb"),
        [
            "misc_assets/enemyFloating_1b.png",  # 0  PLAYER
            "misc_assets/yellowCrystal.png",  # 1  LARGE_ORB
            "misc_assets/enemyWalking_1b.png",  # 2  ENEMY_WEAK   (48x48)
            "misc_assets/enemySpikey_1b.png",  # 3  ENEMY_EGG
            "misc_assets/tileStone_slope.png",  # 4  MAZE_WALL
            "misc_assets/enemyFlying_1.png",  # 5  ENEMY        (64x64)
            "misc_assets/enemyFlying_2.png",  # 6  ENEMY2       (64x64)
            "misc_assets/enemyFlying_3.png",
        ],  # 7  ENEMY3       (64x64)
        TOPDOWN_SIMPLE_BG,
        [_EntSlot(f"orb{i}", 1, 0.4, 0.4, z=0) for i in range(3)]
        + [_EntSlot(f"egg{i}", 3, 0.5, 0.5, z=0) for i in range(3)]
        + [_EntSlot(f"enemy{i}", 5, 0.5, 0.5, z=0) for i in range(3)]
        + [_EntSlot("agent", 0, 0.5, 0.5, z=1)],
        [-1, 4, -1],
        fills=[(2, (0, 255, 0), 0.3)],
    ),
}
#: `image_for_type(ENEMY)` in chaser, as a table over the one shared state dimension it collapses to.
#: `can_eat_enemies()` gives ENEMY_WEAK; otherwise `rem = (cur_time / 2) % 4` with `3 -> 1`, so the
#: reachable image ids are {3, 6, 7, 8} and nothing else -- enumerated over 32 (cur_time, eat_time)
#: pairs in `scratch/procgen/probe14_chaser.py`, 0 unmatched.
_CHASER_ENEMY_SPRITE = (2, 5, 6, 7)  # atlas slots for image ids 3, 6, 7, 8
#: procgen grid object id -> render class. miner folds MOVING_BOULDER/MOVING_DIAMOND into their
#: stationary forms because `image_for_type` gives them the same sprite.
_CLASS_OF = {
    "maze": {P_SPACE: 0, P_WALL: 1, 2: 2},
    "heist": {P_SPACE: 0, P_WALL: 1},
    "miner": {P_SPACE: 0, 1: 1, 3: 1, 2: 2, 4: 2, 9: 3},
    # chaser's MAZE_WALL is 5 and its ORB is 1002; MARKER (1001) is a level-generation
    # scratch value that `game_reset` clears before the first frame, and 5,376 harvested
    # states hold only {5, 100, 1002} (`scratch/procgen/probe13_recon.py`).
    "chaser": {P_SPACE: 0, 5: 1, 1002: 2},
}
_PROCGEN_ID = {
    "maze": [P_SPACE, P_WALL, 2],
    "heist": [P_SPACE, P_WALL],
    "miner": [P_SPACE, 1, 2, 9],
    "chaser": [P_SPACE, 5, 1002],
}


# ===================================================================== assets


def _cache_dir() -> str:
    root = os.environ.get("STABLE_ENVS_CACHE") or os.path.expanduser("~/.cache/stable-envs")
    d = os.path.join(root, "procgen")
    os.makedirs(d, exist_ok=True)
    return d


class _Assets:
    """Sprite atlas per game (packed, cached on disk) and backgrounds (decoded lazily, per index)."""

    _atlas: dict = {}
    _bgatlas: dict = {}

    @classmethod
    def atlas(cls, game: str, device):
        key = (game, str(device))
        if key in cls._atlas:
            return cls._atlas[key]
        path = os.path.join(_cache_dir(), f"sprites_{game}.npz")
        packed = wh = None
        if os.path.exists(path):
            try:
                z = np.load(path)
                packed, wh = z["packed"], z["wh"]
            except Exception:  # noqa: BLE001 partial/corrupt cache
                packed = None
        if packed is None:
            from PIL import Image

            imgs = [
                _premultiply(np.asarray(Image.open(os.path.join(ASSET_ROOT, p)).convert("RGBA"), np.uint8).copy())
                for p in SPECS[game].sprites
            ]
            wh = np.array([[im.shape[1], im.shape[0]] for im in imgs], np.int64)
            H, W = int(wh[:, 1].max()), int(wh[:, 0].max())
            packed = np.zeros((len(imgs), H, W, 4), np.uint8)
            for i, im in enumerate(imgs):
                packed[i, : im.shape[0], : im.shape[1]] = im
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, packed=packed, wh=wh)
            os.replace(tmp, path)
        t = torch.from_numpy(np.ascontiguousarray(packed)).to(device)
        # the flat (N,4) view the kernels index is cached too: rebuilding it per render call is a
        # copy of the whole atlas on the critical path, for no reason
        out = (t, np.asarray(wh), t.reshape(-1, 4).contiguous())
        cls._atlas[key] = out
        return out

    @classmethod
    def bg_atlas(cls, game: str, size: int, g, d: int, device):
        """Every background of ``game``, already resampled to ``size`` ROWS and width-padded.

        The row map is a function of the background's height alone (``main_rect`` is the whole
        frame), so baking it turns the background into one gather per frame instead of a Python
        loop over the distinct indices in the batch. miner's table is 62 x 64 x 2000 x 3 = 23.8 MB;
        maze's and heist's are 0.9 MB.
        """
        key = (game, int(size), str(device))
        if key in cls._bgatlas:
            return cls._bgatlas[key]
        path = os.path.join(_cache_dir(), f"bg_{game}_{size}.npz")
        tab = bw = bh = None
        if os.path.exists(path):
            try:
                z = np.load(path)
                tab, bw, bh = z["tab"], z["bw"], z["bh"]
            except Exception:  # noqa: BLE001 partial/corrupt cache
                tab = None
        if tab is None:
            from PIL import Image

            mr = _screen_rect(0, d, d, d, g)
            rows_img, bw, bh = [], [], []
            for p in SPECS[game].backgrounds:
                a = np.asarray(Image.open(os.path.join(ASSET_ROOT, p)).convert("RGB"), np.uint8)
                r, _, _ = _blit_axis(mr[1], mr[3], a.shape[0], size)
                rows_img.append(a[r])
                bw.append(a.shape[1])
                bh.append(a.shape[0])
            bw, bh = np.asarray(bw, np.int64), np.asarray(bh, np.int64)
            tab = np.zeros((len(rows_img), size, int(bw.max()), 3), np.uint8)
            for i, im in enumerate(rows_img):
                tab[i, :, : im.shape[1]] = im
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                np.savez_compressed(fh, tab=tab, bw=bw, bh=bh)
            os.replace(tmp, path)
        out = (
            torch.from_numpy(np.ascontiguousarray(tab).reshape(-1, 3)).to(device),
            int(tab.shape[2]),
            torch.from_numpy(np.asarray(bw)).to(device),
            torch.from_numpy(np.asarray(bh)).to(device),
        )
        cls._bgatlas[key] = out
        return out


# ===================================================================== precomputed geometry


def _blit_axis(pos, extent, src_extent, size):
    """One axis of ``qt_scale_image``: device index -> source index, as a length-``size`` array.

    Entries outside the rect are meaningless and masked by the caller; they are clamped rather than
    left wild so a gather cannot fault.
    """
    l = int(_qround(pos))
    w = int(_qround(pos + extent)) - l
    out = np.zeros(size, np.int64)
    lo, hi = max(l, 0), min(l + w, size)
    if hi <= lo:
        return out, lo, hi
    sf = float(src_extent) / float(extent)
    step = int(0x00010000 * sf)
    base = (int(np.ceil((lo + 0.5 - pos) * sf * 65536)) - 1) & 0xFFFFFFFF
    idx = np.arange(hi - lo, dtype=np.int64)
    out[lo:hi] = np.clip(((base + idx * step) & 0xFFFFFFFF) >> 16, 0, src_extent - 1)
    return out, lo, hi


def _rot_matrix(rect, deg, sw, sh):
    """``copy`` from ``QRasterPaintEngine::drawImage``: source pixel coords -> device coords."""
    rx, ry, rw, rh = rect
    sina, cosa = (float(np.asarray(v).reshape(-1)[0]) for v in _qt_sincos(np.float64(deg)))
    dx = rx + rw / 2.0 - rw / 2.0 * cosa + rh / 2.0 * sina
    dy = ry + rh / 2.0 - rw / 2.0 * sina - rh / 2.0 * cosa
    M = np.array([[cosa * (rw / sw), sina * (rw / sw)], [-sina * (rh / sh), cosa * (rh / sh)]], np.float64)
    return M, np.array([dx, dy], np.float64)


def _rot_index_np(rect, deg, sw, sh, size):
    """Coverage + source index for one rotated blit whose rect is a constant, on the host."""
    M, T = _rot_matrix(rect, deg, sw, sh)
    corners = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float64) @ M + T
    ys, xs = np.mgrid[0:size, 0:size]
    px, py = xs + 0.5, ys + 0.5
    inside = np.ones((size, size), bool)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        inside &= ((b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])) >= 0
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    inv = np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]]) / det
    iT = -T @ inv
    fx = ((inv[1, 0] * py + inv[0, 0] * px + iT[0]) * 65536).astype(np.int64) >> 16
    fy = ((inv[1, 1] * py + inv[0, 1] * px + iT[1]) * 65536).astype(np.int64) >> 16
    ok = inside & (fx >= 0) & (fx < sw) & (fy >= 0) & (fy < sh)
    return ok, np.clip(fx, 0, sw - 1), np.clip(fy, 0, sh - 1)


class _Geom:
    """Everything about (game, size) that does not depend on the state. Built once, cached.

    The grid is NOT a loop over cells. ``RENDER_EPS`` widens each cell by 0.04 of a cell, so a device
    column is covered by at most two cells (measured, at 64/128/256/512 px, all three games) and the
    whole grid is four composited layers indexed by two column tables and two row tables. Cells enter
    each table in ascending index, which IS the reference's draw order (``for x: for y:``).
    """

    _cache: dict = {}
    K = 2

    @classmethod
    def get(cls, game: str, size: int, device):
        key = (game, int(size), str(device))
        if key not in cls._cache:
            cls._cache[key] = _Geom(game, int(size), device)
        return cls._cache[key]

    def __init__(self, game: str, size: int, device):
        spec = SPECS[game]
        self.spec, self.size, self.device = spec, size, torch.device(device)
        d = spec.dim
        self.g = _prepare(d, d, size)
        wh = _Assets.atlas(game, device)[1]
        S, K = len(spec.sprites), self.K

        cx = -np.ones((size, K), np.int64)
        cy = -np.ones((size, K), np.int64)
        sxi = np.zeros((S, size, K), np.int64)
        syi = np.zeros((S, size, K), np.int64)
        ncol = np.zeros(size, np.int64)
        nrow = np.zeros(size, np.int64)
        for i in range(d):
            rx, ry, rw, rh = _screen_rect(i, i + 1, 1, 1, self.g, eps=RENDER_EPS)
            xa = [_blit_axis(rx, rw, int(wh[s, 0]), size) for s in range(S)]
            ya = [_blit_axis(ry, rh, int(wh[s, 1]), size) for s in range(S)]
            for c in range(xa[0][1], xa[0][2]):
                k = ncol[c]
                if k >= K:
                    raise AssertionError(
                        f"{game}@{size}: device column {c} is covered by more than "
                        f"{K} grid cells; the layer count is wrong"
                    )
                cx[c, k] = i
                for s in range(S):
                    sxi[s, c, k] = xa[s][0][c]
                ncol[c] += 1
            for r in range(ya[0][1], ya[0][2]):
                k = nrow[r]
                if k >= K:
                    raise AssertionError(
                        f"{game}@{size}: device row {r} is covered by more than {K} "
                        f"grid cells; the layer count is wrong"
                    )
                cy[r, k] = i
                for s in range(S):
                    syi[s, r, k] = ya[s][0][r]
                nrow[r] += 1
        # One flat table per (kx, ky) layer, so a layer costs three index_selects rather than a
        # chain of broadcast advanced indexes. `idx[s, r, c]` is the atlas offset a pixel would read
        # if sprite `s` were the one drawn there, which depends on nothing but (sprite, row, col).
        _, Hs, Ws, _ = _Assets.atlas(game, device)[0].shape  # noqa: E501
        self.layers = []
        for kx in range(K):
            for ky in range(K):
                mask = (cx[:, kx] >= 0)[None, :] & (cy[:, ky] >= 0)[:, None]
                cell = (np.clip(cy[:, ky], 0, None)[:, None] * d + np.clip(cx[:, kx], 0, None)[None, :]).astype(
                    np.int32
                )
                idx = (
                    (np.arange(S)[:, None, None] * Hs + syi[:, :, ky][:, :, None]) * Ws + sxi[:, :, kx][:, None, :]
                ).astype(np.int32)
                self.layers.append(
                    (
                        torch.from_numpy(cell.reshape(-1)).to(device),
                        torch.from_numpy(mask.reshape(-1).astype(np.int8)).to(device),
                        torch.from_numpy(idx.reshape(-1)).to(device),
                    )
                )
        # `draw_grid_obj` overrides: a solid colour over a centred sub-rect of the cell. There is no
        # source image, so this is a coverage table rather than a gather: `cellof[p]` is the cell
        # whose sub-rect covers device pixel `p`, or -1. One table is enough because a sub-rect at
        # `frac <= 0.5` cannot reach a neighbour's -- measured for chaser at 1 cell per pixel over
        # all 169 (`probe14_chaser.py`), and asserted here so a new game cannot inherit it silently.
        self.fills = []
        for cls_idx, rgb, frac in spec.fills:
            cellof = np.full(size * size, -1, np.int64)
            for i in range(d):
                for j in range(d):
                    rx, ry, rw, rh = _screen_rect(i, j + 1, 1, 1, self.g, eps=RENDER_EPS)
                    l, t, w, h = _to_rect(rx + rw * (1 - frac) / 2, ry + rh * (1 - frac) / 2, rw * frac, rh * frac)
                    x0, x1 = max(int(l), 0), min(int(l) + int(w), size)
                    y0, y1 = max(int(t), 0), min(int(t) + int(h), size)
                    if x1 <= x0 or y1 <= y0:
                        continue
                    sub = cellof.reshape(size, size)[y0:y1, x0:x1]
                    if (sub >= 0).any():
                        raise AssertionError(
                            f"{game}@{size}: two grid fill sub-rects share a device pixel; the "
                            f"single-table fill layer is wrong for this game"
                        )
                    sub[:] = j * d + i
            self.fills.append(
                (
                    int(cls_idx),
                    torch.from_numpy(cellof.astype(np.int32)).to(device),
                    torch.tensor(rgb, dtype=torch.uint8, device=device),
                )
            )
        self.slot_lut = torch.from_numpy(spec.grid_slot.astype(np.int32)).to(device)
        #: the (n_sprite, 2) size table on the device, for the slots whose sprite is state and whose
        #: source extent is therefore per environment. Hoisted out of `_draw_entities` because a
        #: host->device copy per slot per call is a transfer on the critical path.
        self.wh_dev = torch.from_numpy(np.ascontiguousarray(wh)).to(device)
        self.cols = torch.arange(size, device=device)
        self.rc = torch.arange(size * size, dtype=torch.int32, device=device)
        self.SS = size * size
        #: a valid float pointer for the kernels' unused `gain` argument when HAS_GAIN is False
        self.dummy_gain = torch.ones(1, 3, dtype=torch.float32, device=device)
        self._bg = None

        # entities whose rect is a constant of the size (heist's rotated, use_abs_coords ring keys)
        self.fixed = {}
        for si, sl in enumerate(spec.ent_slots):
            if sl.fixed_xy is None:
                continue
            m, fx, fy = _rot_index_np(
                self._abs_rect(sl),
                float(F(F(sl.rotation) * F(180)) / PI_F),
                int(wh[sl.sprite, 0]),
                int(wh[sl.sprite, 1]),
                size,
            )
            self.fixed[si] = (
                torch.from_numpy(np.ascontiguousarray(m)).to(device),
                torch.from_numpy(fx.astype(np.int32)).to(device),
                torch.from_numpy(fy.astype(np.int32)).to(device),
            )

    @property
    def bg(self):
        """``(flat (N,3) uint8, Wmax, widths (n_bg,), heights (n_bg,))``. Built on first use."""
        if self._bg is None:
            self._bg = _Assets.bg_atlas(self.spec.name, self.size, self.g, self.spec.dim, self.device)
        return self._bg

    def _abs_rect(self, sl):
        """``get_abs_rect(view_dim*(x-rx), view_dim*(y+ry), 2*view_dim*rx, 2*view_dim*ry)``."""
        unit, view_dim = self.g[0], self.g[1]
        x, y = sl.fixed_xy
        vals = (
            F(view_dim * F(F(x) - sl.rx)),
            F(view_dim * F(F(y) + sl.ry)),
            F(F(2) * view_dim * sl.rx),
            F(F(2) * view_dim * sl.ry),
        )
        return tuple(float(F(v * unit)) for v in vals)


# ===================================================================== batched blit maps


def _t_qround(v):
    return torch.where(v >= 0, torch.floor(v + 0.5), torch.ceil(v - 0.5))


def _axis_map(pos, extent, src_extent, S: int, cols):
    """``qt_scale_image`` on one axis for a batch. ``pos``/``extent`` are (B,) float64.

    ``src_extent`` is an int, or a (B,) tensor when the batch mixes source images of different sizes
    — which is exactly miner's case, where 62 backgrounds range from 144x144 to 2000x2000.

    Returns ``(ok, idx)``, both ``(B, S)``. This is the whole axis-aligned rule: everything else in
    the renderer is which rects to feed it and in what order.
    """
    B = pos.shape[0]
    l = _t_qround(pos).long()
    w = _t_qround(pos + extent).long() - l
    lo = l.clamp_min(0)
    hi = torch.minimum(l + w, torch.full_like(l, S))
    se = (
        torch.as_tensor(src_extent, device=pos.device, dtype=torch.float64)
        if not isinstance(src_extent, int)
        else float(src_extent)
    )
    sf = se / extent
    step = torch.trunc(65536.0 * sf).long()
    base = (torch.ceil((lo.double() + 0.5 - pos) * sf * 65536.0).long() - 1) & 0xFFFFFFFF
    idx = ((base.view(B, 1) + (cols.view(1, S) - lo.view(B, 1)) * step.view(B, 1)) & 0xFFFFFFFF) >> 16
    ok = (cols.view(1, S) >= lo.view(B, 1)) & (cols.view(1, S) < hi.view(B, 1))
    if isinstance(src_extent, int):
        return ok, idx.clamp_(0, src_extent - 1)
    return ok, torch.minimum(idx.clamp_min_(0), torch.as_tensor(src_extent, device=pos.device).long().view(B, 1) - 1)


def _rot_maps(rx, ry, rw, rh, rot, sw, sh, S):
    """The rotated path for a batch: pixel-centre coverage + ``fetchTransformed``'s fixed point."""
    dev = rx.device
    B = rx.shape[0]
    deg = (rot.to(torch.float32) * F(180) / PI_F).double()
    sina_np, cosa_np = _qt_sincos(deg.detach().cpu().numpy())
    sina = torch.as_tensor(sina_np, dtype=torch.float64, device=dev)
    cosa = torch.as_tensor(cosa_np, dtype=torch.float64, device=dev)

    dx = rx + rw / 2 - rw / 2 * cosa + rh / 2 * sina
    dy = ry + rh / 2 - rw / 2 * sina - rh / 2 * cosa
    m11, m12 = cosa * (rw / sw), sina * (rw / sw)
    m21, m22 = -sina * (rh / sh), cosa * (rh / sh)

    cxs = torch.stack([dx, sw * m11 + dx, sw * m11 + sh * m21 + dx, sh * m21 + dx], 1)
    cys = torch.stack([dy, sw * m12 + dy, sw * m12 + sh * m22 + dy, sh * m22 + dy], 1)
    ys = torch.arange(S, device=dev, dtype=torch.float64).view(1, S, 1) + 0.5
    xs = torch.arange(S, device=dev, dtype=torch.float64).view(1, 1, S) + 0.5
    inside = torch.ones(B, S, S, dtype=torch.bool, device=dev)
    for i in range(4):
        j = (i + 1) % 4
        ex = (cxs[:, j] - cxs[:, i]).view(B, 1, 1)
        ey = (cys[:, j] - cys[:, i]).view(B, 1, 1)
        inside &= (ex * (ys - cys[:, i].view(B, 1, 1)) - ey * (xs - cxs[:, i].view(B, 1, 1))) >= 0

    det = m11 * m22 - m12 * m21
    i00, i01, i10, i11 = m22 / det, -m12 / det, -m21 / det, m11 / det
    it0 = -(dx * i00 + dy * i10)
    it1 = -(dx * i01 + dy * i11)
    fx = ((i10.view(B, 1, 1) * ys + i00.view(B, 1, 1) * xs + it0.view(B, 1, 1)) * 65536).to(torch.int64) >> 16
    fy = ((i11.view(B, 1, 1) * ys + i01.view(B, 1, 1) * xs + it1.view(B, 1, 1)) * 65536).to(torch.int64) >> 16
    ok = inside & (fx >= 0) & (fx < sw) & (fy >= 0) & (fy < sh)
    return ok, fx.clamp_(0, sw - 1).int(), fy.clamp_(0, sh - 1).int()


# ===================================================================== the fused compositor
#
# The torch expression of one SourceOver blend is nine elementwise kernels over a (B,S,S,3) int32
# canvas plus a four-channel gather, and measured 0.73 GB/s of frame output at B=4096 -- two orders
# of magnitude off the roofline for a 12.3 kB frame. The arithmetic is not the problem; the number
# of full-frame passes is. Both kernels below do the whole blend in registers, so a composite reads
# the canvas once and writes it once.

if HAVE_TRITON:

    @triton.jit
    def _blend_kernel(
        canvas, atlas, idx, mask, gain, n, SS: tl.constexpr, HAS_GAIN: tl.constexpr, BLOCK: tl.constexpr
    ):
        """One primitive's blend, from a per-pixel atlas offset and coverage mask."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        m = tl.load(mask + off, mask=ok, other=0).to(tl.int32)
        i = tl.load(idx + off, mask=ok, other=0).to(tl.int32)
        a = tl.load(atlas + i * 4 + 3, mask=ok, other=0).to(tl.int32) * m
        b = off // SS
        for c in tl.static_range(3):
            s = tl.load(atlas + i * 4 + c, mask=ok, other=0).to(tl.int32) * m
            if HAS_GAIN:
                g = tl.load(gain + b * 3 + c, mask=ok, other=0.0)
                s = tl.minimum((s.to(tl.float32) * g).to(tl.int32), 255)
            d = tl.load(canvas + off * 3 + c, mask=ok, other=0).to(tl.int32)
            t = d * (255 - a)
            t = (t + (t >> 8) + 128) >> 8
            tl.store(canvas + off * 3 + c, tl.minimum(s + (t & 255), 255).to(tl.uint8), mask=ok)

    @triton.jit
    def _grid_kernel(
        canvas,
        atlas,
        grid,
        cell_tab,
        mask_tab,
        idx_tab,
        slot_lut,
        gain,
        n,
        NCELL: tl.constexpr,
        SS: tl.constexpr,
        HAS_GAIN: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One of the four grid layers, index computation and blend fused."""
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        b = off // SS
        p = off % SS
        cell = tl.load(cell_tab + p, mask=ok, other=0).to(tl.int32)
        m = tl.load(mask_tab + p, mask=ok, other=0).to(tl.int32)
        cls = tl.load(grid + b * NCELL + cell, mask=ok, other=0).to(tl.int32)
        slot = tl.load(slot_lut + cls, mask=ok, other=0).to(tl.int32)
        m = m * (slot >= 0).to(tl.int32)
        i = tl.load(idx_tab + tl.maximum(slot, 0) * SS + p, mask=ok, other=0).to(tl.int32)
        a = tl.load(atlas + i * 4 + 3, mask=ok, other=0).to(tl.int32) * m
        for c in tl.static_range(3):
            s = tl.load(atlas + i * 4 + c, mask=ok, other=0).to(tl.int32) * m
            if HAS_GAIN:
                g = tl.load(gain + b * 3 + c, mask=ok, other=0.0)
                s = tl.minimum((s.to(tl.float32) * g).to(tl.int32), 255)
            d = tl.load(canvas + off * 3 + c, mask=ok, other=0).to(tl.int32)
            t = d * (255 - a)
            t = (t + (t >> 8) + 128) >> 8
            tl.store(canvas + off * 3 + c, tl.minimum(s + (t & 255), 255).to(tl.uint8), mask=ok)


if HAVE_TRITON:

    @triton.jit
    def _fill_kernel(
        canvas, grid, cell_tab, colour, cls, n, NCELL: tl.constexpr, SS: tl.constexpr, BLOCK: tl.constexpr
    ):
        """A `draw_grid_obj` solid fill: write ``colour`` where this pixel's cell holds ``cls``.

        No blend, and that is exact rather than a shortcut: ``fillRect`` with an opaque ``QColor``
        composites as ``d = s + BYTE_MUL(d, 255 - 255) = s``.
        """
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = off < n
        b = off // SS
        p = off % SS
        cell = tl.load(cell_tab + p, mask=ok, other=-1).to(tl.int32)
        hit = cell >= 0
        g = tl.load(grid + b * NCELL + tl.maximum(cell, 0), mask=ok & hit, other=-1).to(tl.int32)
        hit = hit & (g == cls)
        for c in tl.static_range(3):
            v = tl.load(colour + c).to(tl.int32)
            d = tl.load(canvas + off * 3 + c, mask=ok, other=0).to(tl.int32)
            tl.store(canvas + off * 3 + c, tl.where(hit, v, d).to(tl.uint8), mask=ok)


def _blend_torch(canvas, atlas, idx, mask, gain):
    """The readable definition the kernels are checked against. Same arithmetic, more passes.

    "Checked against" is a measurement, not an intention: triton imports on the machines this repo
    runs on, so every parity number comes from the fused path and this one is dead code there unless
    somebody forces it. ``scratch/procgen/probe21_chaser_torch_path.py`` does — it flips
    ``HAVE_TRITON`` off and re-renders — and gets **max abs diff 0 on 64 states against both the
    Triton path and the official renderer**, for all three kernels including chaser's fill layer.
    """
    B, S = canvas.shape[0], canvas.shape[1]
    src = atlas.index_select(0, idx.reshape(-1)).view(B, S, S, 4).to(torch.int32)
    mm = mask.unsqueeze(-1).to(torch.int32)
    rgb = src[..., :3] * mm
    if gain is not None:
        rgb = (rgb.float() * gain.view(B, 1, 1, 3)).trunc_().clamp_(0, 255).to(torch.int32)
    canvas.copy_(_over(canvas.to(torch.int32), rgb, src[..., 3:4] * mm).to(torch.uint8))


# ===================================================================== the env


class _ProcgenEnv(Env):
    """Shared renderer. Subclasses declare only which game they are and how the state is laid out."""

    game: str = ""
    default_size = 64
    parity_interior_min = 1.0
    #: how many state dimensions the background costs. 1 everywhere except chaser, whose
    #: `topdown_simple_backgrounds` holds a single image: a dimension that can only ever be 0 is a
    #: dead dimension, and the suite is right to say so.
    n_bg_state = 1
    #: steps discarded before a state is harvested into the pool. 48 is enough for maze/heist/miner
    #: (dug cells, collected keys); chaser needs more, because its ENEMY entities do not exist until
    #: the eggs hatch at `egg_timeout = 50` and a pool without them leaves three slots and the
    #: enemy-sprite dimension dead.
    pool_warmup = 48

    def __init__(self, num_envs: int = 1, size=None, supersample=1, device="cuda", backend="auto", seed=0):
        super().__init__(
            num_envs=num_envs, size=size, supersample=supersample, device=device, backend=backend, seed=seed
        )
        # Public state: the tile grid and the entity slots, which is the division this
        # class is already built around (``n_cells``, ``has_grid``, ``_unpack``) rather than a new
        # one. Per instance because ``n_cells`` is a per-game property, and the ``grid`` key is
        # absent on a game that has none. Every column is covered by one of the two, so nothing is
        # hidden and the packed row -- whose per-game column order the frozen digests speak -- does
        # not move. ``entities`` stays one key: its columns differ per game and are named by
        # ``state_names``, and inventing a per-game nest would be four vocabularies for one row.
        n = int(self.n_cells)
        self.state_layout = (((("grid",), n, 0),) if n else ()) + (
            ((("entities",), int(self.state_dim) - n, n),) if int(self.state_dim) > n else ()
        )
        self._pool = None

    def _close(self) -> None:
        """Drop the per-instance draws. ``Env.close`` calls this once and is idempotent.

        The assets and the baked geometry (``_Assets``, ``_Geom``) are keyed on (game, size, device)
        and SHARED by every instance, so they are not released here -- releasing a cache another live
        env is drawing through is the same class of error as releasing a shared EGL context. What an
        instance owns is the sampled state pool, which is rebuildable from the official env.
        """
        self._pool = None

    @property
    def spec(self) -> _Spec:
        return SPECS[self.game]

    #: whether the grid reaches a pixel at all. True for every SHIPPED game; the hook exists because
    #: several procgen games never write their grid — `game_reset` leaves it `fill_elem(..., SPACE)`
    #: and `draw_grid_obj` returns immediately on SPACE — so declaring those cells would be hundreds
    #: of dead state dimensions. `scratch/procgen/bigfish_env.py` is the worked case.
    has_grid = True

    @property
    def n_cells(self) -> int:
        return self.spec.dim**2 if self.has_grid else 0

    def make_variation_space(self) -> Dict:
        """Knobs that are EXACTLY the identity at their defaults.

        ``use_backgrounds`` is procgen's own option, bit for bit: at 0, ``draw_background`` stops
        after its black ``fillRect``. The two gains multiply the already-premultiplied atlas and the
        background; 1.0 is the identity rather than approximately so, because the product is truncated
        and an integer times 1.0 is that integer. They are deliberately GAINS and not RGB leaves: a
        tint cannot put a requested colour literally into the frame, so an RGB leaf here would fail
        ``check_colors_apply_exactly`` for the right reason, and declaring a knob that cannot honour
        its own contract is worse than not declaring it.
        """
        return Dict(
            {
                "rendering": Dict({"use_backgrounds": Discrete(2, init_value=1)}),
                "sprite_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
                "background_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
            }
        )

    # ---------------------------------------------------------------- render

    def _unpack(self, s: torch.Tensor) -> dict:
        d = self.spec.dim
        n = self.n_cells
        bg = (
            s[:, n].round().long().clamp_(0, len(self.spec.backgrounds) - 1)
            if self.n_bg_state
            else torch.zeros(s.shape[0], dtype=torch.long, device=s.device)
        )
        grid = (
            s[:, :n].round().clamp_(0, len(self.spec.classes) - 1).to(torch.int32).view(-1, d, d)
            if self.has_grid
            else None
        )
        return {"grid": grid, "bg_index": bg}

    def _slot_fields(self, f, sl, B):
        """``(on|None, x, y, rotation|None, reflected|None)`` for one entity slot."""
        raise NotImplementedError

    def _slot_sprite(self, f, sl, B):
        """Per-environment atlas slot for one entity slot, or ``None`` when it is a constant.

        ``draw_image`` takes ``image_for_type(ent->image_type)``, and that is a virtual: chaser's
        returns a different sprite depending on ``cur_time`` and ``eat_time``, so its three ENEMY
        slots draw one of four images and WHICH one is state. Returning a tensor here also makes the
        source extent per environment — the four candidate sprites are 48x48 and 64x64 — which
        ``_axis_map`` already handles because miner's backgrounds needed it.
        """
        return None

    @torch.no_grad()
    def render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,S,S,3)`` uint8, in ``game_draw``'s order.

        There is no ``scene()``. The frame is a sequence of nearest-neighbour blits from a sprite
        atlas composited with Qt's integer SourceOver, which is exact by construction; pushing that
        through a polygon IR would trade an exact answer for an approximate one.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = s.shape[0]
        S = int(size or self.size)
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}.render: supersample={supersample} is meaningless "
                f"— the frame is a nearest-neighbour gather from a bitmap atlas, so "
                f"there is no sub-pixel coverage to estimate."
            )
        gm = _Geom.get(self.game, S, self.device)
        atlas, wh, flat = _Assets.atlas(self.game, self.device)
        sty = self.resolve_style(style, variation, batch=B)
        f = self._unpack(s)

        direct = out is not None and torch.is_tensor(out) and out.dtype == torch.uint8 and out.device == self.device
        # no zero_(): `_draw_background` writes every pixel (the black `fillRect` is `use_bg == 0`)
        canvas = out.view(B, S, S, 3) if direct else torch.empty(B, S, S, 3, dtype=torch.uint8, device=self.device)
        use_bg = sty["rendering.use_backgrounds"].reshape(B) > 0.5
        # .contiguous() is load-bearing, not hygiene. `StyleTensor.__getitem__` returns a COLUMN
        # SLICE of the (B, style_dim) buffer, so its row stride is style_dim rather than 3, and the
        # kernels address the gain as `gain + b*3 + c`. Without this the per-environment style read
        # environment 0's row (and past the end of the buffer) while the shared-style default, whose
        # gain is 1 and therefore never loaded, looked perfectly fine: 254/255 on a permuted batch.
        bgain = sty["background_gain"].reshape(B, 3).to(torch.float32).contiguous()
        sgain = sty["sprite_gain"].reshape(B, 3).to(torch.float32).contiguous()
        if bool((sgain == 1).all()):
            sgain = None
        self._draw_background(canvas, f, gm, use_bg, bgain)
        for z in (-1, 0):
            self._draw_entities(canvas, f, gm, flat, atlas.shape, wh, z, sgain)
            if z == -1 and self.has_grid:
                self._draw_grid(canvas, f, gm, flat, atlas.shape, sgain)
        self._draw_entities(canvas, f, gm, flat, atlas.shape, wh, 1, sgain)
        if out is not None and not direct:
            out.view(B, S, S, 3).copy_(canvas)
            canvas = out.view(B, S, S, 3)
        return canvas.view(*lead, S, S, 3) if lead else canvas

    def _draw_background(self, canvas, f, gm, use_bg, gain):
        """``draw_background``: black ``fillRect``, then one scaled blit of the level's background.

        The ROW map depends on nothing but the background's height, so it is baked into the atlas
        (``_Geom.bg``): each background is stored already resampled to S rows and padded to the
        widest source width. What is left per environment is one column map and one gather, which
        is what removed the per-background-index Python loop — 9 groups for maze, 62 for miner, and
        about 10 ms of pure launch overhead that did not shrink with the batch.
        """
        B, S = canvas.shape[0], canvas.shape[1]
        d = self.spec.dim
        mr = _screen_rect(0, d, d, d, gm.g)
        idx = f["bg_index"]
        flat, Wmax, bw, bh = gm.bg
        bwi, bhi = bw[idx], bh[idx]  # (B,) int64
        bg_ar = bwi.float() / bhi.float()
        world_ar = F(d * 1.0 / d)
        pct = f.get("bg_pct_x")
        p32 = torch.zeros(B, device=canvas.device) if pct is None else pct.float()
        off = (p32 * (bg_ar - float(world_ar))).double()
        x = mr[0] + mr[2] * (-off)
        w = mr[2] * (bg_ar / float(world_ar)).double()
        _, cols = _axis_map(x, w, bwi, S, gm.cols)
        lin = ((idx.view(B, 1, 1) * S + gm.cols.view(1, S, 1)) * Wmax + cols.view(B, 1, S)).to(torch.int32)
        px = flat.index_select(0, lin.reshape(-1)).view(B, S, S, 3)
        if not bool((gain == 1).all()):
            px = (px.float() * gain.view(B, 1, 1, 3)).trunc_().clamp_(0, 255).to(torch.uint8)
        canvas.copy_(px * use_bg.view(B, 1, 1, 1))

    def _draw_grid(self, canvas, f, gm, flat, shape, gain):
        B, S = canvas.shape[0], canvas.shape[1]
        grid = f["grid"].reshape(B, -1).contiguous()
        n = B * gm.SS
        for cell, mask, idx in gm.layers:
            if HAVE_TRITON:
                _grid_kernel[(triton.cdiv(n, 512),)](
                    canvas,
                    flat,
                    grid,
                    cell,
                    mask,
                    idx,
                    gm.slot_lut,
                    gain if gain is not None else gm.dummy_gain,
                    n,
                    NCELL=grid.shape[1],
                    SS=gm.SS,
                    HAS_GAIN=gain is not None,
                    BLOCK=512,
                    num_warps=4,
                )
                continue
            cls = grid.index_select(1, cell)
            slot = gm.slot_lut.index_select(0, cls.reshape(-1)).view(B, -1)
            m = (slot >= 0) & mask.reshape(1, -1).bool()  # noqa: E501  int8 table -> bool
            lin = idx.index_select(0, (slot.clamp_min(0) * gm.SS + gm.rc.view(1, -1)).reshape(-1))
            _blend_torch(canvas, flat, lin, m.view(B, S, S), gain)
        for cls_idx, cellof, colour in gm.fills:
            # The fill sub-rect sits inside its own cell's non-eps rect, so it cannot touch a
            # neighbour's sprite pixels and the layer order against the four gathers above is
            # unobservable -- measured for chaser at 0 interactions (`probe14_chaser.py`).
            if HAVE_TRITON:
                _fill_kernel[(triton.cdiv(n, 512),)](
                    canvas, grid, cellof, colour, cls_idx, n, NCELL=grid.shape[1], SS=gm.SS, BLOCK=512, num_warps=4
                )
                continue
            hit = cellof.view(1, -1) >= 0
            g = grid.gather(1, cellof.clamp_min(0).long().view(1, -1).expand(B, -1))
            hit = hit & (g == cls_idx)
            canvas.view(B, -1, 3).copy_(torch.where(hit.unsqueeze(-1), colour.view(1, 1, 3), canvas.view(B, -1, 3)))

    def _draw_entities(self, canvas, f, gm, flat, shape, wh, z, gain):
        B, S = canvas.shape[0], canvas.shape[1]
        Hs, Ws = int(shape[1]), int(shape[2])
        for si, sl in enumerate(self.spec.ent_slots):
            if sl.z != z:
                continue
            on, x, y, rot, refl = self._slot_fields(f, sl, B)
            spr = self._slot_sprite(f, sl, B)
            if spr is None:
                sw, sh = int(wh[sl.sprite, 0]), int(wh[sl.sprite, 1])
            else:
                sw, sh = gm.wh_dev[spr, 0], gm.wh_dev[spr, 1]
            if sl.fixed_xy is not None:
                m0, fx, fy = gm.fixed[si]
                m = m0.view(1, S, S).expand(B, S, S)
                sxg, syg = fx.view(1, S, S).expand(B, S, S), fy.view(1, S, S).expand(B, S, S)
            else:
                rect = self._ent_rect(sl, x, y, gm, self._slot_radii(f, sl, B))
                m, sxg, syg = self._blit_maps(rect, rot, sw, sh, S, gm)
            if on is not None:
                m = m & on.view(B, 1, 1)
            if refl is not None:
                w1 = (sw - 1) if isinstance(sw, int) else (sw - 1).view(B, 1, 1).int()
                sxg = torch.where(refl.view(B, 1, 1) > 0.5, w1 - sxg, sxg)
            base = sl.sprite if spr is None else spr.view(B, 1, 1).int()
            lin = ((base * Hs + syg) * Ws + sxg).contiguous()
            self._composite(canvas, flat, lin, m.contiguous(), gain, gm)

    def _composite(self, canvas, flat, lin, m, gain, gm):
        """Blend one primitive into the canvas from a per-pixel atlas offset and coverage mask."""
        n = canvas.shape[0] * canvas.shape[1] * canvas.shape[2]
        if HAVE_TRITON:
            _blend_kernel[(triton.cdiv(n, 512),)](
                canvas,
                flat,
                lin,
                m.to(torch.int8),
                gain if gain is not None else gm.dummy_gain,
                n,
                SS=canvas.shape[1] * canvas.shape[2],
                HAS_GAIN=gain is not None,
                BLOCK=512,
                num_warps=4,
            )
        else:
            _blend_torch(canvas, flat, lin, m, gain)

    def _slot_radii(self, f, sl, B):
        """``(rx, ry)`` for one entity slot when they are state, or ``None`` when constants.

        Constant for every entity in every SHIPPED game, which is why the base returns ``None``.
        The hook exists for the games where it is not: bigfish's fish are sampled a radius at spawn
        (``FISH_MIN_R`` to ``FISH_MAX_R``) and its agent grows one ``r_inc`` per fish eaten. See
        ``scratch/procgen/bigfish_env.py`` and ``docs/research/pixel_native.md`` 2.9.
        """
        return None

    def _ent_rect(self, sl, x, y, gm, radii=None):
        """``get_object_rect`` -> ``get_screen_rect``, in float32, widened only at the end.

        Every operand stays float32 whether the radius is a python constant or a tensor: the C++ is
        ``float`` and the float64 answer puts a rect edge a whole pixel out.
        """
        unit, view_dim, x_off, y_off = (float(v) for v in gm.g)
        x32, y32 = x.to(torch.float32), y.to(torch.float32)
        if radii is None:
            srx, sry = float(sl.rx), float(sl.ry)
        else:
            srx, sry = (v.to(torch.float32) for v in radii)
        rx = ((x32 - srx) * unit - x_off).double()
        ry = ((view_dim - (y32 + sry)) * unit + y_off).double()
        if radii is None:
            rw = torch.full_like(rx, float(F(F(F(2) * sl.rx) * F(unit))))
            rh = torch.full_like(rx, float(F(F(F(2) * sl.ry) * F(unit))))
        else:
            rw = ((2.0 * srx) * unit).double()
            rh = ((2.0 * sry) * unit).double()
        return rx, ry, rw, rh

    def _blit_maps(self, rect, rot, sw, sh, S, gm):
        """Coverage + source index, choosing the axis-aligned or rotated rule PER ENVIRONMENT.

        ``draw_image`` branches on ``rotation == 0``, so a batch holding both is not hypothetical:
        heist's agent has rotation 0 until it first moves.
        """
        rx, ry, rw, rh = rect
        B = rx.shape[0]
        okx, sx = _axis_map(rx, rw, sw, S, gm.cols)
        oky, sy = _axis_map(ry, rh, sh, S, gm.cols)
        m = okx.view(B, 1, S) & oky.view(B, S, 1)
        sxg = sx.int().view(B, 1, S).expand(B, S, S)
        syg = sy.int().view(B, S, 1).expand(B, S, S)
        if rot is None:
            return m, sxg, syg
        turned = rot != 0
        if bool(turned.any()):
            mr, fx, fy = _rot_maps(rx, ry, rw, rh, rot, sw, sh, S)
            t = turned.view(B, 1, 1)
            m = torch.where(t, mr, m)
            sxg = torch.where(t, fx, sxg)
            syg = torch.where(t, fy, syg)
        return m, sxg, syg

    # ---------------------------------------------------------------- states

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """States from real procgen rollouts, not from a synthetic prior.

        The layouts come out of ``MazeGen``/``simple_choose`` with strong invariants (a wall ring, one
        goal, an exit reachable from above), so uniform noise over the class alphabet would exercise a
        region no episode visits and prove nothing about the frames anyone would store. Same choice as
        MiniGrid and for the same reason.
        """
        rng = rng or self.rng
        if self._pool is None:
            self._pool = self._load_pool()
        n = len(self._pool)
        idx = rng.choice(n, batch, replace=False) if batch <= n else rng.integers(0, n, batch)
        return self._pool[idx].copy()

    def _load_pool(self, n: int = 512) -> np.ndarray:
        path = os.path.join(_cache_dir(), f"pool_{self.game}_{self.state_dim}.npy")
        if os.path.exists(path):
            try:
                return np.load(path)
            except Exception:  # noqa: BLE001 partial write
                pass
        pool = self._build_pool(n)
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, pool)
        os.replace(tmp, path)
        return pool

    def _build_pool(self, n: int) -> np.ndarray:
        """Roll the official env and harvest states, after a warm-up.

        The warm-up is load-bearing rather than decorative. The background is only visible through
        SPACE cells, and a pool of pristine resets would leave miner's grid almost entirely DIRT and
        heist's keys all uncollected — so ``bg_index``, ``bg_pct_x`` and ``has_key*`` would present as
        dead state dimensions, which is a property of the pool and not of the renderer.
        """
        from procgen import ProcgenGym3Env

        num = 16
        env = ProcgenGym3Env(
            num=num,
            env_name=self.game,
            distribution_mode="hard",
            rand_seed=0,
            start_level=0,
            num_levels=0,
            use_backgrounds=True,
        )
        rng = np.random.default_rng(0)
        rows = []
        for step in range(n // num + self.pool_warmup):
            env.act(rng.integers(0, 15, size=num).astype(np.int32))
            if step < self.pool_warmup:
                continue
            for blob in env.get_state():
                rows.append(self.state_from_blob(_parse(blob)))
            if len(rows) >= n:
                break
        return np.stack(rows[:n]).astype(np.float32)

    # ---------------------------------------------------------------- reference bridge

    @classmethod
    def reference_env(cls, size: int = 64):
        if int(size) != 64:
            raise ValueError(f"{cls.__name__}: procgen's observation is 64x64 only; asked for {size}")
        from procgen import ProcgenGym3Env

        env = ProcgenGym3Env(
            num=1,
            env_name=cls.game,
            distribution_mode="hard",
            rand_seed=0,
            start_level=0,
            num_levels=0,
            use_backgrounds=True,
        )
        env.act(np.array([4], np.int32))
        return _Reference(env, _parse(env.get_state()[0]))

    @classmethod
    def reference_render(cls, ref, states) -> np.ndarray:
        """Drive the official env to each state and read ITS own observation.

        The round-trip is asserted, not assumed. ``set_state`` runs procgen's ``deserialize`` and then
        ``observe()``, so a blob that does not describe the requested state would make every parity
        number compare two different scenes and would present as a rendering discrepancy.
        """
        out = []
        for st in np.asarray(states, np.float32):
            ref.env.set_state([_dump(cls.blob_from_state(ref.template, st))])
            back = cls.state_from_blob(_parse(ref.env.get_state()[0]))
            d = np.abs(back - st)
            if d.max() > 1e-6:
                i = int(np.argmax(d))
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state — "
                    f"'{cls.state_names[i]}' came back {back[i]} for {st[i]} (max |diff| {d.max()})."
                )
            _, obs, _ = ref.env.observe()
            out.append(np.asarray(obs["rgb"][0]).copy())
        return np.stack(out)

    @classmethod
    def state_from_blob(cls, blob: dict) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def blob_from_state(cls, template: dict, st: np.ndarray) -> dict:
        raise NotImplementedError


class _Reference:
    """The official env plus the template blob every crafted state is written on top of."""

    def __init__(self, env, template):
        self.env, self.template = env, template


# ===================================================================== blob <-> dict
#
# `Game::serialize` / `BasicAbstractGame::serialize` / `Entity::serialize` / `Grid::serialize` /
# `RandGen::serialize`, in order. Verified by re-serialising: 24/24 states per game, byte for byte.

_HEAD = [
    ("serialize_version", "i"),
    ("game_name", "s"),
    ("paint_vel_info", "i"),
    ("use_generated_assets", "i"),
    ("use_monochrome_assets", "i"),
    ("restrict_themes", "i"),
    ("use_backgrounds", "i"),
    ("center_agent", "i"),
    ("debug_mode", "i"),
    ("distribution_mode", "i"),
    ("use_sequential_levels", "i"),
    ("use_easy_jump", "i"),
    ("plain_assets", "i"),
    ("physics_mode", "i"),
    ("grid_step", "i"),
    ("level_seed_low", "i"),
    ("level_seed_high", "i"),
    ("game_type", "i"),
    ("game_n", "i"),
    ("lsrg_seeded", "i"),
    ("lsrg", "s"),
    ("rg_seeded", "i"),
    ("rg", "s"),
    ("reward", "f"),
    ("done", "i"),
    ("level_complete", "i"),
    ("action", "i"),
    ("timeout", "i"),
    ("current_level_seed", "i"),
    ("prev_level_seed", "i"),
    ("episodes_remaining", "i"),
    ("episode_done", "i"),
    ("last_reward_timer", "i"),
    ("last_reward", "f"),
    ("default_action", "i"),
    ("fixed_asset_seed", "i"),
    ("cur_time", "i"),
    ("is_waiting_for_step", "i"),
]
_ENTITY = [
    ("x", "f"),
    ("y", "f"),
    ("vx", "f"),
    ("vy", "f"),
    ("rx", "f"),
    ("ry", "f"),
    ("type", "i"),
    ("image_type", "i"),
    ("image_theme", "i"),
    ("render_z", "i"),
    ("will_erase", "i"),
    ("collides_with_entities", "i"),
    ("collision_margin", "f"),
    ("rotation", "f"),
    ("vrot", "f"),
    ("is_reflected", "i"),
    ("fire_time", "i"),
    ("spawn_time", "i"),
    ("life_time", "i"),
    ("expire_time", "i"),
    ("use_abs_coords", "i"),
    ("friction", "f"),
    ("smart_step", "i"),
    ("avoids_collisions", "i"),
    ("auto_erase", "i"),
    ("alpha", "f"),
    ("health", "f"),
    ("theta", "f"),
    ("grow_rate", "f"),
    ("alpha_decay", "f"),
    ("climber_spawn_x", "f"),
]
_MID = [
    ("use_procgen_background", "i"),
    ("background_index", "i"),
    ("bg_tile_ratio", "f"),
    ("bg_pct_x", "f"),
    ("char_dim", "f"),
    ("last_move_action", "i"),
    ("move_action", "i"),
    ("special_action", "i"),
    ("mixrate", "f"),
    ("maxspeed", "f"),
    ("max_jump", "f"),
    ("action_vx", "f"),
    ("action_vy", "f"),
    ("action_vrot", "f"),
    ("center_x", "f"),
    ("center_y", "f"),
    ("random_agent_start", "i"),
    ("has_useful_vel_info", "i"),
    ("step_rand_int", "i"),
    ("arg_seeded", "i"),
    ("arg", "s"),
    ("main_width", "i"),
    ("main_height", "i"),
    ("out_of_bounds_object", "i"),
    ("unit", "f"),
    ("view_dim", "f"),
    ("x_off", "f"),
    ("y_off", "f"),
    ("visibility", "f"),
    ("min_visibility", "f"),
]
_TAIL = {
    "maze": [("maze_dim", "i"), ("world_dim", "i")],
    "heist": [("num_keys", "i"), ("world_dim", "i"), ("has_keys", "vb")],
    "miner": [("diamonds_remaining", "i")],
    # chaser's tail carries two derived vectors before its scalars. Neither reaches a pixel
    # (`free_cells` and `is_space_vec` are pathfinding caches), but both have to be READ to find
    # `eat_time`, which does: it is half of `image_for_type(ENEMY)`.
    "chaser": [
        ("free_cells", "vi"),
        ("is_space_vec", "vb"),
        ("eat_timeout", "i"),
        ("egg_timeout", "i"),
        ("eat_time", "i"),
        ("total_enemies", "i"),
        ("total_orbs", "i"),
        ("orbs_collected", "i"),
        ("maze_dim", "i"),
    ],
    # bigfish is NOT a shipped env (see docs/research/pixel_native.md 2.9); its tail stays here
    # because `scratch/procgen/bigfish_env.py` drives the reference through this parser.
    "bigfish": [("fish_eaten", "i"), ("r_inc", "f")],
}


def _parse(buf: bytes) -> dict:
    import struct

    o = [0]

    def i():
        v = struct.unpack_from("<i", buf, o[0])[0]
        o[0] += 4
        return v

    def fl():
        v = struct.unpack_from("<f", buf, o[0])[0]
        o[0] += 4
        return v

    def sb():
        n = i()
        v = buf[o[0] : o[0] + n]
        o[0] += n
        return v

    rd = {
        "i": i,
        "f": fl,
        "s": sb,
        "vb": lambda: [i() > 0 for _ in range(i())],
        "vi": lambda: [i() for _ in range(i())],
    }
    st = {k: rd[t]() for k, t in _HEAD}
    st["grid_size"] = i()
    st["entities"] = [{k: rd[t]() for k, t in _ENTITY} for _ in range(i())]
    st.update({k: rd[t]() for k, t in _MID})
    gw, gh = i(), i()
    st["grid_w"], st["grid_h"] = gw, gh
    st["grid"] = np.asarray([i() for _ in range(i())], np.int32).reshape(gh, gw)
    st["_tail"] = _TAIL[st["game_name"].decode()]
    st.update({k: rd[t]() for k, t in st["_tail"]})
    st["_rest"] = buf[o[0] :]  # vecgame.cpp's END_OF_BUFFER sentinel
    return st


def _dump(st: dict) -> bytes:
    import struct

    p = []

    def wi(v):
        p.append(struct.pack("<i", int(v)))

    def wf(v):
        p.append(struct.pack("<f", float(v)))

    def ws(v):
        wi(len(v))
        p.append(bytes(v))

    def wvb(v):
        wi(len(v))
        for x in v:
            wi(1 if x else 0)

    def wvi(v):
        wi(len(v))
        for x in v:
            wi(x)

    w = {"i": wi, "f": wf, "s": ws, "vb": wvb, "vi": wvi}
    for k, t in _HEAD:
        w[t](st[k])
    wi(st["grid_size"])
    wi(len(st["entities"]))
    for e in st["entities"]:
        for k, t in _ENTITY:
            w[t](e[k])
    for k, t in _MID:
        w[t](st[k])
    wi(st["grid_w"])
    wi(st["grid_h"])
    g = np.asarray(st["grid"]).reshape(-1)
    wi(g.size)
    for v in g.tolist():
        wi(v)
    for k, t in st["_tail"]:
        w[t](st[k])
    return b"".join(p) + st["_rest"]


def _make_entity(sl, x, y, theme, type_, image_type, rot=0.0, refl=0, agent=False) -> dict:
    e = {k: (0 if t == "i" else 0.0) for k, t in _ENTITY}
    e.update(
        alpha=1.0,
        alpha_decay=1.0,
        x=float(x),
        y=float(y),
        rx=float(sl.rx),
        ry=float(sl.ry),
        type=int(type_),
        image_type=int(image_type),
        image_theme=int(theme),
        render_z=int(sl.z),
        rotation=float(rot),
        is_reflected=int(refl),
        use_abs_coords=int(sl.abs_coords),
        smart_step=int(agent),
    )
    return e


def _grid_to_classes(game, grid):
    # sized by the largest id in the map, not by a round number: chaser's ORB is 1002, because
    # `draw_grid_obj` values live above `USE_ASSET_THRESHOLD` rather than in the sprite range.
    lut = np.zeros(max(_CLASS_OF[game]) + 1, np.float32)
    for k, v in _CLASS_OF[game].items():
        lut[k] = v
    return lut[np.asarray(grid, np.int64)]


def _classes_to_grid(game, cls, dim):
    ids = np.asarray(_PROCGEN_ID[game], np.int32)
    return ids[np.clip(np.rint(cls).astype(np.int64), 0, len(ids) - 1)].reshape(dim, dim)


# ===================================================================== the four games


@register
class ProcgenMaze(_ProcgenEnv):
    """25x25 topdown maze: sand walls, one cheese, a mouse that mirrors with its heading.

    ``bg_pct_x`` is deliberately NOT in the state: every topdown background is square, so
    ``offset_x = bg_pct_x * (bg_ar - world_ar)`` is identically zero and the dimension would be dead.
    """

    game = "maze"
    name = "procgen-maze"
    reference_id = "procgen-maze-v0"
    state_dim = 25 * 25 + 4
    state_names = tuple(f"cell_{y}_{x}" for y in range(25) for x in range(25)) + (
        "bg_index",
        "agent_x",
        "agent_y",
        "agent_reflected",
    )

    def _unpack(self, s):
        f = super()._unpack(s)
        n = self.n_cells
        f["agent"] = (s[:, n + 1], s[:, n + 2], s[:, n + 3])
        return f

    def _slot_fields(self, f, sl, B):
        x, y, r = f["agent"]
        return None, x, y, None, r

    @classmethod
    def state_from_blob(cls, b):
        a = [e for e in b["entities"] if e["type"] == 0][0]
        return np.concatenate(
            [
                _grid_to_classes("maze", b["grid"]).reshape(-1),
                [b["background_index"], a["x"], a["y"], a["is_reflected"]],
            ]
        ).astype(np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n = 625
        b["grid"] = _classes_to_grid("maze", st[:n], 25)
        b["grid_w"] = b["grid_h"] = 25
        b["grid_size"] = n
        b["background_index"] = int(round(float(st[n])))
        sl = SPECS["maze"].ent_slots[0]
        b["entities"] = [
            _make_entity(sl, st[n + 1], st[n + 2], 0, 0, 0, refl=int(round(float(st[n + 3]))), agent=True)
        ]
        return b


@register
class ProcgenHeist(_ProcgenEnv):
    """13x13 maze with coloured locks, keys and a key ring.

    The only place in the four games where Qt's TRANSFORMED texture path runs: the agent calls
    ``Entity::face_direction``, so its sprite is drawn through ``translate/rotate/drawImage`` rather
    than the axis-aligned fast path, and the three ring keys are drawn rotated by pi/2 in absolute
    screen coordinates.
    """

    game = "heist"
    name = "procgen-heist"
    reference_id = "procgen-heist-v0"
    state_dim = 13 * 13 + 1 + 3 + 2 + 9 + 9 + 3
    state_names = (
        tuple(f"cell_{y}_{x}" for y in range(13) for x in range(13))
        + ("bg_index", "agent_x", "agent_y", "agent_rotation", "exit_x", "exit_y")
        + tuple(f"{k}{i}_{c}" for k in ("lock", "key") for i in range(3) for c in ("x", "y", "present"))
        + ("has_key0", "has_key1", "has_key2")
    )
    _BASE = 13 * 13 + 6

    def _unpack(self, s):
        f = super()._unpack(s)
        n = self.n_cells
        b = self._BASE
        f["agent"] = (s[:, n + 1], s[:, n + 2], s[:, n + 3])
        f["exit"] = (s[:, n + 4], s[:, n + 5])
        f["lock"] = [(s[:, b + 3 * i], s[:, b + 3 * i + 1], s[:, b + 3 * i + 2]) for i in range(3)]
        f["key"] = [(s[:, b + 9 + 3 * i], s[:, b + 9 + 3 * i + 1], s[:, b + 9 + 3 * i + 2]) for i in range(3)]
        f["has_key"] = s[:, b + 18 : b + 21]
        return f

    def _slot_fields(self, f, sl, B):
        k = sl.key
        if k.startswith("lock") or k.startswith("key"):
            x, y, on = f["lock" if k.startswith("lock") else "key"][int(k[-1])]
            return on > 0.5, x, y, None, None
        if k.startswith("ring"):
            z = f["has_key"][:, 0] * 0
            return f["has_key"][:, int(k[-1])] > 0.5, z, z, None, None
        if k == "exit":
            return (None, *f["exit"], None, None)
        x, y, rot = f["agent"]
        return None, x, y, rot, None

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        ex = [e for e in ents if e["type"] == 9]
        locks = {e["image_theme"]: e for e in ents if e["type"] == 1}
        keys = {e["image_theme"]: e for e in ents if e["type"] == 2}
        hk = list(b["has_keys"]) + [False] * 3
        row = [
            b["background_index"],
            a["x"],
            a["y"],
            a["rotation"],
            ex[0]["x"] if ex else 0.0,
            ex[0]["y"] if ex else 0.0,
        ]
        for tbl in (locks, keys):
            for i in range(3):
                e = tbl.get(i)
                row += [e["x"] if e else 0.0, e["y"] if e else 0.0, 1.0 if e else 0.0]
        row += [1.0 if hk[i] else 0.0 for i in range(3)]
        return np.concatenate([_grid_to_classes("heist", b["grid"]).reshape(-1), row]).astype(np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n, base = 169, cls._BASE
        b["grid"] = _classes_to_grid("heist", st[:n], 13)
        b["grid_w"] = b["grid_h"] = 13
        b["grid_size"] = n
        b["background_index"] = int(round(float(st[n])))
        S = {s.key: s for s in SPECS["heist"].ent_slots}
        ents = []
        for kind, off, ty in (("lock", 0, 1), ("key", 9, 2)):
            for i in range(3):
                x, y, on = st[base + off + 3 * i : base + off + 3 * i + 3]
                if on > 0.5:
                    ents.append(_make_entity(S[f"{kind}{i}"], x, y, i, ty, ty))
        ents.append(_make_entity(S["exit"], st[n + 4], st[n + 5], 0, 9, 9))
        ents.append(_make_entity(S["agent"], st[n + 1], st[n + 2], 0, 0, 0, rot=float(st[n + 3]), agent=True))
        for i in range(3):
            sl = S[f"ring{i}"]
            ents.append(_make_entity(sl, *sl.fixed_xy, i, 11, 2, rot=sl.rotation))
        b["entities"] = ents
        b["has_keys"] = [bool(st[base + 18 + i] > 0.5) for i in range(3)]
        b["num_keys"] = 3
        return b


@register
class ProcgenChaser(_ProcgenEnv):
    """13x13 pac-man: stone maze, green orbs painted as fillRects, 3 eggs that hatch into chasers.

    Three things here are new relative to maze/heist/miner, and each was measured before it was
    written (``scratch/procgen/probe1{3,4,5}*.py``):

      * **a grid value that is not a sprite.** ``ChaserGame::draw_grid_obj`` paints ORB as a solid
        ``QColor(0, 255, 0)`` over the centred 30% sub-rect of the cell, so the grid is four gathers
        PLUS a coverage table. Qt's aliased ``fillRect`` rounding had never been pinned in this repo
        and the two readings disagree by a whole pixel: ``QRectF::toRect()`` scores **169/169** cells
        and ``toNormalizedFillRect``'s ``qRound(edge + 0.5 - 1/64)`` — which is what Qt's source has
        for ``fillRect(QRectF)`` and is therefore the tempting one — scores **1/169**. The fill is
        1, 2 or 4 device pixels per cell at 64 px, exactly ``(0, 255, 0)``, and **no two cells' fills
        share a pixel**, which is what makes one table enough.
      * **a sprite that is state rather than a constant of the slot.** ``image_for_type(ENEMY)``
        returns ENEMY_WEAK while ``cur_time - eat_time < eat_timeout``, else ``ENEMY + rem`` with
        ``rem = (cur_time / 2) % 4`` folded so ``3 -> 1``. All three enemies share it, so it
        collapses to ONE state dimension over four images — enumerated over 32 ``(cur_time,
        eat_time)`` pairs, 0 unmatched. The four are not the same size (48x48 for the weak one,
        64x64 for the other three), so the blit's source extent is per environment.
      * **no background dimension.** ``load_background_images`` points at
        ``topdown_simple_backgrounds``, which holds one image, and every harvested state has index 0.

    The z=0 slot order is fixed (orbs, eggs, enemies) and that is safe for heist's reason, re-measured
    here: reversing the z=0 entity list changed **0 of 360** real frames, and **0 of 360** had two
    z=0 entities overlapping even in world space.
    """

    game = "chaser"
    name = "procgen-chaser"
    reference_id = "procgen-chaser-v0"
    n_bg_state = 0
    pool_warmup = 128
    state_dim = 13 * 13 + 2 + 9 + 9 + 9 + 1
    state_names = (
        tuple(f"cell_{y}_{x}" for y in range(13) for x in range(13))
        + ("agent_x", "agent_y")
        + tuple(f"{k}{i}_{c}" for k in ("orb", "egg", "enemy") for i in range(3) for c in ("x", "y", "present"))
        + ("enemy_sprite",)
    )
    _BASE = 13 * 13 + 2

    def _unpack(self, s):
        f = super()._unpack(s)
        b = self._BASE
        f["agent"] = (s[:, b - 2], s[:, b - 1])
        for j, k in enumerate(("orb", "egg", "enemy")):
            o = b + 9 * j
            f[k] = [(s[:, o + 3 * i], s[:, o + 3 * i + 1], s[:, o + 3 * i + 2]) for i in range(3)]
        f["enemy_sprite"] = s[:, b + 27].round().clamp_(0, 3).long()
        return f

    def _slot_fields(self, f, sl, B):
        if sl.key == "agent":
            return (None, *f["agent"], None, None)
        x, y, on = f[sl.key[:-1]][int(sl.key[-1])]
        return on > 0.5, x, y, None, None

    _sprite_lut: dict = {}

    def _slot_sprite(self, f, sl, B):
        if not sl.key.startswith("enemy"):
            return None
        key = str(self.device)
        lut = self._sprite_lut.get(key)
        if lut is None:
            lut = torch.as_tensor(_CHASER_ENEMY_SPRITE, device=self.device)
            self._sprite_lut[key] = lut
        return lut[f["enemy_sprite"]]

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        row = [a["x"], a["y"]]
        for ty in (2, 4, 6):  # LARGE_ORB, ENEMY_EGG, ENEMY
            got = [e for e in ents if e["type"] == ty][:3]
            for i in range(3):
                e = got[i] if i < len(got) else None
                row += [e["x"] if e else 0.0, e["y"] if e else 0.0, 1.0 if e else 0.0]
        row.append(cls._enemy_sprite(b))
        return np.concatenate([_grid_to_classes("chaser", b["grid"]).reshape(-1), row]).astype(np.float32)

    @staticmethod
    def _enemy_sprite(b) -> int:
        """``image_for_type(ENEMY)`` as an index into ``_CHASER_ENEMY_SPRITE``."""
        if b["cur_time"] - b["eat_time"] < b["eat_timeout"]:
            return 0
        rem = (int(b["cur_time"]) // 2) % 4
        return 1 + (1 if rem == 3 else rem)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n, base = 169, cls._BASE
        b["grid"] = _classes_to_grid("chaser", st[:n], 13)
        b["grid_w"] = b["grid_h"] = 13
        b["grid_size"] = n
        S = {s.key: s for s in SPECS["chaser"].ent_slots}
        ents = []
        for j, (k, ty) in enumerate((("orb", 2), ("egg", 4), ("enemy", 6))):
            for i in range(3):
                x, y, on = st[base + 9 * j + 3 * i : base + 9 * j + 3 * i + 3]
                if on > 0.5:
                    ents.append(_make_entity(S[f"{k}{i}"], x, y, 0, ty, ty))
        ents.append(_make_entity(S["agent"], st[base - 2], st[base - 1], 0, 0, 0, agent=True))
        b["entities"] = ents
        # Drive `image_for_type(ENEMY)` to the requested sprite through the two fields it reads.
        # `eat_timeout` is 75 and comes from the template rather than being written here.
        k = int(round(float(st[base + 27])))
        if k == 0:
            b["cur_time"], b["eat_time"] = 0, 0
        else:
            b["cur_time"] = 2 * (k - 1)
            b["eat_time"] = b["cur_time"] - int(template["eat_timeout"]) - 1
        return b


@register
class ProcgenMiner(_ProcgenEnv):
    """20x20 boulder-dash: dirt, boulders, diamonds, a digging robot and an exit window.

    The grid folds ``MOVING_BOULDER``/``MOVING_DIAMOND`` into their stationary forms, because
    ``image_for_type`` gives both pairs the same sprite. The frame is unaffected; the dynamics are,
    and this module claims only the frame. ``bg_pct_x`` IS state here: 52 of the 62 platform/space
    backgrounds are non-square, so the horizontal offset is real.
    """

    game = "miner"
    name = "procgen-miner"
    reference_id = "procgen-miner-v0"
    state_dim = 20 * 20 + 7
    state_names = tuple(f"cell_{y}_{x}" for y in range(20) for x in range(20)) + (
        "bg_index",
        "bg_pct_x",
        "agent_x",
        "agent_y",
        "agent_reflected",
        "exit_x",
        "exit_y",
    )

    def _unpack(self, s):
        f = super()._unpack(s)
        n = self.n_cells
        f["bg_pct_x"] = s[:, n + 1]
        f["agent"] = (s[:, n + 2], s[:, n + 3], s[:, n + 4])
        f["exit"] = (s[:, n + 5], s[:, n + 6])
        return f

    def _slot_fields(self, f, sl, B):
        if sl.key == "exit":
            return (None, *f["exit"], None, None)
        x, y, r = f["agent"]
        return None, x, y, None, r

    @classmethod
    def state_from_blob(cls, b):
        ents = b["entities"]
        a = [e for e in ents if e["type"] == 0][0]
        x = [e for e in ents if e["type"] == 6]
        return np.concatenate(
            [
                _grid_to_classes("miner", b["grid"]).reshape(-1),
                [
                    b["background_index"],
                    b["bg_pct_x"],
                    a["x"],
                    a["y"],
                    a["is_reflected"],
                    x[0]["x"] if x else 0.0,
                    x[0]["y"] if x else 0.0,
                ],
            ]
        ).astype(np.float32)

    @classmethod
    def blob_from_state(cls, template, st):
        b = dict(template)
        n = 400
        b["grid"] = _classes_to_grid("miner", st[:n], 20)
        b["grid_w"] = b["grid_h"] = 20
        b["grid_size"] = n
        b["background_index"] = int(round(float(st[n])))
        b["bg_pct_x"] = float(st[n + 1])
        S = {s.key: s for s in SPECS["miner"].ent_slots}
        b["entities"] = [
            _make_entity(S["exit"], st[n + 5], st[n + 6], 0, 6, 6),
            _make_entity(S["agent"], st[n + 2], st[n + 3], 0, 0, 0, refl=int(round(float(st[n + 4]))), agent=True),
        ]
        return b
