"""Craftax — a tile atlas underneath, and an honest accounting of what the state actually costs.

Reference: ``Craftax-Classic-Pixels-v1`` (JAX). The observation is drawn by
``craftax.craftax_classic.renderer.make_craftax_pixel_renderer(7)``: a 7x9 window of the world map
expanded to 7 px per tile, the player sprite composited over the centre tile, then zombies, cows,
skeletons and arrows composited one tile each, then a day/night post-process and a sleeping
desaturation, then a 2-row inventory strip concatenated below. All of it is a gather from fixed
tables plus at most a dozen sprite blits, which is why it belongs in this repo.

**The state that ships with Craftax is a loss, and the docstring says so rather than omitting it.**
Measured: the full ``EnvState`` is 273,276 bytes against a
42,900-byte frame for Craftax, and 20,952 bytes against an 11,907-byte frame for Classic. Storing
``EnvState`` instead of images is **6.37x worse** and **1.76x worse** respectively. The reason is that
``EnvState`` carries the whole world — all nine 48x48 levels of ``map``, ``item_map`` and ``light_map``
for full Craftax — while the frame shows a 9x11 window of one of them. There is no version of "store
the state" that wins if the state is the simulator's state.

What wins is the **render window**: the tiles actually on screen, the sprites drawn over them, the
sixteen small integers the inventory strip reads, the light level, the sleeping flag and the PRNG
key. That is what this env carries — 149 float32 numbers, **596 bytes**, against a 62,208-byte frame
at the default 16 px tile size (**104x**) or an 11,907-byte frame at the 7 px agent size (**20x**).

The key costs **four** slots rather than two, and that is measured rather than stylistic: only
2.008% of random uint32 survive a round trip through float32 (2**32 > 2**24), while 100% of 100,000
do when each is split into two 16-bit halves.

Parity against the official renderer, measured on 64 sampled states at both tile sizes, over the
**whole observation** — world window and inventory strip — and over the whole day cycle: 18 night
rows, 29 dusk rows, 17 daylight rows, sleeping rows throughout.
**100% of all pixels exact in every band, mean |diff| 0.000, max |diff| 0.**

**The daylight path is one gather from a uint8 table** — no float buffer, no PRNG — and the
measured cost of growing the frame from 7 to 9 tile rows is almost all of the extra time. One Apple
arm64 core, one thread: at 16 px and B=256, **1.440 -> 1.871 ms**, of which **1.851 ms is the
112 -> 144 row growth** and 0.020 ms (1.1%) is everything else; at B=1024 the growth accounts for
all of it. What remains is a **flat ~0.02 ms per call**, which matters only at tiny batches
(B=1: 0.069 -> 0.111 ms). Independently measured 0.061 -> 0.166 ms at B=1 because ``_strip_grid``
rebuilt two tensors with 32 indexed assignments on every render and the two post-process colour
constants were allocated per call; all four are now cached per device, and the PRNG key is assembled
only on a lit row.

**"The official frame" is not one object, and the parity number says which one.** Under ``jax.jit``
XLA fuses the renderer's arithmetic differently and it disagrees with its own un-jitted self by up to
1.5e-05 in float32. For a daylight frame that is invisible in uint8. For a sleeping one it is not:
the desaturation halves the dynamic range, and a 1.5e-05 difference then straddles a rounding
boundary. Against this renderer, the jitted reference is 99.7854% of pixels exact at 7 px and
99.7225% at 16 px with max |diff| 1 on the sampled pool used here, where every differing pixel
happened to be on a sleeping row; the un-jitted reference is 100.0000% with max |diff| 0. The
sleeping rows are not the rule, only the commonest case: independent review found the jitted
reference differing on an **awake** row too, at ``light_level = 0.49999997``. **Any lit row can
straddle**, because the mechanism is a ~1e-5 float32 shift meeting a uint8 rounding boundary, and
lighting -- like the sleep desaturation -- compresses the range towards one. :meth:`CraftaxClassic.reference_env` therefore returns the
un-jitted function — the renderer as written — and pays 0.25 s a frame for it. The claim certified is
a **uint8** claim; this renderer does not assert bit equality of the reference's float, because the
reference does not have one value to be equal to.

What this env covers, and what it still does not:

  * the **world layer** — terrain, the player's facing, the sleeping sprite, and the mobs and arrows
    on screen — exactly;
  * the **2-row inventory strip** (rows 49-62 of the 63x63 agent observation, 112-143 of the 144x144
    image observation). It is a function of twelve inventory counts and four intrinsics and of nothing
    else, drawn from a 160-entry ``(cell, count)`` table folded into the same atlas as the world tiles,
    so the whole observation is still one gather. The reference concatenates it **after** the lighting
    chain, so it is the same picture at midnight and at noon — measured, not assumed;
  * the **sleeping player**: both halves of it. ``is_sleeping`` selects a fifth player sprite
    (``player_texture_index = select(is_sleeping, 4, player_direction - 1)``) *and* desaturates the
    world rows. Missing either half gives a plausible frame and a wrong one;
  * **night**, including the noise field. ``light_level < 0.5`` mixes in
    ``jax.random.uniform(state.state_rng, (H, W)) * 95 + 32`` through a vignette, so a night frame is
    a function of the PRNG key as well as of the map — measured: changing ``state_rng`` moves 72% of
    the world layer's channels on a night frame and **0** on a daylight one.
    ``stable_envs/envs/craftax/prng.py`` reproduces JAX's threefry2x32 counter PRNG bit for bit and is
    shared with the full-Craftax renderer. Two bands cost nothing: ``light >= 0.5`` makes
    ``night_static_intensity`` exactly 0, so no PRNG is drawn (58.3% of a 300-step day), and
    ``light == 1.0`` makes the whole chain the identity on the world pixels, so the batch stays on
    the uint8 gather with no float buffer at all;
  * it carries **one mob per tile**. Craftax's own ``mob_map`` keeps zombies, cows and skeletons off
    each other's tiles, but an arrow may share a tile with a mob, and the reference then composites
    both. That state is not representable here and the sampler does not generate it.

Three measured facts that shaped the implementation:

  * **The alphas are not binary.** They take exactly {0, 128/255, 1}, so the composite is a real blend
    and the frame is fractional-valued float32. This repo's frames are uint8, so ``render`` rounds
    (half-to-even, matching numpy) and ``reference_render`` rounds the official float the same way. The
    frame is a rounding of the reference, not a lossless view of it, and that is stated rather than
    hidden behind an all-pixel percentage.
  * **The player tables are zero away from the centre tile** — all five of them, the sleeping sprite
    included — asserted during extraction, so compositing the player over the whole frame (what the
    reference does) and into one tile (what this does) are the same arithmetic: ``x*(1-0) + 0*0 == x``
    exactly in float32.
  * **Night is where a resolution-consistency bound goes to die, and that is the reference's
    property rather than this renderer's.** The noise field is drawn at ``(7t, 9t)``, so element
    ``i`` of the 32 px field is a different threefry counter from element ``i`` of the 16 px field.
    Rendering at 32 px and box-downsampling to 16 px therefore reproduces the 16 px render exactly at
    ``light >= 0.5`` (``exact_frac`` 1.000) and not below it (0.282 at 0.45, 0.035 at 0.10, max abs
    diff 32). No port can design that away without changing pixels. See
    ``pre-repository-cleanup:tests/fixtures/regression/join/step7/craftax-classic-full/findings.md``.
  * **The strip's counts cannot come from a random rollout.** Over 4,000 steps, ten of the twelve
    inventory cells never leave zero, because a random policy never crafts. A pool built from one
    presents ten dead state dimensions, which is indistinguishable from a wrong state contract — the
    failure ``pre-repository-cleanup:docs/research/pixel_native.md`` records for Procgen miner and heist. ``sample_states``
    samples the counts over 0..9 directly instead, which is sound because the strip is a function of
    the counts alone and the simulator caps every count at 9.

Out-of-range class ids and counts **wrap** rather than clamp. Real states are always in range, so the
rule never fires in practice; it exists because a clamping decoder makes a cell that already holds the
last class insensitive to the conformance suite's +25 probe, which would be reported as a dead state
dimension when it is an artifact of the probe.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np
import torch

from ...env import Env
from ...registry import register
from ...render import _resources as _RES
from ...spaces import Box, Dict, Discrete
from . import prng

TILE_PX_AGENT = 7
TILE_PX_IMG = 16
OBS_H, OBS_W = 7, 9
#: rows of inventory strip below the world window, ``INVENTORY_OBS_HEIGHT`` in craftax's constants
STRIP_H = 2
N_BLOCK = 17
N_MOB = 8  #: none, zombie, cow, skeleton, arrow x 4 orientations
#: facing left, right, up, down, **asleep**. The fifth is not decoration: the reference selects
#: ``player_texture_index = select(is_sleeping, 4, player_direction - 1)``, so a sleeping player is a
#: different sprite and not only a post-process, and ``player-sleep.png`` is asserted zero away from
#: the centre tile exactly like the four facing sprites.
N_PLAYER = 5
PLAYER_ASLEEP = 4
#: facing values a state row carries: left, right, up, down. Slot 4 is NOT reachable from a facing
#: value -- it is reached by ``is_sleeping`` -- so ``player_facing`` wraps mod four, not mod N_PLAYER.
N_FACING = 4
#: inventory-strip cells: 4 intrinsics then 12 inventory counts, each holding a count in 0..9 that
#: the simulator caps at 9 (``craftax.craftax_classic.game_logic``: 1633 inventory, 1293 energy,
#: 1319 health). Cell ``i`` is drawn at ``STRIP_LAYOUT[i]``; the two cells the reference never writes
#: stay black.
STRIP_NAMES = (
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
STRIP_LAYOUT = (
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
    (0, 8),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
)
N_STRIP = len(STRIP_NAMES)
N_COUNT = 10  #: 0..9, the whole digit alphabet; ``number_textures[0]`` is the empty texture
#: A ``state_rng`` key is two uint32, and a uint32 does NOT survive a float32 state row: only 2.008%
#: of random uint32 round-trip through one (2**32 > 2**24). Split into 16-bit halves, 100% of 100,000
#: do. So the key costs four slots, not two. See ``stable_envs/envs/craftax/prng.py``.
N_KEY_HALVES = 4
#: ``night_texture`` -- the colour night is tinted towards, ``[[[0, 16, 64]]]`` in craftax's constants.
NIGHT_TINT = (0.0, 16.0, 64.0)
#: Below this ``light_level`` the reference mixes in a noise field; at or above it
#: ``night_static_intensity = max(2*(0.5 - daylight), 0)`` is exactly 0 and the noise cannot
#: contribute, so no PRNG is drawn at all. Over a 300-step day, 58.3% of steps are at or above it.
NIGHT_THRESHOLD = 0.5
#: craftax's day length, used only by ``sample_states`` to give ``light_level`` the distribution the
#: simulator actually produces rather than a uniform one.
DAY_LENGTH = 300

#: block id -> name, straight from craftax.craftax_classic.constants.BlockType
BLOCK_NAMES = (
    "invalid",
    "out_of_bounds",
    "grass",
    "water",
    "stone",
    "tree",
    "wood",
    "path",
    "coal",
    "iron",
    "diamond",
    "crafting_table",
    "furnace",
    "sand",
    "lava",
    "plant",
    "ripe_plant",
)

_BLOCK_TEX_7_SHAPE = (17, 7, 7, 3)
_BLOCK_TEX_7_B64 = (
    "eNrtWUFIFVEUnbLMQCoXSRJRkARBUAs3ReH/bwpaKNmmTduIli1cBC2UFvFpEdJKIkIkRCKkVURIRESISLSQiAiRiGgRESLiQqJ3nTvN+a9/vncapUV/"
    "4DB3ruede9+dN+8934+igThaveTesBt2ETtqoIF1wvtyFJ3UsTXs7f3eHipze9yjWfglzrFoog7arG1eTeTn7aNF01IHC8cSq0gfN1oT/TdcFO3w9uW4"
    "2j7g/zbkn3cGfrlG/fOROnaXvy95dNexLbHy5maJVYTP+sjymfL+af986i9qgrEsNUE+xrXwWU0sdUP/R5fhJ0DqkqIFsAnww9XGlrg2kDMHkFxSfHUZ"
    "MFYH4LvLgHmiPnK2xRkwFnKwj5jbSwDyMdYs4BUAY+0GsDwZf9llwDqw+suckQJruOAyYKwllwE5WLcZlwHbYq1wjpLriYt+X6k92B1FV1xyZ/56OmfV"
    "vu2SMf024MjzovpRE/3YFjlt2s+Kq+Z88fdxl/D2eM43f3/squMyDuojB/3y3d/06HS8DmwtQE153uv9FX+fF57qlJRnXY9Q80Up6aPcUfOY7BXW0EQd"
    "rKfMYT0uiYmaqCPfs7Qf9Dio3+VDl/ieepx21Zy8/Ibm/6O54vHG46rOj9Nr2LJWzfv7vcAvc+gHl4zXrcq5H9jI36zr612P7bq2PXD6/4DqIL9V16Sx"
    "IBa2xdyQI9/cJ38fCfiMg7kt6Lx+K+CjHdZK+LJ/Er1Jj35jbZGP7wXzwTogB23sC6vJiq7JlUC/WfnDgebq+uBxR/dA79TG/McA++IM5wAllwH5uI5j"
    "22VXGyUCxme5oZ9drC3TZ/1Cv6UmCNZfliderCbYF6aZt4aW98j4jHPRtz8RV+uFcdvjDIyPnOMeTXEyD8jzI12LMQf0y15T5oCWwGb5PCMIc64V19Iv"
    "zN9SQ1ZzzAE1WQ4sFubGOKz+lrGEebJ6oh9jsXHL4haZl9Jz0oGY++W6rjbLDTloo2bIcf5+puC8VGR+Yzlb8mR1ztvHjZ6X8u7NpnR9nKnD/6wxZc/P"
    "7DEd+31lm2bePNdrz8k0R8rJ3zv0vHdUz3Msmu1xbaAOy+FwnAHbYg2x7Xr13aJZAeDcxfy74gxdLgPjdxnA9C1+y2XRwfzxkm9ezh1CP57DsHUW+9jr"
    "NV7LOygn69Ql2a8GNtPEnDFPpok5oybrO+bZT9AKYO/6vNct6/8Gkk+Pnk1ZbGzL7JK/H42Tsz7W1qJv4WCsvP1ifIs/b55FYhV5L0Xeo8WWMTzhsm9Q"
    "3sW4+g/Bb8CMswh+5KecC3FyplcqVduhJupMEBv1kd9Gcmb5p36WT+hvM9Sklh3qsD4uGvrL+LX6y+LWexeWfjFOWp+N+G1X9hLPdU+RnnenV2qjjoWP"
    "tiWWcJriP205q51dQ4dpYq1kLZnUs1/Z883p3s9SK9TEtiG/KcfvsBbNvL/tspqzfjXGkn0s9ervNdecbSxJXfu0LdqoKecene7P8cnGEvKR8+/H0i9R"
    "01o9"
)

_PLAYER_TEX_7_SHAPE = (5, 7, 7, 3)
_PLAYER_TEX_7_B64 = (
    "eNpjYCAGxDszMKgD8QMnBqoBSsysd4ZgEHjgyMAgD2QbOONmb3CE2PPDCbc5yOx/QHV+QHYEHnOQ1RDjTjcgrQnEokAsBMRngPr7gPgclHYB4iKoeXbO"
    "xIUDspnIANlMUgEtzBzMaQlXXONiI8cpLoArneBKS7jSIS4/EpOWiHHnUE9LpPqRGHOIiQtqpSVc5RIlaYmYconUtDRaLg3+cmm4piVq5U1caQnZHFqk"
    "JWq5H1daQjeTx5m8tESMOchqiElLILVBSGZiY4/ccgkANRruVg=="
)

_PLAYER_ALPHA_7_SHAPE = (5, 7, 7, 3)
_PLAYER_ALPHA_7_B64 = (
    "eNpjYCAGNNgjMLUAJWYi6yUVU8scYtxPvpmNDQxwTHoYkmoOMXqxi1Ni/vBKS7ROV6NpiXZpiZK4HsxpabRcGi2XhlNaosQc8s2khfmUlS3E2DVaxw32"
    "cgkAKtZf8Q=="
)

_MOB_TEX_7_SHAPE = (8, 7, 7, 3)
_MOB_TEX_7_B64 = (
    "eNrtVTEOwjAM9MgGa6VuLDyDJgsjI49BgpGBNzAxMfECxCOYeAATj+A65RgsHJpWrZqTTo2qy9lOLEckI6NLrJ3IGzy6fnhOvcgSLP23z70SOeA7c/p/"
    "Dayx+LPGUmMqT+0ctFqanG0qz1QokNMD8c/gE9yC5R/58F5eb3wgx+I1ayx5sl6LZanFEjdWf0K8BXitbJ6x+j6j7vMXatk37Oc5fCbgBT4rH1hEkvda"
    "ctN8NI2l9psL1PbGgj25xqZz4FftsfA+MBXa8EzVt0P03PnAjO7B/czku9bW+e76gzbuYsxzaShvx9h62zKXtJk2XnwAR1ebYg=="
)

_MOB_ALPHA_7_SHAPE = (8, 7, 7, 3)
_MOB_ALPHA_7_B64 = (
    "eNrtlsEOABAMQ/dnfo0/d8NFosw20SY7rhF7OiIUZamcekXwHHvResOzZGmF3yHqs9J74nmDGS9+IrDUNc4Cn9c+b34MaPGglWlaLGkxhuXDiuecMcs3"
    "i53z/R1HTxtPKs7epP78z9OTWceMiqIKm/PmyA=="
)

_BLOCK_TEX_16_SHAPE = (17, 16, 16, 3)
_BLOCK_TEX_16_B64 = (
    "eNrtnU/oHVcVx1+bWKOmpim0WFQsWARBSBfZKC19v3kKBSvGjRs3LkRcdtGF0IXBhfxwIaGLEiRIKCUEEXEVSikiRaRIEZEgQaQEKSIiJZQfRaQU53bu"
    "8D6ZzHfeOe/e+37v/XIfHObk5sy53/P33jcz7zez2Y8Wsw8/4Vj5yle+8pWvfOUrX/nKV/5o87NKlSpVqlSpUqVKlSrdNXRxbzb7XHu8sOfnb7T0RPwu"
    "wfGrLd0Xxue380o+17wc57wpc5XQv834vTK58JfAVgJ/6djtCv4SflB9Y5vxe3ta6XktPiyB3zLvrq8FpXv+JteCEn1jk2tBLvyH1YdLrGtHCf9hrQVe"
    "bNvQu7w9n3z4vNTMZl9q/322Pb7X0lMD/sft8ZPt8ftr8Eqn4omHPHUqzBY8b7Q2/7H995Mt/2jLX2j5UxNzKVsUToXNgpnYKE+cym/KlpQ4euPljXUK"
    "Nm/sLHGkzhRfKZ0pcUyJqcKQq2Zz5ZU351NyT/UTC79JX1lyzDKv8mdK/0/xW+meo3qpyjHKq7ko85n2+M+Wvp4Yd0s9psybKz+V3yw9Kte8KWu0mstb"
    "mylrN2XUPsSLwbK+qHnDPvPVlv+iMS5Khjmp/KDiomJNPd5ekWs9SpFXMSrR91J0/q9Z0onFkv7bjNM7oOOLJZ0CPQp6q1lSwNXTQ6BHQNebJd0Ccfz3"
    "IJ7792ZJlPkARFtOLcaJfni9WRJtuWexpGOgW81qIjblQ9r7JkjhZ1yI812Qwq+wUYZ+fhB03EnKJwo/bWf+kKjnXyDK0BbmA+1V+aBsYYwob8FPPDyX"
    "8VLyHKc8Y0S7LDWr4sK5qP+dZpw+uliSqn2l01IXDwmy+Jw6iZnzsr5U7XBc9UZiOyFI6WTPUZhVTBk76meusl8xXhbfkohT1Y7CTL+pea8340TMrHcV"
    "O4tvOS9zRvUBzsX8UWvcDH2HmFX+q3pXmEnMAdW7GAv6lnP9o1mSmou2MK/UvKrvqVgrW3guY6Hwf2yxJPqZ+az0MEaW9d2yxoVrtD2xX9E/xMwcVv1K"
    "5cwDIMbo9WaciJl5S5+/1yxJ5TPxKz8Tm1rTOddnF0tS/Zn9Qe1VLPsK+pP6uX8gTtpCzDxX1aDqLbSFmCmvMDMWas3y3rP4c0sHTXe8PO/yJRzJUyZ8"
    "rjXdkTIcJ89zU+6tKAznn5rNftB0RzUvZcL32qvN8t+ecyljwazO5finWtn/tPxvmjttfBr+7HnKK50/a7rvhFO+stg+5ecxPJzXYjvlOa+KEWWUr6Zs"
    "fHrERuI/HWtyv7kzn/tzp2zsZYjfkkvDWF8bmWtKT38u8Ss/E9u50DPasWcH4+RVfC01SN6ix4uNvNKp6miq1q6tiCNtUblqwalyNVwn+0lLjzW38/SJ"
    "yoHhveBV9atqR9WFBb+lBhVmL2/pq8RMPlx7e6bprpsqzOfa/7t30R3Pt3T/ovN/+Pf7rcy8PX53r9t/ncv4DIDKMWVLOOfT7Tn7MT+Oxbluhjl6/7fj"
    "f2i6sXmsq/B5fK/bszy+Afwql3juhziaO/H/bt7FOhwpQ/y5nmlRPZP4WY8KmwV/iNm/m7zP5Cj8qsd68Zd4psWy7lCGvZG1rPDfDc//fD5ei/pl/H4X"
    "/HU+xv2Vlr7a5JPheC49FX/FX/FX/BV/xV/xV/wVf8Vf8dv13Irfi56N96TCvbLLTXcNIzxH8/MBT/mPtOM32+Mv4v21cP/n5Xhf5K8tvRCfKwr8iwOZ"
    "T8Tva1daOhnvhVyJ90v+Es8N9zD+1nTfxzhvuI7yp8hPjYfv3RcGMsqWKb7XQ1uOR9svTZxL/ENbep3vx/uc+wP/hDi81tJzTbyOGH1C/JRn7BQe+of6"
    "lb1K58cX3fUHzntpkA+KVzHluIoR7SVPmXC/8kbUo2RU/lj00xaOH4/3CF8Y5AZjR52UuTfaPsx5hYFzUZ51xPhShrZT/z3Rb/sDbJSh/qHOPoeZM8Qw"
    "ZvuLA53MPdVbiJ89hOPHon8ujcS9x6lygLlHG1n7HFexIDb6hPJqrnejH346yA2VJ6q+VI1T/r7IX5zoG8O+1+P/QPR2lbeqxukr8lwL1DpF/NTJ+Kpc"
    "ZR5yXPUEVfvMW7XGEQ/jqNYsa6/OwdPPnItxJ55hvd8cyUPFK98Sg8KjzlVrIvGH5yfexlo2tqYrPVN9b9U+RO0BmMNcI9h7idmyf1B9mz5kLFjLltxT"
    "PueaS18Rm/Kz6s/D9XFMXvUu4udehTL0Odc+7l1VvXB9UfsZ4qftqlcPe/uNkfVU4VH7CpW3PJfPU5HU8/+KvHpKzHulGSel/5uLcZo3S6IenmvRn+IT"
    "ZYvCaSGFX40rGy228MNxL2alx0sp8fLiVB8LHoVNxUudq+KYgjml1rw5nyvWlnktfUCdm1JTKX7w9oGUGHl7pjcPLTXlzQ0VX++8Fl9Zaiql55fomdSv"
    "Pik+T+n5ufp/rpznufx4e2npNde7Rlg+pfd13r6Xot+7Rnj9mZJjuciLTck8LKhEXLxzWfKHer7T0leizJfjb19eXmPvpHBSP3mvT5R+Yla8JZ8tPgm/"
    "6QrXHU9E+V/FZ+eU36bwj53LcfKW3mjRT/xq3FsLyieW9UXZZYmvxT/eHH5VEPFw3FJrCrPSk9L/LTXIeHlt8WL2xprjqgZV3lqwldgLKfwWzN4eZclh"
    "77qg+obCqXoU+ZTv8rnsVdiULZY9J/2U0+djNLV2j9ml8t+71ntzXum06CFm7/7EsmZZeqNlrfSuL6oPeGNn2b9Zcl6ta95ebVkfU/b/Sv+27f/vhuv/"
    "3r3TNlz/z3XttPS1CEX9++eGFD7PR96CM0WPN15Kv+LVvLn0eHPYMlf4NO3xayvu6eTQk5LDlrlS9Fjwb8P1f28N5roW6u3nKv9VLZTY/1vwKAzD66XP"
    "O+q3RM9P6SHe+s11/X8dDGP4vfVb4vr/VF9d1UO27fq/ty/V6/9H7/o/fxew63yJ32KEv43b0/2gN+IzVG8mzpWC7e2Ym+H34oq34Kd8Cn4LHsVf6f/G"
    "wIC34Kd8Cn6FwcLvYv54ZWhLrrk2id8icyP+nZf+72P3/DbjJ87Le53sIwttyzbkj8JssbE0fuIk8RqgBb9FpoRvH3aSyhNiLpH/XszsmeSJk3yu/mPJ"
    "BwvOEr/lTPGtwrnJ36IeVv8sgX8X+X0DPbAYJ6+MRf9+k5/UPReFQcmfBSl5ynhxKj/kwpNio8LGvytLPZSnTjWXIs5l0eO1UeG3nKtwqo+yxYtZzev1"
    "ea6P8qGyMSV/LH72frxx8eaPF3O4nng6+ukbi+5vML6ylxYLS4+y1KMXc3i24XtN9zcDKeONhSLaa+lXFvzErHiL/1P6p6Vm6WeLzy22bDJ/VM6Qt+TV"
    "YeWPwqn8r+RL5483/58DWca9+ZMr/4nnJMg7rvInV/5T5mRh8uIJufpMfL/ArvPfamkv/uY5RUbx85bOxPdmWPQrPsWuw9Lp9S35FL/lil0JnUe1Xrzx"
    "2rZ6yZX/JWqhBJ7S/kzJB8u5Fplc/f+w6muTfXuTeVvaD5uMaen+k6vGt20tKLGme32VK682GRcvhhL9MHyPO2iW3wHJ/xr8Fxarx5WMRU/Isav4Xtnz"
    "6twzQkbxlCdvwbwKw7cX3Xsl5vM7fej1D3n6YSxGnHeIQWEe0zml52AFzqH8GYf+Kczr4h+ee+CMaYr+HPGd8o/KJcWvyh9rra1b71b83robOzenrzy9"
    "a518tpx72tCj+nHvvCl+Xkf+tLNXe/WnxHoMm9WfSs8qDCl5MlXL3vz0rvve3Mi11pxJ0OPtM6t8PtVnDgquWalx9ORnyp7Eux9bJ465Yrqu7daef5CA"
    "J6W/efK5xPvL1PvIlPxhvTctZd4U/Rb84TnG38ZnGxWfMq9Ff/9ewHVw8lwvr3ybay6FX8mHdwGexbs+x3h17mHpt7zzkeda5qLMRbwzNPz+4a34OwiO"
    "h/vfr8GW6yvwW95jqGrH4meF04r/LN7x2stb8HhrTfHKD8RDzOQpo/BbsKn8tNhCPMr/Xt5Sy8O8Ve8FXpXzFn8Oc4lzjdmbay2z2D6F85hhrl3Gr+ol"
    "1/tzU/B757L4szT+lLlyvdOWOi39MwX/UX3/b93/1/1/3f/X/X/d/+/W/j88Sxx+q/fD5nZ+m/f/xBnmPBfHFV9i/08M1GPJeQvm8DcVHxPX1jz7Z6+v"
    "vPs34iQ//I62Ln6vr7Z5/698xXnr/r/u/3dv//9/VAog1A=="
)

_PLAYER_TEX_16_SHAPE = (5, 16, 16, 3)
_PLAYER_TEX_16_B64 = (
    "eNrtmkFKA0EQRfsA40YQXAguxYXgBRQyLS5dZOXOO4hLITfwIK48gOQE4hlyDFdihUygHKykmqrqqZZqaKaQIb4aZqZffjolrfHap/QE87jn1d7GY07p"
    "HuY1zAeYZzBX/e/6Co4HMN+D33R4Y17kzVyPCzguB5ZvON4N1x3Xq1lKp3C8HNXU+W+zTa9fij0+D8wnmeanOCU17gX/XcLMuf7WvUiYtfg590zpfbXm"
    "6vYwe+Yv7QXzH0L9AfVLX15/DscbRX5OL5j/FupzmEeZx4zPp/ipdUSyHo172dYUD1WX8luvHRweyfnWI/in5S91Y88uHf4f/P/V/6n1Xcv/8ZqO13rJ"
    "oP6vhSdT/NiNJdeZcnKtvih+SS+lrivxZw5/aS+e+bFDcu5/il/L/7n8XZa/zymfkfj/Lv4uy9cj6h2u5f/U94vw/+AP/6/fyzgT2z6Pnv2fm+N1uT1m"
    "T71ImafuBa+VFrXWqOn/Wvn/VP4vyf89+LNF/tw6f03/t+Cv6f/W+b+1/0f+H/zh/5H/R/4f+X+L+b/F/p/w//D/8P/wf2/+b5GZ4PzH2v81M6vWs7ip"
    "s1B8P5fmt579n5Oft+L/uBeL2tqfrfLqufKzQ/FTv41S9a5+55We/VL/t2bGn6nl/9bM+PO1PLmUWbL/X8v/rZ7fxZ7nhbPXhcM/vrYW9V/7zLX4a/5e"
    "Rt17FH+L/t/i/n/8Do/9/5GfB7+X/P8HLA/27A=="
)

_PLAYER_ALPHA_16_SHAPE = (5, 16, 16, 3)
_PLAYER_ALPHA_16_B64 = (
    "eNrtm0EOwyAMBPlZv5b8vOdK5UDiJbvOIHEfIyUer2CMqnV81rbbgt+/lpSzvbMTmatqcWBercWNrSf/eYy/26OWKv5navnlgV/Xp9bYZpxq/jnD9bPd"
    "ef5qfnWPXv2/adjwZ/jxf2YBZoFrtdxxdee5Bv5aX83gr/P/FH78H//H//HnN7hoJ3+G38el02cWxfyi/n578nv7Zzb/3vw/hR//x//xf/wffvJ//B//"
    "x//xZ/zf8f7P3vz/eU79/KJlTud368uZ/s8dCfi1uUovL4W5x7sA7s/048f/yf/J/8n/8/L/L26IQ84="
)

_MOB_TEX_16_SHAPE = (8, 16, 16, 3)
_MOB_TEX_16_B64 = (
    "eNrtma9OBDEQh1cgcIclAYVB8gAIaBEIBJKH4BFIQCKQaBQKhURdeAgUgqAIgvAMzAaSHS73g5n0z3Xh9yUTyqVpv3Zn2+1u1xFCCCGEEEKIl0nsuh2J"
    "tfi9fBi67k3iQuJ+t+vO5e9KaNtfO4/FHzmPxR+NpTVnnSfazZI/uozaSfHRPEub79LeVvS7aR9UtoxFO+S6H5ED+j2lnOu6oHZK5Iw3r1LaQXUsOZPL"
    "P2X+vTmwKH/db838Qf3mWsO9+eO9pqX3Du+1zpUb9P8b/oT8VVblHnuQe+s64PKjxInEWmjPGbnp34/iEC2g3VBZO6OwXDtdtrRj4UrqbUrc7n53nvZ9"
    "SCzNlPs6/Rq+HT7PGZcSBz+UvXOi66zL/69f/efKK0u/Kc4oh71j8fpbfNA11WV0f3nXDeSD6ljcdLn0+oDmtgQl+rLMf4k1MJe/Xpd0uQQ1+yKkBPvR"
    "FzXZkP6WJW4ce9+8WJRzf4Z9kb9nwTb/iGkYoqbzaRzCO/+lnWffFcybZ+2P8nnVGbnuCzTnuk3tb3HTc64jZYwWf9QOyh+vJ6qP2rTc+3rOUZuW/Nd9"
    "PYUhvD6oTUTvfiexF2z5b1l/UvIhZbwp/pacKb0XoGuB/L3rZM111ZJXOh8sPt76JcYyRv9c+bCofTnXc1oLz3WkTWIcYiwcxyG0/0QF/evkjN4jxphL"
    "ljMm/elPf/rXfMZumdP4e9C/jr/3vE8IIaTN8yA6G7a2D8b4e6B3oSll3WauvXsS50fL808IIYT8t/Mj3z+0wxjfX80++/H9J/3pT/9ajP37Kb+/t5lL"
    "hBBCxnUGr3kG1PsF+u6Qayyl3/9bvkG0PP/ouwMhhJCcfAB94Q66"
)

_MOB_ALPHA_16_SHAPE = (8, 16, 16, 3)
_MOB_ALPHA_16_B64 = (
    "eNrt21kOgzAMRVF21q3Rnfe7UlNhYZPpXIk/BDdhenba4wAAAACAKOcrtvHnHx3LjPM8wnWpdq4YSy/nrLFcOU6F25r+7/P4uY3g/+3Dfw3/nGfkiltr"
    "/zzPmH/b4cm5Xd+/dd6WQ3R/QD3yVObcoTapnp+R/SuydC/n+a5L+zvev5aJZozqnHlnq8lIsWNWzFW1/5OZVraA7JfhNot/9Lyz9JxHXguo7nmumjPv"
    "jKui7/1kLz0re9fWVu0sMZr/nSzUxz+et7Pen/lzW53JK3rp9T3SrHWr6Hj10nPrJnUW7tQIM9Zc/Pmvev/z58+fv/c//5WeBQDYuQafZd1qt98sySEA"
    "gJXyxoz1o/7D+PcSf/78+fP3/eLv/gcA9P0G7fZfjB3mHwDwjw9M5Wyl"
)

_STRIP_ICON_7_SHAPE = (16, 5, 5, 3)
_STRIP_ICON_7_B64 = (
    "eNrlljtoVUEQhlcMoiImCIY0iRhjLLTRBHzjPbsKFoIRrTSIiKJFCCmCiFFMGSzESqwkRQghiIWFBEsri5QWIiLBIqUECwkW4j/sXPb36t7s5tzAJfnh"
    "Y+fsmZmdfZyHMaxHzphBhfUA1xeUWn+riN2ljIEO5Z7EKJfAc+uRnEvWI/4Prec+7H7ruS5x1rMf9njhYQ2j/0PhuQm7rfAcg71Q8aRI/E1FyVQ3Yr+h"
    "xpfgEOwfaF+Dw7B/o31n47E/cW8WDIEW+L/VdeiBvYh2sib2Fq53gGeY4wzaK2AB9hu0d/Ue67ELXCZG4deheWI64gKcp4wOIL4P3F5Fni16pq4lxh6E"
    "31FwB0xgrp+tn/sk5jwN+6Srvy9tuH9Vz6Dsy3mJxfl4Cru3TuwT3J8Hp/U8yvVeOb9njBmw/kyshWT/Zc6Dq8g/gDp/Ie5G4e3tyDOCtgK+Wt+Xojn4"
    "bUXsTGGaWi9Q3x737/n/hOtTzqwLteo7ZNz+bUs7B87WsTda7BdimXhPbHaB6rdrSceq0kLM2wD7bCJ2E/KcVflOsM9HG+B6OCfXxnPZ5gI7iU5iF8E+"
    "XNs0wfm5v9OtTMUGWOwTG+uiC7AP54zVExuX88dqiM1l2f4fjj2uezVV0y/fsxMrrFU7eKXf+JScsXrYP2Xv2muo1sB9sbVi/9hacZ5GnSvRmP6rcGys"
    "P2XvYrGxcTmPSP6Rz7nGnauY9jn//Z21jft2pIjfpSn15ObP9U8Rv1fLiN+3ZdYqVk8zrNWQC5SRc4EyaxWrpxnWqlFjpfjH/nnK/B82w1rlPlMp/uyT"
    "m5/nnjvWWiv3mUrxZ5/c/LxWuWPl6Q/QCcgq"
)

_STRIP_DIGIT_7_SHAPE = (10, 4, 4, 3)
_STRIP_DIGIT_7_B64 = (
    "eNpjYBhOoN4ZOyZVL7XU43IPLr2kqifV79Qyc6DcTy3/0iKd0NoN9IwvSvLdYEhvtI47armfFn6hRZlGrfxOT/Np4R7ql7cAHtqxVQ=="
)

_STRIP_DIGIT_ALPHA_7_SHAPE = (10, 4, 4, 3)
_STRIP_DIGIT_ALPHA_7_B64 = (
    "eNpjYBhOoMEeOyZVL7XU43IPLr2kqifV79Qyc6DcTy3/0iKd0NoN9IwvSvLdYEhvtI47armfFn6hRZlGrfxOT/Np4R7ql7cA+ZCulw=="
)

_STRIP_ICON_16_SHAPE = (16, 12, 12, 3)
_STRIP_ICON_16_B64 = (
    "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"
)

_STRIP_DIGIT_16_SHAPE = (10, 9, 9, 3)
_STRIP_DIGIT_16_B64 = (
    "eNrtmd0NACEIg93tBrvVb4SLMcjXUhJfrfJbYK1IJDJH3mfvVN+vgnXyhmDdxzrxjd03KGLRdEjIXWT/JMcyze4TYrlLh12cpBpLJS+pcx6nWkmIkS6s"
    "m3qj+du0k97Bm28Q5gwq9nKNBZW+UmUmRtMnof+q+EtXPqzmMOqxlrrsXVMyx85fVLiia02ZsHfr8o1wAO+9QPKG7j6R4JNkvf3LByWZZKI="
)

_STRIP_DIGIT_ALPHA_16_SHAPE = (10, 9, 9, 3)
_STRIP_DIGIT_ALPHA_16_B64 = (
    "eNrtmd0NACEIg93sVrvRb4SLMcjXUhJfrfJbYK1IJDJH3mfvVN+vgnXyhmDdxzrxjd03KGLRdEjIXWT/JMcyze4TYrlLh12cpBpLJS+pcx6nWkmIkS6s"
    "m3qj+du0k97Bm28Q5gwq9nKNBZW+UmUmRtMnof+q+EtXPqzmMOqxlrrsXVMyx85fVLiia02ZsHfr8o1wAO+9QPKG7j6R4JNkvf3LB/IrVzQ="
)

#: mob ids, matching the reference's draw order; 4-7 are the arrow's (flip, transpose) orientations
MOB_NONE, MOB_ZOMBIE, MOB_COW, MOB_SKELETON = 0, 1, 2, 3
MOB_ARROW = (4, 5, 6, 7)
#: StaticEnvParams caps for Craftax Classic — the sampler must respect them or no EnvState can be
#: built for the reference comparison
MAX_ZOMBIES, MAX_COWS, MAX_SKELETONS, MAX_ARROWS = 3, 3, 2, 3
#: the block the renderer pads a map view with at the world edge; also the only block besides INVALID
#: whose texture is a flat colour, which the sampler depends on (see ``sample_states``)
BLOCK_OUT_OF_BOUNDS = 1
#: terrain the sampler draws from, with the weights a Craftax map roughly has
_TERRAIN = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
_TERRAIN_P = (0.34, 0.10, 0.20, 0.14, 0.03, 0.07, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02, 0.01, 0.02)


#: the two tile sizes craftax itself declares and ships textures for. Both are exact; anything else is
#: nearest-resampled from the nearer of the two and is not compared against a reference, because there
#: are no textures shipped at that size, hence no published frame to certify against.
REFERENCE_TILE_PX = (TILE_PX_AGENT, TILE_PX_IMG)
_TABLES = {
    TILE_PX_AGENT: (
        (_BLOCK_TEX_7_B64, _BLOCK_TEX_7_SHAPE),
        (_PLAYER_TEX_7_B64, _PLAYER_TEX_7_SHAPE),
        (_PLAYER_ALPHA_7_B64, _PLAYER_ALPHA_7_SHAPE),
        (_MOB_TEX_7_B64, _MOB_TEX_7_SHAPE),
        (_MOB_ALPHA_7_B64, _MOB_ALPHA_7_SHAPE),
    ),
    TILE_PX_IMG: (
        (_BLOCK_TEX_16_B64, _BLOCK_TEX_16_SHAPE),
        (_PLAYER_TEX_16_B64, _PLAYER_TEX_16_SHAPE),
        (_PLAYER_ALPHA_16_B64, _PLAYER_ALPHA_16_SHAPE),
        (_MOB_TEX_16_B64, _MOB_TEX_16_SHAPE),
        (_MOB_ALPHA_16_B64, _MOB_ALPHA_16_SHAPE),
    ),
}

#: Strip ingredients, kept SEPARATE from ``_TABLES`` because they are not tile-sized: an icon is
#: ``int(0.8*t)`` on a side and a digit ``int(0.6*t)``, and neither ratio scales by an exact factor
#: of two (12 and 9 at 16 px; 25 and 19, not 24 and 18, at 32 px). Resampling these ingredients the
#: way ``_Assets`` resamples the tile-sized tables would therefore give a strip at 2S that is not a
#: 2x replication of the strip at S, and ``check_resolution_consistency`` would report a renderer
#: change where there is only a resampling rule. :class:`_StripComposite` instead builds the finished
#: CELL at a reference size and resamples the cell, which is exact 2x replication.
_STRIP_TABLES = {
    TILE_PX_AGENT: (
        (_STRIP_ICON_7_B64, _STRIP_ICON_7_SHAPE),
        (_STRIP_DIGIT_7_B64, _STRIP_DIGIT_7_SHAPE),
        (_STRIP_DIGIT_ALPHA_7_B64, _STRIP_DIGIT_ALPHA_7_SHAPE),
    ),
    TILE_PX_IMG: (
        (_STRIP_ICON_16_B64, _STRIP_ICON_16_SHAPE),
        (_STRIP_DIGIT_16_B64, _STRIP_DIGIT_16_SHAPE),
        (_STRIP_DIGIT_ALPHA_16_B64, _STRIP_DIGIT_ALPHA_16_SHAPE),
    ),
}


def strip_geometry(t: int) -> tuple[int, int, int, int]:
    """``(icon_lo, icon_hi, digit_lo, digit_hi)`` within one ``t x t`` strip cell.

    The reference's own expressions: the icon occupies
    ``[(t - int(0.8t))//2 - 1 : t - (t - int(0.8t) - that)]`` and the digit
    ``[t - int(0.6t) - 1 : t - 1]``, both in each axis. Every write stays inside the cell and no two
    cells overlap, which is what makes the strip a per-cell table rather than a sequence of blits —
    asserted at generation time and again in ``tests/unit/test_craftax_strip.py``.
    """
    lo = (t - int(0.8 * t)) // 2 - 1
    hi = t - (t - int(0.8 * t) - lo)
    n = int(0.6 * t)
    return lo, hi, t - n - 1, t - 1


def _unpack(b64: str, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(b64)), dtype=np.float32).reshape(shape)


class _Assets:
    """Texture tables, decoded once per (device, tile size)."""

    _cache: dict = {}

    @classmethod
    def get(cls, device: torch.device, t: int):
        key = (str(device), int(t))
        if key not in cls._cache:
            src = t if t in _TABLES else min(REFERENCE_TILE_PX, key=lambda s: abs(s - t))
            tabs = [torch.as_tensor(_unpack(b64, shape).copy(), device=device) for b64, shape in _TABLES[src]]
            if t != src:
                ix = ((torch.arange(t, device=device, dtype=torch.float32) + 0.5) * src / t).long().clamp_(0, src - 1)
                tabs = [a[:, ix][:, :, ix] for a in tabs]
            cls._cache[key] = tuple(tabs)
        return cls._cache[key]


#: Player slot meaning "no player is drawn on this tile". ``player_visible = 0`` and every non-centre
#: tile both resolve to it, and it is exactly the identity: the reference computes
#: ``tile*(1 - 0) + player_tex*0``, which is ``tile + 0`` in float32, and so does this slot. It is
#: slot 5 since the sleeping sprite took slot 4.
NO_PLAYER = N_PLAYER


class _StripAssets:
    """Inventory-strip ingredients at a reference size. Never resampled; see :data:`_STRIP_TABLES`."""

    _cache: dict = {}

    @classmethod
    def get(cls, device: torch.device, src: int):
        key = (str(device), int(src))
        if key not in cls._cache:
            cls._cache[key] = tuple(
                torch.as_tensor(_unpack(b64, shape).copy(), device=device) for b64, shape in _STRIP_TABLES[src]
            )
        return cls._cache[key]


_VIGNETTE_CACHE: dict = {}
#: ``(cell_of, blank)`` and the two post-process colour constants, per device. All four are
#: state-independent, and building them per call is not free: ``_strip_grid`` alone costs 39.7 us a
#: render on this CPU (32 indexed assignments into two fresh tensors) and about 34 tiny kernel
#: launches plus two allocations a frame on CUDA, which at the daylight batch sizes this env is
#: certified at is most of the per-call overhead rather than a rounding error. Independent review
#: measured the whole of it: B=1 went 0.061 -> 0.166 ms and B=256 at 16 px 1.460 -> 2.092 ms before
#: these caches existed.
_GRID_CACHE: dict = {}
_CONST_CACHE: dict = {}


def _post_constants(device: torch.device):
    """``(night_tint, sleep_lift)`` on ``device``, built once."""
    key = str(device)
    if key not in _CONST_CACHE:
        _CONST_CACHE[key] = (torch.tensor(NIGHT_TINT, device=device), torch.tensor([0.0, 0.0, 8.0], device=device))
    return _CONST_CACHE[key]


def night_vignette(device: torch.device, h: int, w: int) -> torch.Tensor:
    """``(h, w, 1)`` float32 — craftax's ``night_noise_intensity_texture``, rebuilt not approximated.

    ``1 - exp(-0.5 (x^2 + y^2) / 0.5^2)`` on a ``linspace(-1, 1)`` grid, **transposed**: zero at the
    frame centre and rising to the corners, so night eats the edges of the view first. craftax
    computes it in numpy float64 and lets jax narrow the result to float32, and so does this — the
    same two steps in the same order, rather than a float32 ``exp`` that would round differently.

    Parameterised on ``(h, w)`` because full Craftax draws the same texture at its own window size.
    """
    key = (str(device), int(h), int(w))
    if key not in _VIGNETTE_CACHE:
        xs, ys = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
        v = (1 - np.exp(-0.5 * (xs**2 + ys**2) / (0.5**2)).T).astype(np.float32)
        _VIGNETTE_CACHE[key] = torch.as_tensor(v, device=device).unsqueeze(-1)
    return _VIGNETTE_CACHE[key]


def night_and_sleep(
    world: torch.Tensor,
    light: torch.Tensor,
    sleeping: torch.Tensor,
    keys: torch.Tensor,
    light_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Craftax's day/night chain and sleeping desaturation, in the reference's own order.

    ``world`` ``(n, h, w, 3)`` float32 pre-post-process pixels, ``light`` ``(n,)``, ``sleeping``
    ``(n,)`` bool, ``keys`` ``(n, 2)`` int32 threefry keys. ``light_map`` is the per-pixel multiplier
    full Craftax applies and Classic does not; it is an argument rather than a second function
    because that is the only difference between the two renderers' chains.

    Shared with the full-Craftax renderer: nothing here reads ``CraftaxClassic``'s state layout, and
    the shapes come from ``world``.

    **The arithmetic is the reference's, op for op, and is deliberately not algebraically collapsed.**
    The chain looks like it should fold into ``a1*world + a2*lum + c`` with two scalars and a per-pixel
    constant, which would be three passes instead of ten. It does — in real arithmetic. In float32 it
    is a different rounding at every step, and this renderer's claim is bit equality with the
    reference, not closeness to it. The reductions taken instead are the ones that change no bits:

    * **the noise field is drawn only for rows that can use it.** ``intensity`` is exactly 0 for
      ``light >= 0.5``, so ``(1 - 0)*world + 0*noise`` is ``world`` in float32 whatever the noise was;
      those rows get zeros and no threefry evaluation. 58.3% of a 300-step day is in that band.
    * **int32 threefry** (``craftax/prng.py``): bit-identical to an int64-with-masking implementation
      and 2.3x faster.
    * **a batch that is entirely at ``light == 1.0`` and awake never gets here at all** — ``render``
      keeps it on the uint8 gather, because ``1.0*world + 0.0*night`` is exactly ``world``.
    """
    n, h, w, _ = world.shape
    dev = world.device
    lt = light.view(n, 1, 1, 1)

    # --- night static, drawn only where it can contribute
    dark = light < NIGHT_THRESHOLD
    night = world.clone()
    if bool(dark.any()):
        idx = dark.nonzero(as_tuple=True)[0]
        sub = world.index_select(0, idx)
        noise = (prng.uniform(keys.index_select(0, idx), (h, w)) * 95 + 32).unsqueeze(-1)
        mask = torch.clamp_min(2.0 * (NIGHT_THRESHOLD - lt.index_select(0, idx)), 0.0) * night_vignette(dev, h, w)
        night.index_copy_(0, idx, (1 - mask) * sub + mask * noise)

    # --- enhance: pull towards luminance. In place from here on: the expressions are the
    # reference's, op for op, but each result overwrites its own buffer instead of allocating a new
    # image-sized float32 temporary. At B=256 and 16 px one such temporary is 63.7 MB, which is what
    # made this chain 111x the cost of the gather in the first measurement.
    lum = 0.299 * night[..., 0] + 0.587 * night[..., 1] + 0.114 * night[..., 2]
    night.mul_(0.4).add_(lum.mul_(1 - 0.4).unsqueeze(-1))
    # --- tint
    tint, lift = _post_constants(dev)
    night.mul_(0.5).add_(tint * 0.5)
    # --- blend
    if light_map is not None:
        night.mul_(light_map)
    px = night.mul_(1 - lt).add_(world * lt)

    # --- sleep: a luminance at half brightness, lifted towards blue
    if not bool(sleeping.any()):
        return px
    lum = 0.299 * px[..., 0] + 0.587 * px[..., 1] + 0.114 * px[..., 2]
    sleep = lum.mul_(0.5).unsqueeze(-1) + lift
    return torch.where(sleeping.view(n, 1, 1, 1), sleep, px)


def _strip_cells(device: torch.device, t: int) -> torch.Tensor:
    """``(N_STRIP * N_COUNT, t, t, 3)`` float32 — every ``(cell, count)`` picture the strip can show.

    The strip is a function of sixteen small integers and of nothing else — twelve inventory counts
    and four intrinsics, each capped at 9 by the simulator — and the reference draws each cell with
    two writes that stay inside that cell's own tile, so no two cells interact. The whole strip is
    therefore a 16 x 10 table of finished tiles: 160 entries, 23,520 bytes as uint8 at 7 px and
    122,880 at 16 px, and drawing it is a gather.

    **The strip is not lit.** ``render_craftax_pixels`` concatenates it *after* the day/night chain
    and after the sleeping desaturation, so it is the same picture at midnight and at noon. Measured
    over 64 real states at both tile sizes: bit-identical under every ``light_level`` and
    ``is_sleeping``, and 0 channels move when ``state_rng`` changes
    (``pre-repository-cleanup:tests/fixtures/regression/join/step7/craftax-classic-full/phase0/components.json``). The Crafter port found
    the same thing by failing first (``pre-repository-cleanup:docs/research/crafter.md``); here it was measured first.

    Built at a reference tile size and then resampled *as a finished cell*, never as ingredients: an
    icon edge is ``int(0.8*t)`` and a digit edge ``int(0.6*t)``, and neither scales by an exact
    factor of two (12 and 9 at 16 px; 25 and 19 — not 24 and 18 — at 32 px). Re-laying the cell out
    at 2S would give a strip that is not a 2x replication of the strip at S, and
    ``check_resolution_consistency`` would report a renderer change where there is only a resampling
    rule. Resampling the finished cell is exact 2x replication.
    """
    src = t if t in _STRIP_TABLES else min(REFERENCE_TILE_PX, key=lambda z: abs(z - t))
    icons, digits, dalpha = _StripAssets.get(device, src)
    lo, hi, dlo, dhi = strip_geometry(src)
    cell = torch.zeros(N_STRIP, N_COUNT, src, src, 3, dtype=torch.float32, device=device)
    # count 0 leaves the icon out: the reference selects ``smaller_empty_texture`` on ``> 0``, and
    # digit 0 is the empty texture with zero alpha, so a zero cell is entirely black. That is also
    # what the two cells the reference never writes look like, which is why a blank needs no entry
    # of its own — it is cell 0 holding count 0.
    cell[:, 1:, lo:hi, lo:hi, :] = icons.view(N_STRIP, 1, hi - lo, hi - lo, 3)
    patch = cell[:, :, dlo:dhi, dlo:dhi, :].clone()
    cell[:, :, dlo:dhi, dlo:dhi, :] = patch * (1 - dalpha.view(1, N_COUNT, dhi - dlo, dhi - dlo, 3)) + digits.view(
        1, N_COUNT, dhi - dlo, dhi - dlo, 3
    )
    if t != src:
        ix = ((torch.arange(t, device=device, dtype=torch.float32) + 0.5) * src / t).long().clamp_(0, src - 1)
        cell = cell[:, :, ix][:, :, :, ix]
    return cell.reshape(N_STRIP * N_COUNT, t, t, 3)


#: (player slot, block, mob) composites in the atlas: 6 * 17 * 8 = 816. Six player slots because the
#: sleeping sprite is one of them and :data:`NO_PLAYER` is the sixth.
N_COMPOSITE = (N_PLAYER + 1) * N_BLOCK * N_MOB
#: The strip cells live in the same table, after the world tiles, so one gather draws the whole
#: observation. Tile id ``STRIP_BASE + cell * N_COUNT + count`` is strip cell ``cell`` showing
#: ``count``; ``STRIP_BASE + 0`` is black, which is what the two unwritten cells show.
STRIP_BASE = N_COMPOSITE
N_TILE = N_COMPOSITE + N_STRIP * N_COUNT

#: Largest tile edge the baked atlas is used at. The atlas holds ``N_TILE`` tiles, so it costs
#: ``976 * t*t*3`` bytes as uint8 and four times that as float32 — 750 KiB / 2.9 MiB at the default
#: 16 px, but 192 MiB / 767 MiB at ``size=256``. Above this edge the atlas stops being a small table
#: that stays resident and becomes the largest allocation in the process, so ``render`` falls back to
#: :meth:`CraftaxClassic._render_blend`, which is the pre-bake implementation and is asserted
#: pixel-identical to the atlas path by ``tests/unit/test_craftax_atlas.py``.
ATLAS_MAX_TILE_PX = 64


class _Composite:
    """The **composited** tile atlas: every (player slot, block, mob) tile the renderer can draw.

    The frame is a gather and nothing else. The reference composited
    per frame: it gathered the block texture into a ``(B, 7t, 9t, 3)`` float32 buffer, blended the
    player into the centre tile, then blended a full-frame mob texture and a full-frame mob alpha
    over all of it, applied the channel gain and rounded. Every one of those steps is a
    ``B x 7t x 9t x 3`` float32 temporary, and there are about a dozen of them: **1.94 MB per 16 px
    frame against a 48.4 KB frame**. That is why throughput FELL with batch — 1,876 f/s at B=16 down
    to 1,113 f/s at B=256 on one core of ``node1831`` — rather than rising: 16 frames of temporaries
    are 31 MB and still fit that node's ~35.8 MiB of last-level cache, and 256 frames are 495 MB and
    do not. Measured in ``pre-repository-cleanup:tests/fixtures/regression/join/step6/tilemap/profile_before/``.

    None of that work depends on the state. A tile's pixels are a function of exactly three small
    integers, so the whole composite is a 680-entry table that is built once per (device, tile size)
    and gathered from thereafter. The arithmetic per tile is the reference's arithmetic in the
    reference's order, elementwise in float32, so the result is bit-identical rather than close.

    Two forms are kept:

    * ``u8`` — ``round -> clamp -> uint8`` already applied. Usable only when ``channel_gain`` is
      exactly 1 on every row of the batch, where ``px * 1.0`` is the identity in float32 and the
      rounding therefore commutes with the gather. This is the default appearance and the certified
      one, and it lets ``render`` gather straight into the uint8 output with **no float intermediate
      at all**.
    * ``f32`` — the pre-rounding composite, built lazily and only when a non-default gain is actually
      rendered, because the gain multiplies before the rounding and cannot be folded into a uint8
      table.
    """

    _MAX_ENTRIES = 8
    _cache = _RES.BoundedCache(_MAX_ENTRIES, name="craftax.composite")

    @staticmethod
    def _build(device: torch.device, t: int) -> torch.Tensor:
        """``(N_COMPOSITE * t, t * 3)`` float32, row ``slot * t + y``.

        Laid out one *tile row* per row so the gather in :meth:`CraftaxClassic.render` can address a
        whole output scanline of one tile with a single index and write it where it finally belongs.
        """
        block, ptex, palpha, mtex, malpha = _Assets.get(device, t)
        # player over block -> (6, 17, t, t, 3); slot NO_PLAYER is the block untouched
        with_player = block.unsqueeze(0).repeat(N_PLAYER + 1, 1, 1, 1, 1)
        ap = palpha.view(N_PLAYER, 1, t, t, 3)
        with_player[:N_PLAYER] = with_player[:N_PLAYER] * (1 - ap) + ptex.view(N_PLAYER, 1, t, t, 3) * ap
        # mob over that -> (6, 17, 8, t, t, 3)
        am = malpha.view(1, 1, N_MOB, t, t, 3)
        comp = with_player.unsqueeze(2) * (1 - am) + mtex.view(1, 1, N_MOB, t, t, 3) * am
        # ... and the 160 strip cells after them, in the SAME table, so one gather draws the whole
        # observation. A strip cell is a tile like any other; it just happens to be indexed by
        # (cell, count) rather than by (player, block, mob).
        return torch.cat(
            [comp.reshape(N_COMPOSITE * t, t * 3), _strip_cells(device, t).reshape(N_STRIP * N_COUNT * t, t * 3)]
        ).contiguous()

    @classmethod
    def get(cls, device: torch.device, t: int, want_float: bool):
        """``(f32_or_None, u8)``. ``f32`` is materialised only once a gain path has asked for it.

        ``device`` is always a tensor's own ``.device`` here, which carries an explicit index, so the
        string key cannot conflate ``"cuda"`` on one GPU with ``"cuda"`` on another — the failure
        ``render/tilemap.py:MiniGridAtlas.get`` documents and resolves the long way round.
        """
        key = (str(device), int(t))
        have = cls._cache.get(key)
        if have is not None and (have[0] is not None or not want_float):
            return have
        f32 = cls._build(device, t)
        u8 = have[1] if have is not None else f32.round().clamp_(0, 255).to(torch.uint8)
        return cls._cache.put(key, (f32 if want_float else None, u8))


@register
class CraftaxClassic(Env):
    """Craftax Classic: a 7x9 block window, the player, one mob per tile, and the inventory strip.

    Rendering and Variation handling are provided by stable-envs. Dynamics stay in Craftax/JAX;
    this cell currently renders stored windows (``has_dynamics`` is False).
    """

    family = "craftax"
    domain = "classic"

    name = "craftax-classic"
    #: 63 blocks, 63 mobs, the player's facing, then the sixteen strip counts and the sleeping flag.
    #: **Appended, never reordered**: the first 127 slots keep the meaning and position they had when
    #: the frozen corpus was recorded, which is what makes the world rows of that corpus replayable.
    state_dim = OBS_H * OBS_W * 2 + 1 + N_STRIP + 1 + 1 + N_KEY_HALVES
    state_names = (
        tuple(f"block_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"mob_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + ("player_facing",)
        + tuple(f"inv_{n}" for n in STRIP_NAMES)
        + ("is_sleeping", "light_level")
        + tuple(f"rng_half{i}" for i in range(N_KEY_HALVES))
    )
    #: tile edge in pixels; the frame is ((7+2)*size, 9*size, 3). Craftax ships textures at 7 px (the
    #: size the agent observation is drawn at) and 16 px, and both are exact against the official
    #: renderer. The default is 16 because at 7 px Craftax's dithered art leaves no flat 3x3
    #: neighbourhood in the world layer, the conformance suite's interior mask is then nearly empty,
    #: and its interior statistic comes back NaN — which no floor can satisfy, however exact the
    #: renderer is.
    default_size = TILE_PX_IMG
    reference_id = "Craftax-Classic-Pixels-v1"
    #: REFERENCE-side versions the parity numbers were measured against. ``craftax`` pins the atlas
    #: and the strip layout. ``jax`` is here for a reason that is not obvious and is load-bearing:
    #: see :meth:`reference_env`, which refuses to build a reference whose PRNG path differs.
    reference_versions = {"craftax": "1.6.1", "jax": "0.11.1"}
    #: Certified against un-jitted Craftax. A jitted wrapper of the same renderer can differ by
    #: at most 1 uint8 because ``jax.jit`` fuses the float32 arithmetic differently (up to 1.5e-05),
    #: which a sleeping or lit row's desaturation can turn into a rounding flip. The parity claim
    #: certified above (100.0000% exact, max |diff| 0, see :meth:`reference_env`) is measured against
    #: the renderer as WRITTEN -- un-jitted. Historical jitted-wrapper measurements live in
    #: ``tests.helpers.reference.craftax_jit``.
    #: Where ``check_resolution_consistency`` applies to this env, and where it cannot -- **decided by
    #: an experiment on the OFFICIAL renderer, not by argument from this one**. The check renders at
    #: S and 2S, box-downsamples, and asks how much agrees. Running its exact protocol on craftax's
    #: own renderer (un-jitted, each state's own ``state_rng``, craftax's own
    #: ``load_all_textures_given_size(32)``) over the frozen corpus splits into two
    #: independent mechanisms, each confined to one region of the frame
    #: (``pre-repository-cleanup:tests/fixtures/regression/join/step7/craftax-classic-full/oracle/resolution_oracle_S16.json``):
    #:
    #: * **world rows.** Official 1.0000 at daylight and at dusk; **0.1634 at night**, because the
    #:   noise field is drawn at ``(7t, 9t)`` -- the frame's own resolution -- so element ``i`` of the
    #:   32 px field is a different threefry counter from element ``i`` of the 16 px one. This
    #:   renderer measures **0.1634 too**: not merely also-inconsistent, but inconsistent by exactly
    #:   the reference's amount, which is what reproducing its noise bit for bit implies. The quantity
    #:   the check is defined to measure is undefined when the two renders draw different noise, so
    #:   below :data:`NIGHT_THRESHOLD` the check is **not applicable**, on the reference's evidence.
    #: * **strip rows.** Official **0.6815-0.6946 in every band, daylight included**, because its
    #:   icon is ``int(0.8*t)`` and its digit ``int(0.6*t)`` and neither scales by two (12 and 9 at
    #:   16 px; 25 and 19 at 32 px). This renderer measures **1.0000**, because it resamples the
    #:   finished cell rather than re-laying it out -- see :func:`_strip_cells`. That makes it
    #:   *stricter* than the reference at 2S, at a size craftax **ships no textures for**
    #:   (:meth:`reference_env` refuses any such size, so 32 px is not a declared parity size), and
    #:   it is what keeps the frozen daylight bound meetable. Matching craftax's 32 px re-layout
    #:   instead would break that bound and buy no parity.
    #:
    #: "No shipped textures at 2S" is the accurate reason, and not "no official frame": craftax's own
    #: ``load_all_textures_given_size(32)`` builds a perfectly good 32 px renderer, and against it
    #: this renderer's **world rows are 100% exact in every band, night included** (max abs diff 0 --
    #: ``oracle/parity_at_2S.json``). The whole 2S divergence is the strip, at about 18% of its
    #: pixels, by design. So the 32 px frames do not disagree about the world; there is simply no
    #: shipped art to certify against.
    #:
    #: Parity at the two supported sizes is unaffected and still exact: the same experiment records
    #: ``exact_frac`` 1.0 against the official at S in every band.
    #: The state distribution the **one** registered/default performance tier is measured on, which
    #: is not a choice made for the benchmark: the frozen throughput harness draws
    #: ``probe.sample_states(64)``, so this env's own sampler IS the certification distribution and
    #: its shares are a declared property of the cell. Measured over seeds 0-5 at n=64: **daylight
    #: exactly 0.25** (the band the index stratification pins to ``light_level == 1.0``), **dusk
    #: 0.33-0.44**, **night 0.31-0.43**, **sleeping 0.31-0.46**, over seeds 0-39. Daylight is exact because the band
    #: is assigned by row index; the rest vary with the draw inside their band because the values
    #: come from craftax's own day cycle.
    #:
    #: This matters for reading the tier. A daylight frame is one gather from a uint8 table; a lit
    #: one carries the reference's per-pixel float post-process, and the two differ by about 100x on
    #: one CPU core. About three quarters of this distribution is lit, so the tier is dominated by
    #: the slow band **by construction** -- which is correct, because three quarters of Craftax's day
    #: is lit. A tier quoted on daylight alone would be a number about a state the agent is in a
    #: quarter of the time.
    certification_state_distribution = {
        "draw": "CraftaxClassic.sample_states(64) -- the throughput harness protocol",
        "daylight_share": 0.25,
        "dusk_share": "0.39-0.44",
        "night_share": "0.31-0.36",
        "sleeping_share": "0.33-0.39",
        "measured_over": "seeds 0-5, n=64",
        "band_assignment": "stratified by row index (i % 4); values from craftax's own day cycle",
    }
    #: The measured performance record for the registered default, kept beside the declaration it
    #: belongs to rather than only in a findings file. **None of these numbers certifies anything
    #: until Sami rules on which reference a speed ratio may use** -- see ``s_cpu_note``.
    #:
    #: Oscar ``node2356``, frozen throughput harness ``--device cpu --threads 1``, paired against a
    #: pristine 76dc511 measured either side of it in the same job (spread 1.0045, baseline at 1.44x
    #: its previously certified value, node load 0.86 per core). Artifacts under
    #: ``pre-repository-cleanup:tests/fixtures/regression/join/step7/craftax-classic-full/sweep/``.
    performance_record_cpu = {
        "mixed_default_fps": 1150.6,
        "mixed_default_batch": 64,
        "decode_bar_fps": 4367.4,
        "cores_cpu": 0.263,
        "tier": "CPU_FASTER",
        "qualifier": "BELOW_G4",
        "S_cpu": 2.80,
        "official_jitted_fps": 213.44,
        "daylight_diagnostic_cores_cpu": 24.2,
        "daylight_diagnostic_fps": 105742,
        "night_diagnostic_fps": 656,
        "daylight_over_night": 104.9,
        "why_lower_than_prior": (
            "the prior tier was measured on an observation that could not "
            "represent night; this one is the full observation, about three "
            "quarters of which is lit. The daylight path itself is unchanged."
        ),
        "job": "6395527 (throughput), 6409933 (reference)",
        "host": "node2356 / node2335",
        "reference_conditions": "cgroup affinity [8] -- one core",
    }
    #: **The speed ratio and the parity claim cannot use the same reference, and this env is where
    #: that first bites.** :meth:`reference_env` returns the renderer UN-JITTED, because the jitted
    #: one disagrees with itself in uint8 and cannot be what a parity claim is made against. But
    #: ``bench_step3`` then measures ``official_cpu`` at **2.18 f/s** against 610.6 f/s for the
    #: jitted one, so ``S_cpu`` -- the ratio the throughput ladder prefers wherever a reference exists --
    #: would read **528x** where the honest figure against the renderer a user actually runs is about
    #: **1.9x**. A tier taken on 528x would certify ``CPU_EXCEPTIONAL`` on a speedup over a renderer
    #: nobody runs, on an env whose ``NO_REFERENCE`` ladder says ``CPU_REGRESSION``.
    #:
    #: The general rule this asks for -- parity against the reference **as written**, speed against
    #: the reference **as a user runs it** -- is a certification ruling. Nothing in
    #: the shared harness is edited here to suppress ``S_cpu``; both numbers are recorded so the
    #: ruling can be made on them.
    s_cpu_note = {
        "official_fps_unjitted": 2.18,
        "official_fps_jitted": 610.6,
        "s_cpu_vs_unjitted": 528.0,
        "s_cpu_vs_jitted": 1.9,
        "status": "RULED. The reference is the official renderer as craftax runs it -- jitted, warm, "
        "compile excluded -- timed on the certification host under the required "
        "conditions (one core for CPU, the same card for GPU, the same job). Parity is "
        "unchanged and remains against the UN-JITTED function as a uint8 claim. Labels: "
        "CPU_FASTER (BELOW_G4) and GPU_FASTER.",
        "ruled_labels": {"cpu": "CPU_FASTER", "gpu": "GPU_FASTER"},
        "rejected_reference_timings": {
            "610.6 f/s node2356 CPU": "jax ran on all allocated cpus while ours ran on one; XLA's "
            "CPU backend ignores OMP_NUM_THREADS",
            "386.7 f/s gpu2107 GPU": "env_header.sh exports JAX_PLATFORMS=cpu, so the reference "
            "never touched the card it was compared on",
            "258.9 f/s gpu2104 GPU": "JAX_PLATFORMS unset, but the venv carries jax[cpu]: no CUDA "
            "jaxlib, so jax reported backend 'cpu' anyway",
        },
        # The throughput evaluator, run on the real records three ways. The GPU pair is the proof that
        # this is not a quibble: measured on ONE 3090 in ONE job, the previous tree does 1,174,913 f/s
        # against a jitted reference and labels GPU_FAST, while this branch does 46,187 f/s -- 25x
        # slower -- against an un-jitted one and labels GPU_EXCEPTIONAL.
        "labels_cpu": {"as_measured": "CPU_FASTER", "vs_jitted": "CPU_PARITY", "no_reference": "CPU_REGRESSION"},
        "labels_gpu": {"as_measured": "GPU_EXCEPTIONAL", "vs_jitted": "GPU_FASTER", "no_reference": "GPU_FAST"},
        "must_not_be_recorded_as_the_tier": ["GPU_EXCEPTIONAL (ratio vs the un-jitted reference)"],
        "label_is_invariant_to_the_pairing": (
            "both a sweep-based and a same-process paired ratio "
            "were computed (CPU 5.39 vs 2.80, GPU 304.5 vs 316.0) "
            "and give the same label on both devices"
        ),
        "evidence": "pre-repository-cleanup:tests/fixtures/regression/join/step7/craftax-classic-full/sweep/tier_three_ways_*.json",
    }
    #: The GPU side of :attr:`performance_record_cpu`. Oscar ``gpu2107``, one RTX 3090, paired
    #: against a pristine 76dc511 either side of it in the same allocation: baselines 1,174,913 and
    #: 1,149,220 f/s, spread 1.0224, **0.963x** the 1,206,822 previously certified for that tree.
    performance_record_gpu = {
        "mixed_default_fps": 46187.3,
        "mixed_default_batch": 4096,
        "decode_bar_fps": 2999.9,
        "cores": 15.40,
        "tier": "GPU_FASTER",
        "qualifier": "BATCH_SCALES",
        "R": 315.98,
        "official_jitted_fps": 151.68,
        "tier_no_reference_alternative": "GPU_FAST",
        "tier_if_reference_not_jitted_DO_NOT_USE": "GPU_EXCEPTIONAL",
        "daylight_diagnostic_fps": 796702,
        "night_diagnostic_fps": 24353,
        "batch_scales": 238.0,
        "no_large_batch_regression": 1.093,
        "job": "6395847 (throughput), 6409934 (reference)",
        "host": "gpu2107 / gpu2109",
        "card": "NVIDIA GeForce RTX 3090",
        "reference_conditions": "jax backend 'gpu', CudaDevice(id=0) -- the same card as torch",
    }
    resolution_consistency_applicability = {
        "light_level >= 0.5": "asserted at the frozen DOWNSAMPLE_EXACT_FRAC; measured 1.000",
        "light_level < 0.5": "not_applicable: the reference is itself resolution-inconsistent there "
        "(official world rows 0.1634; this renderer 0.1634, the same number)",
        "evidence": "pre-repository-cleanup:tests/fixtures/regression/join/step7/craftax-classic-full/oracle/resolution_oracle_S16.json",
        "measured_against": "craftax 1.6.1 un-jitted, jax 0.11.1, jax_threefry_partitionable=True",
    }
    #: 100% of all pixels exact over the whole observation, so the strong bound is declared rather
    #: than left at None. It is guarded by ``reference_versions`` above: a different craftax or jax
    #: blocks the check instead of failing it, which is the right verdict for a comparison that is no
    #: longer the comparison the bound was measured on.
    parity_all_exact_min = 1.0
    #: What the parity bound is, and what it is not. The declared versions above are corroborated
    #: here rather than left as bare numbers.
    parity_note = (
        "Measured on 64 sampled states at 7 px and 16 px, over the WHOLE observation -- the 7x9 world "
        "window and the 2-row inventory strip -- and over the whole day cycle: 18 night rows "
        "(light_level < 0.5, noise field drawn), 29 dusk rows (0.5 <= light < 1, lit but noise-free) "
        "and 17 daylight rows, sleeping rows throughout. Against craftax 1.6.1 under jax 0.11.1 with "
        "jax_threefry_partitionable at its default True: 100% of all pixels exact in every band, max "
        "abs diff 0. The claim is a UINT8 claim and deliberately not a float one, because the "
        "reference does not have one float value to be equal to: under jax.jit XLA fuses its "
        "arithmetic differently and it disagrees with its own un-jitted self by up to 1.5e-05, which "
        "is invisible in uint8 by day and visible wherever the picture is compressed towards one value, "
        "which the sleep desaturation and the night blend both do. Against the JITTED reference this "
        "renderer is 99.7854% (7 px) and 99.7225% (16 px) of pixels exact with max abs diff 1 ON "
        "THIS POOL, where every differing pixel was on a sleeping row; independent review found the "
        "jitted reference differing on an awake row at light_level 0.49999997, so any LIT row can "
        "straddle and the sleeping rows are only the commonest case. reference_env therefore returns "
        "the un-jitted function, which is the renderer as written. "
        "jax is a declared reference version because jax_threefry_partitionable selects between two "
        "threefry counter layouts that agree on 0 of 448 draws, and the night noise field is drawn "
        "from one of them."
    )

    # ------------------------------------------------------------------ state
    #
    # The public :data:`State` is the render WINDOW grouped by what it describes, which is what this
    # cell's whole thesis is about (module docstring): the 149 numbers are the tiles on screen, the
    # sprites over them, the sixteen inventory counts, the day-cycle light and the PRNG key — not the
    # simulator's 20,952-byte ``EnvState``. Every offset below is the packed column it
    # has always been, in the order ``state_names`` gives, so ``pack`` writes the frozen row and both
    # frozen corpora round-trip unchanged.
    #
    # There is no offset-``None`` field, because there is no hidden dynamical state: this cell has no
    # simulator at all (see :meth:`step`'s inherited refusal), so the packed row IS the whole state.
    state_layout = (
        (
            (("world", "blocks"), OBS_H * OBS_W, 0),
            (("world", "mobs"), OBS_H * OBS_W, OBS_H * OBS_W),
            (("player", "facing"), 1, 2 * OBS_H * OBS_W),
        )
        + tuple((("inventory", nm), 1, 2 * OBS_H * OBS_W + 1 + i) for i, nm in enumerate(STRIP_NAMES))
        + (
            (("player", "is_sleeping"), 1, 2 * OBS_H * OBS_W + 1 + N_STRIP),
            (("world", "light_level"), 1, 2 * OBS_H * OBS_W + 2 + N_STRIP),
            (("world", "rng_key"), N_KEY_HALVES, 2 * OBS_H * OBS_W + 3 + N_STRIP),
        )
    )

    @property
    def frame_shape(self) -> tuple[int, int]:
        """``(H, W)`` of one frame. ``size`` is the TILE edge here, not the frame edge.

        ``Env.frame_spec`` assumes a square frame of ``size`` pixels, which would report
        ``(16, 16)`` for a picture that is ``(144, 144)``: the buffer a consumer allocates from the
        spec would be 81x too small. The window is ``OBS_H + STRIP_H`` rows by ``OBS_W`` columns of
        ``size``-pixel tiles, and it happens to be square only because 7 + 2 == 9; the arithmetic is
        written out rather than relying on that coincidence.
        """
        return ((OBS_H + STRIP_H) * self.size, OBS_W * self.size)

    def make_variation_space(self) -> Dict:
        """Knobs that are the identity at their defaults and genuinely move pixels off them.

        There is deliberately no per-block colour knob. Craftax's blocks are 16x16 pixel-art PNGs
        resampled to 7 px, not procedurally drawn shapes, so a "grass colour" would have to invent a
        recolouring rule the reference does not have. ``channel_gain`` is the one global recolouring
        that is exactly the identity at 1.0; the two visibility flags are ablations that a world-model
        experiment actually wants (render the same state with and without the agent, or without mobs).
        """
        return Dict(
            {
                "channel_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32, init_value=[1.0, 1.0, 1.0]),
                "player_visible": Discrete(2, init_value=1),
                "mobs_visible": Discrete(2, init_value=1),
            }
        )

    @staticmethod
    def _tiles_to_frame(x: torch.Tensor) -> torch.Tensor:
        """(n,H,W,t,t,3) tile-major -> (n,H*t,W*t,3) raster."""
        n, H, W, t, _, _ = x.shape
        return x.permute(0, 1, 3, 2, 4, 5).reshape(n, H * t, W * t, 3)

    @staticmethod
    def _strip_grid(device):
        """``(cell_of, blank)`` — which strip cell sits at each ``(strip row, column)``.

        Two of the eighteen positions are never written by the reference and stay black; they map to
        cell 0 holding count 0, which IS black, so a blank needs no table entry of its own.

        Cached per device: it is state-independent, and rebuilding it per render cost 39.7 us on the
        CPU and ~34 kernel launches on CUDA. The returned tensors are never written through, which
        ``tests/unit/test_craftax_strip.py:test_the_strip_grid_cache_is_not_written_through``
        asserts, because a write would corrupt every later frame in the process silently.
        """
        key = str(device)
        if key not in _GRID_CACHE:
            cell_of = torch.zeros(STRIP_H, OBS_W, dtype=torch.long, device=device)
            blank = torch.ones(STRIP_H, OBS_W, dtype=torch.bool, device=device)
            for i, (r, c) in enumerate(STRIP_LAYOUT):
                cell_of[r, c] = i
                blank[r, c] = False
            _GRID_CACHE[key] = (cell_of, blank.unsqueeze(0))
        return _GRID_CACHE[key]

    def _decode(self, s, n):
        """The state row -> the small integers the tables are indexed by.

        Every decode is a **wrap**, never a clamp, and deliberately: ``check_state_dims_matter``
        probes a dimension by adding 25 to it, and a clamping decoder leaves a cell that already
        holds the last class insensitive to the probe — which is reported as a dead state dimension
        when it is an artifact of the probe. Real states are always in range, so the rule never
        fires in practice.
        """
        nb = OBS_H * OBS_W
        blk = s[:, :nb].round().remainder(N_BLOCK).long().view(n, OBS_H, OBS_W)
        mob = s[:, nb : 2 * nb].round().remainder(N_MOB).long().view(n, OBS_H, OBS_W)
        # facing wraps mod FOUR, not mod N_PLAYER: slot 4 is the sleeping sprite and is reached by
        # the is_sleeping flag, not by a facing value
        facing = s[:, 2 * nb].round().remainder(N_FACING).long()
        # one round() over the whole tail block rather than one per field: at B=1 the decode is
        # dominated by op launches, not by arithmetic, and every frame pays it including a daylight
        # one that will never look at most of what is decoded
        tail = s[:, 2 * nb + 1 :].round()
        cnt = tail[:, :N_STRIP].remainder(N_COUNT).long()
        j = 2 * nb + 1 + N_STRIP
        sleeping = s[:, j] > 0.5
        # ``light_level`` is the one slot a wrap must be careful with. Clamping is not an option --
        # a clamped daylight state is insensitive to the +25 probe and reports as a dead dimension --
        # but the obvious wrap ``x - floor(x)`` sends 1.0 to 0.0, turning every daylight state into
        # midnight. A tent on period 2 fixes that (it is the identity on [0, 1] and folds everything
        # else back into it), and it is applied ONLY outside [0, 1] so an in-range value passes
        # through bit-exact instead of through two float32 roundings.
        raw = s[:, j + 1]
        tent = 1.0 - (raw.remainder(2.0) - 1.0).abs()
        light = torch.where((raw >= 0.0) & (raw <= 1.0), raw, tent)
        # the key HALVES, not the assembled key: assembling costs six more ops and is needed only on
        # a lit row, so the caller does it when it takes the float path
        halves = tail[:, N_STRIP + 2 : N_STRIP + 2 + N_KEY_HALVES].remainder(65536.0)
        return blk, mob, facing, cnt, sleeping, light, halves

    def _tile_ids(self, blk, mob, facing, cnt, sleeping, sty) -> torch.Tensor:
        """``(n, OBS_H + STRIP_H, OBS_W)`` int32 — one table row per output tile, world and strip.

        Everything that could have been a branch is folded in here instead, so one batch may hold
        any mixture of appearances and of sleeping and awake players:

        * ``mobs_visible = 0`` becomes mob id 0, whose alpha table is exactly zero (asserted, not
          assumed, by ``tests/unit/test_craftax_atlas.py``);
        * ``player_visible = 0`` and every non-centre tile become player slot :data:`NO_PLAYER`;
        * ``is_sleeping`` becomes player slot :data:`PLAYER_ASLEEP` on the centre tile.
        """
        n, dev = blk.shape[0], blk.device
        vis_m = (sty["mobs_visible"] > 0.5).view(n, 1, 1)
        vis_p = (sty["player_visible"] > 0.5).view(n)
        m = torch.where(vis_m, mob, torch.zeros_like(mob))
        slot = torch.full((n, OBS_H, OBS_W), NO_PLAYER, dtype=torch.long, device=dev)
        cr, cc = OBS_H // 2, OBS_W // 2
        here = torch.where(sleeping, torch.full_like(facing, PLAYER_ASLEEP), facing)
        slot[:, cr, cc] = torch.where(vis_p, here, torch.full_like(here, NO_PLAYER))
        world = (slot * N_BLOCK + blk) * N_MOB + m  # (n, OBS_H, OBS_W)

        cell_of, blank = self._strip_grid(dev)
        c = cnt[:, cell_of.reshape(-1)].view(n, STRIP_H, OBS_W)
        c = torch.where(blank, torch.zeros_like(c), c)
        strip = STRIP_BASE + cell_of.unsqueeze(0) * N_COUNT + c  # (n, STRIP_H, OBS_W)
        return torch.cat([world, strip], dim=1).to(torch.int32)

    @staticmethod
    def _post(px: torch.Tensor, sleeping, light, halves) -> torch.Tensor:
        """The reference's post-process on the WORLD rows. ``px`` is ``(n, OBS_H*t, OBS_W*t, 3)``.

        The strip is not passed in, because the reference concatenates it *after* this runs: it is
        never lit and never desaturated. See :func:`night_and_sleep`, which is module-level and
        shape-parameterised so the full-Craftax renderer can call the same function.
        """
        return night_and_sleep(px, light, sleeping, prng.key_from_halves(halves))

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """``(...,144)`` -> ``(...,(7+2)*size, 9*size, 3)`` uint8, in the reference's own order.

        There is no ``scene()``: the frame is a gather from a texture atlas with two alpha blends
        over it, not a rasterisation of vector primitives — the blends are baked
        into the atlas (:class:`_Composite`) instead of being redone per frame. The
        inventory strip's 160 composited cells live in the *same* table, so the whole observation —
        world window and strip — is still **one** ``index_select``: a strip cell is a tile like any
        other, indexed by ``(cell, count)`` instead of by ``(player, block, mob)``.

        Two things cannot fold into the gather, and both are guarded so that the common case keeps
        the uint8 fast path:

        * ``channel_gain`` multiplies *before* the rounding, so a gain other than exactly 1 gathers
          the pre-rounding float table instead;
        * a sleeping player desaturates the world rows after compositing, so a batch containing one
          gathers float too. Which path is taken changes only the speed: the uint8 table is
          *defined* as ``round -> clamp -> uint8`` of the float one, so a non-sleeping row at unit
          gain comes out identical either way, and ``tests/unit/test_craftax_atlas.py`` asserts it
          on a batch that mixes sleeping and awake rows.
        """
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        n = s.shape[0]
        t = int(size or self.size)
        if supersample not in (None, 1):
            raise ValueError(
                f"CraftaxClassic.render: supersample={supersample} is meaningless here — the frame is a "
                f"gather from a bitmap atlas, so there is no sub-pixel coverage to estimate."
            )
        sty = self.resolve_style(style, variation, batch=n)

        blk, mob, facing, cnt, sleeping, light, halves = self._decode(s, n)
        H = (OBS_H + STRIP_H) * t
        if t > ATLAS_MAX_TILE_PX:
            img = self._render_blend(blk, mob, facing, cnt, sleeping, light, halves, t, sty)
        else:
            img = self._render_atlas(blk, mob, facing, cnt, sleeping, light, halves, t, sty, out)
        if out is not None and img.data_ptr() != out.data_ptr():
            out.view(n, H, OBS_W * t, 3).copy_(img)
            img = out.view(n, H, OBS_W * t, 3)
        return img.view(*lead, *img.shape[1:]) if lead else img

    def _render_atlas(self, blk, mob, facing, cnt, sleeping, light, halves, t, sty, out) -> torch.Tensor:
        """The gather. One ``index_select`` per frame at the default appearance, and no float buffer."""
        n, dev = blk.shape[0], blk.device
        H = (OBS_H + STRIP_H) * t
        tile = self._tile_ids(blk, mob, facing, cnt, sleeping, sty)

        # One index entry per (frame, tile row, output scanline within the tile, tile column), which
        # IS the output's own row order — so the gather writes the final raster directly instead of
        # building a tile-major buffer and transposing it.
        rows = tile.unsqueeze(2) * t + torch.arange(t, dtype=torch.int32, device=dev).view(1, 1, t, 1)
        rows = rows.reshape(-1)

        # One host read, not three: each of these is a device->host synchronisation on CUDA, and the
        # question they answer together is a single one -- "can this batch stay on the uint8 gather".
        # It can when the gain is exactly 1 (it multiplies before the rounding), nothing is asleep
        # (the desaturation multiplies before the rounding too) and it is full daylight, where
        # ``1.0*world + 0.0*night`` is exactly ``world`` in float32 and the whole chain is the
        # identity -- measured bit-exact on 24 real states at both tile sizes.
        gain = sty["channel_gain"]
        plain = bool(torch.all(gain == 1.0) & ~sleeping.any() & torch.all(light == 1.0))
        if plain:
            atlas = _Composite.get(dev, t, want_float=False)[1]
            img = (
                out.view(n, H, OBS_W * t, 3)
                if out is not None
                else torch.empty(n, H, OBS_W * t, 3, dtype=torch.uint8, device=dev)
            )
            torch.index_select(atlas, 0, rows, out=img.view(-1, t * 3))
            return img
        f32 = _Composite.get(dev, t, want_float=True)[0]
        px = torch.index_select(f32, 0, rows).view(n, H, OBS_W * t, 3)
        w = OBS_H * t
        px[:, :w] = self._post(px[:, :w], sleeping, light, halves)
        px = px * gain.to(px.dtype).view(n, 1, 1, 3)
        return px.round_().clamp_(0, 255).to(torch.uint8)

    def _render_blend(self, blk, mob, facing, cnt, sleeping, light, halves, t, sty) -> torch.Tensor:
        """Composite per frame, the readable definition of what the atlas bakes.

        Kept because the baked atlas costs ``976 * t*t*3`` bytes and stops being a table worth
        keeping somewhere above :data:`ATLAS_MAX_TILE_PX` (see there). It is the readable definition
        of what the atlas bakes, and ``tests/unit/test_craftax_atlas.py`` asserts the two agree on
        every pixel — including sleeping and awake rows and both visibility flags mixed within one
        batch, out-of-range class ids and a random gain.
        """
        n = blk.shape[0]
        block, ptex, palpha, mtex, malpha = _Assets.get(blk.device, t)
        px = self._tiles_to_frame(block[blk])  # (n,H*t,W*t,3) float32

        # The two visibility flags scale the sprite's own alpha instead of branching, so a batch can
        # hold both settings. Zeroing the alpha is exactly the skip: ``sl*(1-0) + x*0`` is ``sl`` in
        # float32, not merely close to it.
        cr, cc = OBS_H // 2, OBS_W // 2
        sl = px[:, cr * t : (cr + 1) * t, cc * t : (cc + 1) * t, :]
        pid = torch.where(sleeping, torch.full_like(facing, PLAYER_ASLEEP), facing)
        a = palpha[pid] * (sty["player_visible"] > 0.5).to(px.dtype).view(n, 1, 1, 1)
        px[:, cr * t : (cr + 1) * t, cc * t : (cc + 1) * t, :] = sl * (1 - a) + ptex[pid] * a

        a = self._tiles_to_frame(malpha[mob]) * (sty["mobs_visible"] > 0.5).to(px.dtype).view(n, 1, 1, 1)
        px = px * (1 - a) + self._tiles_to_frame(mtex[mob]) * a

        px = self._post(px, sleeping, light, halves)

        # the strip, drawn after the post-process exactly as the reference concatenates it after
        cells = _strip_cells(blk.device, t)
        cell_of, blank = self._strip_grid(blk.device)
        c = cnt[:, cell_of.reshape(-1)].view(n, STRIP_H, OBS_W)
        c = torch.where(blank, torch.zeros_like(c), c)
        strip = self._tiles_to_frame(cells[cell_of.unsqueeze(0) * N_COUNT + c])
        px = torch.cat([px, strip], dim=1)

        px = px * sty["channel_gain"].to(px.dtype).view(n, 1, 1, 3)

        # the reference frame is fractional float32; uint8 is a rounding of it, half-to-even to match
        # numpy, and reference_render rounds the official float the same way
        return px.round_().clamp_(0, 255).to(torch.uint8)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Random terrain windows with the reference's mob budget respected.

        Mob counts are capped at ``StaticEnvParams``' ``max_zombies=3, max_cows=3, max_skeletons=2,
        max_arrows=3`` and no mob is placed on the centre tile. Both are requirements of the parity
        path, not cosmetics: a state with four zombies cannot be written into a Craftax ``EnvState`` at
        all, and a mob standing on the player would make the ``player_visible`` knob inert on that
        frame and be reported as an unimplemented variation.
        """
        rng = rng or self.rng
        nb = OBS_H * OBS_W
        st = np.zeros((batch, self.state_dim), np.float32)
        terrain = rng.choice(_TERRAIN, size=(batch, OBS_H, OBS_W), p=_TERRAIN_P).astype(np.float32)
        # Half the windows sit against a world edge, where the renderer pads the view with
        # OUT_OF_BOUNDS. That is a real state, and it is also the only block whose texture has a flat
        # 3x3 interior (measured: blocks 0 and 1 are the only uniform ones, 225 interior pixels each at
        # 16 px). Without it every pixel of every frame borders a different colour, the conformance
        # suite's interior mask is empty, and its interior statistic comes back NaN — so a renderer that
        # is exact on every pixel still cannot report an interior agreement.
        for b in range(batch):
            if rng.random() < 0.5:
                k = int(rng.integers(1, 3))
                side = int(rng.integers(4))
                if side == 0:
                    terrain[b, :k, :] = BLOCK_OUT_OF_BOUNDS
                elif side == 1:
                    terrain[b, -k:, :] = BLOCK_OUT_OF_BOUNDS
                elif side == 2:
                    terrain[b, :, :k] = BLOCK_OUT_OF_BOUNDS
                else:
                    terrain[b, :, -k:] = BLOCK_OUT_OF_BOUNDS
        st[:, :nb] = terrain.reshape(batch, nb)
        centre = (OBS_H // 2) * OBS_W + OBS_W // 2
        cells = [i for i in range(nb) if i != centre]
        for b in range(batch):
            budget = [
                (MOB_ZOMBIE, int(rng.integers(0, MAX_ZOMBIES + 1))),
                (MOB_COW, int(rng.integers(0, MAX_COWS + 1))),
                (MOB_SKELETON, int(rng.integers(0, MAX_SKELETONS + 1))),
                (None, int(rng.integers(1, MAX_ARROWS + 1))),
            ]
            k = sum(c for _, c in budget)
            where = rng.choice(len(cells), size=min(k, len(cells)), replace=False)
            j = 0
            for kind, count in budget:
                for _ in range(count):
                    if j >= len(where):
                        break
                    mid = kind if kind is not None else MOB_ARROW[int(rng.integers(4))]
                    st[b, nb + cells[int(where[j])]] = mid
                    j += 1
        st[:, nb * 2] = rng.integers(0, N_FACING, size=batch).astype(np.float32)

        # The strip counts are sampled DIRECTLY over 0..9 rather than taken from a rollout. That is
        # not a shortcut: measured over a 4,000-step random-policy rollout, ten of the twelve
        # inventory cells never leave zero (wood reaches 2, sapling 1, the rest nothing) because a
        # random policy never crafts, so a pool built from one would present ten dead state
        # dimensions -- exactly the failure ``pre-repository-cleanup:docs/research/pixel_native.md`` records for Procgen
        # miner and heist, and it looks identical to a wrong state contract. Sampling the counts is
        # sound here because the strip is a function of the counts alone (each cell's two writes stay
        # inside its own tile, asserted at generation time) and because the simulator caps every
        # count at 9, so every sampled row is a row a real ``EnvState`` can hold.
        st[:, nb * 2 + 1 : nb * 2 + 1 + N_STRIP] = rng.integers(0, N_COUNT, size=(batch, N_STRIP)).astype(np.float32)
        # A quarter asleep, but not by an unconditional draw. ``player_facing`` reaches no pixel
        # while the player is asleep -- the reference selects sprite slot 4 whatever the facing is,
        # which is a true statement about the renderer and not a defect -- so a ``batch=1`` pool that
        # happened to sample a sleeping state reports ``player_facing`` dead. Row 0 is therefore
        # always awake and row 3 always asleep, which makes both observable in every batch from one
        # row upward. ``is_sleeping`` itself stays observable either way, because
        # ``check_state_dims_matter`` falls back to a ``-25`` nudge that crosses the 0.5 threshold
        # from whichever side the row is on.
        sleeping = rng.random(batch) < 0.25
        sleeping[::4] = False
        if batch > 3:
            sleeping[3::4] = True
        st[:, nb * 2 + 1 + N_STRIP] = sleeping.astype(np.float32)

        # ``light_level`` takes the values the simulator actually produces -- craftax computes it as
        # ``1 - |cos(pi * ((t/300) % 1 + 0.3))|**3`` -- but the BAND is assigned by row index rather
        # than left to chance, and that is a requirement rather than a preference.
        # ``check_state_dims_matter`` probes ``env.B`` sampled states, and the four PRNG-key slots
        # reach no pixel above ``light_level = 0.5``: measured, 32 of 32 night states change when the
        # key changes and 0 of 32 daylight ones do. With the band left to the draw, a ``batch=1``
        # instance reports those four dimensions dead about two times in three, and what that
        # measures is the sampler's luck rather than the renderer. Stratifying makes row 0 a night
        # row, so every dimension is observable in every batch this returns, down to a batch of one.
        # It is the same device as the ``OUT_OF_BOUNDS`` band above, which exists so that the
        # conformance suite's interior statistic is computable at all.
        j = nb * 2 + 2 + N_STRIP
        cycle = 1 - np.abs(np.cos(np.pi * ((np.arange(DAY_LENGTH) / DAY_LENGTH) % 1 + 0.3))) ** 3
        night_vals = cycle[cycle < NIGHT_THRESHOLD]
        dusk_vals = cycle[(cycle >= NIGHT_THRESHOLD) & (cycle < 1.0)]
        band = np.arange(batch) % 4
        light = rng.choice(cycle, size=batch)  # band 3: the raw distribution
        light[band == 0] = rng.choice(night_vals, size=int((band == 0).sum()))
        light[band == 1] = rng.choice(dusk_vals, size=int((band == 1).sum()))
        light[band == 2] = 1.0
        st[:, j] = light.astype(np.float32)
        # the PRNG key, as four 16-bit halves. A bare uint32 would not survive this float32 row.
        st[:, j + 1 : j + 1 + N_KEY_HALVES] = rng.integers(0, 65536, size=(batch, N_KEY_HALVES)).astype(np.float32)
        return st

    # reference bridge

    @staticmethod
    def reference_env(size: int = TILE_PX_IMG):
        """The official Craftax renderer at ``size`` px per tile, plus a template ``EnvState``.

        Returned as a tuple rather than a gym env because Craftax is a functional JAX env: the renderer
        is a pure function of ``EnvState`` and needs no instance. The template comes from a real
        ``reset`` so every field the renderer does not read still holds a valid value.

        Only the two sizes craftax **ships textures for** are accepted. That is the accurate reason
        and it is narrower than "no official frame": craftax's ``load_all_textures_given_size`` will
        build a renderer at any size, and this one's world rows are exact against a constructed 32 px
        reference in every band (``oracle/parity_at_2S.json``). What is missing at an unshipped size
        is *shipped art to certify against* — the frame such a reference draws is a choice craftax
        never published, so agreeing with it would certify against a second implementation rather
        than against the environment.

        **The renderer is deliberately NOT jitted, and that is a correctness decision rather than a
        performance one.** "The official frame" is not one object: under ``jax.jit`` XLA fuses the
        arithmetic differently and the reference disagrees with its own un-jitted self by up to
        1.5e-05 in float32. That is invisible in uint8 for a full-daylight frame, and it is *not*
        invisible wherever the picture is compressed towards one value — the sleep desaturation
        halves the dynamic range and the night blend pulls everything towards the tint — because a
        1.5e-05 difference then straddles a rounding boundary. Measured over 64 states at both tile
        sizes, against this renderer: **un-jitted 100.0000% of all pixels exact, max abs diff 0;
        jitted 99.7854% at 7 px and 99.7225% at 16 px, max abs diff 1.** On that pool every differing
        pixel was on a sleeping row, but that is the commonest case rather than the rule: independent
        review found the jitted reference differing on an **awake** row at
        ``light_level = 0.49999997``. Any **lit** row can straddle. The un-jitted function is the
        renderer as written, so it is the one certified against; jit is an optimisation that changes
        the picture. It costs about 0.25 s a frame instead of 0.008 s, which
        is the right trade for a check that runs on tens of states and is the whole basis of a parity
        claim.

        ``jax_threefry_partitionable`` is asserted rather than assumed. jax has two threefry code
        paths and they give different bits for the same key (0/448 agreement between them, measured);
        the flag's default flipped to ``True`` in jax 0.4.36. With it off, the reference's night noise
        is a different picture and a parity number against it would be meaningless rather than wrong.
        """
        if int(size) not in REFERENCE_TILE_PX:
            raise ValueError(
                f"craftax draws at {REFERENCE_TILE_PX[0]} px (the agent observation) or "
                f"{REFERENCE_TILE_PX[1]} px per tile; craftax ships no textures at {size} px, so "
                f"there is no published frame to certify a {size} px render against"
            )
        import jax
        from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
        from craftax.craftax_env import make_craftax_env_from_name

        if not bool(jax.config.jax_threefry_partitionable):
            raise RuntimeError(
                "CraftaxClassic.reference_env: jax_threefry_partitionable is off, so this jax draws "
                "the night noise field with the pre-0.4.36 threefry counter layout. That is a "
                "different field for the same state_rng (0 of 448 draws agree between the two "
                "paths), so a parity number taken against it would not be a parity number. Run with "
                "the flag at its default (True), or declare the other path and re-measure."
            )
        env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
        _obs, template = env.reset(jax.random.PRNGKey(0), env.default_params)
        return (make_craftax_pixel_renderer(int(size)), template, int(size))

    @staticmethod
    def reference_render(env, states) -> np.ndarray:
        """Official frames for the given render states — the WHOLE observation, no crop.

        The render state is written back into a full ``EnvState``: the block window goes into the middle
        of the map with the player standing at its centre, each occupied mob tile becomes one mob of the
        right class at the matching absolute position, arrow orientations are recovered by choosing an
        ``arrow_direction`` whose ``(flip, transpose)`` classification is the one encoded, and the
        sixteen strip counts go into ``player_health``/``food``/``drink``/``energy`` and ``inventory``.

        The crop of the official frame to the world layer is gone: the strip is drawn, so the parity
        number is over all ``(OBS_H + STRIP_H) * tile_px`` rows.

        Nothing is forced any more. ``light_level`` and ``state_rng`` are written back from the state
        row like everything else, so a night frame is compared against the reference's own night
        frame — noise field included — rather than against a daylight stand-in.
        """
        import jax
        import jax.numpy as jnp

        render, tmpl, tile_px = env

        def _i(field, v):
            """``v`` in the dtype the template's field already has, so the pytree stays well typed."""
            return jnp.asarray(v, jnp.asarray(field).dtype)

        s = np.asarray(states, dtype=np.float32).reshape(-1, CraftaxClassic.state_dim)
        nb = OBS_H * OBS_W
        H, W = np.asarray(tmpl.map).shape
        py, px_ = H // 2, W // 2
        r0, c0 = py - OBS_H // 2, px_ - OBS_W // 2
        #: mob id -> an ``arrow_direction`` producing the orientation that id encodes. The textures are
        #: generated as ``(flip, transpose)`` in the order (0,0), (0,1), (1,0), (1,1), and the reference
        #: classifies a direction as ``flip = (d0 > 0) or (d1 > 0)``, ``transpose = d1 != 0`` — so the
        #: inverse is this table and NOT the one that pairs ids with directions in the obvious order.
        #: Getting it wrong swaps ids 5 and 6 and shows up as two whole wrong tiles per frame, which is
        #: what the first parity run measured (0.24% of pixels, max |diff| 255).
        ARROW_DIR = {4: (-1, 0), 5: (0, -1), 6: (1, 0), 7: (0, 1)}

        def mobs_for(occupied, cap):
            pos = np.zeros((cap, 2), np.int32)
            mask = np.zeros((cap,), bool)
            for i, (r, c) in enumerate(occupied[:cap]):
                pos[i] = (r0 + r, c0 + c)
                mask[i] = True
            return pos, mask

        out = []
        for row in s:
            blk = np.round(row[:nb]).astype(np.int32).reshape(OBS_H, OBS_W) % N_BLOCK
            mob = np.round(row[nb : 2 * nb]).astype(np.int32).reshape(OBS_H, OBS_W) % N_MOB
            pid = int(np.round(row[2 * nb])) % N_FACING
            cnt = np.round(row[2 * nb + 1 : 2 * nb + 1 + N_STRIP]).astype(np.int32) % N_COUNT
            j = 2 * nb + 1 + N_STRIP
            sleeping = bool(row[j] > 0.5)
            raw = float(row[j + 1])
            light = raw if 0.0 <= raw <= 1.0 else 1.0 - abs((raw % 2.0) - 1.0)
            halves = np.round(row[j + 2 : j + 2 + N_KEY_HALVES]).astype(np.int64) % 65536
            key = np.array([(halves[0] << 16) | halves[1], (halves[2] << 16) | halves[3]], np.uint32)

            m = np.asarray(tmpl.map).copy()
            m[r0 : r0 + OBS_H, c0 : c0 + OBS_W] = blk
            occ = {k: [] for k in (MOB_ZOMBIE, MOB_COW, MOB_SKELETON)}
            arrows = []
            for r in range(OBS_H):
                for c in range(OBS_W):
                    v = int(mob[r, c])
                    if v in occ:
                        occ[v].append((r, c))
                    elif v in ARROW_DIR:
                        arrows.append((r, c, v))
            zp, zm = mobs_for(occ[MOB_ZOMBIE], MAX_ZOMBIES)
            cp, cm = mobs_for(occ[MOB_COW], MAX_COWS)
            sp, sm = mobs_for(occ[MOB_SKELETON], MAX_SKELETONS)
            ap = np.zeros((MAX_ARROWS, 2), np.int32)
            am = np.zeros((MAX_ARROWS,), bool)
            ad = np.zeros((MAX_ARROWS, 2), np.int32)
            for i, (r, c, v) in enumerate(arrows[:MAX_ARROWS]):
                ap[i], am[i], ad[i] = (r0 + r, c0 + c), True, ARROW_DIR[v]

            def mob(tmpl_mobs, pos, mask):
                return tmpl_mobs.replace(position=jnp.asarray(pos, jnp.int32), mask=jnp.asarray(mask, bool))

            st = tmpl.replace(
                map=jnp.asarray(m, jnp.asarray(tmpl.map).dtype),
                player_position=jnp.asarray([py, px_], jnp.int32),
                player_direction=jnp.asarray(pid + 1, jnp.asarray(tmpl.player_direction).dtype),
                is_sleeping=jnp.bool_(sleeping),
                light_level=jnp.float32(light),
                state_rng=jnp.asarray(key, jnp.uint32),
                player_health=_i(tmpl.player_health, cnt[0]),
                player_food=_i(tmpl.player_food, cnt[1]),
                player_drink=_i(tmpl.player_drink, cnt[2]),
                player_energy=_i(tmpl.player_energy, cnt[3]),
                inventory=tmpl.inventory.replace(
                    **{n: _i(getattr(tmpl.inventory, n), cnt[4 + i]) for i, n in enumerate(STRIP_NAMES[4:])}
                ),
                zombies=mob(tmpl.zombies, zp, zm),
                cows=mob(tmpl.cows, cp, cm),
                skeletons=mob(tmpl.skeletons, sp, sm),
                arrows=mob(tmpl.arrows, ap, am),
                arrow_directions=jnp.asarray(ad, jnp.asarray(tmpl.arrow_directions).dtype),
            )
            out.append(np.round(np.asarray(jax.device_get(render(st)))).clip(0, 255).astype(np.uint8))
        return np.stack(out)
