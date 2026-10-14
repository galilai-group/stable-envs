"""Craftax Classic: a texture-atlas renderer over stored render windows.

Reference: ``Craftax-Classic-Pixels-v1`` (JAX), drawn by
``craftax.craftax_classic.renderer.make_craftax_pixel_renderer``: a 7x9 window of the map, the player
sprite over the centre tile, zombies, cows, skeletons and arrows composited one tile each, a
day/night post-process, a sleeping desaturation, then a 2-row inventory strip below. Every frame is a
gather from fixed tables with the alpha blends baked into an atlas.

The state is the render window, not the simulator's ``EnvState``: 149 float32 values (63 blocks, 63
mobs, facing, sixteen strip counts, sleeping flag, light level, PRNG key). A uint32 does not fit a
float32 slot exactly (2**32 > 2**24), so the key is stored as four 16-bit halves.

Frames are uint8 roundings of the reference's float32 output (alphas take {0, 128/255, 1}). Parity
is exact in uint8 against the un-jitted official renderer; under ``jax.jit`` XLA fuses the float32
arithmetic differently (up to 1.5e-05), which flips roundings where the picture is compressed towards
one value (sleep desaturation, night blend, any lit row). See :meth:`CraftaxClassic.reference_env`.

Coverage and invariants:

  * the world layer: terrain, the player's facing, the sleeping sprite, and the mobs and arrows on
    screen;
  * the inventory strip (rows 49-62 of the 63x63 agent observation, 112-143 of the 144x144 image
    observation) depends only on twelve counts and four intrinsics, drawn from a 160-entry
    ``(cell, count)`` table in the same atlas. The reference concatenates it after the lighting
    chain, so it is the same at midnight and noon;
  * ``is_sleeping`` selects a fifth player sprite and desaturates the world rows;
  * night: ``light_level < 0.5`` mixes in ``jax.random.uniform(state.state_rng, (H, W)) * 95 + 32``
    through a vignette, so night frames depend on the PRNG key (``prng.py`` reproduces threefry2x32
    bit for bit and is shared with full Craftax). At ``light >= 0.5`` no PRNG is drawn, and at
    ``light == 1.0`` the whole chain is the identity, so the batch stays on the uint8 gather;
  * one mob per tile. The reference allows an arrow and a mob on one tile; that state is not
    representable here and ``sample_states`` does not generate it;
  * the player tables are zero away from the centre tile, so compositing the player over the whole
    frame (reference) or into one tile (here) is the same arithmetic;
  * night is not resolution-consistent in the reference itself: the noise field is drawn at
    ``(7t, 9t)``, so element ``i`` at 32 px uses a different threefry counter from element ``i`` at 16 px;
  * ``sample_states`` draws strip counts directly over 0..9, since a random policy never crafts and
    ten of the twelve inventory cells would stay at zero.

Out-of-range class ids and counts wrap rather than clamp, so a cell already at its last class still
responds to the conformance suite's +25 probe.
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
#: facing left, right, up, down, asleep. ``is_sleeping`` selects the fifth sprite:
#: ``player_texture_index = select(is_sleeping, 4, player_direction - 1)``.
N_PLAYER = 5
PLAYER_ASLEEP = 4
#: Facing values a state row carries. Slot 4 is reached by ``is_sleeping``, so ``player_facing``
#: wraps mod four, not mod N_PLAYER.
N_FACING = 4
#: Inventory-strip cells: 4 intrinsics then 12 inventory counts, each 0..9 (capped by the
#: simulator). Cell ``i`` is drawn at ``STRIP_LAYOUT[i]``; the two cells the reference never
#: writes stay black.
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
#: A uint32 survives a float32 state row only 2% of the time (2**32 > 2**24), so the two-uint32
#: ``state_rng`` key is stored as four 16-bit halves.
N_KEY_HALVES = 4
#: ``night_texture`` -- the colour night is tinted towards, ``[[[0, 16, 64]]]`` in craftax's constants.
NIGHT_TINT = (0.0, 16.0, 64.0)
#: Below this ``light_level`` the reference mixes in a noise field; at or above it
#: ``night_static_intensity`` is exactly 0 and no PRNG is drawn.
NIGHT_THRESHOLD = 0.5
#: craftax's day length; ``sample_states`` draws ``light_level`` from the simulator's day cycle.
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


#: Tile sizes craftax ships textures for; both are exact. Other sizes are nearest-resampled and
#: not compared against a reference.
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

#: Strip ingredients are not tile-sized (icon ``int(0.8*t)``, digit ``int(0.6*t)``: 12 and 9 at
#: 16 px, 25 and 19 at 32 px), so they are never resampled. ``_strip_cells`` builds the finished
#: cell at a reference size and resamples that, keeping S -> 2S an exact 2x replication.
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
    """``(icon_lo, icon_hi, digit_lo, digit_hi)`` within one ``t x t`` strip cell, per the reference.

    Every write stays inside the cell and no two cells overlap (asserted at generation time and in
    ``tests/unit/test_craftax_strip.py``).
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


#: Player slot meaning "no player on this tile": ``tile*(1 - 0) + player_tex*0``, the identity in float32.
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
#: State-independent ``(cell_of, blank)`` and post-process colour constants, cached per device.
_GRID_CACHE: dict = {}
_CONST_CACHE: dict = {}


def _post_constants(device: torch.device):
    """``(night_tint, sleep_lift)`` on ``device``, built once."""
    key = str(device)
    if key not in _CONST_CACHE:
        _CONST_CACHE[key] = (torch.tensor(NIGHT_TINT, device=device), torch.tensor([0.0, 0.0, 8.0], device=device))
    return _CONST_CACHE[key]


def night_vignette(device: torch.device, h: int, w: int) -> torch.Tensor:
    """``(h, w, 1)`` float32 copy of craftax's ``night_noise_intensity_texture``.

    ``1 - exp(-0.5 (x^2 + y^2) / 0.5^2)`` on a ``linspace(-1, 1)`` grid, transposed. Computed in
    float64 and narrowed to float32, as craftax does.
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
    """Craftax Classic's day/night chain and sleeping desaturation, in the reference's order.

    ``world`` ``(n, h, w, 3)`` float32 pre-post-process pixels, ``light`` ``(n,)``, ``sleeping``
    ``(n,)`` bool, ``keys`` ``(n, 2)`` int32 threefry keys. ``light_map`` is an optional per-pixel
    multiplier applied before the blend.

    The arithmetic is the reference's, op for op, and is not algebraically collapsed: the folded
    form rounds differently in float32. The only reductions are ones that change no bits:

    * noise is drawn only for rows with ``light < 0.5``; elsewhere ``intensity`` is exactly 0, so
      ``(1 - 0)*world + 0*noise`` is ``world`` whatever the noise was;
    * int32 threefry (``prng.py``) is bit-identical to an int64-with-masking implementation;
    * a batch entirely at ``light == 1.0`` and awake never gets here (``1.0*world + 0.0*night`` is
      exactly ``world``).
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

    # --- enhance: pull towards luminance. In place from here on: the same expressions as the
    # reference, but each result overwrites its own buffer rather than allocating an image-sized temporary.
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
    """``(N_STRIP * N_COUNT, t, t, 3)`` float32: every ``(cell, count)`` picture the strip can show.

    The strip depends on sixteen counts (each capped at 9) and the reference draws each cell with
    two writes inside its own tile, so it is a 16 x 10 table of finished tiles. It is concatenated
    after the day/night chain and the sleep desaturation, so it is never lit and does not depend on
    ``state_rng``.

    Built at a reference tile size and resampled as a finished cell: icon and digit edges do not
    scale by exactly two, so re-laying the cell out at 2S would not be a 2x replication of S.
    """
    src = t if t in _STRIP_TABLES else min(REFERENCE_TILE_PX, key=lambda z: abs(z - t))
    icons, digits, dalpha = _StripAssets.get(device, src)
    lo, hi, dlo, dhi = strip_geometry(src)
    cell = torch.zeros(N_STRIP, N_COUNT, src, src, 3, dtype=torch.float32, device=device)
    # Count 0 leaves the icon out (the reference selects ``smaller_empty_texture`` unless ``> 0``)
    # and digit 0 is the empty texture with zero alpha, so a zero cell is black. The two cells the
    # reference never writes reuse cell 0 holding count 0.
    cell[:, 1:, lo:hi, lo:hi, :] = icons.view(N_STRIP, 1, hi - lo, hi - lo, 3)
    patch = cell[:, :, dlo:dhi, dlo:dhi, :].clone()
    cell[:, :, dlo:dhi, dlo:dhi, :] = patch * (1 - dalpha.view(1, N_COUNT, dhi - dlo, dhi - dlo, 3)) + digits.view(
        1, N_COUNT, dhi - dlo, dhi - dlo, 3
    )
    if t != src:
        ix = ((torch.arange(t, device=device, dtype=torch.float32) + 0.5) * src / t).long().clamp_(0, src - 1)
        cell = cell[:, :, ix][:, :, :, ix]
    return cell.reshape(N_STRIP * N_COUNT, t, t, 3)


#: (player slot, block, mob) composites in the atlas: 6 * 17 * 8 = 816; the sixth player slot is
#: :data:`NO_PLAYER`.
N_COMPOSITE = (N_PLAYER + 1) * N_BLOCK * N_MOB
#: The strip cells follow the world tiles in the same table, so one gather draws the whole
#: observation. Tile id ``STRIP_BASE + cell * N_COUNT + count`` is strip cell ``cell`` showing
#: ``count``; ``STRIP_BASE + 0`` is black.
STRIP_BASE = N_COMPOSITE
N_TILE = N_COMPOSITE + N_STRIP * N_COUNT

#: Largest tile edge the baked atlas is used at: it costs ``976 * t*t*3`` bytes as uint8 and four
#: times that as float32. Above it ``render`` falls back to :meth:`CraftaxClassic._render_blend`,
#: which ``tests/unit/test_craftax_atlas.py`` asserts is pixel-identical.
ATLAS_MAX_TILE_PX = 64


class _Composite:
    """The composited tile atlas: every (player slot, block, mob) tile, then every strip cell.

    A world tile's pixels depend on three small integers, so the reference's per-frame blends are
    done once per (device, tile size) into an 816-entry table (plus the 160 strip cells) and each
    frame is a gather. The per-tile arithmetic is the reference's, elementwise float32 in the same
    order, so the result is bit-identical.

    Two forms are kept:

    * ``u8``: ``round -> clamp -> uint8`` already applied. Usable only when ``channel_gain`` is
      exactly 1 on every row (``px * 1.0`` is the identity, so rounding commutes with the gather);
    * ``f32``: the pre-rounding composite, built lazily when a non-default gain is rendered, since
      the gain multiplies before the rounding.
    """

    _MAX_ENTRIES = 8
    _cache = _RES.BoundedCache(_MAX_ENTRIES, name="craftax.composite")

    @staticmethod
    def _build(device: torch.device, t: int) -> torch.Tensor:
        """``(N_COMPOSITE * t, t * 3)`` float32, row ``slot * t + y``.

        One tile row per row, so the gather in :meth:`CraftaxClassic.render` addresses a whole
        output scanline of one tile with a single index.
        """
        block, ptex, palpha, mtex, malpha = _Assets.get(device, t)
        # player over block -> (6, 17, t, t, 3); slot NO_PLAYER is the block untouched
        with_player = block.unsqueeze(0).repeat(N_PLAYER + 1, 1, 1, 1, 1)
        ap = palpha.view(N_PLAYER, 1, t, t, 3)
        with_player[:N_PLAYER] = with_player[:N_PLAYER] * (1 - ap) + ptex.view(N_PLAYER, 1, t, t, 3) * ap
        # mob over that -> (6, 17, 8, t, t, 3)
        am = malpha.view(1, 1, N_MOB, t, t, 3)
        comp = with_player.unsqueeze(2) * (1 - am) + mtex.view(1, 1, N_MOB, t, t, 3) * am
        # the 160 strip cells follow in the same table, indexed by (cell, count)
        return torch.cat(
            [comp.reshape(N_COMPOSITE * t, t * 3), _strip_cells(device, t).reshape(N_STRIP * N_COUNT * t, t * 3)]
        ).contiguous()

    @classmethod
    def get(cls, device: torch.device, t: int, want_float: bool):
        """``(f32_or_None, u8)``. ``f32`` is materialised once a gain path asks for it.

        ``device`` is a tensor's own ``.device`` (explicit index), so the string key cannot
        conflate ``"cuda"`` on different GPUs.
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

    Renders stored render windows; ``has_dynamics`` is False. The delegated JAX simulator is
    registered separately in :mod:`stable_envs.envs.craftax.semantics`.
    """

    family = "craftax"
    domain = "classic"

    name = "craftax-classic"
    #: 63 blocks, 63 mobs, the player's facing, the sixteen strip counts, the sleeping flag, the
    #: light level, then the PRNG key halves. Appended, never reordered.
    state_dim = OBS_H * OBS_W * 2 + 1 + N_STRIP + 1 + 1 + N_KEY_HALVES
    state_names = (
        tuple(f"block_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"mob_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + ("player_facing",)
        + tuple(f"inv_{n}" for n in STRIP_NAMES)
        + ("is_sleeping", "light_level")
        + tuple(f"rng_half{i}" for i in range(N_KEY_HALVES))
    )
    #: Tile edge in pixels; the frame is ((7+2)*size, 9*size, 3). Craftax ships textures at 7 px (the
    #: agent observation) and 16 px, both exact. The default is 16 because at 7 px the dithered art
    #: has no flat 3x3 neighbourhood and the conformance suite's interior statistic is NaN.
    default_size = TILE_PX_IMG
    reference_id = "Craftax-Classic-Pixels-v1"
    #: Reference versions the parity was measured against. ``jax`` matters because
    #: :meth:`reference_env` refuses a PRNG path that differs.
    reference_versions = {"craftax": "1.6.1", "jax": "0.11.1"}
    #: Throughput-harness state distribution: ``sample_states`` assigns bands by row index (daylight
    #: exactly 0.25); the rest vary with the draw inside their band. About three quarters of it is lit.
    certification_state_distribution = {
        "draw": "CraftaxClassic.sample_states(64) -- the throughput harness protocol",
        "daylight_share": 0.25,
        "dusk_share": "0.39-0.44",
        "night_share": "0.31-0.36",
        "sleeping_share": "0.33-0.39",
        "measured_over": "seeds 0-5, n=64",
        "band_assignment": "stratified by row index (i % 4); values from craftax's own day cycle",
    }
    #: The reference is itself not resolution-consistent at night (world rows 0.1634), so the bound
    #: applies at daylight and dusk only.
    resolution_consistency_applicability = {
        "light_level >= 0.5": "asserted at the frozen DOWNSAMPLE_EXACT_FRAC; measured 1.000",
        "light_level < 0.5": "not_applicable: the reference is itself resolution-inconsistent there "
        "(official world rows 0.1634; this renderer 0.1634, the same number)",
        "evidence": "official-renderer resolution oracle at 16 px per tile against 32 px",
        "measured_against": "craftax 1.6.1 un-jitted, jax 0.11.1, jax_threefry_partitionable=True",
    }
    #: Exact over the whole observation. A different craftax or jax skips the check through
    #: ``reference_versions`` rather than failing it.
    parity_all_exact_min = 1.0
    parity_note = (
        "Measured on 64 sampled states at 7 px and 16 px over the whole observation (world window and "
        "inventory strip) and the whole day cycle (18 night, 29 dusk, 17 daylight rows, sleeping rows "
        "throughout) against craftax 1.6.1 under jax 0.11.1 with jax_threefry_partitionable True: 100% "
        "of pixels exact, max abs diff 0. The claim is a UINT8 claim against the UN-JITTED reference: "
        "under jax.jit XLA fuses the arithmetic differently (up to 1.5e-05), which flips a uint8 "
        "rounding wherever the picture is compressed towards one value (sleep desaturation, night "
        "blend, any lit row). Against the jitted reference this renderer is 99.7854% (7 px) and "
        "99.7225% (16 px) exact with max abs diff 1. jax is a declared reference version because "
        "jax_threefry_partitionable selects between two threefry counter layouts that agree on 0 of "
        "448 draws."
    )

    # Public State groups the render window by what it describes; offsets follow ``state_names``.
    # There is no hidden state, so the packed row is the whole state.
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
        """``(H, W)`` of one frame. ``size`` is the tile edge, not the frame edge: the window is
        ``OBS_H + STRIP_H`` rows by ``OBS_W`` columns of ``size``-pixel tiles.
        """
        return ((OBS_H + STRIP_H) * self.size, OBS_W * self.size)

    def make_variation_space(self) -> Dict:
        """Knobs that are the identity at their defaults and move pixels off them.

        There is no per-block colour knob: the blocks are pixel-art PNGs, so a recolouring rule would
        be invented. ``channel_gain`` is the one global recolouring that is exact at 1.0; the two
        visibility flags are ablations.
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
        """``(cell_of, blank)``: which strip cell sits at each ``(strip row, column)``.

        The two positions the reference never writes map to cell 0 holding count 0, which is black.
        Cached per device; the tensors must never be written through
        (``test_the_strip_grid_cache_is_not_written_through``).
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

        Every decode wraps rather than clamps: ``check_state_dims_matter`` probes by adding 25, and a
        clamping decoder would leave a cell already at its last class insensitive to it.
        """
        nb = OBS_H * OBS_W
        blk = s[:, :nb].round().remainder(N_BLOCK).long().view(n, OBS_H, OBS_W)
        mob = s[:, nb : 2 * nb].round().remainder(N_MOB).long().view(n, OBS_H, OBS_W)
        # facing wraps mod four, not mod N_PLAYER: slot 4 is reached by is_sleeping
        facing = s[:, 2 * nb].round().remainder(N_FACING).long()
        # one round() over the whole tail block: at B=1 the decode is dominated by op launches
        tail = s[:, 2 * nb + 1 :].round()
        cnt = tail[:, :N_STRIP].remainder(N_COUNT).long()
        j = 2 * nb + 1 + N_STRIP
        sleeping = s[:, j] > 0.5
        # light_level: ``x - floor(x)`` would send 1.0 to 0.0 (daylight -> midnight), so a period-2 tent
        # (identity on [0, 1]) is applied only outside [0, 1], keeping in-range values bit-exact.
        raw = s[:, j + 1]
        tent = 1.0 - (raw.remainder(2.0) - 1.0).abs()
        light = torch.where((raw >= 0.0) & (raw <= 1.0), raw, tent)
        # the key halves, not the assembled key: assembling is needed only on a lit row
        halves = tail[:, N_STRIP + 2 : N_STRIP + 2 + N_KEY_HALVES].remainder(65536.0)
        return blk, mob, facing, cnt, sleeping, light, halves

    def _tile_ids(self, blk, mob, facing, cnt, sleeping, sty) -> torch.Tensor:
        """``(n, OBS_H + STRIP_H, OBS_W)`` int32: one table row per output tile, world and strip.

        Branches are folded into the ids so one batch may mix appearances and sleeping/awake players:

        * ``mobs_visible = 0`` becomes mob id 0, whose alpha table is exactly zero
          (``tests/unit/test_craftax_atlas.py``);
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
        """The reference's post-process on the world rows; ``px`` is ``(n, OBS_H*t, OBS_W*t, 3)``.

        The strip is excluded because the reference concatenates it after this runs.
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
        """``(...,149)`` -> ``(...,(7+2)*size, 9*size, 3)`` uint8.

        There is no ``scene()``: the frame is a gather from a texture atlas with the alpha blends
        baked in (:class:`_Composite`), strip included, so it is one ``index_select``.

        A ``channel_gain`` other than 1, a sleeping player or a light level below 1 cannot fold into
        the gather, because each acts before the rounding. Those batches gather the float table
        instead; the uint8 table is ``round -> clamp -> uint8`` of it, so a plain row is identical
        either way.
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
        """Gather the uint8 atlas for a plain batch, else the float atlas plus the post-process."""
        n, dev = blk.shape[0], blk.device
        H = (OBS_H + STRIP_H) * t
        tile = self._tile_ids(blk, mob, facing, cnt, sleeping, sty)

        # One index per (frame, tile row, scanline within the tile, tile column) in the output's own
        # row order, so the gather writes the final raster directly.
        rows = tile.unsqueeze(2) * t + torch.arange(t, dtype=torch.int32, device=dev).view(1, 1, t, 1)
        rows = rows.reshape(-1)

        # One host read: can the batch stay on the uint8 gather? Yes when the gain is exactly 1, nothing
        # is asleep, and it is full daylight (``1.0*world + 0.0*night`` is exactly ``world``).
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
        """Composite per frame: the readable definition of what the atlas bakes.

        Used above :data:`ATLAS_MAX_TILE_PX`; ``tests/unit/test_craftax_atlas.py`` asserts the two
        paths agree on every pixel.
        """
        n = blk.shape[0]
        block, ptex, palpha, mtex, malpha = _Assets.get(blk.device, t)
        px = self._tiles_to_frame(block[blk])  # (n,H*t,W*t,3) float32

        # The visibility flags scale the sprite's alpha instead of branching, so a batch can mix
        # settings; zero alpha is exactly the skip.
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

        # uint8 is a half-to-even rounding of the fractional float32 frame, as numpy rounds
        return px.round_().clamp_(0, 255).to(torch.uint8)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Random terrain windows within the reference's mob budget.

        Counts are capped at ``StaticEnvParams`` (3 zombies, 3 cows, 2 skeletons, 3 arrows), since a
        larger state cannot be written into a Craftax ``EnvState``, and no mob stands on the centre
        tile, which would make ``player_visible`` inert on that frame.
        """
        rng = rng or self.rng
        nb = OBS_H * OBS_W
        st = np.zeros((batch, self.state_dim), np.float32)
        terrain = rng.choice(_TERRAIN, size=(batch, OBS_H, OBS_W), p=_TERRAIN_P).astype(np.float32)
        # Half the windows sit against a world edge, padded with OUT_OF_BOUNDS: a real state, and the only
        # block with a flat 3x3 interior. Without it the conformance suite's interior mask is empty and
        # its interior statistic is NaN.
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

        # Counts are sampled directly over 0..9: a random-policy rollout leaves ten of the twelve
        # inventory cells at zero (dead state dimensions). Sound because the strip is a function of the
        # counts alone and the simulator caps every count at 9.
        st[:, nb * 2 + 1 : nb * 2 + 1 + N_STRIP] = rng.integers(0, N_COUNT, size=(batch, N_STRIP)).astype(np.float32)
        # A quarter asleep, stratified by row: ``player_facing`` reaches no pixel while asleep, so row 0 is
        # always awake and row 3 always asleep, which keeps both observable from batch=1 upward.
        sleeping = rng.random(batch) < 0.25
        sleeping[::4] = False
        if batch > 3:
            sleeping[3::4] = True
        st[:, nb * 2 + 1 + N_STRIP] = sleeping.astype(np.float32)

        # ``light_level`` takes the simulator's values (``1 - |cos(pi * ((t/300) % 1 + 0.3))|**3``), with the
        # band assigned by row index: the PRNG-key slots reach no pixel above 0.5, so row 0 is a night row
        # to make every dimension observable even at batch=1.
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

        A tuple rather than a gym env because the renderer is a pure function of ``EnvState``. The
        template comes from a real ``reset`` so unread fields hold valid values. Only the two sizes
        craftax ships textures for are accepted.

        The renderer is not jitted: under ``jax.jit`` XLA fuses the arithmetic differently (up to
        1.5e-05 in float32), which flips uint8 roundings where the picture is compressed towards one
        value (sleep desaturation, night blend, any lit row). It costs about 0.25 s a frame.

        ``jax_threefry_partitionable`` must be on: the two jax threefry paths give different bits for
        the same key.
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
        """Official frames for the given render states: the whole observation, strip included.

        The render state is written back into a full ``EnvState``: the block window goes into the
        middle of the map with the player at its centre, each occupied mob tile becomes one mob of
        its class at the matching absolute position, arrow orientations are recovered through an
        ``arrow_direction`` with the encoded ``(flip, transpose)``, and the sixteen strip counts go
        into ``player_health``/``food``/``drink``/``energy`` and ``inventory``. ``light_level`` and
        ``state_rng`` are written back too, so a night frame is compared against the reference's own
        night frame.
        """
        import jax
        import jax.numpy as jnp

        render, tmpl, tile_px = env

        def _i(field, v):
            """``v`` in the dtype the template's field has, so the pytree stays well typed."""
            return jnp.asarray(v, jnp.asarray(field).dtype)

        s = np.asarray(states, dtype=np.float32).reshape(-1, CraftaxClassic.state_dim)
        nb = OBS_H * OBS_W
        H, W = np.asarray(tmpl.map).shape
        py, px_ = H // 2, W // 2
        r0, c0 = py - OBS_H // 2, px_ - OBS_W // 2
        # mob id -> an ``arrow_direction`` with the orientation that id encodes. Textures are generated as
        # ``(flip, transpose)`` in the order (0,0), (0,1), (1,0), (1,1), and the reference classifies a
        # direction as ``flip = (d0 > 0) or (d1 > 0)``, ``transpose = d1 != 0``, so the inverse is this
        # table, not the obvious pairing (which swaps ids 5 and 6).
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
