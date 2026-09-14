"""Craftax — a tile atlas underneath, and an honest accounting of what the state actually costs.

Reference: ``Craftax-Classic-Pixels-v1`` (JAX). The observation is drawn by
``craftax.craftax_classic.renderer.make_craftax_pixel_renderer(7)``: a 7x9 window of the world map
expanded to 7 px per tile, the player sprite composited over the centre tile, then zombies, cows,
skeletons and arrows composited one tile each, then a day/night post-process, then a 2-row inventory
strip. The world layer — everything before the post-process — is a gather from a fixed 17-entry block
atlas plus at most a dozen sprite blits, which is why it belongs in this repo.

**The state that ships with Craftax is a loss, and the docstring says so rather than omitting it.**
Measured (``scratch/pixel_native/probe_craftax.py``): the full ``EnvState`` is 273,276 bytes against a
42,900-byte frame for Craftax, and 20,952 bytes against an 11,907-byte frame for Classic. Storing
``EnvState`` instead of images is **6.37x worse** and **1.76x worse** respectively. The reason is that
``EnvState`` carries the whole world — all nine 48x48 levels of ``map``, ``item_map`` and ``light_map``
for full Craftax — while the frame shows a 9x11 window of one of them. There is no version of "store
the state" that wins if the state is the simulator's state.

What wins is the **render window**: the tiles actually on screen plus the sprites drawn over them. That
is what this env carries — 127 float32 numbers, **508 bytes**, against a 48,384-byte frame at the
default 16 px tile size (**95x**) or a 9,261-byte frame at the 7 px agent size (**18x**).

Parity against the official renderer, measured on 64 sampled states at both tile sizes:
**100% of all pixels exact, mean |diff| 0.000, max |diff| 0.** Throughput on one H200: **707,019
frames/s at batch 16384** (34 GB/s of frame output).

The env is a *partial-observation* renderer, and precisely:

  * it **can** reconstruct the world layer of the observation — terrain, the player's facing, and the
    mobs and arrows on screen — exactly;
  * it **cannot** reconstruct the 2-row inventory strip (rows 49-63 of the 63x63 agent observation).
    That strip is a function of twelve inventory counts and four intrinsics, not of the map window, and
    it is not rendered here. ``reference_render`` crops the official frame to the world layer so the
    comparison is like for like rather than quietly favourable — the 100% figure above is 100% of the
    world layer, not of the observation;
  * it **cannot** reconstruct night frames or a sleeping player. ``light_level < 0.5`` mixes in a noise
    field drawn from ``jax.random.uniform(state.state_rng, ...)``, so reproducing it bit-exactly means
    reimplementing JAX's threefry2x32 counter PRNG; and ``is_sleeping`` desaturates the entire frame,
    which couples the sleeping sprite to a global post-process. Both are excluded, and the exclusion is
    load-bearing: with ``light_level = 1.0`` and ``is_sleeping = False`` the whole post-process chain is
    the identity on the map pixels (``daylight * map + (1 - daylight) * night`` with ``daylight = 1``),
    verified numerically to 0 over 20 states, which is what makes exact parity possible at all;
  * it carries **one mob per tile**. Craftax's own ``mob_map`` keeps zombies, cows and skeletons off
    each other's tiles, but an arrow may share a tile with a mob, and the reference then composites
    both. That state is not representable here and the sampler does not generate it.

Two measured facts that shaped the implementation:

  * **The alphas are not binary.** They take exactly {0, 128/255, 1}, so the composite is a real blend
    and the frame is fractional-valued float32. This repo's frames are uint8, so ``render`` rounds
    (half-to-even, matching numpy) and ``reference_render`` rounds the official float the same way. The
    frame is a rounding of the reference, not a lossless view of it, and that is stated rather than
    hidden behind an all-pixel percentage.
  * **The player tables are zero away from the centre tile**, asserted during extraction, so compositing
    the player over the whole frame (what the reference does) and into one tile (what this does) are the
    same arithmetic — ``x*(1-0) + 0*0 == x`` exactly in float32.

Out-of-range class ids **wrap** rather than clamp. Real states are always in range, so the rule never
fires in practice; it exists because a clamping decoder makes a cell that already holds the last class
insensitive to the conformance suite's +25 probe, which would be reported as a dead state dimension
when it is an artifact of the probe.
"""
from __future__ import annotations

import base64
import zlib

import numpy as np
import torch

from ..core.env import StableEnv, register
from ..spaces import Box, Dict, Discrete

TILE_PX_AGENT = 7
TILE_PX_IMG = 16
OBS_H, OBS_W = 7, 9
N_BLOCK = 17
N_MOB = 8            #: none, zombie, cow, skeleton, arrow x 4 orientations
N_PLAYER = 4         #: facing left, right, up, down

#: block id -> name, straight from craftax.craftax_classic.constants.BlockType
BLOCK_NAMES = ("invalid", "out_of_bounds", "grass", "water", "stone", "tree", "wood", "path", "coal", "iron", "diamond", "crafting_table", "furnace", "sand", "lava", "plant", "ripe_plant")

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

_PLAYER_TEX_7_SHAPE = (4, 7, 7, 3)
_PLAYER_TEX_7_B64 = (
    "eNpjYCAGxDszMKgD8QMnBqoBSsysd4ZgEHjgyMAgD2QbOONmb3CE2PPDCbc5yOx/QHV+QHYEHnOQ1RDjTjcgrQnEokAsBMRngPr7gPgclHYB4iKoeXbO"
    "xIUDspnIANlMUgEtzBzMaQlXXONiI8cpLoArneBKS7jSIS4/EpOWiHHnUE9LpPqRGHOIiQtqpSVc5RIlaYmYconUtDRaLg3+cmm4piVq5U1caQnZHMJp"
    "CQAdLbtp"
)

_PLAYER_ALPHA_7_SHAPE = (4, 7, 7, 3)
_PLAYER_ALPHA_7_B64 = (
    "eNpjYCAGNNgjMLUAJWYi6yUVU8scYtxPvpmNDQxwTHoYkmoOMXqxi1Ni/vBKS7ROV6NpiXZpiZK4HsxpabRcGi2XhlNaosQc8s1ExgDVRBDC"
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
    "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"
)

_PLAYER_TEX_16_SHAPE = (4, 16, 16, 3)
_PLAYER_TEX_16_B64 = (
    "eNrtmkFqAlEMht8BppuC4KLQpbgQegELzhOXXbvrHYrLgjfwIF31AKUnkJ7BY3RVjDhCOpCaR5KZTMmDxwsy4JdhNJ8/pqS13uqUNrDHNa/2tl5ySmvY"
    "j7CfYU9gH+rf9RzOG9gfwW+6vDFv83mf1gzOz4blB86n5r7j+rBI6R7Oh1ZNXf++OPf6rdjja8N8l2l+ilNS417w6xJmzv237kXCrMXPeWZKn6sTV3WF"
    "2TN/aS+Y/xbqPdS7urz+as6lIj+nF8y/gnoKe5R5zPh6ip+aI5J51O7lUlM8VF3Kbz07ODyS661X8PfLX+rGnl06/D/4/6v/U/Ndy//xTMezXrKo97Xw"
    "ZIofu7HkPlNOrtUXxS/ppdR1Jf7M4S/txTM/dkjO80/xa/k/l7/K8u9zymck/v8Xf5Xl84j6Dtfyf+r3Rfh/8If/d99LOxO7fB49+z83x6vy8Jg99SJl"
    "7rsXPCstaq3Vpf9r5f99+b8k//fgzxb589D5u/R/C/4u/d86/7f2/8j/gz/8P/L/yP8j/x9i/m/x/5/w//D/8P/wf2/+b5GZ4PzH2v81M6uhZ3F9Z6H4"
    "eS7Nbz37Pyc/13HpI6XA2tY="
)

_PLAYER_ALPHA_16_SHAPE = (4, 16, 16, 3)
_PLAYER_ALPHA_16_B64 = (
    "eNrtmkEOgCAMBPmZX8OfezaRA9CFLQ5J77NNtFNCKVGnXn3lduD3z5KltzOVkTkqiwNzbxY3tjP571o+yyNLFP+eLG8e+HVzqo+txanmbzOM93Zl/9X8"
    "6hnd+3/TsOHP8OP/7ALsAmNZZlzdea+BP9ZXc/DH+X8Wfvwf/8f/8ec/uOhJ/gy/j0tn31kU+4v6+z2T39s/c/Ovvf/Pwo//4//4P/4PP/f/+D/+j//j"
    "z/i/4/uftff/+zn1+4uWOTu/21yeYXsAOmmoGg=="
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
#: is no official frame at that size to compare against.
REFERENCE_TILE_PX = (TILE_PX_AGENT, TILE_PX_IMG)
_TABLES = {
    TILE_PX_AGENT: ((_BLOCK_TEX_7_B64, _BLOCK_TEX_7_SHAPE), (_PLAYER_TEX_7_B64, _PLAYER_TEX_7_SHAPE),
                    (_PLAYER_ALPHA_7_B64, _PLAYER_ALPHA_7_SHAPE), (_MOB_TEX_7_B64, _MOB_TEX_7_SHAPE),
                    (_MOB_ALPHA_7_B64, _MOB_ALPHA_7_SHAPE)),
    TILE_PX_IMG: ((_BLOCK_TEX_16_B64, _BLOCK_TEX_16_SHAPE), (_PLAYER_TEX_16_B64, _PLAYER_TEX_16_SHAPE),
                  (_PLAYER_ALPHA_16_B64, _PLAYER_ALPHA_16_SHAPE), (_MOB_TEX_16_B64, _MOB_TEX_16_SHAPE),
                  (_MOB_ALPHA_16_B64, _MOB_ALPHA_16_SHAPE)),
}


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
            tabs = [torch.as_tensor(_unpack(b64, shape).copy(), device=device)
                    for b64, shape in _TABLES[src]]
            if t != src:
                ix = ((torch.arange(t, device=device, dtype=torch.float32) + 0.5)
                      * src / t).long().clamp_(0, src - 1)
                tabs = [a[:, ix][:, :, ix] for a in tabs]
            cls._cache[key] = tuple(tabs)
        return cls._cache[key]


#: Player slot meaning "no player is drawn on this tile". ``player_visible = 0`` and every non-centre
#: tile both resolve to it, and it is exactly the identity: the reference computes
#: ``tile*(1 - 0) + player_tex*0``, which is ``tile + 0`` in float32, and so does slot 4.
NO_PLAYER = N_PLAYER
#: (player slot, block, mob) composites in the atlas: 5 * 17 * 8.
N_COMPOSITE = (N_PLAYER + 1) * N_BLOCK * N_MOB

#: Largest tile edge the baked atlas is used at. The atlas holds ``N_COMPOSITE`` tiles, so it costs
#: ``680 * t*t*3`` bytes as uint8 and four times that as float32 — 522 KiB / 2.0 MiB at the default
#: 16 px, but 134 MiB / 534 MiB at ``size=256``. Above this edge the atlas stops being a small table
#: that stays resident and becomes the largest allocation in the process, so ``render`` falls back to
#: :meth:`CraftaxClassic._render_blend`, which is the pre-bake implementation and is asserted
#: pixel-identical to the atlas path by ``tests/unit/test_craftax_atlas.py``.
ATLAS_MAX_TILE_PX = 64


class _Composite:
    """The **composited** tile atlas: every (player slot, block, mob) tile the renderer can draw.

    The frame is a gather and nothing else. The reference — and this env before Step 6 — composited
    per frame: it gathered the block texture into a ``(B, 7t, 9t, 3)`` float32 buffer, blended the
    player into the centre tile, then blended a full-frame mob texture and a full-frame mob alpha
    over all of it, applied the channel gain and rounded. Every one of those steps is a
    ``B x 7t x 9t x 3`` float32 temporary, and there are about a dozen of them: **1.94 MB per 16 px
    frame against a 48.4 KB frame**. That is why throughput FELL with batch — 1,876 f/s at B=16 down
    to 1,113 f/s at B=256 on one core of ``node1831`` — rather than rising: 16 frames of temporaries
    are 31 MB and still fit that node's ~35.8 MiB of last-level cache, and 256 frames are 495 MB and
    do not. Measured in ``examples/results/step6/tilemap/profile_before/``.

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

    _cache: dict = {}
    #: How many (device, tile size) tables to keep. Both forms of one table come to ~40 MiB at
    #: :data:`ATLAS_MAX_TILE_PX`, and a caller sweeping sizes would otherwise retain one per size
    #: for the life of the process. Dropping the whole dict rather than evicting one entry is the
    #: rule ``render/tilemap.py``'s recoloured-atlas cache already uses, and it is safe because a
    #: render holds its own reference to the table it is gathering from.
    _MAX_ENTRIES = 8

    @staticmethod
    def _build(device: torch.device, t: int) -> torch.Tensor:
        """``(N_COMPOSITE * t, t * 3)`` float32, row ``slot * t + y``.

        Laid out one *tile row* per row so the gather in :meth:`CraftaxClassic.render` can address a
        whole output scanline of one tile with a single index and write it where it finally belongs.
        """
        block, ptex, palpha, mtex, malpha = _Assets.get(device, t)
        # player over block -> (5, 17, t, t, 3); slot 4 is the block untouched
        with_player = block.unsqueeze(0).repeat(N_PLAYER + 1, 1, 1, 1, 1)
        ap = palpha.view(N_PLAYER, 1, t, t, 3)
        with_player[:N_PLAYER] = (with_player[:N_PLAYER] * (1 - ap)
                                  + ptex.view(N_PLAYER, 1, t, t, 3) * ap)
        # mob over that -> (5, 17, 8, t, t, 3)
        am = malpha.view(1, 1, N_MOB, t, t, 3)
        comp = (with_player.unsqueeze(2) * (1 - am)
                + mtex.view(1, 1, N_MOB, t, t, 3) * am)
        return comp.reshape(N_COMPOSITE * t, t * 3).contiguous()

    @classmethod
    def get(cls, device: torch.device, t: int, want_float: bool):
        """``(f32_or_None, u8)``. ``f32`` is materialised only once a gain path has asked for it."""
        key = (str(device), int(t))
        have = cls._cache.get(key)
        if have is not None and (have[0] is not None or not want_float):
            return have
        f32 = cls._build(device, t)
        u8 = have[1] if have is not None else f32.round().clamp_(0, 255).to(torch.uint8)
        if have is None and len(cls._cache) >= cls._MAX_ENTRIES:
            cls._cache.clear()
        cls._cache[key] = (f32 if want_float else None, u8)
        return cls._cache[key]


@register
class CraftaxClassic(StableEnv):
    """Craftax Classic's world layer: a 7x9 block window, the player, and one mob per tile."""

    name = "craftax-classic"
    state_dim = OBS_H * OBS_W * 2 + 1
    state_names = (tuple(f"block_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
                   + tuple(f"mob_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
                   + ("player_facing",))
    #: tile edge in pixels; the frame is (7*size, 9*size, 3). Craftax ships textures at 7 px (the size
    #: the agent observation is drawn at) and 16 px, and both are exact against the official renderer.
    #: The default is 16 because at 7 px Craftax's dithered art leaves no flat 3x3 neighbourhood in the
    #: whole frame, the conformance suite's interior mask is then empty, and its interior statistic
    #: comes back NaN — which no floor can satisfy, however exact the renderer is.
    default_size = TILE_PX_IMG
    reference_id = "Craftax-Classic-Pixels-v1"

    def make_variation_space(self) -> Dict:
        """Knobs that are the identity at their defaults and genuinely move pixels off them.

        There is deliberately no per-block colour knob. Craftax's blocks are 16x16 pixel-art PNGs
        resampled to 7 px, not procedurally drawn shapes, so a "grass colour" would have to invent a
        recolouring rule the reference does not have. ``channel_gain`` is the one global recolouring
        that is exactly the identity at 1.0; the two visibility flags are ablations that a world-model
        experiment actually wants (render the same state with and without the agent, or without mobs).
        """
        return Dict({
            "channel_gain": Box(low=0.0, high=2.0, shape=(3,), dtype=np.float32,
                                init_value=[1.0, 1.0, 1.0]),
            "player_visible": Discrete(2, init_value=1),
            "mobs_visible": Discrete(2, init_value=1),
        })

    @staticmethod
    def _tiles_to_frame(x: torch.Tensor) -> torch.Tensor:
        """(n,H,W,t,t,3) tile-major -> (n,H*t,W*t,3) raster."""
        n, H, W, t, _, _ = x.shape
        return x.permute(0, 1, 3, 2, 4, 5).reshape(n, H * t, W * t, 3)

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,127)`` -> ``(...,7*size,9*size,3)`` uint8, in the reference's compositing order.

        There is no ``scene()``: the frame is a gather from a texture atlas with two alpha blends over
        it, not a rasterisation of vector primitives — and since Step 6 the two blends are baked into
        the atlas (:class:`_Composite`) instead of being redone per frame, so at the default
        appearance the whole render is one ``index_select`` straight into the uint8 output.

        Three things fold into the gather index rather than into a branch, so a batch may hold any
        mixture of them:

        * ``mobs_visible = 0`` becomes mob id 0, whose alpha table is exactly zero (asserted at build
          time by ``tests/unit/test_craftax_atlas.py``), which is what scaling the alpha to zero did;
        * ``player_visible = 0`` becomes player slot :data:`NO_PLAYER`;
        * every non-centre tile is player slot :data:`NO_PLAYER` too.

        ``channel_gain`` cannot fold in — it multiplies before the rounding — so a batch carrying any
        gain other than exactly 1 gathers the pre-rounding float composite instead and then applies
        the gain, which is still one gather and three elementwise passes rather than a dozen.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        n = s.shape[0]
        t = int(size or self.size)
        if supersample not in (None, 1):
            raise ValueError(
                f"CraftaxClassic.render: supersample={supersample} is meaningless here — the frame is a "
                f"gather from a bitmap atlas, so there is no sub-pixel coverage to estimate.")
        sty = self.resolve_style(style, variation, batch=n)

        nb = OBS_H * OBS_W
        blk = s[:, :nb].round().remainder(N_BLOCK).long().view(n, OBS_H, OBS_W)
        mob = s[:, nb:nb + nb].round().remainder(N_MOB).long().view(n, OBS_H, OBS_W)
        pid = s[:, nb + nb].round().remainder(N_PLAYER).long()

        if t > ATLAS_MAX_TILE_PX:
            img = self._render_blend(blk, mob, pid, t, sty)
        else:
            img = self._render_atlas(blk, mob, pid, t, sty, out)
        if out is not None and img.data_ptr() != out.data_ptr():
            out.view(n, OBS_H * t, OBS_W * t, 3).copy_(img)
            img = out.view(n, OBS_H * t, OBS_W * t, 3)
        return img.view(*lead, *img.shape[1:]) if lead else img

    def _render_atlas(self, blk, mob, pid, t, sty, out) -> torch.Tensor:
        """The gather. One ``index_select`` per frame at the default gain, and no float buffer."""
        n = blk.shape[0]
        dev = blk.device
        vis_m = (sty["mobs_visible"] > 0.5).view(n, 1, 1)
        vis_p = (sty["player_visible"] > 0.5).view(n)
        m = torch.where(vis_m, mob, torch.zeros_like(mob))
        slot = torch.full((n, OBS_H, OBS_W), NO_PLAYER, dtype=torch.long, device=dev)
        cr, cc = OBS_H // 2, OBS_W // 2
        slot[:, cr, cc] = torch.where(vis_p, pid, torch.full_like(pid, NO_PLAYER))
        tile = ((slot * N_BLOCK + blk) * N_MOB + m).to(torch.int32)      # (n, H, W)

        # One index entry per (frame, tile row, output scanline within the tile, tile column), which
        # IS the output's own row order — so the gather writes the final raster directly instead of
        # building a tile-major buffer and transposing it.
        rows = (tile.unsqueeze(2) * t
                + torch.arange(t, dtype=torch.int32, device=dev).view(1, 1, t, 1))
        rows = rows.reshape(-1)

        gain = sty["channel_gain"]
        if bool(torch.all(gain == 1.0)):
            atlas = _Composite.get(dev, t, want_float=False)[1]
            img = (out.view(n, OBS_H * t, OBS_W * t, 3) if out is not None
                   else torch.empty(n, OBS_H * t, OBS_W * t, 3, dtype=torch.uint8, device=dev))
            torch.index_select(atlas, 0, rows, out=img.view(-1, t * 3))
            return img
        f32 = _Composite.get(dev, t, want_float=True)[0]
        px = torch.index_select(f32, 0, rows).view(n, OBS_H * t, OBS_W * t, 3)
        px = px * gain.to(px.dtype).view(n, 1, 1, 3)
        return px.round_().clamp_(0, 255).to(torch.uint8)

    def _render_blend(self, blk, mob, pid, t, sty) -> torch.Tensor:
        """Composite per frame, the way this renderer did before Step 6.

        Kept because the baked atlas costs ``680 * t*t*3`` bytes and stops being a table worth keeping
        somewhere above :data:`ATLAS_MAX_TILE_PX` (see there). It is the readable definition of what
        the atlas bakes, and ``tests/unit/test_craftax_atlas.py`` asserts the two agree on every
        pixel — including the two visibility flags mixed within one batch, out-of-range class ids and
        a random gain.
        """
        n = blk.shape[0]
        block, ptex, palpha, mtex, malpha = _Assets.get(blk.device, t)
        px = self._tiles_to_frame(block[blk])                              # (n,H*t,W*t,3) float32

        # The two visibility flags scale the sprite's own alpha instead of branching, so a batch can
        # hold both settings. Zeroing the alpha is exactly the skip: ``sl*(1-0) + x*0`` is ``sl`` in
        # float32, not merely close to it.
        cr, cc = OBS_H // 2, OBS_W // 2
        sl = px[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t, :]
        a = palpha[pid] * (sty["player_visible"] > 0.5).to(px.dtype).view(n, 1, 1, 1)
        px[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t, :] = sl * (1 - a) + ptex[pid] * a

        a = (self._tiles_to_frame(malpha[mob])
             * (sty["mobs_visible"] > 0.5).to(px.dtype).view(n, 1, 1, 1))
        px = px * (1 - a) + self._tiles_to_frame(mtex[mob]) * a

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
            budget = [(MOB_ZOMBIE, int(rng.integers(0, MAX_ZOMBIES + 1))),
                      (MOB_COW, int(rng.integers(0, MAX_COWS + 1))),
                      (MOB_SKELETON, int(rng.integers(0, MAX_SKELETONS + 1))),
                      (None, int(rng.integers(1, MAX_ARROWS + 1)))]
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
        st[:, nb * 2] = rng.integers(0, N_PLAYER, size=batch).astype(np.float32)
        return st

    # reference bridge

    @staticmethod
    def reference_env(size: int = TILE_PX_IMG):
        """The official Craftax renderer at ``size`` px per tile, plus a template ``EnvState``.

        Returned as a tuple rather than a gym env because Craftax is a functional JAX env: the renderer
        is a pure function of ``EnvState`` and needs no instance. The template comes from a real
        ``reset`` so every field the renderer does not read still holds a valid value.

        Only the two sizes craftax ships textures for are accepted. Anything else has no official frame
        to be compared against, and inventing one by resampling the reference would be comparing this
        renderer against a second implementation of itself.
        """
        if int(size) not in REFERENCE_TILE_PX:
            raise ValueError(
                f"craftax draws at {REFERENCE_TILE_PX[0]} px (the agent observation) or "
                f"{REFERENCE_TILE_PX[1]} px per tile; there is no official frame to compare a "
                f"{size} px render against")
        import jax
        from craftax.craftax_classic.renderer import make_craftax_pixel_renderer
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
        _obs, template = env.reset(jax.random.PRNGKey(0), env.default_params)
        return (jax.jit(make_craftax_pixel_renderer(int(size))), template, int(size))

    @staticmethod
    def reference_render(env, states) -> np.ndarray:
        """Official frames for the given render states, cropped to the world layer.

        The render state is written back into a full ``EnvState``: the block window goes into the middle
        of the map with the player standing at its centre, each occupied mob tile becomes one mob of the
        right class at the matching absolute position, and arrow orientations are recovered by choosing
        an ``arrow_direction`` whose ``(flip, transpose)`` classification is the one encoded. Daylight
        and not-sleeping are forced, which makes the post-process the identity on the map pixels.
        """
        import jax
        import jax.numpy as jnp
        render, tmpl, tile_px = env
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
            mob = np.round(row[nb:2 * nb]).astype(np.int32).reshape(OBS_H, OBS_W) % N_MOB
            pid = int(np.round(row[2 * nb])) % N_PLAYER

            m = np.asarray(tmpl.map).copy()
            m[r0:r0 + OBS_H, c0:c0 + OBS_W] = blk
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
                return tmpl_mobs.replace(position=jnp.asarray(pos, jnp.int32),
                                         mask=jnp.asarray(mask, bool))

            st = tmpl.replace(
                map=jnp.asarray(m, jnp.asarray(tmpl.map).dtype),
                player_position=jnp.asarray([py, px_], jnp.int32),
                player_direction=jnp.asarray(pid + 1, jnp.asarray(tmpl.player_direction).dtype),
                is_sleeping=jnp.bool_(False),
                light_level=jnp.float32(1.0),
                zombies=mob(tmpl.zombies, zp, zm),
                cows=mob(tmpl.cows, cp, cm),
                skeletons=mob(tmpl.skeletons, sp, sm),
                arrows=mob(tmpl.arrows, ap, am),
                arrow_directions=jnp.asarray(ad, jnp.asarray(tmpl.arrow_directions).dtype))
            frame = np.asarray(jax.device_get(render(st)))[: OBS_H * tile_px]
            out.append(np.round(frame).clip(0, 255).astype(np.uint8))
        return np.stack(out)
