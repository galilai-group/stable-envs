"""Phase 0 — the official FULL-Craftax pixel renderer, instrumented to hand back its intermediates.

Same device as W1's Classic probe: the function body is the *installed* one, taken with
``inspect.getsource`` and transformed by anchored substitution rather than retyped, so a craftax
upgrade that moves a stage makes the anchors fail loudly instead of silently measuring a stale copy.

Capture points, in the order the renderer computes them:

  ``after_blocks``   — the block gather (``full_map_block_textures``), before the item layer
  ``after_items``    — after the ladder/torch layer
  ``after_player``   — after the player sprite
  ``after_mobs``     — after melee + passive + ranged
  ``world``          — after both projectile layers: the last state before any lighting
  ``after_lightmap`` — after the per-tile ``light_map`` multiply
  ``after_night``    — after ``daylight * map + (1 - daylight) * night``
  ``after_sleep``    — after the sleep multiply
  ``inv``            — the 4-row inventory strip, before it is concatenated
  ``frame``          — the official observation
"""
from __future__ import annotations

import inspect
import re

import craftax.craftax.renderer as _ref

_ANCHORS = [
    ("    def render_craftax_pixels(state):\n",
     "    def render_craftax_pixels(state):\n        _cap = {}\n"),
    ("        # Items\n",
     "        _cap[\"after_blocks\"] = map_pixels\n        # Items\n"),
    ("        # Render player\n",
     "        _cap[\"after_items\"] = map_pixels\n        # Render player\n"),
    ("        # Render mobs\n",
     "        _cap[\"after_player\"] = map_pixels\n        # Render mobs\n"),
    ("        def _add_projectile_to_pixels(carry, projectile_index):\n",
     "        _cap[\"after_mobs\"] = map_pixels\n"
     "        def _add_projectile_to_pixels(carry, projectile_index):\n"),
    ("        # Apply darkness (underground)\n",
     "        _cap[\"world\"] = map_pixels\n        # Apply darkness (underground)\n"),
    ("        # Apply night\n",
     "        _cap[\"after_lightmap\"] = map_pixels\n        # Apply night\n"),
    ("        # Apply sleep\n",
     "        _cap[\"after_night\"] = map_pixels\n        # Apply sleep\n"),
    ("        # Render mob map\n",
     "        _cap[\"after_sleep\"] = map_pixels\n        # Render mob map\n"),
    ("        pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)\n",
     "        pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)\n"
     "        _cap[\"inv\"] = inv_pixels\n        _cap[\"frame\"] = pixels\n"),
    ("\n        return pixels\n", "\n        return _cap\n"),
]


def make_instrumented_renderer(block_pixel_size: int, do_night_noise: bool = True):
    src = inspect.getsource(_ref.make_craftax_pixel_renderer)
    src = re.sub(r"^def make_craftax_pixel_renderer",
                 "def _make_instrumented", src, count=1)
    for old, new in _ANCHORS:
        if src.count(old) != 1:
            raise RuntimeError(
                f"instrumentation anchor is not unique in craftax's full renderer source "
                f"({src.count(old)} occurrences): {old!r}. The installed craftax has changed; "
                f"re-derive the anchors rather than trusting this probe.")
        src = src.replace(old, new, 1)
    ns = dict(vars(_ref))
    exec(compile(src, "<instrumented craftax full renderer>", "exec"), ns)
    return ns["_make_instrumented"](block_pixel_size, do_night_noise)
