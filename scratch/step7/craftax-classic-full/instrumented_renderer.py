"""Phase 0 — the official Classic pixel renderer, instrumented to hand back its intermediates.

The function body is the *installed* one, taken with ``inspect.getsource`` and transformed by
anchored string substitution rather than retyped, so a craftax upgrade that moves a stage makes the
anchors fail loudly instead of silently measuring a stale copy of the renderer.

Capture points, in the order the renderer computes them:

  ``world``        — pixels after block gather + player + zombies + cows + skeletons + arrows,
                     immediately before ``# Apply night``. This is exactly what
                     ``stable_envs.envs.craftax`` reproduces today.
  ``noise_scaled`` — ``jax.random.uniform(state.state_rng, (H, W)) * 95 + 32``
  ``after_night``  — after ``daylight * map + (1 - daylight) * night_pixels``
  ``after_sleep``  — after the ``is_sleeping`` desaturation
  ``inv``          — the inventory strip, before it is concatenated
  ``frame``        — the official observation
"""
from __future__ import annotations

import inspect
import re

import craftax.craftax_classic.renderer as _ref

_ANCHORS = [
    ("    def render_craftax_pixels(state):\n",
     "    def render_craftax_pixels(state):\n        _cap = {}\n"),
    ("        # Apply night\n",
     "        _cap[\"world\"] = map_pixels\n        # Apply night\n"),
    ("        night_static_mask = (",
     "        _cap[\"noise_scaled\"] = night_with_static\n        night_static_mask = ("),
    ("        # Apply sleep\n",
     "        _cap[\"after_night\"] = map_pixels\n        # Apply sleep\n"),
    ("        # Render mob map\n",
     "        _cap[\"after_sleep\"] = map_pixels\n        # Render mob map\n"),
    ("        pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)\n",
     "        pixels = jnp.concatenate([map_pixels, inv_pixels], axis=0)\n"
     "        _cap[\"inv\"] = inv_pixels\n        _cap[\"frame\"] = pixels\n"),
    ("\n        return pixels\n", "\n        return _cap\n"),
]


def make_instrumented_renderer(block_pixel_size: int):
    src = inspect.getsource(_ref.make_craftax_pixel_renderer)
    src = re.sub(r"^def make_craftax_pixel_renderer",
                 "def _make_instrumented", src, count=1)
    for old, new in _ANCHORS:
        if src.count(old) != 1:
            raise RuntimeError(
                f"instrumentation anchor is not unique in craftax's renderer source "
                f"({src.count(old)} occurrences): {old!r}. The installed craftax has changed; "
                f"re-derive the anchors rather than trusting this probe.")
        src = src.replace(old, new, 1)
    ns = dict(vars(_ref))
    exec(compile(src, "<instrumented craftax renderer>", "exec"), ns)
    return ns["_make_instrumented"](block_pixel_size)
