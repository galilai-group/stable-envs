"""Everything that exists only because stable-envs and another package must understand each other.

Two directions, and keeping them apart is what the module is for:

  * **stable-worldmodel drives us.** ``StableEnvGym`` presents one stable-envs cell as a gymnasium env
    and ``BatchedEnvPool`` presents ``num_envs`` of them as a vector env, so ``swm.World`` can collect,
    plan and evaluate against a batched GPU renderer through the API it already speaks.
  * **we re-draw somebody else's env.** ``swap_renderer`` keeps the official env's dynamics and
    replaces only its pixels, which is the honest path when the simulator is the reference and only
    the renderer is ours.

``VariationMap`` is the third piece: SWM's variation names and ours describe the same appearance knobs
in different words, and the map is the checked translation between them (``style_from_variation``,
``variation_from_style``, ``check_mapping``, ``swm_variation_space``).

NOTHING HERE IS IMPORTED BY THE CORE. ``env``, ``dynamics``, ``specs``, ``registry`` and ``_config``
know nothing about SWM, and this module imports ``gymnasium``/``stable_worldmodel`` lazily, so a user
with neither installed pays nothing. The ideal end state is that this file SHRINKS as SWM consumes
``Env`` natively.

The implementation lives in the private package ``stable_envs/_interop/`` because it is ~2,450 lines
across the facade and four variation-map tables; this module is its single public face.
"""
from __future__ import annotations

from ._interop.worldmodel import (INFO_KEYS, VARIATION_MAPS, BatchedEnvPool,  # noqa: F401
                                  StableEnvGym, VariationMap, attach_batched_pool, check_mapping,
                                  pusht_state, register_gym_ids, style_from_swm_env,
                                  style_from_variation, swap_renderer, swm_variation_space,
                                  variation_from_style, variation_map)

__all__ = ["INFO_KEYS", "VARIATION_MAPS", "BatchedEnvPool", "StableEnvGym", "VariationMap",
           "attach_batched_pool", "check_mapping", "pusht_state", "register_gym_ids",
           "style_from_swm_env", "style_from_variation", "swap_renderer", "swm_variation_space",
           "variation_from_style", "variation_map"]
