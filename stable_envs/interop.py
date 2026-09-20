"""The public cross-package facade for stable-envs.

Two directions, and keeping them apart is what the module is for:

  * **IDs resolve without constructing an environment.** ``SWM_IDS`` and
    ``canonical_names`` expose the certified stable-worldmodel-to-stable-envs mapping.
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
know nothing about SWM. The ID mapping is pure Python, and the adapter implementation (and therefore
``gymnasium``) is imported only when an adapter name is first used; ``stable_worldmodel`` is imported
only by operations that inspect a live SWM environment. The ideal end state is that this file SHRINKS
as SWM consumes ``Env`` natively.

The implementation lives in the private package ``stable_envs/_interop/`` because it is ~2,450 lines
across the facade and four variation-map tables; this module is its single public face.
"""

from __future__ import annotations

import importlib
from typing import Any

from .registry import SWM_IDS, SYMBOLIC_IDS, canonical_names  # noqa: F401

_ADAPTER_EXPORTS = frozenset(
    {
        "INFO_KEYS",
        "VARIATION_MAPS",
        "BatchedEnvPool",
        "StableEnvGym",
        "VariationMap",
        "attach_batched_pool",
        "check_mapping",
        "pusht_state",
        "register_gym_ids",
        "style_from_swm_env",
        "style_from_variation",
        "swap_renderer",
        "swm_variation_space",
        "variation_from_style",
        "variation_map",
    }
)


def __getattr__(name: str) -> Any:
    """Load optional gym/SWM glue only when a caller asks for it."""
    if name not in _ADAPTER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    implementation = importlib.import_module("stable_envs._interop.worldmodel")
    value = getattr(implementation, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = ["SWM_IDS", "SYMBOLIC_IDS", "canonical_names", *_ADAPTER_EXPORTS]
