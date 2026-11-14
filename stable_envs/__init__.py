"""Batched environments with explicit state, variation and state-driven rendering.

Environment classes are attributes named by their registry key, for example ``stable_envs.PushT``.
"""

from __future__ import annotations

from typing import Any

from . import registry as _registry
from . import tuning
from .__about__ import __version__  # noqa: F401
from ._core.config import get_config, set  # noqa: A004,F401
from ._core.cpus import limit_thread_defaults as _limit_thread_defaults
from .env import Env, State, StepResult, VariationLike  # noqa: F401
from .registry import import_errors, info, list_envs, list_families  # noqa: F401
from .registry import make as _make
from .specs import CompositeSpec, Spec  # noqa: F401

# Size the Numba and Torch thread pools to the CPUs a container quota allows, before a family imports Numba.
_limit_thread_defaults()

# Discover available families; report missing optional dependencies through import_errors().
_registry.discover()


def make(name: str, **kwargs) -> Env:
    """Construct a registered environment by class name.

    ``make("CartPole", num_envs=64)`` is the same as ``stable_envs.CartPole(num_envs=64)``.

    Args:
        name: Class name from ``list_envs``, not a ``reference_id`` such as ``"CartPole-v1"``.
        **kwargs: Forwarded to the constructor, such as ``num_envs``, ``size``, ``device`` and ``seed``.

    Raises:
        KeyError: If ``name`` is not registered. ``import_errors`` shows families that didn't import.
    """
    return _make(name, **kwargs)


def tune(env: Env, *, knobs=(), repeats: int = 5, write: bool = True, verbose: bool = True) -> dict[str, int]:
    """Time the tuning knobs ``env`` reads and cache the fastest values.

    Args:
        env: Environment whose device, ``size`` and ``num_envs`` define the workload.
        knobs: Knobs to sweep (all that ``env`` reads if empty).
        repeats: Timed calls per value.
        write: Save the results to the tuning cache.
        verbose: Print the result table.

    Returns:
        A dict mapping each swept knob to its chosen value.

    Raises:
        ValueError: If ``knobs`` names an unknown knob.
    """
    from ._tune import tune as _tune

    return _tune(env, knobs=knobs, repeats=repeats, write=write, echo=print if verbose else None)


def __getattr__(name: str) -> Any:
    """Resolve environment classes by canonical registry name."""
    if name in _registry.REGISTRY:
        return _registry.REGISTRY[name]
    raise AttributeError(
        f"module 'stable_envs' has no attribute {name!r}. Environments are looked up by class "
        f"name: snv.list_envs() shows the {len(_registry.REGISTRY)} registered and "
        f"snv.import_errors() shows the families that did not import."
    )


def __dir__() -> list[str]:
    return sorted({*__all__, *_registry.REGISTRY})


__all__ = [
    "Env",
    "State",
    "StepResult",
    "VariationLike",
    "Spec",
    "CompositeSpec",
    "make",
    "info",
    "list_envs",
    "list_families",
    "import_errors",
    "set",
    "get_config",
    "tuning",
    "tune",
    "__version__",
]
