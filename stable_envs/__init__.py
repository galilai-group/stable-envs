from __future__ import annotations

from typing import Any

from . import registry as _registry
from .__about__ import __version__  # noqa: F401
from ._config import get_config, set  # noqa: A004,F401
from .env import Env, State, StepResult, VariationLike  # noqa: F401
from .registry import import_errors, info, list_envs, list_families  # noqa: F401
from .registry import make as _make
from .specs import CompositeSpec, Spec  # noqa: F401

# Discover available families; report missing optional dependencies through import_errors().
_registry.discover()


def make(name: str, **kwargs) -> Env:
    """Construct a registered environment, for example make("PushT", num_envs=64)."""
    return _make(name, **kwargs)


make.__doc__ = (_make.__doc__ or "") + "\n\n    " + (make.__doc__ or "")


def __getattr__(name: str) -> Any:
    """Resolve environment classes by canonical registry name."""
    if name in _registry.REGISTRY:
        return _registry.REGISTRY[name]
    raise AttributeError(
        f"module 'stable_envs' has no attribute {name!r}. Environments are looked up by their class "
        f"name — snv.list_envs() shows the {len(_registry.REGISTRY)} registered; "
        f"snv.import_errors() shows the families whose optional dependency is missing."
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
    "__version__",
]
