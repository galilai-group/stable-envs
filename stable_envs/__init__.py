"""stable-envs: a batched environment library with explicit state, dynamics, and variation, plus fast state-driven rendering.

Five nouns: **Env**, **State**, **Action**, **Variation**, **Frame**.
Stateful simulation (``reset`` / ``step``) and explicit-State rendering (``render``).
``VariationLike`` is the type of ``render(..., variation=)``: a mapping of overrides, or a flat / batched row.

Public surface::

    import stable_envs as snv

    env   = snv.PushT(num_envs=64)              # device="auto" → CUDA if available, else CPU
    env   = snv.make("PushT")                   # strings are canonical class names
    state = env.reset(seed=0)                   # State: TensorDict, keys grouped by entity
    out   = env.step(actions)                   # stateful; read out["state"]
    traj  = env.step(action_sequence, frames=None)  # [B, T, A] known actions
    px    = env.render(state)                   # Frame: (B,H,W,3) uint8 on device
    env.close()

``snv.info("PushT")`` and ``snv.list_envs()`` answer what exists without constructing anything;
``snv.set(verbose="INFO", default_device="auto")`` is the whole configuration.

``stable_envs.testing`` is a developer-facing measurement library, not advertised at ``snv.*``.
Policy rollouts, planning, dataset storage and training belong to the sibling stable-* projects
and are not part of this API.
"""

from __future__ import annotations

from typing import Any

from . import registry as _registry
from .__about__ import __version__  # noqa: F401
from ._config import get_config, set  # noqa: A004,F401
from .env import Env, State, StepResult, VariationLike  # noqa: F401
from .registry import import_errors, info, list_envs, list_families  # noqa: F401
from .registry import make as _make
from .specs import CompositeSpec, Spec  # noqa: F401

# Auto-discovery runs at import: ``import stable_envs`` gives you a populated registry and
# ``snv.PushT``, and a family whose optional dependency is missing costs that family alone
# (``snv.import_errors()``).
_registry.discover()


def make(name: str, **kwargs) -> Env:
    """``snv.make("PushT", num_envs=64)``. See :mod:`stable_envs.registry`."""
    return _make(name, **kwargs)


make.__doc__ = (_make.__doc__ or "") + "\n\n    " + (make.__doc__ or "")


def __getattr__(name: str) -> Any:
    """Environment classes — resolved lazily from the registry by canonical class name."""
    if name in _registry.REGISTRY:
        return _registry.REGISTRY[name]
    raise AttributeError(
        f"module 'stable_envs' has no attribute {name!r}. Environments are looked up by their class "
        f"name — snv.list_envs() shows the {len(_registry.REGISTRY)} registered; "
        f"snv.import_errors() shows the families whose optional dependency is missing."
    )


def __dir__() -> list[str]:
    """The documented namespace and the canonical environment class names."""
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
