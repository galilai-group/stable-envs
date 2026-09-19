"""stable-envs: famous research environments as GPU-batched, state-driven renderers.

The claim the repo is built on: for a large fraction of the environments the field trains on, a frame
is a pure function of a compact state vector. If that is true, an image dataset is redundant with a
state dataset that is orders of magnitude smaller, and the frames can be regenerated on the GPU faster
than they could be read from disk. PushT is 5 floats per step against a 96x96x3 frame — 1,382x at 96px,
7,526x at 224px — and renders at 3.2 M frames/s end-to-end on one H200 (5.6 M for the rasterizer alone).

Public surface::

    import stable_envs as snv

    env   = snv.PushT(num_envs=64, device="cuda")   # concrete classes are canonical
    env   = snv.make("swm/PushT-v1", device="cuda") # strings are for configs and interop
    state = env.reset(seed=0)                       # a TensorDict, keys grouped by entity
    out   = env.step(actions)                       # stateful
    nxt   = env.transition(state, action)           # functional, pure in `state`
    traj  = env.unroll(state, action_sequence)      # open loop, known actions
    px    = env.render(state)                       # (B,H,W,3) uint8 on device
    env.close()

``snv.spec("PushT")`` and ``snv.list_envs()`` answer what exists without constructing anything;
``snv.set(verbose="INFO")`` and ``snv.get_config()`` are the whole configuration.

``import stable_envs.testing``, ``stable_envs.rollout``, ``stable_envs.data`` and
``stable_envs.interop`` are package imports: they resolve, and they are not advertised in
``dir(snv)``. ``Env.sample`` / ``rollout.py`` stay in the package and are not first-class.
"""
from __future__ import annotations

from typing import Any

from .__about__ import __version__  # noqa: F401
from ._config import get_config, set  # noqa: A004,F401
from .dynamics import Dynamics  # noqa: F401
from .env import Env, State, StepResult  # noqa: F401
from .registry import (SWM_IDS, import_errors, list_envs, list_families,  # noqa: F401
                       make as _make, spec)
from .specs import CompositeSpec, Spec  # noqa: F401
from . import registry as _registry

# Auto-discovery runs at import: ``import stable_envs`` gives you a populated registry and
# ``snv.PushT``, and a family whose optional dependency is missing costs that family alone
# (``snv.import_errors()``).
_registry.discover()

#: Names that still RESOLVE off ``snv.*`` but are not advertised in ``dir(snv)``. They are helpers
#: tests and notebooks reach for; they are not the happy path.
_HELPER_ATTRS = {
    "Scene2D": ("stable_envs.render.scene", "Scene2D"),
    "SceneBuilder": ("stable_envs.render.scene", "SceneBuilder"),
    "StyleTensor": ("stable_envs._style", "StyleTensor"),
    "StateDataset": ("stable_envs.data", "StateDataset"),
    "TrajectoryDataset": ("stable_envs.data", "TrajectoryDataset"),
    "collate_batched": ("stable_envs.data", "collate_batched"),
    "render_states": ("stable_envs.data", "render_states"),
    "BasePolicy": ("stable_envs.rollout", "BasePolicy"),
    "RandomPolicy": ("stable_envs.rollout", "RandomPolicy"),
    "ScriptedPolicy": ("stable_envs.rollout", "ScriptedPolicy"),
    "sample": ("stable_envs.rollout", "sample"),
    "check_states": ("stable_envs.env", "check_states"),
    "FIDELITY_TIERS": ("stable_envs.env", "FIDELITY_TIERS"),
    "RENDER_DECODERS": ("stable_envs.env", "RENDER_DECODERS"),
    "registry_table": ("stable_envs.registry", "registry_table"),
    "format_registry": ("stable_envs.registry", "format_registry"),
}


def make(name: str, **kwargs) -> Env:
    """``snv.make("PushT", num_envs=64)``. See :mod:`stable_envs.registry`."""
    return _make(name, **kwargs)


make.__doc__ = (_make.__doc__ or "") + "\n\n    " + (make.__doc__ or "")


def __getattr__(name: str) -> Any:
    """Environment classes and unadvertised helpers — resolved lazily.

    ``snv.PushT`` is a lookup in the registry rather than a hand-maintained import list, so the
    top-level namespace and the registry cannot disagree and adding an environment stays a one-file
    change.
    """
    import importlib

    if name in _registry.REGISTRY:
        return _registry.REGISTRY[name]
    if name in _HELPER_ATTRS:
        mod, attr = _HELPER_ATTRS[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(
        f"module 'stable_envs' has no attribute {name!r}. Environments are looked up by their class "
        f"name — snv.list_envs() shows the {len(_registry.REGISTRY)} registered; "
        f"snv.import_errors() shows the families whose optional dependency is missing.")


def __dir__() -> list[str]:
    """The documented namespace and the environments.

    Built from ``__all__`` rather than from ``globals()``, because Python itself binds every imported
    submodule as an attribute of its package (importing ``stable_envs.spaces`` anywhere puts
    ``spaces`` in this module's globals), so a ``globals()``-based listing advertises whatever the
    process happened to import. The environments are added because the concrete classes ARE the
    primary public API: ``snv.PushT`` must be completable.
    """
    return sorted({*__all__, *_registry.REGISTRY})


__all__ = ["Env", "State", "StepResult", "Spec", "CompositeSpec", "Dynamics", "make", "spec",
           "list_envs", "list_families", "import_errors", "set", "get_config", "SWM_IDS",
           "__version__"]
