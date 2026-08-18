"""Env auto-discovery: drop a module in this package and its ``@register``ed classes are available.

There is no hand-maintained import list. Adding an env meant editing a shared ``__init__.py``, which is
a two-place step (write the env, remember to register it) and, with several people adding envs at once,
a file everyone edits and nobody owns. Auto-discovery removes both problems.

Failures are recorded, never silent. Most env modules need an optional dependency — ``gym_pusht`` and
``pymunk`` for PushT, ``minigrid`` for the grid family, ``mujoco``/``ogbench`` for the mazes — and a
machine without one should lose exactly that env, not the whole package. ``import_errors()`` reports
what failed and why, and the conformance suite prints it, so "my env vanished" is answerable in one
line instead of being mistaken for a registration bug.
"""
from __future__ import annotations

import importlib
import pkgutil

#: module name -> the exception that stopped it importing
_IMPORT_ERRORS: dict[str, str] = {}


def _discover() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if mod.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{mod.name}")
        except Exception as e:                     # optional dependency, or a broken env module
            _IMPORT_ERRORS[mod.name] = f"{type(e).__name__}: {e}"


def import_errors() -> dict[str, str]:
    """``module -> error`` for env modules that could not be imported."""
    return dict(_IMPORT_ERRORS)


_discover()
