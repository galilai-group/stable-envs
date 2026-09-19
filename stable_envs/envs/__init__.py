"""Env auto-discovery: drop a module in this package and its ``@register``ed classes are available.

There is no hand-maintained import list. Adding an env meant editing a shared ``__init__.py``, which is
a two-place step (write the env, remember to register it) and, with several people adding envs at once,
a file everyone edits and nobody owns. Auto-discovery removes both problems.

Failures are recorded, never silent. Most env modules need an optional dependency — ``gym_pusht`` and
``pymunk`` for PushT, ``minigrid`` for the grid family, ``mujoco``/``ogbench`` for the mazes — and a
machine without one should lose exactly that env, not the whole package. ``import_errors()`` reports
what failed and why, and the conformance suite prints it, so "my env vanished" is answerable in one
line instead of being mistaken for a registration bug.

The walk itself now lives in ``stable_envs/registry.py`` (Step 9), because discovery is a registry
operation and the registry is what the rest of the package asks. This module keeps the two names it
has always exported so that ``stable_envs.envs.import_errors()`` keeps answering.
"""

from __future__ import annotations

from ..registry import discover, import_errors  # noqa: F401

discover()
