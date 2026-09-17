"""Deprecated import path. ``stable_envs.envs.simple_point_maze`` IS ``stable_envs.envs.navigation``.

Step 9 merged the three ``stable_worldmodel`` navigation ports into one module (plan §3). This file
is a ``sys.modules`` ALIAS rather than a set of re-exports, so ``DEFAULT_WALL_SHAPE``,
``DEFAULT_WALL_POS``, ``SimplePointMaze`` and everything else the old module exposed still resolve
through it, and there is only one module object to monkeypatch. The five constants whose names
collided with the TwoRoom/Piecewise half (``AGENT_C``, ``WALL_C``, ``BG_C``, ``AGENT_R``,
``AGENT_SPEED``) carry a ``MAZE_`` prefix in the merged module; nothing outside the module ever read
them. Wave 3 removes this file; nothing new should import it.
"""
import sys

from . import navigation as _navigation

sys.modules[__name__] = _navigation
