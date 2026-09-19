"""The OGBench family: 25 maze cells plus the two ``stable_worldmodel`` manipulation cells.

TWO MODULES, BECAUSE THERE ARE TWO SCENES AND TWO RENDERERS. ``maze`` is OGBench's own
``pointmaze``/``antmaze``/``humanoidmaze`` under a fixed top-down camera: a MuJoCo-baked plate plus
the analytic ray-caster in ``render/raycast3d.py`` for the handful of agent primitives, certified at
the ``parity`` tier and CPU-capable. ``manipulation`` is swm's ``CubeEnv``/``SceneEnv``: a
MuJoCo-baked plate composited with ``mujoco_warp``'s batch ray-caster for the movers, certified at
the ``visual`` tier and CUDA-only by declaration (``CPU3D-OGB-1``). They share the family name, the
upstream package and the plate-bake idea, and nothing else — no base class, no renderer, no cache.
Merging them would produce a class that is two classes with a flag.

WHAT THE IMPORT PATHS ARE, AND WHY THEY ARE NOT THE SAME SHAPE
--------------------------------------------------------------

``stable_envs/envs/ogbench.py`` became THIS PACKAGE. A module and a package cannot share a name in
one directory, so — unlike Workstream A's ``swm_nav``/``navigation`` merge — there is no second path
to alias: the package IS ``stable_envs.envs.ogbench``, and the obligation the alias would have met is
met directly by the re-exports below. Every module-level name the Step-8 module exported resolves off
this package, and ``scratch/step9/envs-ogbench/oracle.py`` checks that name by name, read off the
baseline SOURCE so the list cannot drift from what that file actually had.

The manipulation half lives at ``stable_envs.envs.ogbench.manipulation``.

THE RE-EXPORTS ARE THE SAME OBJECTS, not copies, and that matters for two of them in particular:
``_SCENE_CACHE`` and ``_ENV_CACHE`` are mutated through this package by
``tests/unit/test_ogbench_scene_cache.py``, which saves, clears and restores the cache. A re-export
binds the same object, so a mutation through either path is seen by both; only a REBIND would not be,
and ``tests/unit/test_envs_ogbench.py`` asserts the identity so that a future re-export written as a
copy fails loudly instead of quietly rendering the wrong thing.
"""

from __future__ import annotations

from . import manipulation, maze  # noqa: F401  (imported for their registrations)

# ---- the manipulation half: the two cells and what a reader of this package would look for ----
from .manipulation import (  # noqa: F401
    IDENTITY_QUAT,
    MJW_CHUNK,
    SPEC,
    VISIBLE_GROUPS,
    OGBManipBase,
)

# ---- the maze half: every module-level name ``stable_envs/envs/ogbench.py`` exported ----------
from .maze import (  # noqa: F401
    _ANT_JOINTS,
    _ENV_CACHE,
    _FREE,
    _HUMANOID_JOINTS,
    _NOMINAL,
    _SCENE_CACHE,
    _SCENE_CACHE_MAX,
    BODY_STATE,
    CELLS,
    MAT_SPECULAR,
    SHADE_AMBIENT,
    SHADE_DIFFUSE,
    SHADE_EXPONENT,
    SHADE_SPECULAR,
    OGBenchMaze,
    _build_reference,
    _make_classes,
    _mat_and_tex,
    _offsets,
    _Scene,
    _scene_env,
)

#: The 25 maze cells and the two manipulation cells, by name, so ``stable_envs.envs.ogbench.<Cell>``
#: resolves exactly as it did in Step 8. Taken from each module's own ``CELL_CLASSES`` rather than
#: transcribed, so a cell cannot exist and be missing here.
CELL_CLASSES = {**maze.CELL_CLASSES, **manipulation.CELL_CLASSES}
globals().update(CELL_CLASSES)

__all__ = ["OGBenchMaze", "OGBManipBase", "maze", "manipulation", *sorted(CELL_CLASSES)]
