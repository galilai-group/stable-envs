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

``stable_envs/envs/ogb_manip.py`` keeps ITS name, because no package claims it: it is a
``sys.modules`` alias of ``stable_envs.envs.ogbench.manipulation``, so the private names its tests
reach for (``_bundle``, ``_context``, ``_button_tables``, ``_appearance_digest``, ``SPEC``, …) all
still resolve and a monkeypatch through either path is seen by both — there is one module object.
Wave 3 removes that file.

THE RE-EXPORTS ARE THE SAME OBJECTS, not copies, and that matters for two of them in particular:
``_SCENE_CACHE`` and ``_ENV_CACHE`` are mutated through this package by
``tests/unit/test_ogbench_scene_cache.py``, which saves, clears and restores the dict. A re-export
binds the same dict, so a mutation through either path is seen by both; only a REBIND would not be,
and ``tests/unit/test_envs_ogbench.py`` asserts the identity so that a future re-export written as a
copy fails loudly instead of quietly rendering the wrong thing.
"""
from __future__ import annotations

from . import manipulation, maze  # noqa: F401  (imported for their registrations)

# ---- the maze half: every module-level name ``stable_envs/envs/ogbench.py`` exported ----------
from .maze import (  # noqa: F401
    BODY_STATE,
    CELLS,
    MAT_SPECULAR,
    OGBenchMaze,
    SHADE_AMBIENT,
    SHADE_DIFFUSE,
    SHADE_EXPONENT,
    SHADE_SPECULAR,
    _ANT_JOINTS,
    _build_reference,
    _ENV_CACHE,
    _FREE,
    _HUMANOID_JOINTS,
    _make_classes,
    _mat_and_tex,
    _NOMINAL,
    _offsets,
    _Scene,
    _SCENE_CACHE,
    _SCENE_CACHE_MAX,
    _scene_env,
)

# ---- the manipulation half: the two cells and what a reader of this package would look for ----
from .manipulation import (  # noqa: F401
    MJW_CHUNK,
    OGBManipBase,
    SPEC,
    VISIBLE_GROUPS,
    IDENTITY_QUAT,
)

#: The 25 maze cells and the two manipulation cells, by name, so ``stable_envs.envs.ogbench.<Cell>``
#: resolves exactly as it did in Step 8. Taken from each module's own ``CELL_CLASSES`` rather than
#: transcribed, so a cell cannot exist and be missing here.
CELL_CLASSES = {**maze.CELL_CLASSES, **manipulation.CELL_CLASSES}
globals().update(CELL_CLASSES)

__all__ = ["OGBenchMaze", "OGBManipBase", "maze", "manipulation", *sorted(CELL_CLASSES)]
