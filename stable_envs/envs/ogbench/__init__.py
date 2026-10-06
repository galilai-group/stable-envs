from __future__ import annotations

from ...registry import REGISTRY, register
from . import manipulation, maze  # noqa: F401  (imported for their registrations)
from .manipulation import (  # noqa: F401
    IDENTITY_QUAT,
    MJW_CHUNK,
    SPEC,
    VISIBLE_GROUPS,
    OGBManipBase,
)
from .manipulation_semantics import CELL_CLASSES as _MANIPULATION_CPU_CLASSES
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

# Register the semantic subclasses; re-export cache objects by identity for shared ownership.
for _name, _cls in _MANIPULATION_CPU_CLASSES.items():
    assert REGISTRY[_name] is manipulation.CELL_CLASSES[_name]
    del REGISTRY[_name]
    register(_cls)
CELL_CLASSES = {**maze.CELL_CLASSES, **_MANIPULATION_CPU_CLASSES}
globals().update(CELL_CLASSES)

__all__ = ["OGBenchMaze", "OGBManipBase", "maze", "manipulation", *sorted(CELL_CLASSES)]
