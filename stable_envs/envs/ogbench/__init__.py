"""OGBench locomotion mazes and Cube/Scene manipulation environments."""

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
    BODY_STATE,
    CELLS,
    MAT_SPECULAR,
    SHADE_AMBIENT,
    SHADE_DIFFUSE,
    SHADE_EXPONENT,
    SHADE_SPECULAR,
    OGBenchMaze,
)

# Replace the GPU-only Cube/Scene registrations with the simulation-backed subclasses.
for _name, _cls in _MANIPULATION_CPU_CLASSES.items():
    assert REGISTRY[_name] is manipulation.CELL_CLASSES[_name]
    del REGISTRY[_name]
    register(_cls)
CELL_CLASSES = {**maze.CELL_CLASSES, **_MANIPULATION_CPU_CLASSES}
globals().update(CELL_CLASSES)

__all__ = ["OGBenchMaze", "OGBManipBase", "maze", "manipulation", *sorted(CELL_CLASSES)]
