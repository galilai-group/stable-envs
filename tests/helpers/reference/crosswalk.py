"""SWM id -> stable-envs cell correspondence.

Read from ``tests/fixtures/swm/frozen_cells.json`` (42 rows, 40 pixel, 2 symbolic). The
matrix spells some cells differently from the registered class names; the mapping lives
here because ``snv.make`` resolves neither SWM ids nor matrix spellings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stable_envs.env import Env

_ROOT = Path(__file__).resolve().parents[3]
_MATRIX = _ROOT / "tests" / "fixtures" / "swm" / "frozen_cells.json"


def _swm_ids_from_matrix() -> dict[str, tuple[str, ...]]:
    rows = json.loads(_MATRIX.read_text())["rows"]
    return {r["swm_id"]: tuple(r["stable_envs_cells"]) for r in rows}


#: SWM id -> frozen cell spelling(s), as recorded in the frozen mapping.
SWM_IDS: dict[str, tuple[str, ...]] = _swm_ids_from_matrix()

SYMBOLIC_IDS = tuple(sorted(k for k, v in SWM_IDS.items() if not v))

#: Cell spelling in the frozen matrix -> registered class name.
HISTORICAL_CLASS_NAMES: dict[str, str] = {
    "DMCAcrobotSwingup": "DMCAcrobotBalance",
    "DMCHumanoidStand": "DMCHumanoidWalk",
    "AntmazeGiantNavigate": "AntMazeGiantNavigate",
    "AntmazeGiantStitch": "AntMazeGiantStitch",
    "AntmazeLargeExplore": "AntMazeLargeExplore",
    "AntmazeLargeNavigate": "AntMazeLargeNavigate",
    "AntmazeLargeStitch": "AntMazeLargeStitch",
    "AntmazeMediumExplore": "AntMazeMediumExplore",
    "AntmazeMediumNavigate": "AntMazeMediumNavigate",
    "AntmazeMediumStitch": "AntMazeMediumStitch",
    "AntmazeTeleportExplore": "AntMazeTeleportExplore",
    "AntmazeTeleportNavigate": "AntMazeTeleportNavigate",
    "AntmazeTeleportStitch": "AntMazeTeleportStitch",
    "HumanoidmazeGiantNavigate": "HumanoidMazeGiantNavigate",
    "HumanoidmazeGiantStitch": "HumanoidMazeGiantStitch",
    "HumanoidmazeLargeNavigate": "HumanoidMazeLargeNavigate",
    "HumanoidmazeLargeStitch": "HumanoidMazeLargeStitch",
    "HumanoidmazeMediumNavigate": "HumanoidMazeMediumNavigate",
    "HumanoidmazeMediumStitch": "HumanoidMazeMediumStitch",
    "MujocoAnt": "MuJoCoAnt",
    "MujocoHalfCheetah": "MuJoCoHalfCheetah",
    "MujocoHopper": "MuJoCoHopper",
    "MujocoHumanoid": "MuJoCoHumanoid",
    "MujocoInvertedDoublePendulum": "MuJoCoInvertedDoublePendulum",
    "MujocoInvertedPendulum": "MuJoCoInvertedPendulum",
    "MujocoPusher": "MuJoCoPusher",
    "MujocoReacher": "MuJoCoReacher",
    "MujocoSwimmer": "MuJoCoSwimmer",
    "MujocoWalker2d": "MuJoCoWalker2d",
    "PFRocketLanding": "PyFlytRocketLanding",
    "PointmazeGiantNavigate": "PointMazeGiantNavigate",
    "PointmazeGiantStitch": "PointMazeGiantStitch",
    "PointmazeLargeNavigate": "PointMazeLargeNavigate",
    "PointmazeLargeStitch": "PointMazeLargeStitch",
    "PointmazeMediumNavigate": "PointMazeMediumNavigate",
    "PointmazeMediumStitch": "PointMazeMediumStitch",
    "PointmazeTeleportNavigate": "PointMazeTeleportNavigate",
    "PointmazeTeleportStitch": "PointMazeTeleportStitch",
}


def canonical_from_frozen(name: str) -> str:
    """Registered class name for a frozen-matrix cell spelling."""
    return HISTORICAL_CLASS_NAMES.get(name, name)


def frozen_from_canonical(name: str) -> str:
    """Frozen-matrix cell spelling for a registered class name."""
    for old, new in HISTORICAL_CLASS_NAMES.items():
        if new == name:
            return old
    return name


def class_for_frozen_cell(name: str) -> type[Env] | None:
    """Look up a registered class by class name or frozen-matrix cell spelling."""
    from stable_envs.registry import REGISTRY

    return REGISTRY.get(name) or REGISTRY.get(canonical_from_frozen(name))


def canonical_names(name: str) -> tuple[str, ...]:
    """The class name(s) ``name`` refers to: itself if registered, else its SWM cells."""
    from stable_envs.registry import REGISTRY

    if name in REGISTRY:
        return (name,)
    live = canonical_from_frozen(name)
    if live in REGISTRY:
        return (live,)
    return tuple(canonical_from_frozen(cell) for cell in SWM_IDS.get(name, ()))


def swm_ids_for_class(name: str) -> tuple[str, ...]:
    """SWM ids whose frozen cells name this class, under either spelling."""
    frozen = frozen_from_canonical(name)
    return tuple(sorted(i for i, cells in SWM_IDS.items() if name in cells or frozen in cells))
