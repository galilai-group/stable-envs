from __future__ import annotations

from ...render import raycast3d as R3
from ._aligned import DMCAlignedSemantics
from ._cells import PARITY_NOTE, parity_cell

# This cell has a separate all-pixel floor; the interior agreement requirement is unchanged.
ALL_EXACT_FLOOR = {"DMCFingerTurnEasy": 0.89}

DMCFingerSpin = parity_cell("DMCFingerSpin", "finger", "spin", bases=(DMCAlignedSemantics,), SIMULATOR="finger-spin")
DMCFingerTurnEasy = parity_cell(
    "DMCFingerTurnEasy",
    "finger",
    "turn_easy",
    parity_all_exact_min=ALL_EXACT_FLOOR["DMCFingerTurnEasy"],
    parity_note=PARITY_NOTE,
    bases=(DMCAlignedSemantics,),
    SIMULATOR="finger-turn_easy",
)
# The registered wrapper's target alternatives at the task's radius; the sphere is the compiled model's own.
_HARD_TARGET = 0.03

DMCFingerTurnHard = parity_cell(
    "DMCFingerTurnHard",
    "finger",
    "turn_hard",
    bases=(DMCAlignedSemantics,),
    GEOM_COLORS={"spinner": ("cap1", "cap2")},
    SHAPE_KNOBS={"target": ((R3.BOX, (_HARD_TARGET,) * 3), (R3.SPHERE, (_HARD_TARGET, 0.005, 0.005)))},
)
