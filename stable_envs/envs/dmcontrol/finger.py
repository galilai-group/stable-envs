from __future__ import annotations

from ._aligned import DMCAlignedSemantics
from ._cells import PARITY_NOTE, parity_cell

# This cell has a separate all-pixel floor; the interior agreement requirement is unchanged.
ALL_EXACT_FLOOR = {"DMCFingerTurnEasy": 0.89}

DMCFingerSpin = parity_cell("DMCFingerSpin", "finger", "spin")
DMCFingerTurnEasy = parity_cell(
    "DMCFingerTurnEasy",
    "finger",
    "turn_easy",
    parity_all_exact_min=ALL_EXACT_FLOOR["DMCFingerTurnEasy"],
    parity_note=PARITY_NOTE,
)
DMCFingerTurnHard = parity_cell("DMCFingerTurnHard", "finger", "turn_hard", bases=(DMCAlignedSemantics,))
