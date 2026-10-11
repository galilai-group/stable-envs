from __future__ import annotations

from ._aligned import DMCAlignedSemantics
from ._cells import parity_cell

# The tendon is drawn as a catenary that hangs along gravity.
DMCBallInCupCatch = parity_cell(
    "DMCBallInCupCatch",
    "ball_in_cup",
    "catch",
    bases=(DMCAlignedSemantics,),
    FRAME_PHYSICS=frozenset({"gravity.x", "gravity.y", "gravity.z"}),
)
