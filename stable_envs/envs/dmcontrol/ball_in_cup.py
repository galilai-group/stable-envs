from __future__ import annotations

from ._aligned import DMCAlignedSemantics
from ._cells import parity_cell

DMCBallInCupCatch = parity_cell("DMCBallInCupCatch", "ball_in_cup", "catch", bases=(DMCAlignedSemantics,))
