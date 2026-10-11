from __future__ import annotations

from ._aligned import DMCAlignedSemantics
from ._cells import parity_cell

DMCPointMassEasy = parity_cell(
    "DMCPointMassEasy", "point_mass", "easy", bases=(DMCAlignedSemantics,), SIMULATOR="point_mass-easy"
)
