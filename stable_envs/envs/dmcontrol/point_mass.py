"""DM Control ``point_mass`` — the parity tier.

The smallest scene in the family: the plate is 99.85% of the frame, and what is left is the mass,
its shadow and its reflection in the reflectance-0.2 floor.
"""

from __future__ import annotations

from ._cells import parity_cell

DMCPointMassEasy = parity_cell("DMCPointMassEasy", "point_mass", "easy")
