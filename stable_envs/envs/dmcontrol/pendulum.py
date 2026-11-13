from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

DMCPendulumSwingup = parity_cell("DMCPendulumSwingup", "pendulum", "swingup", bases=(FirstFamilySemantics,))
