from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

DMCCartpoleSwingup = parity_cell("DMCCartpoleSwingup", "cartpole", "swingup", bases=(FirstFamilySemantics,))
DMCCartpoleBalance = parity_cell("DMCCartpoleBalance", "cartpole", "balance")
