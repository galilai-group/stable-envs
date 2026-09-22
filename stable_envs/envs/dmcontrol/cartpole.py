"""DM Control ``cartpole`` — the parity tier.

Two tasks on one model, drawn by ``suite``: three baked plates and an analytic ray-cast of the
cart, the pole and their shadow and reflection. The cart's front face is a BOX at normal
incidence, which is the one surface in this family where a shading constant can be read straight
off the reference — ``SPECULAR_SCALE`` in ``_model`` was measured on it.
"""

from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

DMCCartpoleSwingup = parity_cell(
    "DMCCartpoleSwingup", "cartpole", "swingup", bases=(FirstFamilySemantics,)
)
DMCCartpoleBalance = parity_cell("DMCCartpoleBalance", "cartpole", "balance")
