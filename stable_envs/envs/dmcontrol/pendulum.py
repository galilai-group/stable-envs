"""DM Control ``pendulum`` — the parity tier.

One task, one mover. Nothing is drawn per episode, so the state is ``qpos`` alone.
"""

from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

DMCPendulumSwingup = parity_cell(
    "DMCPendulumSwingup", "pendulum", "swingup", bases=(FirstFamilySemantics,)
)
