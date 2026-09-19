"""DM Control ``pendulum`` — the parity tier.

One task, one mover. Nothing is drawn per episode, so the state is ``qpos`` alone.
"""

from __future__ import annotations

from ._cells import parity_cell

DMCPendulumSwingup = parity_cell("DMCPendulumSwingup", "pendulum", "swingup")
