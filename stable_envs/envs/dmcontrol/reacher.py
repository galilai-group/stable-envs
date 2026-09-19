"""DM Control ``reacher`` — the parity tier.

Both tasks DRAW PER EPISODE — ``initialize_episode`` moves the target geom — so the state is
``qpos`` plus that draw and the target is a mover (``_episode_slots``). A top-down camera over a
floor that fills the frame is also why this domain ships at parity and not through MJWarp: the
spotlight's radial falloff IS the picture here, the plate holds it exactly because MuJoCo painted
it, and MJWarp renders it nearly flat (mean 28.4/255).
"""

from __future__ import annotations

from ._cells import parity_cell

DMCReacherEasy = parity_cell("DMCReacherEasy", "reacher", "easy")
DMCReacherHard = parity_cell("DMCReacherHard", "reacher", "hard")
