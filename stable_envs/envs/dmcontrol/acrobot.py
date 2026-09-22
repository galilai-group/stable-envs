"""DM Control ``acrobot`` — the parity tier.

One task. Its ``target`` site is in geom group 3, which ``mujoco.Renderer``'s default
``MjvOption`` does not show, so neither does this renderer (``VISIBLE_GROUPS``).
"""

from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

DMCAcrobotSwingup = parity_cell("DMCAcrobotSwingup", "acrobot", "swingup", bases=(FirstFamilySemantics,))
