"""DM Control ``ball_in_cup`` — the parity tier.

The one shipped cell that draws a SPATIAL TENDON. MuJoCo does not draw it as one connector: a
slack tendon hangs as a chain of 28 capsules along a catenary and a taut one is a single capsule
recoloured by its limit constraint. ``_tendon`` is that mover kind, ported from ``mjv_catenary``
to 0 ULP.
"""
from __future__ import annotations

from ._cells import parity_cell

DMCBallInCupCatch = parity_cell("DMCBallInCupCatch", "ball_in_cup", "catch")
