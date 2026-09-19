"""DM Control ``finger`` — the parity tier, and the family's one declared all-pixel floor.

Three tasks. The two ``turn`` tasks DRAW PER EPISODE — ``initialize_episode`` moves a target site —
so their state is ``qpos`` plus that draw and the site is a mover (``_episode_slots``).

``turn_easy`` is the one cell in the family that declares ``parity_all_exact_min``, and the reason is
the size of its robot rather than the renderer: all-pixel exactness is interior exactness WEIGHTED BY
HOW MUCH OF THE FRAME MOVES, and its seven movers cover 11.7% of an 84 px frame against cartpole's
2.5%. ``parity_interior_min`` is NOT relaxed on any of the three — it measures exactly 1.000000. The
other two clear the shared 0.90 bar by 0.024-0.029 and declare nothing. See ``PARITY_NOTE``.
"""

from __future__ import annotations

from ._cells import PARITY_NOTE, parity_cell

#: All-pixel floors that differ from the suite's shared 0.90, with the mechanism in ``PARITY_NOTE``.
#: ONE cell, because one cell needs it. The INTERIOR floor is untouched at 1.0 on every cell here and
#: is the number that describes the renderer; this only stops a scene being failed for the size of its
#: robot, and the other two finger cells clear the shared bar by 0.021-0.026 and declare nothing.
ALL_EXACT_FLOOR = {"DMCFingerTurnEasy": 0.89}

DMCFingerSpin = parity_cell("DMCFingerSpin", "finger", "spin")
DMCFingerTurnEasy = parity_cell(
    "DMCFingerTurnEasy",
    "finger",
    "turn_easy",
    parity_all_exact_min=ALL_EXACT_FLOOR["DMCFingerTurnEasy"],
    parity_note=PARITY_NOTE,
)
DMCFingerTurnHard = parity_cell("DMCFingerTurnHard", "finger", "turn_hard")
