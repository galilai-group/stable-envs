"""Historical sparse Acrobot Balance semantics with the preserved Acrobot renderer."""

from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

# TASK selects the historical renderer asset; the simulator task is Balance(sparse=True).
DMCAcrobotBalance = parity_cell("DMCAcrobotBalance", "acrobot", "swingup", bases=(FirstFamilySemantics,))
DMCAcrobotBalance.name = "dmc-acrobot-balance"
DMCAcrobotBalance.reference_id = "dm_control/acrobot-swingup_sparse"
DMCAcrobotBalance.__doc__ = """Historical SWM Acrobot sparse Balance, using authoritative CPU physics.

The public identity names acrobot.Balance(sparse=True), selected by
swm/AcrobotDMControl-v0. Full State includes physical, RNG and lifecycle fields.
CUDA selects State/render placement; simulation remains delegated to CPU.
"""
