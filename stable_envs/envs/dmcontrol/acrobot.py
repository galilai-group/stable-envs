from __future__ import annotations

from ._cells import parity_cell
from ._first_family import FirstFamilySemantics

# TASK selects the renderer model; the simulator adapter supplies Balance(sparse=True).
DMCAcrobotBalance = parity_cell("DMCAcrobotBalance", "acrobot", "swingup", bases=(FirstFamilySemantics,))
DMCAcrobotBalance.name = "dmc-acrobot-balance"
DMCAcrobotBalance.reference_id = "dm_control/acrobot-swingup_sparse"
DMCAcrobotBalance.__doc__ = """SWM Acrobot sparse Balance with CPU dm_control physics.

Simulates acrobot.Balance(sparse=True), the task swm/AcrobotDMControl-v0 selects.
Full State includes physical, RNG and lifecycle fields. CUDA selects State and
render placement; simulation runs on CPU.
"""
