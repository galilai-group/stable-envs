from __future__ import annotations

import torch
from tensordict import TensorDict

from ...render import raycast3d as R3
from ._cells import parity_cell
from ._delegated import DelegatedSemantics


class _ReacherSemantics(DelegatedSemantics):
    """Reacher on CPU dm_control lanes.

    The four-column renderer projection cannot resume physics or implicit resets.
    """

    def make_variation_space(self):
        space = super().make_variation_space()
        if self._simulator_key() == "reacher":
            space["visibility"]["target"].set_init_value(0)
        return space

    def _layout(self):
        return tuple((key, width, 0 if key == ("qpos",) else None) for key, (width, _) in self._fields().items())

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full Reacher State")
        return torch.cat((state["qpos"], state["task", "target_pos"][:, :2]), -1).to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, 4)
        # This projection omits velocity, RNG, lifecycle, and target z; it is not restorable State.
        return TensorDict(
            {"qpos": rows[:, :2].clone(), "target_xy": rows[:, 2:].clone()}, [len(rows)], device=self.device
        )


DMCReacherEasy = parity_cell("DMCReacherEasy", "reacher", "easy", bases=(_ReacherSemantics,), SIMULATOR="reacher-easy")
# The registered wrapper's target alternatives at the task's size; the sphere is the compiled model's own.
_HARD_TARGET = 0.015

DMCReacherHard = parity_cell(
    "DMCReacherHard",
    "reacher",
    "hard",
    bases=(_ReacherSemantics,),
    SHAPE_KNOBS={"target": ((R3.BOX, (_HARD_TARGET,) * 3), (R3.SPHERE, (_HARD_TARGET, 0.0, 0.0)))},
)

DMCReacherHard.__doc__ = """DM Control Reacher hard with authoritative CPU simulation and batched rendering.

Public State includes positions, velocities, integration warm-start values, clock,
RNG, lifecycle counters, cumulative reward and the target that changes on implicit
resets. Device selects State/render placement; physics runs on CPU.
"""
