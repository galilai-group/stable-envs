from __future__ import annotations

import torch
from tensordict import TensorDict

from ...render import raycast3d as R3
from ._cells import parity_cell
from ._delegated import DelegatedSemantics


class _ReacherSemantics(DelegatedSemantics):
    """Reacher with native CPU, CUDA MuJoCo Warp or CUDA MJX physics."""

    def make_variation_space(self):
        space = super().make_variation_space()
        space["visibility"]["target"].set_init_value(0)
        return space

    @classmethod
    def reference_env(cls, size: int | None = None):
        env = super().reference_env(size)
        # The source frame is stable-worldmodel's default, which makes the target material transparent.
        env.physics.named.model.mat_rgba["target", 3] = 0.0
        return env

    def _layout(self):
        return tuple((key, width, 0 if key == ("qpos",) else None) for key, (width, _) in self._fields().items())

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full Reacher State")
        return torch.cat((state["qpos"], state["task", "target_pos"][:, :2]), -1).to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, 4)
        # This projection omits velocity, RNG, lifecycle, and target z; it is not restorable state.
        return TensorDict(
            {"qpos": rows[:, :2].clone(), "target_xy": rows[:, 2:].clone()}, [len(rows)], device=self.device
        )


# The registered wrapper's target alternatives at the task's size; the sphere is the compiled model's own.
_TARGET_RADIUS = 0.015

DMCReacher = parity_cell(
    "DMCReacher",
    "reacher",
    "hard",
    bases=(_ReacherSemantics,),
    SHAPE_KNOBS={"target": ((R3.BOX, (_TARGET_RADIUS,) * 3), (R3.SPHERE, (_TARGET_RADIUS, 0.0, 0.0)))},
)

DMCReacher.name = "dmc-reacher"
DMCReacher.__doc__ = """Reacher with native float64 CPU or float32 MuJoCo Warp/MJX CUDA physics.

On CUDA the default ``simulation_backend=None`` uses MuJoCo Warp at every batch size. ``"auto"`` uses native
physics below ``mjwarp_min_envs`` (8 by default) and MuJoCo Warp from there. ``"native"``, ``"mjwarp"`` and
``"mjx"`` select explicitly. ``device`` controls output and rendering.
"""
