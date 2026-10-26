from __future__ import annotations

import torch
from tensordict import TensorDict

from ...render import raycast3d as R3
from ._cells import parity_cell
from ._delegated import DelegatedSemantics
from .simulation.execution import make_dmc_adapter, reacher_backend


class _ReacherSemantics(DelegatedSemantics):
    """Reacher with native CPU or CUDA MJX physics."""

    execution_mode = "AUTO_NATIVE_MJX"

    def __init__(self, *args, simulation_backend="auto", mjx_min_envs=32, **kwargs):
        """Select physics at construction and return tensors on the requested device.

        ``"auto"`` uses native CPU physics below ``mjx_min_envs`` and MJX for
        larger CUDA batches. ``"native"`` and ``"mjx"`` select explicitly.
        """
        self.simulation_backend = simulation_backend
        self.mjx_min_envs = mjx_min_envs
        super().__init__(*args, **kwargs)
        self.execution_mode = "JAX_MJX" if self.simulation_backend == "mjx" else "DELEGATED_CPU"

    def _make_simulator(self):
        self.simulation_backend = reacher_backend(self.B, self.device, self.simulation_backend, self.mjx_min_envs)
        return make_dmc_adapter(
            self._simulator_key(),
            self.B,
            self.device,
            self._seed,
            simulation_backend=self.simulation_backend,
            mjx_min_envs=self.mjx_min_envs,
        )

    def make_variation_space(self):
        space = super().make_variation_space()
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
DMCReacher.__doc__ = """Reacher with native float64 CPU or float32 MJX CUDA physics.

``simulation_backend="auto"`` uses native CPU physics below ``mjx_min_envs``.
The cutoff defaults to 32. Larger CUDA batches use MJX. Set ``simulation_backend``
to ``"native"`` or ``"mjx"`` to choose explicitly. ``device`` controls output and rendering.
"""
