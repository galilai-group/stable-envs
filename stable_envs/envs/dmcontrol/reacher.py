from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._adapters.dmc.execution import make_dmc_adapter
from ..._adapters.dmc.first_family import REACHER_FIELDS
from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict
from ._cells import parity_cell


class _ReacherSemantics:
    """Delegate hard-task simulation to CPU while exposing full State on the requested device.

    The four-column renderer projection cannot resume physics or implicit resets.
    """

    action_dim = 2
    action_dtype = torch.float64
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulator = make_dmc_adapter("reacher", self.B, self.device, self._seed)

    def make_variation_space(self):
        base = super().make_variation_space()
        base["visibility"]["target"].set_init_value(0)
        return Dict(
            {
                **dict(base.items()),
                "agent": Dict(
                    {
                        "arm_density": Box(low=500, high=1500, shape=(1,), init_value=[1000.0]),
                        "finger_density": Box(low=500, high=1500, shape=(1,), init_value=[1000.0]),
                    }
                ),
            }
        )

    def set_variation(self, variation=None, **overrides):
        result = super().set_variation(variation, **overrides)
        if hasattr(self, "_simulator"):
            layout = {name: (off, width) for name, off, width in self.variation_space.flat_layout()}
            cols = [layout[f"agent.{part}_density"][0] for part in ("arm", "finger")]
            self._simulator.configure_density(self.get_style()[:, cols].cpu().numpy())
        return result

    def _layout(self):
        return tuple(
            (key if isinstance(key, tuple) else (key,), width, 0 if key == "qpos" else None)
            for key, (width, _) in REACHER_FIELDS.items()
        )

    def _layout_dtypes(self):
        return {key if isinstance(key, tuple) else (key,): dtype for key, (_, dtype) in REACHER_FIELDS.items()}

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

    def reset(self, seed=None):
        self._check_open("reset")
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step_pipeline(self) -> StepPipeline | None:
        owner = self._simulator
        return step_pipeline(owner)

    def _step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()


DMCReacherEasy = parity_cell("DMCReacherEasy", "reacher", "easy")
DMCReacherHard = parity_cell("DMCReacherHard", "reacher", "hard", bases=(_ReacherSemantics,))

DMCReacherHard.__doc__ = """DM Control Reacher hard with authoritative CPU simulation and batched rendering.

Public State includes positions, velocities, integration warm-start values, clock,
RNG, lifecycle counters, cumulative reward and the target that changes on implicit
resets. Device selects State/render placement; physics runs on CPU.
"""
