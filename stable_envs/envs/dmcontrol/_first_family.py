from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict
from .simulation.execution import make_dmc_adapter
from .simulation.first_family import FIRST_FAMILY_FIELDS


class FirstFamilySemantics:
    """Delegate Acrobot, Pendulum, and Cartpole simulation to CPU adapters."""

    action_dim = 1
    action_dtype = torch.float64
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulator = make_dmc_adapter(self.DOMAIN, self.B, self.device, self._seed)

    @property
    def _resume_fields(self):
        return FIRST_FAMILY_FIELDS[self.DOMAIN]

    def make_variation_space(self):
        base = super().make_variation_space()
        return Dict(
            {
                **dict(base.items()),
                "gravity": Dict(
                    {
                        "x": Box(low=-5.0, high=5.0, shape=(1,), init_value=[0.0]),
                        "y": Box(low=-5.0, high=5.0, shape=(1,), init_value=[0.0]),
                        "z": Box(low=-20.0, high=0.0, shape=(1,), init_value=[-9.81]),
                    }
                ),
            }
        )

    @property
    def reset_value_paths(self):
        return frozenset(self.physics_variation_leaves)

    @property
    def reset_sample_paths(self):
        return frozenset(self.physics_variation_leaves)

    def _layout(self):
        return tuple(
            (key if isinstance(key, tuple) else (key,), width, 0 if key == "qpos" else None)
            for key, (width, _) in self._resume_fields.items()
        )

    def _layout_dtypes(self):
        return {key if isinstance(key, tuple) else (key,): dtype for key, (_, dtype) in self._resume_fields.items()}

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError(f"Expected full {self.DOMAIN} State")
        return state["qpos"].to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        nq = self._resume_fields["qpos"][0]
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, nq)
        return TensorDict({"qpos": rows.clone()}, [len(rows)], device=self.device)

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        self.draw_reset_values(seed, options)
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)
        self._adopt_physics_state(state)

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
