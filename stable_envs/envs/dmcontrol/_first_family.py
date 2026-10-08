from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._adapters import StepPipeline, step_pipeline
from ..._adapters.dmc.execution import make_dmc_adapter
from ..._adapters.dmc.first_family import FIRST_FAMILY_FIELDS
from ...spaces import Box, Dict


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

    def set_variation(self, variation=None, **overrides):
        result = super().set_variation(variation, **overrides)
        if hasattr(self, "_simulator"):
            layout = {name: (off, width) for name, off, width in self.variation_space.flat_layout()}
            cols = [layout[f"gravity.{axis}"][0] for axis in "xyz"]
            gravity = self.get_style()[:, cols].to(torch.float64)
            # Preserve nominal gravity in float64 when a float32 style row encodes its default.
            encoded_default = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32, device=self.device)
            exact_default = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64, device=self.device)
            gravity = torch.where(self.get_style()[:, cols] == encoded_default, exact_default, gravity)
            self._simulator.configure_gravity(gravity.cpu().numpy())
        return result

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
