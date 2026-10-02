"""Historical run/hop/walk semantics over the existing locomotion renderers."""

from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._adapters.dmcontrol_continuation import CONFIG, CONTINUATION_FIELDS, DMCContinuationAdapter
from ...spaces import Box, Dict


class DMCContinuationSemantics:
    action_dtype = torch.float64
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"

    @property
    def action_dim(self):
        return CONFIG[self.DOMAIN]["nu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulator = DMCContinuationAdapter(self.DOMAIN, self.B, self.device, self._seed)

    @property
    def _resume_fields(self):
        return CONTINUATION_FIELDS[self.DOMAIN]

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
            layout = {name: (offset, width) for name, offset, width in self.variation_space.flat_layout()}
            columns = [layout[f"gravity.{axis}"][0] for axis in "xyz"]
            encoded = self.get_style()[:, columns]
            gravity = encoded.to(torch.float64)
            encoded_default = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32, device=self.device)
            exact_default = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64, device=self.device)
            gravity = torch.where(encoded == encoded_default, exact_default, gravity)
            self._simulator.configure_gravity(gravity.cpu().numpy())
        return result

    def _layout(self):
        return tuple(
            (key if isinstance(key, tuple) else (key,), width, 0 if key == "qpos" else None)
            for key, (width, _) in self._resume_fields.items()
        )

    def _layout_dtypes(self):
        return {
            key if isinstance(key, tuple) else (key,): dtype for key, (_, dtype) in self._resume_fields.items()
        }

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

    def _step_pipeline(self):
        owner = self._simulator
        return owner if hasattr(owner, "submit_step") else None

    def _step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()

