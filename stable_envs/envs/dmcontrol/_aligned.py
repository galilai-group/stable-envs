from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict, Discrete
from .simulation.aligned import ALIGNED_FIELDS, CONFIG
from .simulation.execution import make_dmc_adapter


class DMCAlignedSemantics:
    action_dtype = torch.float64
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"

    @property
    def action_dim(self):
        return CONFIG[self.DOMAIN]["nu"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulator = make_dmc_adapter(self.DOMAIN, self.B, self.device, self._seed)

    @property
    def _resume_fields(self):
        return ALIGNED_FIELDS[self.DOMAIN]

    def make_variation_space(self):
        base = super().make_variation_space()
        extra = {}
        if self.DOMAIN in ("ball_in_cup", "quadruped"):
            extra["gravity"] = Dict(
                {
                    "x": Box(low=-5.0, high=5.0, shape=(1,), init_value=[0.0]),
                    "y": Box(low=-5.0, high=5.0, shape=(1,), init_value=[0.0]),
                    "z": Box(low=-20.0, high=0.0, shape=(1,), init_value=[-9.81]),
                }
            )
        elif self.DOMAIN == "finger":
            base["visibility"]["target"] = Discrete(2, init_value=0)
            extra["agent_physics"] = Dict(
                {"proximal_density": Box(low=500.0, high=1500.0, shape=(1,), init_value=[1000.0])}
            )
        else:
            extra["agent_physics"] = Dict(
                {"upper_arm_density": Box(low=500.0, high=1500.0, shape=(1,), init_value=[1000.0])}
            )
        return Dict({**dict(base.items()), **extra})

    @classmethod
    def _hideable(cls):
        entries = super()._hideable()
        if cls.DOMAIN != "finger":
            return entries
        # The target material is shared by the marker and tip sites; hide every site using it.
        model = cls._model()
        material = model.material("target").id
        sites = tuple(i for i in range(model.nsite) if model.site_matid[i] == material)
        return tuple(
            (kind, name, sites if kind == "site" and name == "target" else ids) for kind, name, ids in entries
        )

    def set_variation(self, variation=None, **overrides):
        result = super().set_variation(variation, **overrides)
        if not hasattr(self, "_simulator"):
            return result
        layout = {name: (offset, width) for name, offset, width in self.variation_space.flat_layout()}
        style = self.get_style()
        if self.DOMAIN in ("ball_in_cup", "quadruped"):
            columns = [layout[f"gravity.{axis}"][0] for axis in "xyz"]
            encoded = style[:, columns]
            gravity = encoded.to(torch.float64)
            encoded_default = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float32, device=self.device)
            exact_default = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64, device=self.device)
            gravity = torch.where(encoded == encoded_default, exact_default, gravity)
            self._simulator.configure_gravity(gravity.cpu().numpy())
        elif self.DOMAIN == "finger":
            column = layout["agent_physics.proximal_density"][0]
            self._simulator.configure_proximal_density(style[:, column].cpu().numpy())
        else:
            column = layout["agent_physics.upper_arm_density"][0]
            self._simulator.configure_upper_arm_density(style[:, column].cpu().numpy())
        return result

    def _layout(self):
        offset = 0
        layout = []
        for key, (width, _) in self._resume_fields.items():
            render_offset = None
            if key == "qpos":
                render_offset = offset
                offset += width
            elif self.DOMAIN == "finger" and key == ("model", "target_site_xz"):
                render_offset = offset
                offset += width
            elif self.DOMAIN == "manipulator" and key in (("model", "target_body_xz"), ("model", "target_body_qwqy")):
                render_offset = offset
                offset += width
            layout.append((key if isinstance(key, tuple) else (key,), width, render_offset))
        return tuple(layout)

    def _layout_dtypes(self):
        return {key if isinstance(key, tuple) else (key,): dtype for key, (_, dtype) in self._resume_fields.items()}

    def _pack_render_state(self, state):
        rows = [state["qpos"]]
        if self.DOMAIN == "finger":
            rows.append(state["model", "target_site_xz"])
        elif self.DOMAIN == "manipulator":
            rows.extend((state["model", "target_body_xz"], state["model", "target_body_qwqy"]))
        return torch.cat(rows, dim=1).to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, self.state_dim)
        nq = self._resume_fields["qpos"][0]
        values = {"qpos": rows[:, :nq].clone()}
        if self.DOMAIN == "finger":
            values["model"] = {"target_site_xz": rows[:, nq : nq + 2].clone()}
        elif self.DOMAIN == "manipulator":
            values["model"] = {
                "target_body_xz": rows[:, nq : nq + 2].clone(),
                "target_body_qwqy": rows[:, nq + 2 : nq + 4].clone(),
            }
        return TensorDict(values, [len(rows)], device=self.device)

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
