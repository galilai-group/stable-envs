from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict
from .simulation.execution import make_dmc_adapter
from .simulation.factory import ACTION_DIMS, FIELDS, RENDER_KEYS
from .simulation.physics import LEAVES


class DelegatedSemantics:
    """Delegate simulation to CPU dm_control lanes; ``device`` selects State and render placement.

    The physics Variation leaves of the simulator are reset-time leaves and travel in State."""

    action_dtype = torch.float64
    action_low, action_high = -1.0, 1.0
    execution_mode = "DELEGATED_CPU"
    #: The simulator key; the domain when empty.
    SIMULATOR = ""

    @classmethod
    def _simulator_key(cls) -> str:
        return cls.SIMULATOR or cls.DOMAIN

    @property
    def action_dim(self):
        return ACTION_DIMS[self._simulator_key()]

    @property
    def _physics_paths(self) -> frozenset[str]:
        return frozenset(leaf.path for leaf in LEAVES.get(self._simulator_key(), ()))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._simulator = make_dmc_adapter(self._simulator_key(), self.B, self.device, self._seed)

    @property
    def _resume_fields(self):
        return FIELDS[self._simulator_key()]

    def make_variation_space(self):
        space = dict(super().make_variation_space().items())
        groups: dict[str, dict] = {}
        for leaf in LEAVES.get(self._simulator_key(), ()):
            group, name = leaf.path.split(".")
            groups.setdefault(group, {})[name] = Box(
                low=leaf.low, high=leaf.high, shape=(1,), init_value=[leaf.default]
            )
        for group, leaves in groups.items():
            space[group] = Dict({**(dict(space[group].items()) if group in space else {}), **leaves})
        return Dict(space)

    @property
    def reset_value_paths(self):
        return frozenset(self.physics_variation_leaves)

    @property
    def reset_sample_paths(self):
        return frozenset(self.physics_variation_leaves)

    def _fields(self):
        return {key if isinstance(key, tuple) else (key,): spec for key, spec in self._resume_fields.items()}

    def _render_keys(self):
        return (("qpos",), *RENDER_KEYS.get(self._simulator_key(), ()))

    def _layout(self):
        fields, offsets, offset = self._fields(), {}, 0
        for key in self._render_keys():
            offsets[key] = offset
            offset += fields[key][0]
        return tuple((key, width, offsets.get(key)) for key, (width, _) in fields.items())

    def _layout_dtypes(self):
        return {key: dtype for key, (_, dtype) in self._fields().items()}

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError(f"Expected full {self.DOMAIN} State")
        return torch.cat([state[key] for key in self._render_keys()], dim=1).to(self.device, torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, self.state_dim)
        fields, values, offset = self._fields(), {}, 0
        for key in self._render_keys():
            width = fields[key][0]
            node = values
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = rows[:, offset : offset + width].clone()
            offset += width
        return TensorDict(values, [len(rows)], device=self.device)

    def render(self, state=None, **kwargs):
        """A State's own physics values draw the leaves that move pixels, unless the call names them."""
        state = self.get_state() if state is None else state
        if "style" not in kwargs and isinstance(state, TensorDict):
            keys = set(state.keys(True, True))
            variation = dict(kwargs.pop("variation", None) or {})
            for leaf in self.FRAME_PHYSICS:
                key = ("physics", *leaf.split("."))
                if key in keys:
                    variation.setdefault(leaf, state[key].to(torch.float32))
            kwargs["variation"] = variation or None
        return super().render(state, **kwargs)

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
        return step_pipeline(self._simulator)

    def _step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()
