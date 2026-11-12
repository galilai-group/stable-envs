from __future__ import annotations

import torch
from tensordict import TensorDict

from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict
from .simulation.execution import dmc_backend, make_dmc_adapter
from .simulation.factory import ACTION_DIMS, FIELDS, RENDER_KEYS
from .simulation.physics import LEAVES


class DelegatedSemantics:
    """Select native DMC or CUDA Warp physics; ``backend`` still selects rendering.

    The physics Variation leaves of the simulator are reset-time leaves and travel in State."""

    action_dtype = torch.float64
    action_low, action_high = -1.0, 1.0
    execution_mode = "AUTO_NATIVE_MJWARP"
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

    def __init__(self, *args, simulation_backend=None, mjwarp_min_envs=None, **kwargs):
        """Select physics at construction.

        ``None`` uses native physics on CPU and MuJoCo Warp on CUDA at every batch size. ``"auto"`` uses
        native physics below ``mjwarp_min_envs`` environments and MuJoCo Warp at or above it on CUDA; the
        default is the measured cutoff of the task. ``"native"``, ``"mjwarp"`` and (Reacher) ``"mjx"``
        select explicitly.
        """
        self.simulation_backend = simulation_backend
        self.mjwarp_min_envs = mjwarp_min_envs
        super().__init__(*args, **kwargs)
        self._simulator = self._make_simulator()
        self.execution_mode = {"native": "DELEGATED_CPU", "mjx": "JAX_MJX", "mjwarp": "MUJOCO_WARP"}[
            self.simulation_backend
        ]

    def _make_simulator(self):
        self.simulation_backend = dmc_backend(
            self._simulator_key(),
            self.B,
            self.device,
            self.simulation_backend,
            mjwarp_min_envs=self.mjwarp_min_envs,
        )
        return make_dmc_adapter(
            self._simulator_key(),
            self.B,
            self.device,
            self._seed,
            simulation_backend=self.simulation_backend,
        )

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
        leaves = sorted(self.FRAME_PHYSICS)
        if leaves and "style" not in kwargs and isinstance(state, TensorDict):
            keys = set(state.keys(True, True))
            variation = dict(kwargs.pop("variation", None) or {})
            drawn = [p for p in leaves if ("physics", *p.split(".")) in keys and p not in variation]
            if drawn:
                values = torch.cat([state[("physics", *p.split("."))] for p in drawn], 1)
                values = values.to(device=self.device, dtype=torch.float32)
                rows = self._style.values.shape[0]
                current = torch.cat([self._style[p].reshape(rows, -1) for p in drawn], 1)
                # The live style already draws a State this env produced; skip the per-call override.
                if values.shape != current.shape or not torch.equal(values, current):
                    variation.update({p: values[:, i : i + 1] for i, p in enumerate(drawn)})
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

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active, capture_state=False)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()
