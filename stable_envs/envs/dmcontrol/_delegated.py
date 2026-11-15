from __future__ import annotations

from functools import cached_property

import torch
from tensordict import TensorDict

from ..._core.physics import state_key
from ..._core.state_schema import StateSchema, get_field, schema_of
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import check_drawn_fields
from ...spaces import Box, Dict
from ...specs import CompositeSpec
from .simulation.execution import dmc_backend, dmc_path, make_dmc_adapter
from .simulation.factory import ACTION_DIMS, FIELDS, RENDER_KEYS
from .simulation.physics import LEAVES
from .simulation.state import field_name, state_schema


class DelegatedSemantics:
    """Select native DMC or CUDA Warp physics; ``backend`` still selects rendering.

    The physics variation leaves of the simulator are reset-time leaves and travel in state."""

    #: Physics implementations that ``simulation_backend`` accepts besides ``"auto"``.
    simulation_backends: tuple[str, ...] = ("native", "mjwarp")
    state_solver_caches = ("warmstart",)
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

    def __init__(self, *args, simulation_backend="auto", **kwargs):
        """Select physics at construction.

        ``"auto"`` uses native physics on CPU. On CUDA it uses native physics for small batches and MuJoCo Warp for
        large ones, from the measured ``large_backend_min_envs`` switch point. ``"native"``, ``"mjwarp"`` and
        (Reacher) ``"mjx"`` select explicitly.
        """
        self.simulation_backend = simulation_backend
        super().__init__(*args, **kwargs)
        self._simulator = self._make_simulator()
        self.execution_mode = {"native": "DELEGATED_CPU", "mjx": "JAX_MJX", "mjwarp": "MUJOCO_WARP"}[
            self.simulation_backend
        ]

    def _make_simulator(self):
        self.simulation_backend = dmc_backend(
            self._simulator_key(), self.B, self.device, self.simulation_backend, env_type=type(self)
        )
        self.simulation_backend_path = dmc_path(self._simulator_key())
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

    @property
    def _state_schema(self) -> StateSchema:
        """The layout of the State: its named fields, folded into one float64, one int64 and one bool group."""
        return state_schema(self._simulator_key(), self._resume_fields)

    @property
    def state_spec(self) -> CompositeSpec:
        """Specs of the State's group leaves, without the batch axis."""
        return self._state_schema.spec(self.device)

    def _validate_state(self, state, *, batch=None):
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
        self._state_schema.validate(state, finite=True)

    def _validate_render_state(self, state):
        fields = self._drawn_fields()
        if isinstance(state, TensorDict) and schema_of(state) is not None:
            # The drawn fields of a folded State, so that a missing or broken field the frame doesn't read passes.
            drawn: dict = {}
            for key, _ in fields:
                try:
                    value = get_field(state, field_name(key))
                except KeyError:
                    raise ValueError(f"State is missing {field_name(key)}, which the renderer draws") from None
                node = drawn
                for part in key[:-1]:
                    node = node.setdefault(part, {})
                node[key[-1]] = value
            state = TensorDict(drawn, state.batch_size, device=state.device)
        check_drawn_fields(state, fields)

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

    def _render_spans(self):
        """``(field name, start, stop)``: the State columns of each part of the compact render row, in order."""
        fields = self._fields()
        return tuple((field_name(key), 0, fields[key][0]) for key in self._render_keys())

    @cached_property
    def _render_runs(self):
        """The float64 group path and size, and the ``(start, stop)`` group columns that make the render row."""
        schema, runs = self._state_schema, []
        for name, start, stop in self._render_spans():
            where = schema.slices[name]
            start, stop = where.start + start, where.start + stop
            if runs and runs[-1][1] == start:
                runs[-1][1] = stop
            else:
                runs.append([start, stop])
        group = schema.groups[where.group]
        return group.path, group.size, tuple(map(tuple, runs))

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            raise TypeError(f"Expected full {self.DOMAIN} State")
        path, size, runs = self._render_runs
        group = state.get(path, None)
        if torch.is_tensor(group) and group.shape[-1] == size:
            parts = [group[:, start:stop] for start, stop in runs]
        else:
            # A render projection holds one leaf per field.
            parts = [get_field(state, name)[:, start:stop] for name, start, stop in self._render_spans()]
        rows = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
        return rows.to(self.device, torch.float32)

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

    def _render_fields(self):
        """The State leaves ``_pack_render_state`` reads."""
        return self._render_keys()

    def _drawn_fields(self):
        """The packed leaves and the physics leaves that move pixels, as ``(key, width)``."""
        fields = self._fields()
        keys = (*self._render_fields(), *(state_key(p) for p in sorted(self.FRAME_PHYSICS)))
        return tuple((key, fields[key][0]) for key in keys)

    def _render_state_input(self, state, **kwargs):
        """A State's own physics values draw the leaves that move pixels, unless the call names them."""
        state = self.get_state() if state is None else state
        leaves = [] if "style" in kwargs else sorted(self.FRAME_PHYSICS)
        variation = dict(kwargs.pop("variation", None) or {}) if leaves else {}
        drawn = [p for p in leaves if p not in variation]
        self._validate_render_state(state)
        if drawn:
            values = torch.cat([get_field(state, field_name(state_key(p))) for p in drawn], 1)
            values = values.to(device=self.device, dtype=torch.float32)
            rows = self._style.values.shape[0]
            current = torch.cat([self._style[p].reshape(rows, -1) for p in drawn], 1)
            # The live style already draws a State this env produced; skip the per-call override.
            if values.shape != current.shape or not torch.equal(values, current):
                variation.update({p: values[:, i : i + 1] for i, p in enumerate(drawn)})
        if leaves:
            kwargs["variation"] = variation or None
        return self._render_checked(self._pack_render_state(state), **kwargs)

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

    @property
    def _step_outputs_owned(self) -> bool:
        return getattr(self._simulator, "outputs_owned", False)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active, capture_state=False)

    def _host_step(self, actions, move):
        """One step on a host simulator: ``move`` puts its arrays on the device in one copy."""
        step = getattr(self._simulator, "step_leaves", None)
        if step is None:
            return None
        self._check_open("step")
        return move(step(actions))[0]

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()
