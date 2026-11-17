from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.style import StyleTensor
from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict
from ...specs import Spec
from .simulation.local import (
    _PHYSICS_SPACES,
    CLASSIC_FIELDS,
    STATES,
    physics_field,
    validate_classic_state,
)

_EXECUTION_MODES = {"gym": "DELEGATED_CPU", "numpy": "NATIVE_NUMPY", "numba": "NATIVE_NUMBA", "torch": "NATIVE_TORCH"}


class ClassicControlSemantics:
    """Pair the exact renderer with batched or Gymnasium steps."""

    #: ``__init__`` sets the mode of the selected step backend.
    execution_mode = "AUTO_BATCHED"
    #: Every simulator returns new output tensors on each step.
    _step_outputs_owned = True
    #: Physics implementations that ``simulation_backend`` accepts besides ``"auto"``.
    simulation_backends: tuple[str, ...] = ("gym", "numpy", "torch")

    def __init__(self, *args, simulation_backend: str | None = "auto", **kwargs):
        """Select the step backend.

        ``"auto"`` selects from the device and the batch size. On the CPU, small batches step one Gymnasium
        environment per batch row and large batches step in NumPy. On CUDA, small batches step with the CPU choice and
        move their results to the GPU, and large batches step and keep their State on the GPU in Torch.
        The class ships the switch points (``CPU_MIN_ENVS`` and ``CUDA_MIN_ENVS``); the ``large_backend_min_envs``
        tuning setting and ``stable-envs tune`` take precedence.

        ``"gym"`` steps one Gymnasium environment per batch row. ``"numpy"`` or ``"torch"`` selects a batched
        backend on any device. NumPy steps equal Gymnasium steps, and Torch steps agree with them to rounding.
        Gymnasium resets every backend, so a seed gives the same initial states on each.
        """
        super().__init__(*args, **kwargs)
        self._group = (self.CLASSIC_DOMAIN, "float64")
        sim = [self._schema.slices[name].start for name in self.sim_state_names]
        if sim != list(range(len(sim))):
            raise TypeError(f"{type(self).__name__}: the simulation variables must be the first float64 columns")
        self._drawn = tuple(self.sim_state_names[i] for i in self.render_from_sim)
        from .simulation.batched import classic_backend
        from .simulation.execution import make_step_adapter

        self.simulation_backend = classic_backend(self.CLASSIC_DOMAIN, self.device, self.B, simulation_backend)
        self.execution_mode = _EXECUTION_MODES[self.simulation_backend]
        sink = (self._sim, self._render_state, self.sim_state_names, self.render_from_sim)
        self._simulator = make_step_adapter(
            self.CLASSIC_DOMAIN, self.simulation_backend, self.B, self.device, self._seed, sink
        )
        self._configure_simulator()
        self.reset(seed=self._seed)

    @property
    def _schema(self):
        return STATES[self.CLASSIC_DOMAIN]

    def _with_physics_variation(self, base):
        physics = {
            name: Box(low=low, high=high, shape=(1,), dtype=np.float32, init_value=[default])
            for name, (low, high, default) in _PHYSICS_SPACES[self.CLASSIC_DOMAIN].items()
        }
        return Dict({**dict(base.items()), "physics": Dict(physics)})

    @property
    def dynamics_variation_leaves(self):
        return frozenset(f"physics.{name}" for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN])

    @property
    def reset_value_paths(self):
        return self.dynamics_variation_leaves

    @property
    def reset_sample_paths(self):
        return self.dynamics_variation_leaves

    def _configure_simulator(self):
        layout = {name: (off, width) for name, off, width in self.variation_space.flat_layout()}
        style = self.get_variation()
        values = {
            name: style[:, layout[f"physics.{name}"][0]].detach().cpu().numpy()
            for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN]
        }
        self._simulator.configure(values)

    def _adopt_physics_state(self, state):
        """Show a restored state's physics in the variation for this episode; defaults stay."""
        values = self._style.values.clone()
        offsets = {path: off for path, off, _ in self.variation_space.flat_layout()}
        for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN]:
            leaf = f"physics.{name}"
            column = self._schema.get(state, physics_field(name))
            values[:, offsets[leaf]] = column.reshape(-1).to(values.device, values.dtype)
        self._style = StyleTensor(values, self.variation_space.flat_layout())
        self._variation_is_default = False

    def _configure_variation(self, exact=None):
        super()._configure_variation(exact)
        if hasattr(self, "_simulator"):
            self._configure_simulator()

    def _layout(self):
        offsets = {self.sim_state_names[index]: packed for packed, index in enumerate(self.render_from_sim)}
        return tuple(
            (tuple(name.split("/")), width, offsets.get(name))
            for name, (width, _) in CLASSIC_FIELDS[self.CLASSIC_DOMAIN].items()
        )

    def _layout_dtypes(self):
        return {tuple(name.split("/")): dtype for name, (_, dtype) in CLASSIC_FIELDS[self.CLASSIC_DOMAIN].items()}

    def _validate_state(self, state, *, batch=None):
        validate_classic_state(self.CLASSIC_DOMAIN, state, batch=batch)

    def _validate_render_state(self, state):
        """Check the structure of ``state`` and that the fields the renderer draws are finite."""
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
        if len(state.batch_size) != 1:
            raise ValueError(f"Expected State batch [N], got {list(state.batch_size)}")
        self._schema.validate(state)
        drawn = [(name, self._schema.get(state, name)) for name in self._drawn]
        if not bool(torch.stack([torch.isfinite(value).all() for _, value in drawn]).all()):
            for name, value in drawn:
                if not torch.isfinite(value).all():
                    raise ValueError(f"State field {name} contains non-finite values")

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            return super()._pack_render_state(state)
        group = state.get(self._group, None)
        if not torch.is_tensor(group):
            # A render projection (``_unpack_render_state``) holds the drawn fields, not the group.
            return super()._pack_render_state(state)
        return group[..., list(self.render_from_sim)].to(self.device, torch.float32)

    @property
    def state_spec(self):
        return self._schema.spec(self.device)

    @property
    def action_spec(self):
        if self.CLASSIC_DOMAIN == "cartpole":
            return Spec(shape=(1,), dtype=torch.int64, low=0, high=1, discrete_n=2, device=self.device)
        if self.CLASSIC_DOMAIN in {"acrobot", "mountain_car"}:
            return Spec(shape=(1,), dtype=torch.int64, low=0, high=2, discrete_n=3, device=self.device)
        bound = 2.0 if self.CLASSIC_DOMAIN == "pendulum" else 1.0
        return Spec(shape=(1,), dtype=torch.float32, low=-bound, high=bound, device=self.device)

    def _sync_authoritative_state(self, state):
        self._rows_stale = False
        sim = state.get(self._group)[:, : len(self.sim_state_names)].to(self.device, torch.float64)
        super().set_sim_state(sim)

    def _sync_stale_rows(self):
        """Copy the simulator's State into the simulation and render rows when a step left them behind."""
        if self.__dict__.get("_rows_stale"):
            self._sync_authoritative_state(self._simulator.get_state())

    def _get_render_state(self):
        self._sync_stale_rows()
        return super()._get_render_state()

    def _set_render_state(self, states, variation=None, style=None):
        self._rows_stale = False
        return super()._set_render_state(states, variation=variation, style=style)

    def _sim_columns(self, drawn):
        self._sync_stale_rows()
        return super()._sim_columns(drawn)

    def _render(self, states=None, *args, **kwargs):
        if states is None:
            self._sync_stale_rows()
        return super()._render(states, *args, **kwargs)

    def sim_state(self):
        raise NotImplementedError(f"{type(self).__name__}.sim_state is unavailable; use get_state()")

    def set_sim_state(self, value):
        raise NotImplementedError(
            f"{type(self).__name__}.set_sim_state cannot restore lifecycle/RNG; use set_state(full_state)"
        )

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        self.draw_reset_values(seed, options)
        state = self._simulator.reset(seed=seed, options=options)
        self._sync_authoritative_state(state)
        return state

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)
        self._sync_authoritative_state(state)
        self._adopt_physics_state(state)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _stack_states(self, states):
        """Stack the States of a step sequence; the GPU simulator stacks its own States with one copy."""
        from ..._core.stepping import _stack_states

        stack = getattr(self._simulator, "stack_states", None)
        stacked = None if stack is None else stack(states)
        return _stack_states(states) if stacked is None else stacked

    def _step(self, actions, *, active=None, capture_state=True):
        self._check_open("step")
        result = self._simulator.step(actions, active=active, capture_state=capture_state)
        # The rows follow the simulator only when they are read.
        if not getattr(self._simulator, "writes_render_state", False):
            self._rows_stale = True
        return result

    def _host_step(self, actions, move):
        """One action without frames: the result from the simulator's arrays in one pass, or None.

        A CPU simulator gives host arrays that ``move`` copies into the result. The CUDA simulator builds the result
        from one copy of its packed buffer.
        """
        self._check_open("step")
        simulator = self._simulator
        if self.device.type == "cpu":
            leaves = getattr(simulator, "host_leaves", None)
            got = None if leaves is None else leaves(actions, getattr(move, "needs_state", True))
            if got is None:
                return None
            self._rows_stale = True
            return move(got)[0]
        result = getattr(simulator, "step_result", None)
        if result is None or not hasattr(move, "final_state"):
            return None
        return result(actions, move.sequence, move.state_indices, move.final_state)

    def _advance_without_snapshot(self, actions, *, active=None):
        # The CUDA simulator copies its whole State with the outputs either way; a later snapshot costs more.
        capture = getattr(self._simulator, "writes_render_state", False)
        return self._step(actions, active=active, capture_state=capture)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
