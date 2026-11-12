from __future__ import annotations

import numpy as np
import torch

from ..._core.style import StyleTensor
from ..._execution.protocol import StepPipeline, step_pipeline
from ...spaces import Box, Dict
from ...specs import Spec
from .simulation.local import (
    _PHYSICS_SPACES,
    CLASSIC_FIELDS,
    ClassicControlAdapter,
    physics_state_key,
    validate_classic_state,
)

_EXECUTION_MODES = {"gym": "DELEGATED_CPU", "numpy": "NATIVE_NUMPY", "torch": "NATIVE_TORCH"}


class ClassicControlSemantics:
    """Pair the exact renderer with batched or Gymnasium steps."""

    #: ``__init__`` sets the mode of the selected step backend.
    execution_mode = "AUTO_BATCHED"
    #: Every simulator returns new output tensors on each step.
    _step_outputs_owned = True

    def __init__(self, *args, simulation_backend: str | None = None, torch_min_envs: int | None = None, **kwargs):
        """Select the step backend.

        By default, CUDA lanes step, reset and keep their State on the GPU in Torch, and CPU lanes step in
        NumPy, or with Gymnasium below ``NUMPY_MIN_ENVS`` lanes, where it is faster. ``"auto"`` uses Torch
        only for ``torch_min_envs`` (default ``TORCH_MIN_ENVS``) or more CUDA lanes, and NumPy on the host
        for fewer lanes.

        ``"gym"`` steps one Gymnasium environment per lane. ``"numpy"`` or ``"torch"`` selects a batched
        backend on any device. NumPy steps equal Gymnasium steps, and Torch steps agree with them to rounding.
        On the CPU, Gymnasium resets every backend. On CUDA, Torch resets the lanes on the GPU.
        """
        super().__init__(*args, **kwargs)
        from .simulation.batched import TORCH_MIN_ENVS, BatchedClassicAdapter, classic_backend
        from .simulation.execution import make_classic_adapter

        cutoff = TORCH_MIN_ENVS if torch_min_envs is None else torch_min_envs
        self.simulation_backend = classic_backend(self.CLASSIC_DOMAIN, self.device, self.B, simulation_backend, cutoff)
        self.execution_mode = _EXECUTION_MODES[self.simulation_backend]
        if self.simulation_backend == "gym":
            self._simulator = make_classic_adapter(
                self.CLASSIC_DOMAIN, self.B, self.device, self._seed, ClassicControlAdapter
            )
        elif self.simulation_backend == "torch" and self.device.type == "cuda":
            from .simulation.device import DeviceClassicAdapter

            sink = (self._sim, self._render_state, self.sim_state_names, self.render_from_sim)
            self._simulator = DeviceClassicAdapter(self.CLASSIC_DOMAIN, self.B, self.device, self._seed, sink)
        else:
            self._simulator = BatchedClassicAdapter(
                self.CLASSIC_DOMAIN, self.B, self.device, self._seed, self.simulation_backend
            )
        self._configure_simulator()
        self.reset(seed=self._seed)

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
        """Show a restored State's physics in the Variation for this episode; defaults stay."""
        values = self._style.values.clone()
        offsets = {path: off for path, off, _ in self.variation_space.flat_layout()}
        for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN]:
            leaf = f"physics.{name}"
            values[:, offsets[leaf]] = state[physics_state_key(leaf)].reshape(-1).to(values.device, values.dtype)
        self._style = StyleTensor(values, self.variation_space.flat_layout())
        self._variation_is_default = False

    def _configure_variation(self, exact=None):
        super()._configure_variation(exact)
        if hasattr(self, "_simulator"):
            self._configure_simulator()

    def _layout(self):
        offsets = {self.sim_state_names[index]: packed for packed, index in enumerate(self.render_from_sim)}
        return tuple(
            (
                key if isinstance(key, tuple) else (key,),
                width,
                offsets.get(key) if isinstance(key, str) else None,
            )
            for key, (width, _) in CLASSIC_FIELDS[self.CLASSIC_DOMAIN].items()
        )

    def _layout_dtypes(self):
        return {
            key if isinstance(key, tuple) else (key,): dtype
            for key, (_, dtype) in CLASSIC_FIELDS[self.CLASSIC_DOMAIN].items()
        }

    def _validate_state(self, state, *, batch=None):
        validate_classic_state(self.CLASSIC_DOMAIN, state, batch=batch)

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
        sim = torch.cat(
            [state[name].to(self.device, torch.float64).reshape(self.B, 1) for name in self.sim_state_names], dim=1
        )
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

    def _step(self, actions, *, active=None):
        self._check_open("step")
        result = self._simulator.step(actions, active=active)
        # The rows follow the simulator only when they are read.
        if not getattr(self._simulator, "writes_render_state", False):
            self._rows_stale = True
        return result

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
