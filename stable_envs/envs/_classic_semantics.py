from __future__ import annotations

import numpy as np
import torch

from .._adapters.classic.local import _PHYSICS_SPACES, CLASSIC_FIELDS, ClassicControlAdapter, validate_classic_state
from .._execution.protocol import StepPipeline, step_pipeline
from ..spaces import Box, Dict
from ..specs import Spec


class ClassicControlSemantics:
    """Compose an existing exact renderer with the authoritative CPU simulator."""

    execution_mode = "DELEGATED_CPU"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .._adapters.classic.execution import make_classic_adapter

        self._simulator = make_classic_adapter(
            self.CLASSIC_DOMAIN, self.B, self.device, self._seed, ClassicControlAdapter
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

    def _configure_simulator(self):
        layout = {name: (off, width) for name, off, width in self.variation_space.flat_layout()}
        style = self.get_variation()
        values = {
            name: style[:, layout[f"physics.{name}"][0]].detach().cpu().numpy()
            for name in _PHYSICS_SPACES[self.CLASSIC_DOMAIN]
        }
        self._simulator.configure(values)

    def set_variation(self, variation=None, **overrides):
        result = super().set_variation(variation, **overrides)
        if hasattr(self, "_simulator"):
            self._configure_simulator()
        return result

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
        sim = torch.cat(
            [state[name].to(self.device, torch.float64).reshape(self.B, 1) for name in self.sim_state_names], dim=1
        )
        super().set_sim_state(sim)

    def sim_state(self):
        raise NotImplementedError(
            f"{type(self).__name__}.sim_state is unavailable for recovered semantics; use get_state()"
        )

    def set_sim_state(self, value):
        raise NotImplementedError(
            f"{type(self).__name__}.set_sim_state cannot restore lifecycle/RNG; use set_state(full_state)"
        )

    def reset(self, seed=None, options=None):
        self._check_open("reset")
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

    def _step_pipeline(self) -> StepPipeline | None:
        owner = self._simulator
        return step_pipeline(owner)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        result = self._simulator.step(actions, active=active)
        self._sync_authoritative_state(result["state"])
        return result


    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
        super()._close()
