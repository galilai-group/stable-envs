"""DMC lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from ...._execution.process import ProcessAdapter
from .factory import make_local


def _lanes(domain, count, seed):
    return make_local(domain, count, seed=seed)


class DMCProcessAdapter(ProcessAdapter):
    """Run DMC lanes in worker processes; physics stays on CPU when outputs use CUDA."""

    _worker_family = _step_family = _state_family = "DMC"

    def __init__(self, domain, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        self.domain = domain
        super().__init__(partial(_lanes, domain), num_envs, device, int(seed), processes=processes, cpus=cpus)

    def _inspect(self, prototype):
        self.action_dim = prototype.envs[0].action_spec().shape[0]
        self.fields = {key: (value.shape[1], value.dtype) for key, value in prototype.get_state().items(True, True)}
        physics = prototype.physics
        self.physics_layout = None if physics is None else physics.layout
        self._physics_dim = 0 if physics is None else physics.spec.dim

    def _host_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"DMC actions must be finite with shape {(self.num_envs, self.action_dim)}")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or not active.all():
                raise ValueError(
                    "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
                )
        return actions, None

    def configure_physics(self, values):
        self._check_idle()
        if self.physics_layout is None:
            raise ValueError(f"{self.domain} declares no physics Variation")
        values = np.asarray(values, np.float64).reshape(self.num_envs, self._physics_dim).copy()
        if not np.isfinite(values).all():
            raise ValueError("Configuration values must be finite")
        self.call("configure_physics", values)
