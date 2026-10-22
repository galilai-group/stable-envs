"""DMC lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np

from ...._execution.process import ProcessAdapter
from .factory import make_local


def _lanes(domain, count, seed):
    return make_local(domain, count, seed=seed)


class DMCProcessAdapter(ProcessAdapter):
    """Run DMC lanes in worker processes; physics stays on CPU when outputs use CUDA."""

    name = "DMC"

    def __init__(self, domain, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        self.domain = domain
        super().__init__(partial(_lanes, domain), num_envs, device, int(seed), processes=processes, cpus=cpus)

    def configure_physics(self, values):
        self._check_idle()
        if self.physics_layout is None:
            raise ValueError(f"{self.domain} declares no physics Variation")
        values = np.asarray(values, np.float64).reshape(self.num_envs, self.physics_dim).copy()
        if not np.isfinite(values).all():
            raise ValueError("Configuration values must be finite")
        self.call("configure_physics", values)
