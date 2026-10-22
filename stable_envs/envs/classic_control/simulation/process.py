"""Classic-control Acrobot lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np

from ...._execution.process import ProcessAdapter
from .local import ClassicControlAdapter, validate_classic_state


def _lanes(domain, count, seed):
    adapter = ClassicControlAdapter(domain, count)
    try:
        adapter.reset(seed)
    except BaseException:
        adapter.close()
        raise
    return adapter


class AcrobotProcessAdapter(ProcessAdapter):
    """Run Acrobot CPU simulators in worker processes with shared State and task columns."""

    name = "Acrobot"

    def __init__(self, domain, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        if domain != "acrobot":
            raise ValueError("AcrobotProcessAdapter is bounded to Acrobot")
        seed = int(seed)
        np.random.SeedSequence(seed)  # rejects invalid seeds before workers spawn
        super().__init__(partial(_lanes, domain), num_envs, device, seed, processes=processes, cpus=cpus)

    def validate(self, state):
        validate_classic_state(self.domain, state, batch=self.num_envs)

    def reset(self, seed=None, options=None):
        self._check_idle()
        if seed is not None:
            seed = int(seed)
            np.random.SeedSequence(seed)
        return self._reset(seed, [(options,)] * len(self._slices))

    def configure(self, values):
        self._check_idle()
        columns = {key: np.asarray(value) for key, value in values.items()}
        if any(value.shape != (self.num_envs,) for value in columns.values()):
            raise ValueError("Physical parameters must have one scalar per lane")
        self.call("configure", columns)
