"""Rocket lanes in worker processes."""

from __future__ import annotations

from functools import partial

import torch

from ...._execution.process import ProcessAdapter
from .adapter import RocketAdapter, check_values, state_columns

#: Lanes one process may own. A staged restore or reset holds two PyBullet clients per lane, frames
#: use one more client, and PyBullet allows 1024 clients in one process.
LANES_PER_PROCESS = 511


def _lanes(config, frame, count, seed):
    return RocketAdapter(count, "cpu", seed, config, frame)


class RocketProcessAdapter(ProcessAdapter):
    """Run Rocket replay owners in worker processes with the whole State in shared buffers."""

    name = "Rocket"

    def __init__(self, batch, device, seed, config=None, *, processes=8, cpus=None, frame=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("Rocket supports CPU dynamics and CPU/CUDA State storage")
        if type(processes) is int and processes > 0 and -(-batch // processes) > LANES_PER_PROCESS:
            raise ValueError(f"Rocket processes may own at most {LANES_PER_PROCESS} lanes each")
        super().__init__(partial(_lanes, config, frame), batch, device, seed, processes=processes, cpus=cpus)

    @property
    def template(self):
        """A lane whose configuration and Variation space describe the batch."""
        self._check_open()
        return self._metadata.template

    def validate(self, state):
        self._check_open()
        return state_columns(state, self.fields, self._metadata.template, self.num_envs)

    def rgba(self):
        return self.call("rgba").to(self.device)

    def configure(self, values):
        self._check_idle()
        check_values(self._metadata.template, values, self.num_envs)
        self.call("configure", values)

    def reset(self, seed=None, options=None, values=None):
        self._check_idle()
        template = self._metadata.template
        if values is not None and len(values) != self.num_envs:
            raise ValueError("Expected one Rocket Variation row per lane")
        for lane in range(self.num_envs):
            template.check_reset(
                None if seed is None else seed + lane, options, None if values is None else values[lane]
            )
        return self._reset(seed, [(options, None if values is None else values[rows]) for rows in self._slices])

    def draw(self, body, links, colors, size, ss):
        """Draw explicit poses with TinyRenderer in this process."""
        self._check_open()
        return self._metadata.draw(body, links, colors, size, ss)
