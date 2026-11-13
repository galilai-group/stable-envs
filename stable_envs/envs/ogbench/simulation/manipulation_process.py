"""OGBench Cube and Scene lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from ...._execution.process import ProcessAdapter, process_worker
from .manipulation import OGBManipulationAdapter

_worker = process_worker


def _lanes(family, reward_task_id, count, seed):
    return OGBManipulationAdapter(family, count, seed=seed, reward_task_id=reward_task_id)


class OGBManipulationProcessAdapter(ProcessAdapter):
    """Run Cube/Scene worlds in worker processes with staged checkpoint restore.

    The parent keeps one metadata instance for validation and configuration.
    """

    name = "Cube/Scene"
    execution_mode = "DELEGATED_CPU"

    def __init__(self, family, num_envs=1, device="cpu", seed=0, reward_task_id=None, *, processes=8, cpus=None):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        factory = partial(_lanes, family, reward_task_id)
        super().__init__(factory, num_envs, device, seed, processes=processes, cpus=cpus)

    def _worker_target(self):
        return _worker

    def validate(self, state):
        self._check_open()
        # Metadata validation is independent of workers advancing the next transition.
        return self._metadata.validate(state, _batch_size=self.num_envs)

    def reset(self, seed=None, options=None):
        self._check_idle()
        if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        lanes = self._metadata.envs[0].lane_options(options, self.num_envs)
        return self._reset(seed, [(lanes[rows],) for rows in self._slices])

    def configure(self, values):
        self._check_idle()
        if len(values) != self.num_envs:
            raise ValueError("Variation batch mismatch")
        for value in values:
            self._metadata.envs[0].validate_options({"variation": [], "variation_values": value})
        self.call("configure", values)
        # ``variation_values`` reads the shared state.
        self._request("get_state")

    def make_variation_space(self):
        self._check_open()
        return self._metadata.make_variation_space()

    def variation_values(self, layout, *, anchor=False):
        self._check_idle()
        namespace = "anchor" if anchor else "current"
        values = np.concatenate(
            [self._views["state", "variation", namespace, name].reshape(self.num_envs, -1) for name, _, _ in layout],
            axis=1,
        )
        return torch.from_numpy(values).to(self.device)
