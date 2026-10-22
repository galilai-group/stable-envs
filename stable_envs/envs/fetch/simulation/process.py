"""Fetch lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.process import ProcessAdapter
from .adapter import FetchAdapter, episode_limits


def _lanes(task, reward_type, threads, count, seed):
    adapter = FetchAdapter(count, "cpu", seed, task, reward_type)
    try:
        adapter._set_workers(min(threads, count))
    except BaseException:
        adapter.close()
        raise
    return adapter


class FetchProcessAdapter(ProcessAdapter, FetchAdapter):
    """Fetch simulators distributed across persistent spawned CPU processes.

    Optional threads run within each worker. Rendering and output-device
    transfer stay in the parent.
    """

    name = "Fetch"
    validate = FetchAdapter.validate

    def __init__(
        self,
        num_envs,
        device="cpu",
        seed=0,
        task="PickAndPlace",
        reward_type="sparse",
        *,
        processes=8,
        threads_per_process=1,
        cpus=None,
    ):
        if type(threads_per_process) is not int or threads_per_process < 1:
            raise ValueError("threads_per_process must be positive")
        self._workers = processes * threads_per_process
        factory = partial(_lanes, task, reward_type, threads_per_process)
        super().__init__(factory, num_envs, device, seed, processes=processes, cpus=cpus)
        self._mass = np.repeat(self._metadata.object_mass, self.num_envs)

    def reference_model(self):
        self._check_open()
        return self._metadata.reference_model()

    @property
    def object_mass(self) -> np.ndarray:
        self._check_open()
        return self._mass.copy()

    def configure_object_mass(self, values):
        self._check_open()
        values = np.asarray(values, np.float64).reshape(self.num_envs).copy()
        self.call("configure_object_mass", values)
        self._mass = values

    def reset_visual_colors(self, values):
        self._check_open()
        if len(values) != self.num_envs:
            raise ValueError("Expected one variation dictionary per lane")
        return self.call("reset_visual_colors", values)

    def _set_episode_limits(self, limits):
        self._check_open()
        self.call("_set_episode_limits", episode_limits(limits, self.num_envs))

    def reset(self, seed: int | None = None, variations=None) -> TensorDict:
        self._check_open()
        if variations is not None and len(variations) != self.num_envs:
            raise ValueError("Expected one variation dictionary per lane")
        state = self._reset(seed, [(None if variations is None else variations[rows],) for rows in self._slices])
        self._mass = self.call("object_mass")
        return state

    def advance(self, actions: torch.Tensor | np.ndarray, *, active=None) -> TensorDict:
        return self.step(actions, active=active, capture_state=False)
