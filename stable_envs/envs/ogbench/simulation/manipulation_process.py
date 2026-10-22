"""OGBench Cube and Scene lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.process import ProcessAdapter, process_worker
from .manipulation import OGBManipulationAdapter

_worker = process_worker


def _lanes(family, reward_task_id, count, seed):
    return OGBManipulationAdapter(family, count, seed=seed, reward_task_id=reward_task_id)


class OGBManipulationProcessAdapter(ProcessAdapter):
    """Run Cube/Scene worlds in worker processes with staged checkpoint restore.

    The parent keeps one metadata instance for validation and configuration.
    """

    _worker_family = "OGBench manipulation"
    _step_family = _state_family = "Cube/Scene"
    _keep_metadata = True
    execution_mode = "DELEGATED_CPU"

    def __init__(self, family, num_envs=1, device="cpu", seed=0, reward_task_id=None, *, processes=8, cpus=None):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        self.family = family
        super().__init__(
            partial(_lanes, family, reward_task_id), num_envs, device, seed, processes=processes, cpus=cpus
        )

    def _worker_target(self):
        return _worker

    def _inspect(self, prototype):
        self.fields, self.nq = prototype.fields, prototype.nq

    def _task_template(self, prototype):
        # Stepping would change the metadata instance; these are the task outputs of every step.
        return TensorDict(
            {
                name: torch.zeros(1, dtype=dtype)
                for name, dtype in (
                    ("reward", torch.float64),
                    ("success", torch.bool),
                    ("terminated", torch.bool),
                    ("truncated", torch.bool),
                )
            },
            [1],
        )

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
        # ``variation_values`` reads the shared State.
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

    def _host_step(self, actions, active):
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if (
            actions.shape != (self.num_envs, 5)
            or actions.dtype not in (np.float32, np.float64)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        return actions, self._host_active(active)
