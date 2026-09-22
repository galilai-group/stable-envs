"""Pinned historical oracle for the first bounded DMC recovery cohort."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

WRAPPERS = {
    "acrobot": ("acrobot", "AcrobotDMControlWrapper"),
    "pendulum": ("pendulum", "PendulumDMControlWrapper"),
    "cartpole": ("cartpole", "CartpoleDMControlWrapper"),
}


class DMCFirstFamilyReference:
    def __init__(self, domain, num_envs=1, options=None):
        module_name, class_name = WRAPPERS[domain]
        module = importlib.import_module(f"stable_worldmodel.envs.dmcontrol.{module_name}")
        wrapper = getattr(module, class_name)
        self.domain = domain
        self.num_envs = num_envs
        self.envs = [wrapper(seed=i) for i in range(num_envs)]
        self.options = options

    def reset(self, seed=None):
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + i, options=self.options)
        return self.get_state()

    def get_state(self):
        snapshots = []
        for wrapper in self.envs:
            env = wrapper.env._env
            physics = env.physics
            algorithm, keys, pos, has_gauss, cached = env.task.random.get_state()
            assert algorithm == "MT19937"
            snapshots.append(
                TensorDict(
                    {
                        "qpos": physics.data.qpos.copy(),
                        "qvel": physics.data.qvel.copy(),
                        "time": np.array([physics.data.time], np.float64),
                        "warmstart": physics.data.qacc_warmstart.copy(),
                        "task": {"cumulative_reward": np.array([wrapper._cumulative_reward], np.float64)},
                        "lifecycle": {
                            "step_count": np.array([env._step_count], np.int64),
                            "reset_next_step": np.array([env._reset_next_step], bool),
                        },
                        "rng": {
                            "keys": keys.astype(np.int64),
                            "position": np.array([pos], np.int64),
                            "has_gauss": np.array([has_gauss], bool),
                            "cached_gaussian": np.array([cached], np.float64),
                        },
                    },
                    batch_size=[],
                ).clone()
            )
        return torch.stack(snapshots)

    def set_state(self, state):
        for wrapper, row in zip(self.envs, state.cpu().unbind(0), strict=True):
            env = wrapper.env._env
            wrapper.set_state(row["qpos"].numpy(), row["qvel"].numpy())
            env.physics.data.time = row["time"].item()
            env.physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
            env._step_count = row["lifecycle", "step_count"].item()
            env._reset_next_step = row["lifecycle", "reset_next_step"].item()
            wrapper._cumulative_reward = row["task", "cumulative_reward"].item()
            env.task.random.set_state(
                (
                    "MT19937",
                    row["rng", "keys"].numpy().astype(np.uint32),
                    row["rng", "position"].item(),
                    int(row["rng", "has_gauss"].item()),
                    row["rng", "cached_gaussian"].item(),
                )
            )

    def step(self, actions):
        outputs = [env.step(np.asarray(a, np.float64)) for env, a in zip(self.envs, actions, strict=True)]
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([out[1] for out in outputs], dtype=torch.float64),
                "terminated": torch.tensor([out[2] for out in outputs], dtype=torch.bool),
                "truncated": torch.tensor([out[3] for out in outputs], dtype=torch.bool),
                "observation": torch.as_tensor(np.stack([out[0] for out in outputs])),
                "score": torch.tensor([out[4]["score"] for out in outputs], dtype=torch.float64),
            },
            batch_size=[self.num_envs],
        )

    @property
    def step_limit(self):
        limits = {int(env.env._env._step_limit) for env in self.envs}
        assert len(limits) == 1
        return limits.pop()

    def close(self):
        for env in self.envs:
            env.close()
