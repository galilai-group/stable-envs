"""Pinned SWM oracle bridge for Cheetah run, Hopper hop and Walker walk."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

WRAPPERS = {
    "cheetah": ("cheetah", "CheetahDMControlWrapper"),
    "hopper": ("hopper", "HopperDMControlWrapper"),
    "walker": ("walker", "WalkerDMControlWrapper"),
}

EXTRA_INFO = {
    "cheetah": ("speed", "torso_height", "bfoot_height", "ffoot_height", "angmomentum"),
    "hopper": ("speed", "height", "angmomentum"),
    "walker": (
        "speed",
        "torso_height",
        "torso_upright",
        "left_foot_height",
        "right_foot_height",
        "left_thigh_height",
        "right_thigh_height",
    ),
}


class DMCContinuationReference:
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
            algorithm, keys, position, has_gauss, cached = env.task.random.get_state()
            assert algorithm == "MT19937"
            values = {
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
                            "position": np.array([position], np.int64),
                            "has_gauss": np.array([has_gauss], bool),
                            "cached_gaussian": np.array([cached], np.float64),
                        },
                    }
            snapshots.append(TensorDict(values, batch_size=[]).clone())
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
        outputs = [
            wrapper.step(np.asarray(action, np.float64))
            for wrapper, action in zip(self.envs, actions, strict=True)
        ]
        infos = [output[4] for output in outputs]
        assert all(info["env_name"].lower() == self.domain for info in infos)
        info = TensorDict(
            {
                key: torch.as_tensor(np.stack([row[key] for row in infos]))
                for key in ("qpos", "qvel", *EXTRA_INFO[self.domain])
            },
            [self.num_envs],
        )
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([output[1] for output in outputs], dtype=torch.float64),
                "terminated": torch.tensor([output[2] for output in outputs], dtype=torch.bool),
                "truncated": torch.tensor([output[3] for output in outputs], dtype=torch.bool),
                "observation": torch.as_tensor(np.stack([output[0] for output in outputs])),
                "success": torch.tensor([row["success"] for row in infos], dtype=torch.float64),
                "score": torch.tensor([row["score"] for row in infos], dtype=torch.float64),
                "info": info,
            },
            [self.num_envs],
        )

    @property
    def step_limit(self):
        limits = {int(wrapper.env._env._step_limit) for wrapper in self.envs}
        assert len(limits) == 1
        return limits.pop()

    @property
    def action_dim(self):
        dims = {int(wrapper.action_space.shape[0]) for wrapper in self.envs}
        assert len(dims) == 1
        return dims.pop()

    def warmstarts(self):
        return np.stack([wrapper.env.physics.data.qacc_warmstart.copy() for wrapper in self.envs])

    def controls(self):
        return np.stack([wrapper.env.physics.data.ctrl.copy() for wrapper in self.envs])

    def task_caches(self):
        return [
            {key: value for key, value in wrapper.env.task.__dict__.items() if key != "_random"}
            for wrapper in self.envs
        ]

    def close(self):
        for env in self.envs:
            env.close()
