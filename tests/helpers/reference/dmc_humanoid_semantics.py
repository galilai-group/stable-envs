"""Pinned SWM oracle bridge for the historical Humanoid Walk default."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict


class DMCHumanoidReference:
    def __init__(self, num_envs=1, options=None):
        from stable_worldmodel.envs.dmcontrol.humanoid import HumanoidDMControlWrapper

        self.num_envs = int(num_envs)
        self.envs = [HumanoidDMControlWrapper(seed=i) for i in range(self.num_envs)]
        self.options = options

    def reset(self, seed=None):
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + i, options=self.options)
        return self.get_state()

    def get_state(self):
        rows = []
        for wrapper in self.envs:
            env, physics = wrapper.env._env, wrapper.env.physics
            algorithm, keys, position, has_gauss, cached = env.task.random.get_state()
            assert algorithm == "MT19937"
            rows.append(
                TensorDict(
                    {
                        "qpos": physics.data.qpos.copy(),
                        "qvel": physics.data.qvel.copy(),
                        "time": np.array([physics.data.time], np.float64),
                        "warmstart": physics.data.qacc_warmstart.copy(),
                        "task": {
                            "cumulative_reward": np.array([wrapper._cumulative_reward], np.float64)
                        },
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
                    },
                    batch_size=[],
                ).clone()
            )
        return torch.stack(rows)

    def set_state(self, state):
        for wrapper, row in zip(self.envs, state.cpu().unbind(0), strict=True):
            env, physics = wrapper.env._env, wrapper.env.physics
            wrapper.set_state(row["qpos"].numpy(), row["qvel"].numpy())
            physics.data.time = row["time"].item()
            physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
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
        assert all(info["env_name"].lower() == "humanoid" for info in infos)
        info = TensorDict(
            {
                key: torch.as_tensor(np.stack([row[key] for row in infos]))
                for key in ("qpos", "qvel", "speed", "torso_upright", "head_height")
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
        values = {int(wrapper.env._env._step_limit) for wrapper in self.envs}
        assert len(values) == 1
        return values.pop()

    @property
    def action_dim(self):
        values = {int(wrapper.action_space.shape[0]) for wrapper in self.envs}
        assert len(values) == 1
        return values.pop()

    def close(self):
        for env in self.envs:
            env.close()
