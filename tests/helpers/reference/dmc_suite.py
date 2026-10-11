"""dm_control's own registered tasks as an independent State/task oracle.

Each lane is ``suite.load(domain, task)`` seeded with ``seed + lane``, driven like the
stable-envs cells: actions normalized to [-1, 1] and two control steps per action.
"""

import numpy as np
import torch
from tensordict import TensorDict

#: Episode-drawn model entries each task writes: ``State key -> (named model field, index)``.
EPISODE_FIELDS = {
    ("finger", "turn_easy"): {("model", "target_site_xz"): ("site_pos", ("target", ["x", "z"]))},
    ("reacher", "easy"): {("task", "target_pos"): ("geom_pos", "target")},
    ("swimmer", "swimmer6"): {
        ("model", "target_geom_xy"): ("geom_pos", ("target", ["x", "y"])),
        ("model", "target_light_xy"): ("light_pos", ("target_light", ["x", "y"])),
    },
}


class SuiteReference:
    def __init__(self, domain, task, num_envs, seed=0):
        from dm_control import suite
        from dm_control.suite.wrappers import action_scale

        self.domain, self.task, self.fields = domain, task, EPISODE_FIELDS.get((domain, task), {})
        self.envs = [
            action_scale.Wrapper(
                suite.load(domain, task, task_kwargs={"random": seed + lane}), minimum=-1.0, maximum=1.0
            )
            for lane in range(num_envs)
        ]
        self.cumulative = np.zeros(num_envs)

    def reset(self, seed):
        for lane, env in enumerate(self.envs):
            env.task._random = np.random.RandomState(seed + lane)
            env.reset()
        self.cumulative[:] = 0
        return self.get_state()

    def get_state(self):
        rows = []
        for lane, wrapper in enumerate(self.envs):
            env, physics = wrapper._env, wrapper.physics
            algorithm, keys, position, has_gauss, cached = env.task.random.get_state()
            assert algorithm == "MT19937"
            row = {
                "qpos": physics.data.qpos.copy(),
                "qvel": physics.data.qvel.copy(),
                "time": np.array([physics.data.time]),
                "warmstart": physics.data.qacc_warmstart.copy(),
                "task": {"cumulative_reward": np.array([self.cumulative[lane]])},
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
            state = TensorDict(row, []).clone()
            for key, (field, index) in self.fields.items():
                state[key] = torch.tensor(np.array(getattr(physics.named.model, field)[index], np.float64))
            rows.append(state)
        return torch.stack(rows)

    def step(self, actions):
        rewards, observations = [], []
        for lane, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                step = env.step(np.asarray(actions[lane], np.float64))
                reward += step.reward or 0.0
            self.cumulative[lane] += reward
            rewards.append(reward)
            observations.append(np.concatenate([v.flatten() for v in step.observation.values()], dtype=np.float32))
        return self.get_state(), torch.tensor(rewards, dtype=torch.float64), torch.as_tensor(np.stack(observations))

    @property
    def step_limit(self):
        return int(self.envs[0]._env._step_limit)

    def close(self):
        for env in self.envs:
            env.close()
