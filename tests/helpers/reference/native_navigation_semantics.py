"""Independent pinned-SWM oracle for the three historical native navigation defaults."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

SOURCES = {
    "two_room": ("stable_worldmodel.envs.two_room.env", "TwoRoomEnv", {}),
    "piecewise": ("stable_worldmodel.envs.piecewise.piecewise_env", "PiecewiseEnv", {}),
    "simple_point_maze": (
        "stable_worldmodel.envs.simple_point_maze",
        "SimplePointMazeEnv",
        {"render_mode": "rgb_array", "show_goal": True},
    ),
}


class NativeNavigationReference:
    """One independent pinned SWM environment per batch lane."""

    def __init__(self, domain, num_envs=1, options=None, persistent_values=None):
        module_name, class_name, kwargs = SOURCES[domain]
        cls = getattr(importlib.import_module(module_name), class_name)
        self.domain = domain
        self.num_envs = int(num_envs)
        self.options = options
        self.envs = [cls(**kwargs) for _ in range(self.num_envs)]
        for env in self.envs:
            for path, value in (persistent_values or {}).items():
                leaf = env.variation_space
                for part in path.split("."):
                    leaf = leaf[part]
                leaf.set_init_value(np.asarray(value, dtype=leaf.dtype).reshape(leaf.shape))
        self.cached_goal_positions = [np.asarray([0.5, 0.5], np.float32) for _ in self.envs]
        if domain == "simple_point_maze":
            for lane, env in enumerate(self.envs):
                leaf = env.variation_space["agent"]["position"]
                original = leaf.sample

                def recorded(*args, _original=original, _lane=lane, **kwargs):
                    value = _original(*args, **kwargs)
                    self.cached_goal_positions[_lane] = np.asarray(value, np.float32).copy()
                    return value

                leaf.sample = recorded

    def reset(self, seed=None):
        for lane, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + lane, options=self.options)
        return self.get_state()

    def get_state(self):
        if self.domain == "simple_point_maze":
            return TensorDict(
                {
                    "agent": {
                        "position": torch.as_tensor(
                            np.stack([env.state for env in self.envs]), dtype=torch.float64
                        )
                    },
                    "goal": {
                        "position": torch.as_tensor(
                            np.stack([env.variation_space["goal"]["position"].value for env in self.envs]),
                            dtype=torch.float32,
                        )
                    },
                    "task": {
                        "cached_goal_agent_position": torch.as_tensor(
                            np.stack(self.cached_goal_positions), dtype=torch.float32
                        ),
                        "state_is_float64": torch.tensor(
                            [[np.asarray(env.state).dtype == np.float64] for env in self.envs], dtype=torch.bool
                        ),
                    },
                },
                batch_size=[self.num_envs],
            )
        return TensorDict(
            {
                "agent": {
                    "position": torch.stack([env.agent_position.detach().clone() for env in self.envs])
                },
                "target": {
                    "position": torch.stack([env.target_position.detach().clone() for env in self.envs])
                },
            },
            batch_size=[self.num_envs],
        )

    def set_state(self, state):
        host = state.cpu()
        for lane, env in enumerate(self.envs):
            if self.domain == "simple_point_maze":
                is_float64 = bool(host[lane]["task", "state_is_float64"].item())
                agent = host[lane]["agent", "position"].numpy().astype(
                    np.float64 if is_float64 else np.float32, copy=True
                )
                goal = host[lane]["goal", "position"].numpy().copy()
                cached = host[lane]["task", "cached_goal_agent_position"].numpy().copy()
                env.variation_space["goal"]["position"]._value = goal
                env.state = cached
                env.variation_space["agent"]["position"]._value = cached.copy()
                env._goal = env.render("rgb_array")
                env.state = agent
                env.variation_space["agent"]["position"]._value = agent.copy()
                self.cached_goal_positions[lane] = cached
            else:
                env.agent_position = host[lane]["agent", "position"].clone()
                env.target_position = host[lane]["target", "position"].clone()

    def _actions(self, actions):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        assert actions.shape == (self.num_envs, 2)
        return [actions[lane] for lane in range(self.num_envs)]

    def step(self, actions, active=None):
        outputs = [
            env.step(action) if active is None or active[lane] else self._last_outputs[lane]
            for lane, (env, action) in enumerate(zip(self.envs, self._actions(actions), strict=True))
        ]
        self._last_outputs = outputs
        observation = torch.as_tensor(np.stack([out[0] for out in outputs])).clone()
        common = {
            "state": self.get_state(),
            "reward": torch.tensor([out[1] for out in outputs], dtype=torch.float64),
            "terminated": torch.tensor([out[2] for out in outputs], dtype=torch.bool),
            "truncated": torch.tensor([out[3] for out in outputs], dtype=torch.bool),
            "observation": observation,
        }
        if self.domain == "simple_point_maze":
            common["info"] = {
                "goal": torch.as_tensor(np.stack([out[4]["goal"] for out in outputs])).clone()
            }
        else:
            common["info"] = {
                "proprio": torch.as_tensor(np.stack([out[4]["proprio"] for out in outputs])).clone(),
                "state": torch.as_tensor(np.stack([out[4]["state"] for out in outputs])).clone(),
                "goal_state": torch.as_tensor(np.stack([out[4]["goal_state"] for out in outputs])).clone(),
                "distance_to_target": torch.tensor(
                    [out[4]["distance_to_target"] for out in outputs], dtype=torch.float64
                ),
            }
        return TensorDict(common, batch_size=[self.num_envs])

    def close(self):
        for env in self.envs:
            env.close()
