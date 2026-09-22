"""Pinned SWM oracle bridge for four source-aligned DMC defaults."""

from __future__ import annotations

import importlib

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs._adapters.dmcontrol_aligned import ALIGNED_FIELDS

WRAPPERS = {
    "ball_in_cup": ("ball_in_cup", "BallInCupDMControlWrapper"),
    "finger": ("finger", "FingerDMControlWrapper"),
    "manipulator": ("manipulator", "ManipulatorDMControlWrapper"),
    "quadruped": ("quadruped", "QuadrupedDMControlWrapper"),
}
EXTRA_INFO = {
    "ball_in_cup": (),
    "finger": ("hinge_velocity", "target_position", "tip_position"),
    "manipulator": (),
    "quadruped": ("torso_velocity",),
}


class DMCAlignedReference:
    def __init__(self, domain, num_envs=1, options=None):
        module_name, class_name = WRAPPERS[domain]
        module = importlib.import_module(f"stable_worldmodel.envs.dmcontrol.{module_name}")
        wrapper = getattr(module, class_name)
        self.domain, self.num_envs, self.options = domain, num_envs, options
        self.envs = [wrapper(seed=i) for i in range(num_envs)]

    def reset(self, seed=None):
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + i, options=self.options)
        return self.get_state()

    def get_state(self):
        rows = []
        for wrapper in self.envs:
            env, fields = wrapper.env._env, ALIGNED_FIELDS[self.domain]
            physics, rng = env.physics, env.task.random.get_state()
            values = {
                "qpos": physics.data.qpos.copy(),
                "qvel": physics.data.qvel.copy(),
                "time": [physics.data.time],
                "warmstart": physics.data.qacc_warmstart.copy(),
                ("lifecycle", "step_count"): [env._step_count],
                ("lifecycle", "reset_next_step"): [env._reset_next_step],
                ("task", "cumulative_reward"): [wrapper._cumulative_reward],
                ("rng", "keys"): rng[1].astype(np.int64),
                ("rng", "position"): [rng[2]],
                ("rng", "has_gauss"): [rng[3]],
                ("rng", "cached_gaussian"): [rng[4]],
            }
            if self.domain == "finger":
                values[("model", "target_site_xz")] = physics.named.model.site_pos[
                    "target", ["x", "z"]
                ].copy()
            elif self.domain == "manipulator":
                values[("model", "target_body_xz")] = physics.named.model.body_pos[
                    "target_ball", ["x", "z"]
                ].copy()
                values[("model", "target_body_qwqy")] = physics.named.model.body_quat[
                    "target_ball", ["qw", "qy"]
                ].copy()
            elif self.domain == "quadruped":
                values["activation"] = physics.data.act.copy()
            row = TensorDict({}, [])
            for key, (_, dtype) in fields.items():
                row[key] = torch.as_tensor(values[key], dtype=dtype).clone()
            rows.append(row)
        return torch.stack(rows)

    def set_state(self, state):
        for wrapper, row in zip(self.envs, state.cpu().unbind(0), strict=True):
            env, physics = wrapper.env._env, wrapper.env.physics
            physics.data.qpos[:] = row["qpos"].numpy()
            physics.data.qvel[:] = row["qvel"].numpy()
            physics.data.time = row["time"].item()
            if self.domain == "finger":
                physics.named.model.site_pos["target", ["x", "z"]] = row[
                    "model", "target_site_xz"
                ].numpy()
            elif self.domain == "manipulator":
                physics.named.model.body_pos["target_ball", ["x", "z"]] = row[
                    "model", "target_body_xz"
                ].numpy()
                physics.named.model.body_quat["target_ball", ["qw", "qy"]] = row[
                    "model", "target_body_qwqy"
                ].numpy()
            elif self.domain == "quadruped":
                physics.data.act[:] = row["activation"].numpy()
            physics.forward()
            physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
            env._step_count = row["lifecycle", "step_count"].item()
            env._reset_next_step = row["lifecycle", "reset_next_step"].item()
            wrapper._cumulative_reward = row["task", "cumulative_reward"].item()
            env.task.random.set_state((
                "MT19937", row["rng", "keys"].numpy().astype(np.uint32),
                row["rng", "position"].item(), int(row["rng", "has_gauss"].item()),
                row["rng", "cached_gaussian"].item(),
            ))

    def step(self, actions):
        outputs = [wrapper.step(np.asarray(action, np.float64)) for wrapper, action in zip(self.envs, actions, strict=True)]
        infos = [output[4] for output in outputs]
        info = TensorDict({
            key: torch.as_tensor(np.stack([row[key] for row in infos]))
            for key in ("qpos", "qvel", *EXTRA_INFO[self.domain])
        }, [self.num_envs])
        return TensorDict({
            "state": self.get_state(),
            "reward": torch.tensor([output[1] for output in outputs], dtype=torch.float64),
            "terminated": torch.tensor([output[2] for output in outputs], dtype=torch.bool),
            "truncated": torch.tensor([output[3] for output in outputs], dtype=torch.bool),
            "observation": torch.as_tensor(np.stack([output[0] for output in outputs])),
            "success": torch.tensor([row["success"] for row in infos], dtype=torch.float64),
            "score": torch.tensor([row["score"] for row in infos], dtype=torch.float64),
            "info": info,
        }, [self.num_envs])

    @property
    def action_dim(self):
        return self.envs[0].action_space.shape[0]

    @property
    def step_limit(self):
        return int(self.envs[0].env._env._step_limit)

    def close(self):
        for env in self.envs:
            env.close()
