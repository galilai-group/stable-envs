"""Exact delegated semantics for the historical SWM Humanoid Walk default."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from tensordict import TensorDict

from .dmcontrol_state import export_state

HUMANOID_CONFIG = {
    "nq": 28,
    "nv": 27,
    "nu": 21,
    "na": 0,
    "time_limit": 25.0,
    "control_timestep": 0.025,
}

HUMANOID_FIELDS = {
    "qpos": (28, torch.float64),
    "qvel": (27, torch.float64),
    "time": (1, torch.float64),
    "warmstart": (27, torch.float64),
    ("lifecycle", "step_count"): (1, torch.int64),
    ("lifecycle", "reset_next_step"): (1, torch.bool),
    ("task", "cumulative_reward"): (1, torch.float64),
    ("rng", "keys"): (624, torch.int64),
    ("rng", "position"): (1, torch.int64),
    ("rng", "has_gauss"): (1, torch.bool),
    ("rng", "cached_gaussian"): (1, torch.float64),
}


def _apply_registered_defaults(model):
    """Apply the pinned SWM wrapper's first-reset MJCF writes exactly."""
    grid = model.find("texture", "grid")
    grid.rgb1 = [0.1, 0.2, 0.3]
    grid.rgb2 = [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = np.array([0.7, 0.5, 0.3, 1.0], np.float32)

    floor = model.find("geom", "floor")
    friction = (
        np.array([1.0, 0.005, 0.0001], np.float32)
        if floor.friction is None
        else np.asarray(floor.friction, np.float32).copy()
    )
    friction[0] = np.float32(1.0)
    floor.friction = friction

    model.find("geom", "torso").density = 1000.0
    model.find("geom", "right_lower_arm").density = 1000.0
    model.find("light", "top").diffuse = np.full(3, 0.7, np.float32)


def _make_env(seed):
    """Build the historical Walk task without importing stable-worldmodel."""
    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite import humanoid
    from dm_control.suite.wrappers import action_scale

    xml, assets = humanoid.get_model_and_assets()
    model = mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(humanoid.__file__),
        assets=assets or {},
    )
    _apply_registered_defaults(model)
    with tempfile.TemporaryDirectory() as directory:
        filename = "humanoid.xml"
        mjcf.export_with_assets(model, directory, out_file_name=filename)
        physics = humanoid.Physics.from_xml_path(os.path.join(directory, filename))
    task = humanoid.Humanoid(move_speed=1, pure_state=False, random=seed)
    env = control.Environment(
        physics,
        task,
        time_limit=HUMANOID_CONFIG["time_limit"],
        control_timestep=HUMANOID_CONFIG["control_timestep"],
    )
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def _numeric_info(env, cumulative):
    physics = env.physics
    return {
        "success": float("nan"),
        "qpos": physics.data.qpos.copy(),
        "qvel": physics.data.qvel.copy(),
        "score": float(cumulative / 1000.0),
        "speed": np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]]),
        "torso_upright": physics.torso_upright(),
        "head_height": physics.head_height(),
    }


class DMCHumanoidAdapter:
    """B independent CPU simulators for the historical Humanoid Walk default."""

    def __init__(self, num_envs, device="cpu", seed=0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [_make_env(int(seed) + i) for i in range(self.num_envs)]
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("DMCHumanoidAdapter is closed")

    def configure_gravity(self, values):
        values = np.asarray(values, np.float64).reshape(self.num_envs, 3)
        if not np.isfinite(values).all():
            raise ValueError("Gravity must be finite")
        for env, gravity in zip(self.envs, values, strict=True):
            env.physics.model.opt.gravity[:] = gravity

    def reset(self, seed=None):
        self._check_open()
        for i, env in enumerate(self.envs):
            if seed is not None:
                env.task._random = np.random.RandomState(int(seed) + i)
            env.reset()
        self._cumulative.fill(0)
        return self.get_state()

    def get_state(self):
        self._check_open()
        return export_state(self.envs, self._cumulative, HUMANOID_FIELDS, self.device)

    def set_state(self, state):
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("Humanoid Walk requires full TensorDict State with batch [num_envs]")
        for key, (width, dtype) in HUMANOID_FIELDS.items():
            if key not in state.keys(True, True):
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if (
                value.shape != (self.num_envs, width)
                or value.dtype != dtype
                or not torch.isfinite(value).all()
            ):
                raise ValueError(f"Invalid State field {key}")
        host = state.cpu()
        for i, env in enumerate(self.envs):
            row, physics, base = host[i], env.physics, env._env
            physics.data.qpos[:] = row["qpos"].numpy()
            physics.data.qvel[:] = row["qvel"].numpy()
            physics.data.time = row["time"].item()
            physics.forward()
            physics.data.qacc_warmstart[:] = row["warmstart"].numpy()
            base._step_count = row["lifecycle", "step_count"].item()
            base._reset_next_step = row["lifecycle", "reset_next_step"].item()
            env.task.random.set_state(
                (
                    "MT19937",
                    row["rng", "keys"].numpy().astype(np.uint32),
                    row["rng", "position"].item(),
                    int(row["rng", "has_gauss"].item()),
                    row["rng", "cached_gaussian"].item(),
                )
            )
            self._cumulative[i] = row["task", "cumulative_reward"].item()

    def step(self, actions):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        expected_shape = (self.num_envs, HUMANOID_CONFIG["nu"])
        if actions.shape != expected_shape or not np.isfinite(actions).all():
            raise ValueError(f"Humanoid action must have finite shape {expected_shape}")
        rewards, observations, infos = [], [], []
        for i, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                result = env.step(actions[i].astype(env.action_spec().dtype))
                reward += result.reward or 0.0
            self._cumulative[i] += reward
            rewards.append(reward)
            observations.append(
                np.concatenate([value.flatten() for value in result.observation.values()], dtype=np.float32)
            )
            infos.append(_numeric_info(env, self._cumulative[i]))
        info = TensorDict(
            {
                key: torch.as_tensor(np.stack([row[key] for row in infos]), device=self.device)
                for key in ("qpos", "qvel", "speed", "torso_upright", "head_height")
            },
            [self.num_envs],
            device=self.device,
        )
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor(rewards, dtype=torch.float64, device=self.device),
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(np.stack(observations), device=self.device),
                "success": torch.tensor(
                    [row["success"] for row in infos], dtype=torch.float64, device=self.device
                ),
                "score": torch.tensor(
                    [row["score"] for row in infos], dtype=torch.float64, device=self.device
                ),
                "info": info,
            },
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
