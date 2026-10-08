from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from tensordict import TensorDict

from .state import export_state

CONFIG = {
    "cheetah": {"nq": 9, "nv": 9, "nu": 6, "time_limit": 25.0, "control_timestep": None},
    "hopper": {"nq": 7, "nv": 7, "nu": 4, "time_limit": 20.0, "control_timestep": 0.02},
    "walker": {"nq": 9, "nv": 9, "nu": 6, "time_limit": 25.0, "control_timestep": 0.025},
}


def _fields(domain):
    cfg = CONFIG[domain]
    fields = {
        "qpos": (cfg["nq"], torch.float64),
        "qvel": (cfg["nv"], torch.float64),
        "time": (1, torch.float64),
        "warmstart": (cfg["nv"], torch.float64),
        ("lifecycle", "step_count"): (1, torch.int64),
        ("lifecycle", "reset_next_step"): (1, torch.bool),
        ("task", "cumulative_reward"): (1, torch.float64),
        ("rng", "keys"): (624, torch.int64),
        ("rng", "position"): (1, torch.int64),
        ("rng", "has_gauss"): (1, torch.bool),
        ("rng", "cached_gaussian"): (1, torch.float64),
    }
    return fields


CONTINUATION_FIELDS = {domain: _fields(domain) for domain in CONFIG}


def _apply_registered_defaults(domain, model):
    """Apply registered reset values before compiling the model."""
    model.find("texture", "grid").rgb1 = [0.1, 0.2, 0.3]
    model.find("texture", "grid").rgb2 = [0.2, 0.3, 0.4]
    color = [0.6, 0.3, 0.3, 1.0] if domain == "cheetah" else [0.7, 0.5, 0.3, 1.0]
    model.find("material", "self").rgba = np.asarray(color, np.float32)

    floor_name = "ground" if domain == "cheetah" else "floor"
    floor = model.find("geom", floor_name)
    friction = (
        np.array([1.0, 0.005, 0.0001], np.float32)
        if floor.friction is None
        else np.asarray(floor.friction, np.float32).copy()
    )
    friction[0] = 1.0
    floor.friction = friction

    density_geoms = {
        "cheetah": ("torso", "bfoot"),
        "hopper": ("torso", "foot"),
        "walker": ("torso", "left_foot"),
    }[domain]
    for name in density_geoms:
        model.find("geom", name).density = 1000.0

    if domain == "walker":
        floor.zaxis = np.array([0.0, 0.0, 1.0], np.float32)
    light_name = "top" if domain == "hopper" else "light"
    model.find("light", light_name).diffuse = np.full(3, 0.7, np.float32)


def _make_env(domain, seed):
    """Construct Cheetah Run, Hopper Hop or Walker Walk with registered defaults."""
    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite.wrappers import action_scale

    if domain == "cheetah":
        from dm_control.suite import cheetah as module

        task = module.Cheetah(random=seed)
    elif domain == "hopper":
        from dm_control.suite import hopper as module

        task = module.Hopper(hopping=True, random=seed)
    elif domain == "walker":
        from dm_control.suite import walker as module

        task = module.PlanarWalker(move_speed=1, random=seed)
    else:
        raise ValueError(f"Unsupported continuation domain {domain!r}")

    xml, assets = module.get_model_and_assets()
    model = mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(module.__file__),
        assets=assets or {},
    )
    _apply_registered_defaults(domain, model)
    with tempfile.TemporaryDirectory() as directory:
        name = f"{domain}.xml"
        mjcf.export_with_assets(model, directory, out_file_name=name)
        physics = module.Physics.from_xml_path(os.path.join(directory, name))
    cfg = CONFIG[domain]
    kwargs = {"time_limit": cfg["time_limit"]}
    if cfg["control_timestep"] is not None:
        kwargs["control_timestep"] = cfg["control_timestep"]
    env = control.Environment(physics, task, **kwargs)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def _numeric_info(domain, env, cumulative):
    physics = env.physics
    common = {
        "success": float("nan"),
        "qpos": physics.data.qpos.copy(),
        "qvel": physics.data.qvel.copy(),
        "score": float(cumulative / 1000.0),
    }
    if domain == "cheetah":
        extra = {
            "speed": physics.speed(),
            "torso_height": physics.named.data.xpos["torso", "z"],
            "bfoot_height": physics.named.data.xpos["bfoot", "z"],
            "ffoot_height": physics.named.data.xpos["ffoot", "z"],
            "angmomentum": physics.named.data.subtree_angmom["torso"][1],
        }
    elif domain == "hopper":
        extra = {
            "speed": physics.speed(),
            "height": physics.height(),
            "angmomentum": physics.named.data.subtree_angmom["torso"][1],
        }
    else:
        extra = {
            "speed": physics.horizontal_velocity(),
            "torso_height": physics.torso_height(),
            "torso_upright": physics.torso_upright(),
            "left_foot_height": physics.named.data.xpos["left_foot", "z"],
            "right_foot_height": physics.named.data.xpos["right_foot", "z"],
            "left_thigh_height": physics.named.data.xpos["left_thigh", "z"],
            "right_thigh_height": physics.named.data.xpos["right_thigh", "z"],
        }
    return {**common, **{key: np.asarray(value, np.float64) for key, value in extra.items()}}


class DMCContinuationAdapter:
    """Independent CPU Cheetah, Hopper or Walker simulators.

    Each action advances the source twice; ``device`` selects output placement.
    State includes lifecycle, warmstart and task RNG for continuation."""

    def __init__(self, domain, num_envs, device="cpu", seed=0):
        if domain not in CONFIG:
            raise ValueError(f"Unsupported continuation domain {domain!r}")
        self.domain = domain
        self.fields = CONTINUATION_FIELDS[domain]
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [_make_env(domain, int(seed) + i) for i in range(self.num_envs)]
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._gravity = np.tile(np.array([0.0, 0.0, -9.81]), (self.num_envs, 1))
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("DMCContinuationAdapter is closed")

    def configure_gravity(self, values):
        values = np.asarray(values, np.float64).reshape(self.num_envs, 3)
        if not np.isfinite(values).all():
            raise ValueError("Gravity must be finite")
        for env, gravity in zip(self.envs, values, strict=True):
            env.physics.model.opt.gravity[:] = gravity
        self._gravity = values.copy()

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
        return export_state(self.envs, self._cumulative, self.fields, self.device)

    def set_state(self, state):
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("DMC continuation requires full TensorDict State with batch [num_envs]")
        for key, (width, dtype) in self.fields.items():
            if key not in state.keys(True, True):
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if value.shape != (self.num_envs, width) or value.dtype != dtype or not torch.isfinite(value).all():
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

    def step(self, actions, *, capture_state=True):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        nu = CONFIG[self.domain]["nu"]
        if actions.shape != (self.num_envs, nu) or not np.isfinite(actions).all():
            raise ValueError(f"{self.domain} action must have finite shape (num_envs, {nu})")
        rewards, observations, infos = [], [], []
        for i, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                result = env.step(actions[i].astype(env.action_spec().dtype))
                reward += result.reward or 0.0
            self._cumulative[i] += reward
            rewards.append(reward)
            observations.append(np.concatenate([value.flatten() for value in result.observation.values()], dtype=np.float32))
            infos.append(_numeric_info(self.domain, env, self._cumulative[i]))
        info = TensorDict(
            {
                key: torch.as_tensor(np.stack([row[key] for row in infos]), device=self.device)
                for key in infos[0]
                if key not in ("success", "score")
            },
            [self.num_envs],
            device=self.device,
        )
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.tensor(rewards, dtype=torch.float64, device=self.device),
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(np.stack(observations), device=self.device),
                "success": torch.tensor([row["success"] for row in infos], dtype=torch.float64, device=self.device),
                "score": torch.tensor([row["score"] for row in infos], dtype=torch.float64, device=self.device),
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
