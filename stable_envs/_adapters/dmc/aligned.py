from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from tensordict import TensorDict

from ..._execution.protocol import SimulatorAdapter
from .state import export_state

CONFIG = {
    "ball_in_cup": {"nq": 4, "nv": 4, "nu": 2, "na": 0, "time_limit": 20.0, "dt": 0.02},
    "finger": {"nq": 3, "nv": 3, "nu": 2, "na": 0, "time_limit": 20.0, "dt": 0.02},
    "manipulator": {"nq": 11, "nv": 11, "nu": 5, "na": 0, "time_limit": 10.0, "dt": 0.01},
    "quadruped": {"nq": 23, "nv": 22, "nu": 12, "na": 12, "time_limit": 20.0, "dt": 0.02},
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
    if domain == "finger":
        fields[("model", "target_site_xz")] = (2, torch.float64)
    elif domain == "manipulator":
        fields[("model", "target_body_xz")] = (2, torch.float64)
        fields[("model", "target_body_qwqy")] = (2, torch.float64)
    elif domain == "quadruped":
        fields["activation"] = (cfg["na"], torch.float64)
    return fields


ALIGNED_FIELDS = {domain: _fields(domain) for domain in CONFIG}


def _common_defaults(model, color):
    grid = model.find("texture", "grid")
    grid.rgb1, grid.rgb2 = [0.1, 0.2, 0.3], [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = np.array([*color, 1.0], np.float32)
    model.find("light", "light").diffuse = np.full(3, 0.7, np.float32)


def _make_env(domain, seed, *, proximal_density=1000.0, upper_arm_density=1000.0):
    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite.wrappers import action_scale

    if domain == "ball_in_cup":
        from dm_control.suite import ball_in_cup as module

        xml, assets = module.get_model_and_assets()
        task = module.BallInCup(random=seed)
    elif domain == "finger":
        from dm_control.suite import finger as module

        xml, assets = module.get_model_and_assets()
        task = module.Turn(target_radius=0.03, random=seed)
    elif domain == "manipulator":
        from dm_control.suite import manipulator as module

        xml, assets = module.make_model(use_peg=False, insert=False)
        task = module.Bring(use_peg=False, insert=False, fully_observable=True, random=seed)
    elif domain == "quadruped":
        from dm_control.suite import quadruped as module

        xml, assets = module.make_model(floor_size=10.0), module.common.ASSETS
        task = module.Move(desired_speed=0.5, random=seed)
    else:
        raise ValueError(f"Unsupported aligned DMC domain {domain!r}")
    model = mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(module.__file__),
        assets=assets or {},
    )
    if domain == "ball_in_cup":
        _common_defaults(model, (0.7, 0.5, 0.3))
        for name in ("cup_part_0", "cup_part_1", "cup_part_2", "cup_part_3", "cup_part_4"):
            model.find("geom", name).density = 1000.0
        ball = model.find("geom", "ball")
        ball.density, ball.size = 1000.0, np.array([0.025], np.float32)
        target = model.find("site", "target")
        target.rgba = np.array([0.6, 0.3, 0.3, 0.0], np.float32).astype(np.float64)
        target.type = "sphere"
        target.size = np.array([0.05, 0.006, 0.05], np.float32)
    elif domain == "finger":
        _common_defaults(model, (0.7, 0.5, 0.3))
        model.find("geom", "proximal").density = float(proximal_density)
        model.find("geom", "fingertip").density = 1000.0
        for name in ("cap1", "cap2"):
            geom = model.find("geom", name)
            geom.density, geom.rgba = 1000.0, [0.7, 0.5, 0.3, 1.0]
        model.find("joint", "hinge").frictionloss = float(np.float32(0.1))
        model.find("material", "target").rgba = [0.6, 0.3, 0.3, 0.0]
        target = model.find("site", "target")
        target.type, target.size = "sphere", [0.03]
    elif domain == "manipulator":
        _common_defaults(model, (0.7, 0.5, 0.3))
        model.find("geom", "upper_arm").density = float(upper_arm_density)
        model.find("geom", "hand").density = 1000.0
        model.find("material", "target").rgba = [0.6, 0.3, 0.3, 0.0]
        target = model.find("geom", "target_ball")
        target.type, target.size = "sphere", np.array([0.015], np.float32)
    else:
        _common_defaults(model, (0.6, 0.3, 0.3))
        floor = model.find("geom", "floor")
        friction = (
            np.array([1.0, 0.005, 0.0001], np.float32)
            if floor.friction is None
            else np.asarray(floor.friction, np.float32).copy()
        )
        friction[0] = 1.0
        floor.friction = friction
        model.find("geom", "torso").density = 1000.0
        model.find("geom", "foot_back_left").density = 1000.0
    with tempfile.TemporaryDirectory() as directory:
        filename = f"{domain}.xml"
        mjcf.export_with_assets(model, directory, out_file_name=filename)
        physics = module.Physics.from_xml_path(os.path.join(directory, filename))
    cfg = CONFIG[domain]
    env = control.Environment(
        physics, task, time_limit=cfg["time_limit"], control_timestep=cfg["dt"]
    )
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


def _numeric_info(domain, env, cumulative):
    physics = env.physics
    info = {
        "qpos": physics.data.qpos.copy(),
        "qvel": physics.data.qvel.copy(),
    }
    success = float(physics.in_target()) if domain == "ball_in_cup" else float("nan")
    if domain == "finger":
        info.update(
            hinge_velocity=np.asarray(physics.hinge_velocity()),
            target_position=physics.target_position().copy(),
            tip_position=physics.tip_position().copy(),
        )
    elif domain == "quadruped":
        info["torso_velocity"] = physics.torso_velocity().copy()
    return info, success, float(cumulative / 1000.0)


class DMCAlignedAdapter(SimulatorAdapter):
    """CPU adapters for BallInCup, Finger, Manipulator and Quadruped.

    Each action advances its source twice; ``device`` selects output placement.
    Density reconfiguration rebuilds and resets every lane."""

    def __init__(self, domain, num_envs, device="cpu", seed=0):
        self.domain, self.num_envs, self.device = domain, int(num_envs), torch.device(device)
        self.fields = ALIGNED_FIELDS[domain]
        self._seed = int(seed)
        self._proximal_density = np.full(self.num_envs, 1000.0)
        self._upper_arm_density = np.full(self.num_envs, 1000.0)
        self.envs = [_make_env(domain, self._seed + i) for i in range(self.num_envs)]
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("DMCAlignedAdapter is closed")

    def configure_gravity(self, values):
        values = np.asarray(values, np.float64).reshape(self.num_envs, 3)
        for env, gravity in zip(self.envs, values, strict=True):
            env.physics.model.opt.gravity[:] = gravity

    def configure_proximal_density(self, values, *, force=False):
        values = np.asarray(values, np.float64).reshape(self.num_envs)
        if not force and np.array_equal(values, self._proximal_density):
            return
        for env in self.envs:
            env.close()
        self.envs = [
            _make_env(self.domain, self._seed + i, proximal_density=value)
            for i, value in enumerate(values)
        ]
        self._proximal_density = values.copy()
        self.reset(seed=self._seed)

    def configure_upper_arm_density(self, values, *, force=False):
        values = np.asarray(values, np.float64).reshape(self.num_envs)
        if not force and np.array_equal(values, self._upper_arm_density):
            return
        for env in self.envs:
            env.close()
        self.envs = [
            _make_env(self.domain, self._seed + i, upper_arm_density=value)
            for i, value in enumerate(values)
        ]
        self._upper_arm_density = values.copy()
        self.reset(seed=self._seed)

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
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("Aligned DMC requires full TensorDict State with batch [num_envs]")
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
            base._step_count = row["lifecycle", "step_count"].item()
            base._reset_next_step = row["lifecycle", "reset_next_step"].item()
            self._cumulative[i] = row["task", "cumulative_reward"].item()
            env.task.random.set_state((
                "MT19937", row["rng", "keys"].numpy().astype(np.uint32),
                row["rng", "position"].item(), int(row["rng", "has_gauss"].item()),
                row["rng", "cached_gaussian"].item(),
            ))

    def step(self, actions, *, capture_state=True):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        nu = CONFIG[self.domain]["nu"]
        if actions.shape != (self.num_envs, nu) or not np.isfinite(actions).all():
            raise ValueError(f"{self.domain} action must have finite shape (num_envs, {nu})")
        rewards, observations, infos, successes, scores = [], [], [], [], []
        for i, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                result = env.step(actions[i].astype(env.action_spec().dtype))
                reward += result.reward or 0.0
            self._cumulative[i] += reward
            info, success, score = _numeric_info(self.domain, env, self._cumulative[i])
            rewards.append(reward)
            observations.append(np.concatenate([value.flatten() for value in result.observation.values()], dtype=np.float32))
            infos.append(info)
            successes.append(success)
            scores.append(score)
        info = TensorDict(
            {key: torch.as_tensor(np.stack([row[key] for row in infos]), device=self.device) for key in infos[0]},
            [self.num_envs], device=self.device,
        )
        return TensorDict({
            **({"state": self.get_state()} if capture_state else {}),
            "reward": torch.tensor(rewards, dtype=torch.float64, device=self.device),
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "observation": torch.as_tensor(np.stack(observations), device=self.device),
            "success": torch.tensor(successes, dtype=torch.float64, device=self.device),
            "score": torch.tensor(scores, dtype=torch.float64, device=self.device),
            "info": info,
        }, [self.num_envs], device=self.device)

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
