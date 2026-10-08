from __future__ import annotations

import os
import tempfile

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from .state import export_state

# These fields restore simulation; compact renderer columns cannot.
REACHER_FIELDS = {
    "qpos": (2, torch.float64),
    "qvel": (2, torch.float64),
    "time": (1, torch.float64),
    "warmstart": (2, torch.float64),
    ("task", "target_pos"): (3, torch.float64),
    ("lifecycle", "step_count"): (1, torch.int64),
    ("lifecycle", "reset_next_step"): (1, torch.bool),
    ("task", "cumulative_reward"): (1, torch.float64),
    ("rng", "keys"): (624, torch.int64),
    ("rng", "position"): (1, torch.int64),
    ("rng", "has_gauss"): (1, torch.bool),
    ("rng", "cached_gaussian"): (1, torch.float64),
}


def _resume_fields(nq, nv):
    """Continuation fields shared by Acrobot, Pendulum and Cartpole."""
    return {
        "qpos": (nq, torch.float64),
        "qvel": (nv, torch.float64),
        "time": (1, torch.float64),
        "warmstart": (nv, torch.float64),
        ("lifecycle", "step_count"): (1, torch.int64),
        ("lifecycle", "reset_next_step"): (1, torch.bool),
        ("task", "cumulative_reward"): (1, torch.float64),
        ("rng", "keys"): (624, torch.int64),
        ("rng", "position"): (1, torch.int64),
        ("rng", "has_gauss"): (1, torch.bool),
        ("rng", "cached_gaussian"): (1, torch.float64),
    }


FIRST_FAMILY_FIELDS = {
    "acrobot": _resume_fields(2, 2),
    "pendulum": _resume_fields(1, 1),
    "cartpole": _resume_fields(2, 2),
}


def _make_first_family(domain, seed):
    """Construct the registered sparse Acrobot/Cartpole or dense Pendulum task."""
    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite.wrappers import action_scale

    if domain == "acrobot":
        from dm_control.suite import acrobot as module

        task = module.Balance(sparse=True, random=seed)
        time_limit = 10
    elif domain == "pendulum":
        from dm_control.suite import pendulum as module

        task = module.SwingUp(random=seed)
        time_limit = 20
    elif domain == "cartpole":
        from dm_control.suite import cartpole as module

        task = module.Balance(swing_up=True, sparse=True, random=seed)
        time_limit = 10
    else:
        raise ValueError(f"Unsupported recovered DMC domain {domain!r}")

    xml, assets = module.get_model_and_assets()
    model = mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(module.__file__),
        assets=assets or {},
    )
    # Pinned wrappers apply these registered defaults before their first reset.
    model.find("texture", "grid").rgb1 = [0.1, 0.2, 0.3]
    model.find("texture", "grid").rgb2 = [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = [0.7, 0.5, 0.3, 1.0]
    model.find("light", "light").diffuse = [0.7, 0.7, 0.7]
    with tempfile.TemporaryDirectory() as directory:
        name = f"{domain}.xml"
        mjcf.export_with_assets(model, directory, out_file_name=name)
        physics = module.Physics.from_xml_path(os.path.join(directory, name))
    env = control.Environment(physics, task, time_limit=time_limit)
    return action_scale.Wrapper(env, minimum=-1.0, maximum=1.0)


class FirstFamilyAdapter(SimulatorAdapter):
    """Independent CPU simulators with two source steps per action.

    ``device`` controls returned tensor placement. State includes lifecycle and
    task RNG fields; it is separate from the renderer’s compact pose columns."""

    def __init__(self, domain, num_envs, device="cpu", seed=0):
        if domain not in FIRST_FAMILY_FIELDS:
            raise ValueError(f"Unsupported recovered DMC domain {domain!r}")
        self.domain = domain
        self.fields = FIRST_FAMILY_FIELDS[domain]
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [_make_first_family(domain, int(seed) + i) for i in range(self.num_envs)]
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._gravity = np.tile(np.array([0.0, 0.0, -9.81]), (self.num_envs, 1))
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("FirstFamilyAdapter is closed")

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
            raise TypeError("Recovered DMC tasks require full TensorDict State with batch [num_envs]")
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
        if actions.shape != (self.num_envs, 1) or not np.isfinite(actions).all():
            raise ValueError("Recovered DMC action must have finite shape (num_envs, 1)")
        rewards, observations = [], []
        for i, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                result = env.step(actions[i].astype(env.action_spec().dtype))
                reward += result.reward or 0.0
            self._cumulative[i] += reward
            rewards.append(reward)
            observations.append(np.concatenate([v.flatten() for v in result.observation.values()], dtype=np.float32))
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.tensor(rewards, dtype=torch.float64, device=self.device),
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(np.stack(observations), device=self.device),
                "score": torch.tensor(self._cumulative / 1000.0, device=self.device),
            },
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True


def _make_reacher(seed, arm_density=1000.0, finger_density=1000.0):
    """Construct Reacher’s hard task with the registered model and action scaling."""
    from dm_control import mjcf
    from dm_control.rl import control
    from dm_control.suite import reacher
    from dm_control.suite.wrappers import action_scale

    xml, assets = reacher.get_model_and_assets()
    model = mjcf.from_xml_string(
        xml.replace(b'file="./common/', b'file="common/'),
        model_dir=os.path.dirname(reacher.__file__),
        assets=assets or {},
    )
    model.find("geom", "arm").density = arm_density
    model.find("geom", "finger").density = finger_density
    model.find("geom", "target").type = "sphere"
    model.find("geom", "target").size = np.array([0.015], np.float32)
    # Apply registered appearance defaults before compiling the model.
    model.find("texture", "grid").rgb1 = [0.1, 0.2, 0.3]
    model.find("texture", "grid").rgb2 = [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = np.array([0.7, 0.5, 0.3, 1], np.float32)
    model.find("material", "target").rgba = np.r_[np.array([0.6, 0.3, 0.3], np.float32), 0.0]
    model.find("light", "light").diffuse = 0.7 * np.ones(3, np.float32)
    with tempfile.TemporaryDirectory() as directory:
        mjcf.export_with_assets(model, directory, out_file_name="reacher.xml")
        physics = reacher.Physics.from_xml_path(os.path.join(directory, "reacher.xml"))
    task = reacher.Reacher(target_size=0.015, random=seed)
    return action_scale.Wrapper(control.Environment(physics, task, time_limit=20), minimum=-1.0, maximum=1.0)


class ReacherAdapter(SimulatorAdapter):
    """Independent CPU Reacher simulators; ``device`` selects output placement.

    Each action advances the source twice. State includes target, lifecycle and RNG."""

    def __init__(self, num_envs, device="cpu", seed=0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [_make_reacher(int(seed) + i) for i in range(self.num_envs)]
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._density = np.full((self.num_envs, 2), 1000.0)
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("ReacherAdapter is closed")

    def configure_density(self, values):
        """Recompile changed arm/finger densities and restore the current State."""
        values = np.asarray(values, np.float64).reshape(self.num_envs, 2)
        if np.array_equal(values, self._density):
            return
        saved = self.get_state()
        for i, (new, old) in enumerate(zip(values, self._density, strict=True)):
            if not np.array_equal(new, old):
                replacement = _make_reacher(0, *new)
                replacement.reset()
                self.envs[i].close()
                self.envs[i] = replacement
        self._density = values.copy()
        self.set_state(saved)

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
        return export_state(self.envs, self._cumulative, REACHER_FIELDS, self.device)

    def set_state(self, state):
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("Reacher requires full TensorDict State with batch [num_envs]")
        # Validate the entire batch before mutating any authoritative simulator.
        for key, (width, dtype) in REACHER_FIELDS.items():
            if key not in state.keys(True, True):
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if value.shape != (self.num_envs, width) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid State field {key}")
        host = state.cpu()
        for i, env in enumerate(self.envs):
            s, physics, base = host[i], env.physics, env._env
            physics.named.model.geom_pos["target"] = s["task", "target_pos"].numpy()
            physics.data.qpos[:] = s["qpos"].numpy()
            physics.data.qvel[:] = s["qvel"].numpy()
            physics.data.time = s["time"].item()
            physics.forward()
            physics.data.qacc_warmstart[:] = s["warmstart"].numpy()
            base._step_count = s["lifecycle", "step_count"].item()
            base._reset_next_step = s["lifecycle", "reset_next_step"].item()
            env.task.random.set_state(
                (
                    "MT19937",
                    s["rng", "keys"].numpy().astype(np.uint32),
                    s["rng", "position"].item(),
                    int(s["rng", "has_gauss"].item()),
                    s["rng", "cached_gaussian"].item(),
                )
            )
            self._cumulative[i] = s["task", "cumulative_reward"].item()

    def step(self, actions, *, capture_state=True):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, 2) or not np.isfinite(actions).all():
            raise ValueError("Reacher action must have finite shape (num_envs, 2)")
        rewards, observations = [], []
        for i, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                result = env.step(actions[i].astype(env.action_spec().dtype))
                reward += result.reward or 0.0
            self._cumulative[i] += reward
            rewards.append(reward)
            observations.append(np.concatenate([v.flatten() for v in result.observation.values()], dtype=np.float32))
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.tensor(rewards, dtype=torch.float64, device=self.device),
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(np.stack(observations), device=self.device),
                "score": torch.tensor(self._cumulative / 1000.0, device=self.device),
            },
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
