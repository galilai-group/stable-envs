from __future__ import annotations

from functools import partial

import numpy as np
import torch

from . import physics as _physics
from .batch import XZ, ZZ, scalar_tolerance, tolerance
from .lanes import DMCLanes
from .registered import compile_physics, normalized, source_model
from .state import core_fields


def suite_env(domain, task, seed):
    """dm_control's own registered task, with actions normalized to [-1, 1]."""
    from dm_control import suite

    return normalized(suite.load(domain, task, task_kwargs={"random": seed}))


def _with_physics(key, fields):
    return {**fields, **_physics.state_fields(key)}


_FIRST_CORE = {
    "acrobot": core_fields(2, 2),
    "pendulum": core_fields(1, 1),
    "cartpole": core_fields(2, 2),
    "cartpole-balance": core_fields(2, 2),
}
FIRST_FAMILY_FIELDS = {key: _with_physics(key, fields) for key, fields in _FIRST_CORE.items()}

# These fields restore simulation; compact renderer columns cannot.
_REACHER_CORE = {**core_fields(2, 2), ("task", "target_pos"): (3, torch.float64)}
REACHER_FIELDS = {"reacher": _with_physics("reacher", _REACHER_CORE)}


def _make_first_family(domain, seed):
    """Construct the registered sparse Acrobot/Cartpole or dense Pendulum task."""
    from dm_control.rl import control

    if domain == "cartpole-balance":
        return suite_env("cartpole", "balance", seed)
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
        raise ValueError(f"Unsupported DMC domain {domain!r}")

    xml, assets = module.get_model_and_assets()
    model = source_model(module, xml, assets)
    # Pinned wrappers apply these registered defaults before their first reset.
    model.find("texture", "grid").rgb1 = [0.1, 0.2, 0.3]
    model.find("texture", "grid").rgb2 = [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = [0.7, 0.5, 0.3, 1.0]
    model.find("light", "light").diffuse = [0.7, 0.7, 0.7]
    physics = compile_physics(module.Physics, model, f"{domain}.xml")
    return normalized(control.Environment(physics, task, time_limit=time_limit))


class FirstFamilyAdapter(DMCLanes):
    """Acrobot, Pendulum and Cartpole lanes."""

    def __init__(self, domain, num_envs, device="cpu", seed=0):
        if domain not in FIRST_FAMILY_FIELDS:
            raise ValueError(f"Unsupported DMC domain {domain!r}")
        self.domain = domain
        super().__init__(domain, num_envs, device, seed, lambda s: _make_first_family(domain, s), _FIRST_CORE[domain])


class CartpoleBatch:
    """Cartpole swingup-sparse and smooth balance batch lanes."""

    def __init__(self, b, models, sparse):
        self.sparse = sparse
        self.qpos, self.qvel, self.ctrl, self.xmat = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl"), b.bind("xmat")

    def reward(self):
        cart, cosine = self.qpos[:, 0], self.xmat[:, 2:, ZZ]
        if self.sparse:
            return tolerance(cart, (-0.25, 0.25)) * tolerance(cosine, (0.995, 1)).prod(axis=1)
        centered = (1 + scalar_tolerance(cart, margin=2)) / 2
        small_control = (4 + tolerance(self.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic")[:, 0]) / 5
        small_velocity = (1 + tolerance(self.qvel[:, 1:], margin=5).min(axis=1)) / 2
        return ((cosine + 1) / 2).mean(axis=1) * small_control * small_velocity * centered

    def observation(self):
        poles = self.xmat[:, 2:][:, :, [ZZ, XZ]].reshape(len(self.xmat), -1)
        return np.concatenate([self.qpos[:, :1], poles, self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return None


class AcrobotBatch:
    """Acrobot swingup-sparse batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.arms = [m.body("upper_arm").id, m.body("lower_arm").id]
        self.target, self.tip = m.site("target").id, m.site("tip").id
        self.radius = np.array([model.site_size[self.target, 0] for model in models])
        self.qvel, self.xmat, self.site_xpos = b.bind("qvel"), b.bind("xmat"), b.bind("site_xpos")

    def reward(self):
        v = self.site_xpos[:, self.target] - self.site_xpos[:, self.tip]
        return (np.sqrt(np.vecdot(v, v)) <= self.radius).astype(np.float64)

    def observation(self):
        arms = self.xmat[:, self.arms]
        return np.concatenate([arms[:, :, XZ], arms[:, :, ZZ], self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return None


class PendulumBatch:
    """Pendulum swingup batch lanes."""

    def __init__(self, b, models):
        self.pole = models[0].body("pole").id
        self.qvel, self.xmat = b.bind("qvel"), b.bind("xmat")

    def reward(self):
        return tolerance(self.xmat[:, self.pole, ZZ], (np.cos(np.deg2rad(8)), 1))

    def observation(self):
        return np.concatenate([self.xmat[:, self.pole, [ZZ, XZ]], self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return None


def _make_reacher(seed):
    """Construct Reacher with the registered model and action scaling."""
    from dm_control.rl import control
    from dm_control.suite import reacher

    xml, assets = reacher.get_model_and_assets()
    model = source_model(reacher, xml, assets)
    model.find("geom", "arm").density = 1000.0
    model.find("geom", "finger").density = 1000.0
    model.find("geom", "target").type = "sphere"
    model.find("geom", "target").size = np.array([0.015], np.float32)
    # Apply registered appearance defaults before compiling the model.
    model.find("texture", "grid").rgb1 = [0.1, 0.2, 0.3]
    model.find("texture", "grid").rgb2 = [0.2, 0.3, 0.4]
    model.find("material", "self").rgba = np.array([0.7, 0.5, 0.3, 1], np.float32)
    model.find("material", "target").rgba = np.r_[np.array([0.6, 0.3, 0.3], np.float32), 0.0]
    model.find("light", "light").diffuse = 0.7 * np.ones(3, np.float32)
    physics = compile_physics(reacher.Physics, model, "reacher.xml")
    task = reacher.Reacher(target_size=0.015, random=seed)
    return normalized(control.Environment(physics, task, time_limit=20))


class ReacherAdapter(DMCLanes):
    """Reacher lanes; state includes the target, which an implicit reset redraws."""

    def __init__(self, num_envs, device="cpu", seed=0):
        super().__init__("reacher", num_envs, device, seed, _make_reacher, _REACHER_CORE)


class ReacherBatch:
    """Reacher batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.target, self.finger = m.geom("target").id, m.geom("finger").id
        self.radii = np.array([model.geom_size[[self.target, self.finger], 0].sum() for model in models])
        self.qpos, self.qvel, self.geom_xpos = b.bind("qpos"), b.bind("qvel"), b.bind("geom_xpos")

    def _to_target(self):
        return self.geom_xpos[:, self.target, :2] - self.geom_xpos[:, self.finger, :2]

    def reward(self):
        v = self._to_target()
        return (np.sqrt(np.vecdot(v, v)) <= self.radii).astype(np.float64)

    def observation(self):
        return np.concatenate([self.qpos, self._to_target(), self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return None


FIRST_FAMILY_BATCH = {
    "acrobot": AcrobotBatch,
    "pendulum": PendulumBatch,
    "cartpole": partial(CartpoleBatch, sparse=True),
    "cartpole-balance": partial(CartpoleBatch, sparse=False),
    "reacher": ReacherBatch,
}
