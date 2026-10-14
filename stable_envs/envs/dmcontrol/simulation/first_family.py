from __future__ import annotations

import numpy as np
import torch

from . import physics as _physics
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
REACHER_FIELDS = {
    "reacher": _with_physics("reacher", _REACHER_CORE),
    "reacher-easy": _with_physics("reacher-easy", _REACHER_CORE),
}


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


def _make_reacher(seed):
    """Construct Reacher’s hard task with the registered model and action scaling."""
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
    """Reacher lanes; State includes the target, which an implicit reset redraws."""

    def __init__(self, num_envs, device="cpu", seed=0, task="hard"):
        key = "reacher" if task == "hard" else f"reacher-{task}"
        make = _make_reacher if task == "hard" else (lambda s: suite_env("reacher", task, s))
        super().__init__(key, num_envs, device, seed, make, _REACHER_CORE)
