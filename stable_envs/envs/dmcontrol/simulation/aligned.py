from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import physics as _physics
from .first_family import suite_env
from .lanes import DMCLanes
from .registered import apply_appearance, apply_floor_friction, compile_physics, normalized, source_model
from .state import core_fields

CONFIG: dict[str, dict[str, Any]] = {
    "ball_in_cup": {"nq": 4, "nv": 4, "nu": 2, "na": 0, "time_limit": 20.0, "dt": 0.02},
    "finger": {"nq": 3, "nv": 3, "nu": 2, "na": 0, "time_limit": 20.0, "dt": 0.02},
    "manipulator": {"nq": 11, "nv": 11, "nu": 5, "na": 0, "time_limit": 10.0, "dt": 0.01},
    "quadruped": {"nq": 23, "nv": 22, "nu": 12, "na": 12, "time_limit": 20.0, "dt": 0.02},
    "finger-spin": {"nq": 3, "nv": 3, "nu": 2, "na": 0},
    "finger-turn_easy": {"nq": 3, "nv": 3, "nu": 2, "na": 0},
    "point_mass-easy": {"nq": 2, "nv": 2, "nu": 2, "na": 0},
    "swimmer-swimmer6": {"nq": 8, "nv": 8, "nu": 5, "na": 0},
}

#: Episode-drawn model entries each simulator carries, in compact render-row order.
MODEL_KEYS = {
    "finger": (("model", "target_site_xz"),),
    "finger-turn_easy": (("model", "target_site_xz"),),
    "manipulator": (("model", "target_body_xz"), ("model", "target_body_qwqy")),
    "swimmer-swimmer6": (("model", "target_geom_xy"), ("model", "target_light_xy")),
}


def _core(domain):
    cfg = CONFIG[domain]
    return {
        **core_fields(cfg["nq"], cfg["nv"], cfg["na"]),
        **dict.fromkeys(MODEL_KEYS.get(domain, ()), (2, torch.float64)),
    }


ALIGNED_FIELDS = {domain: {**_core(domain), **_physics.state_fields(domain)} for domain in CONFIG}


def _make_env(domain, seed):
    if "-" in domain:
        return suite_env(*domain.split("-"), seed)
    from dm_control.rl import control

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
    model = source_model(module, xml, assets)
    if domain == "ball_in_cup":
        apply_appearance(model)
        for name in ("cup_part_0", "cup_part_1", "cup_part_2", "cup_part_3", "cup_part_4"):
            model.find("geom", name).density = 1000.0
        ball = model.find("geom", "ball")
        ball.density, ball.size = 1000.0, np.array([0.025], np.float32)
        target = model.find("site", "target")
        target.rgba = np.array([0.6, 0.3, 0.3, 0.0], np.float32).astype(np.float64)
        target.type = "sphere"
        target.size = np.array([0.05, 0.006, 0.05], np.float32)
    elif domain == "finger":
        apply_appearance(model)
        model.find("geom", "proximal").density = 1000.0
        model.find("geom", "fingertip").density = 1000.0
        for name in ("cap1", "cap2"):
            geom = model.find("geom", name)
            geom.density, geom.rgba = 1000.0, [0.7, 0.5, 0.3, 1.0]
        model.find("joint", "hinge").frictionloss = float(np.float32(0.1))
        model.find("material", "target").rgba = [0.6, 0.3, 0.3, 0.0]
        target = model.find("site", "target")
        target.type, target.size = "sphere", [0.03]
    elif domain == "manipulator":
        apply_appearance(model)
        model.find("geom", "upper_arm").density = 1000.0
        model.find("geom", "hand").density = 1000.0
        model.find("material", "target").rgba = [0.6, 0.3, 0.3, 0.0]
        target = model.find("geom", "target_ball")
        target.type, target.size = "sphere", np.array([0.015], np.float32)
    else:
        apply_appearance(model, (0.6, 0.3, 0.3))
        apply_floor_friction(model.find("geom", "floor"))
        model.find("geom", "torso").density = 1000.0
        model.find("geom", "foot_back_left").density = 1000.0
    physics = compile_physics(module.Physics, model, f"{domain}.xml")
    cfg = CONFIG[domain]
    return normalized(control.Environment(physics, task, time_limit=cfg["time_limit"], control_timestep=cfg["dt"]))


class DMCAlignedAdapter(DMCLanes):
    """BallInCup, Finger, Manipulator, Quadruped, PointMass and Swimmer lanes."""

    def __init__(self, domain, num_envs, device="cpu", seed=0):
        if domain not in CONFIG:
            raise ValueError(f"Unsupported aligned DMC domain {domain!r}")
        self.domain = domain
        super().__init__(domain, num_envs, device, seed, lambda s: _make_env(domain, s), _core(domain))
