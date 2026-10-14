from __future__ import annotations

from typing import Any

import numpy as np

from . import physics as _physics
from .lanes import DMCLanes
from .registered import apply_appearance, apply_floor_friction, compile_physics, normalized, source_model
from .state import core_fields

CONFIG: dict[str, dict[str, Any]] = {
    "cheetah": {"nq": 9, "nv": 9, "nu": 6, "time_limit": 25.0, "control_timestep": None},
    "hopper": {"nq": 7, "nv": 7, "nu": 4, "time_limit": 20.0, "control_timestep": 0.02},
    "walker": {"nq": 9, "nv": 9, "nu": 6, "time_limit": 25.0, "control_timestep": 0.025},
}


CONTINUATION_FIELDS = {
    domain: {**core_fields(cfg["nq"], cfg["nv"]), **_physics.state_fields(domain)} for domain, cfg in CONFIG.items()
}


def _apply_registered_defaults(domain, model):
    """Apply registered reset values before compiling the model."""
    apply_appearance(model, light="top" if domain == "hopper" else "light")
    floor = model.find("geom", "ground" if domain == "cheetah" else "floor")
    apply_floor_friction(floor)

    density_geoms = {
        "cheetah": ("torso", "bfoot"),
        "hopper": ("torso", "foot"),
        "walker": ("torso", "left_foot"),
    }[domain]
    for name in density_geoms:
        model.find("geom", name).density = 1000.0

    if domain == "walker":
        floor.zaxis = np.array([0.0, 0.0, 1.0], np.float32)


def _make_env(domain, seed):
    """Construct Cheetah Run, Hopper Hop or Walker Walk with registered defaults."""
    from dm_control.rl import control

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
    model = source_model(module, xml, assets)
    _apply_registered_defaults(domain, model)
    physics = compile_physics(module.Physics, model, f"{domain}.xml")
    cfg = CONFIG[domain]
    kwargs = {"time_limit": cfg["time_limit"]}
    if cfg["control_timestep"] is not None:
        kwargs["control_timestep"] = cfg["control_timestep"]
    return normalized(control.Environment(physics, task, **kwargs))


class DMCContinuationAdapter(DMCLanes):
    """Cheetah, Hopper and Walker lanes."""

    def __init__(self, domain, num_envs, device="cpu", seed=0):
        if domain not in CONFIG:
            raise ValueError(f"Unsupported continuation domain {domain!r}")
        self.domain = domain
        cfg = CONFIG[domain]
        super().__init__(
            domain, num_envs, device, seed, lambda s: _make_env(domain, s), core_fields(cfg["nq"], cfg["nv"])
        )

    def _info(self, env, lane):
        physics, domain = env.physics, self.domain
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
        common = {"qpos": physics.data.qpos.copy(), "qvel": physics.data.qvel.copy()}
        return float("nan"), {**common, **{key: np.asarray(value, np.float64) for key, value in extra.items()}}
