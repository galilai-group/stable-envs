from __future__ import annotations

import numpy as np

from . import physics as _physics
from .lanes import DMCLanes
from .registered import apply_appearance, apply_floor_friction, compile_physics, normalized, source_model
from .state import core_fields

HUMANOID_CONFIG = {
    "nq": 28,
    "nv": 27,
    "nu": 21,
    "na": 0,
    "time_limit": 25.0,
    "control_timestep": 0.025,
}

HUMANOID_FIELDS = {**core_fields(28, 27), **_physics.state_fields("humanoid")}


def _apply_registered_defaults(model):
    """Apply registered reset values before compiling the model."""
    apply_appearance(model, light="top")
    apply_floor_friction(model.find("geom", "floor"))
    model.find("geom", "torso").density = 1000.0
    model.find("geom", "right_lower_arm").density = 1000.0


def _make_env(seed):
    """Construct Humanoid Walk with registered defaults and normalized actions."""
    from dm_control.rl import control
    from dm_control.suite import humanoid

    xml, assets = humanoid.get_model_and_assets()
    model = source_model(humanoid, xml, assets)
    _apply_registered_defaults(model)
    physics = compile_physics(humanoid.Physics, model, "humanoid.xml")
    task = humanoid.Humanoid(move_speed=1, pure_state=False, random=seed)
    env = control.Environment(
        physics,
        task,
        time_limit=HUMANOID_CONFIG["time_limit"],
        control_timestep=HUMANOID_CONFIG["control_timestep"],
    )
    return normalized(env)


class DMCHumanoidAdapter(DMCLanes):
    """Humanoid Walk lanes."""

    def __init__(self, num_envs, device="cpu", seed=0):
        super().__init__("humanoid", num_envs, device, seed, _make_env, core_fields(28, 27))

    def _info(self, env, lane):
        physics = env.physics
        return float("nan"), {
            "qpos": physics.data.qpos.copy(),
            "qvel": physics.data.qvel.copy(),
            "speed": np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]]),
            "torso_upright": physics.torso_upright(),
            "head_height": physics.head_height(),
        }
