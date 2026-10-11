from __future__ import annotations

import os
import tempfile

import numpy as np

from . import physics as _physics
from .lanes import DMCLanes
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
    """Construct Humanoid Walk with registered defaults and normalized actions."""
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
