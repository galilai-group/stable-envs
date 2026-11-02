from __future__ import annotations

import numpy as np

from . import physics as _physics
from .batch import ZX, ZY, ZZ, scalar_tolerance, tolerance
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


class HumanoidBatch:
    """Humanoid Walk batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.torso, self.head = m.body("torso").id, m.body("head").id
        self.limbs = [m.body(side + limb).id for side in ("left_", "right_") for limb in ("hand", "foot")]
        self.com_vel = int(m.sensor("torso_subtreelinvel").adr[0]) + np.arange(3)
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.xpos, self.xmat, self.sensordata = b.bind("xpos"), b.bind("xmat"), b.bind("sensordata")

    def _speed(self):
        v = self.sensordata[:, self.com_vel][:, :2]
        # A vector's np.linalg.norm is sqrt(dot), which vecdot reproduces and an axis norm does not.
        return np.sqrt(np.vecdot(v, v))

    def reward(self):
        inf = float("inf")
        standing = scalar_tolerance(self.xpos[:, self.head, 2], bounds=(1.4, inf), margin=1.4 / 4)
        upright = tolerance(self.xmat[:, self.torso, ZZ], (0.9, inf), sigmoid="linear", margin=1.9, value_at_margin=0)
        small_control = (4 + tolerance(self.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean(axis=1)) / 5
        move = tolerance(self._speed(), bounds=(1, inf), margin=1, value_at_margin=0, sigmoid="linear")
        return small_control * (standing * upright) * ((5 * move + 1) / 6)

    def observation(self):
        n, frame = len(self.qpos), self.xmat[:, self.torso].reshape(-1, 3, 3)
        extremities = np.matmul(self.xpos[:, self.limbs] - self.xpos[:, self.torso, None], frame).reshape(n, -1)
        head, vertical = self.xpos[:, self.head, 2:], self.xmat[:, self.torso, [ZX, ZY, ZZ]]
        parts = [self.qpos[:, 7:], head, extremities, vertical, self.sensordata[:, self.com_vel], self.qvel]
        return np.concatenate(parts, axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "speed": self._speed(),
            "torso_upright": self.xmat[:, self.torso, ZZ],
            "head_height": self.xpos[:, self.head, 2],
        }
