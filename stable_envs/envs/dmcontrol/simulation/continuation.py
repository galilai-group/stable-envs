from __future__ import annotations

from typing import Any

import numpy as np

from . import physics as _physics
from .batch import XX, XZ, ZZ, scalar_tolerance, tolerance
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


class HopperBatch:
    """Hopper Hop batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.torso, self.foot = m.body("torso").id, m.body("foot").id
        self.speed_adr = int(m.sensor("torso_subtreelinvel").adr[0])
        self.touch = [int(m.sensor(name).adr[0]) for name in ("touch_toe", "touch_heel")]
        self.qpos, self.qvel, self.xipos = b.bind("qpos"), b.bind("qvel"), b.bind("xipos")
        self.sensordata, self.angmom = b.bind("sensordata"), b.bind("subtree_angmom")

    def _height(self):
        return self.xipos[:, self.torso, 2] - self.xipos[:, self.foot, 2]

    def reward(self):
        standing = tolerance(self._height(), (0.6, 2))
        speed = self.sensordata[:, self.speed_adr]
        return standing * tolerance(speed, bounds=(2, float("inf")), margin=1, value_at_margin=0.5, sigmoid="linear")

    def observation(self):
        touch = np.log1p(self.sensordata[:, self.touch])
        return np.concatenate([self.qpos[:, 1:], self.qvel, touch], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "speed": self.sensordata[:, self.speed_adr],
            "height": self._height(),
            "angmomentum": self.angmom[:, self.torso, 1],
        }


class CheetahBatch:
    """Cheetah Run batch lanes."""

    #: Settling steps of dm_control's reset that the batch runs between ``begin_reset`` and ``end_reset``.
    RESET_STEPS = 199

    @staticmethod
    def begin_reset(env):
        """dm_control's Cheetah reset up to its 200 settling steps, and ``mj_step2`` of the first one.

        dm_control's legacy step ends the first settling step with ``mj_step2`` on the stages of the state before the
        draw. The other 199 steps and the closing ``mj_step1`` are ``mj_step`` calls, which the batch runs.
        """
        import mujoco
        from dm_control.rl.control import PhysicsError

        physics, task = env.physics, env.task
        assert physics.legacy_step and physics.model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4
        try:
            physics.reset()
        except PhysicsError:
            pass
        assert physics.model.nq == physics.model.njnt
        is_limited = physics.model.jnt_limited == 1
        lower, upper = physics.model.jnt_range[is_limited].T
        physics.data.qpos[is_limited] = task.random.uniform(lower, upper)
        with physics.check_invalid_state():
            mujoco.mj_step2(physics.model.ptr, physics.data.ptr)

    @staticmethod
    def end_reset(env):
        """The rest of dm_control's Cheetah reset after the settling steps."""
        from dm_control.suite import cheetah

        physics, task = env.physics, env.task
        physics.data.time = 0
        task._timeout_progress = 0
        super(cheetah.Cheetah, task).initialize_episode(physics)
        physics.after_reset()

    def __init__(self, b, models):
        m = models[0]
        self.torso, self.bfoot, self.ffoot = (m.body(name).id for name in ("torso", "bfoot", "ffoot"))
        self.speed_adr = int(m.sensor("torso_subtreelinvel").adr[0])
        self.qpos, self.qvel, self.xpos = b.bind("qpos"), b.bind("qvel"), b.bind("xpos")
        self.sensordata, self.angmom = b.bind("sensordata"), b.bind("subtree_angmom")

    def reward(self):
        speed = self.sensordata[:, self.speed_adr]
        return tolerance(speed, bounds=(10, float("inf")), margin=10, value_at_margin=0, sigmoid="linear")

    def observation(self):
        return np.concatenate([self.qpos[:, 1:], self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "speed": self.sensordata[:, self.speed_adr],
            "torso_height": self.xpos[:, self.torso, 2],
            "bfoot_height": self.xpos[:, self.bfoot, 2],
            "ffoot_height": self.xpos[:, self.ffoot, 2],
            "angmomentum": self.angmom[:, self.torso, 1],
        }


class WalkerBatch:
    """Walker Walk batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.torso = m.body("torso").id
        limbs = ("left_foot", "right_foot", "left_thigh", "right_thigh")
        self.limbs = {f"{name}_height": m.body(name).id for name in limbs}
        self.speed_adr = int(m.sensor("torso_subtreelinvel").adr[0])
        self.qpos, self.qvel, self.xpos = b.bind("qpos"), b.bind("qvel"), b.bind("xpos")
        self.xmat, self.sensordata = b.bind("xmat"), b.bind("sensordata")

    def reward(self):
        standing = scalar_tolerance(self.xpos[:, self.torso, 2], bounds=(1.2, float("inf")), margin=0.6)
        stand_reward = (3 * standing + (1 + self.xmat[:, self.torso, ZZ]) / 2) / 4
        speed = self.sensordata[:, self.speed_adr]
        move = tolerance(speed, bounds=(1, float("inf")), margin=0.5, value_at_margin=0.5, sigmoid="linear")
        return stand_reward * (5 * move + 1) / 6

    def observation(self):
        orientations = self.xmat[:, 1:][:, :, [XX, XZ]].reshape(len(self.xmat), -1)
        return np.concatenate([orientations, self.xpos[:, self.torso, 2:], self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "speed": self.sensordata[:, self.speed_adr],
            "torso_height": self.xpos[:, self.torso, 2],
            "torso_upright": self.xmat[:, self.torso, ZZ],
            **{key: self.xpos[:, body, 2] for key, body in self.limbs.items()},
        }


CONTINUATION_BATCH = {"cheetah": CheetahBatch, "hopper": HopperBatch, "walker": WalkerBatch}
