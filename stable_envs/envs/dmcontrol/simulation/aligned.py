from __future__ import annotations

from functools import cache, partial
from typing import Any

import numpy as np
import torch

from . import physics as _physics
from .batch import ZZ, scalar_tolerance, tolerance
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


def _non_contacting_height(physics, orientation):
    """dm_control's quadruped ``_find_non_contacting_height`` in direct MuJoCo calls.

    Each attempt resets the data and runs both forward passes of ``reset_context`` with actuation disabled, so the
    data end as dm_control leaves them. MuJoCo releases the GIL in these calls, so lanes reset in parallel threads.
    """
    import mujoco

    m, d = physics.model.ptr, physics.data.ptr
    adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "root")])
    flags = m.opt.disableflags
    m.opt.disableflags = flags | mujoco.mjtDisableBit.mjDSBL_ACTUATION
    try:
        z, attempts, contacts = 0.0, 0, 1
        while contacts > 0:
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
            d.qpos[adr : adr + 3] = 0.0, 0.0, z
            d.qpos[adr + 3 : adr + 7] = orientation
            mujoco.mj_forward(m, d)
            contacts = d.ncon
            z += 0.01
            attempts += 1
            if attempts > 10000:
                raise RuntimeError("Failed to find a non-contacting configuration.")
    finally:
        m.opt.disableflags = flags


@cache
def _quadruped_move():
    """dm_control's quadruped ``Move`` task with the height search above."""
    from dm_control.suite import quadruped

    class Move(quadruped.Move):
        def initialize_episode(self, physics):
            orientation = self.random.randn(4)
            orientation /= np.linalg.norm(orientation)
            _non_contacting_height(physics, orientation)
            super(quadruped.Move, self).initialize_episode(physics)

    return Move


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
        task = _quadruped_move()(desired_speed=0.5, random=seed)
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

    def _info(self, env, lane):
        physics = env.physics
        info = {"qpos": physics.data.qpos.copy(), "qvel": physics.data.qvel.copy()}
        success = float(physics.in_target()) if self.domain == "ball_in_cup" else float("nan")
        if self.domain.startswith("finger"):
            info.update(
                hinge_velocity=np.asarray(physics.hinge_velocity()),
                target_position=physics.target_position().copy(),
                tip_position=physics.tip_position().copy(),
            )
        elif self.domain == "quadruped":
            info["torso_velocity"] = physics.torso_velocity().copy()
        return success, info


def _norm(v):
    """A vector's np.linalg.norm, sqrt(dot), per lane."""
    return np.sqrt(np.vecdot(v, v))


class BallInCupBatch:
    """Ball-in-cup catch batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.target, self.ball, ball = m.site("target").id, m.body("ball").id, m.geom("ball").id
        self.room = np.array([model.site_size[self.target, [0, 2]] - model.geom_size[ball, 0] for model in models])
        self.qpos, self.qvel = b.bind("qpos"), b.bind("qvel")
        self.site_xpos, self.xpos = b.bind("site_xpos"), b.bind("xpos")

    def reward(self):
        offset = self.site_xpos[:, self.target][:, [0, 2]] - self.xpos[:, self.ball][:, [0, 2]]
        return np.all(np.abs(offset) < self.room, axis=1).astype(np.float64)

    def observation(self):
        return np.concatenate([self.qpos, self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return self.reward(), {"qpos": self.qpos, "qvel": self.qvel}


class FingerBatch:
    """Finger turn and spin batch lanes."""

    def __init__(self, b, models, spin):
        m, self.spin = models[0], spin
        adr = lambda *names: [int(m.sensor(name).adr[0]) for name in names]
        self.position, self.touch = adr("proximal", "distal"), adr("touchtop", "touchbottom")
        self.velocity = adr("proximal_velocity", "distal_velocity", "hinge_velocity")
        self.tip, self.target, self.spinner = (np.array([a, a + 2]) for a in adr("tip", "target", "spinner"))
        self.radius = np.array([model.site_size[m.site("target").id, 0] for model in models])
        self.qpos, self.qvel, self.s = b.bind("qpos"), b.bind("qvel"), b.bind("sensordata")

    def _tip(self):
        return self.s[:, self.tip] - self.s[:, self.spinner]

    def _target(self):
        return self.s[:, self.target] - self.s[:, self.spinner]

    def _distance(self):
        return _norm(self._target() - self._tip()) - self.radius

    def reward(self):
        reached = self.s[:, self.velocity[2]] <= -15.0 if self.spin else self._distance() <= 0
        return reached.astype(np.float64)

    def observation(self):
        parts = [self.s[:, self.position], self._tip(), self.s[:, self.velocity], np.log1p(self.s[:, self.touch])]
        turn = [] if self.spin else [self._target(), self._distance()[:, None]]
        return np.concatenate([*parts, *turn], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "hinge_velocity": self.s[:, self.velocity[2]],
            "target_position": self._target(),
            "tip_position": self._tip(),
        }


class ManipulatorBatch:
    """Manipulator bring-ball batch lanes, fully observable."""

    def __init__(self, b, models):
        m = models[0]
        arm = ("arm_root", "arm_shoulder", "arm_elbow", "arm_wrist", "finger", "fingertip", "thumb", "thumbtip")
        self.arm_q = [m.jnt_qposadr[m.joint(name).id] for name in arm]
        self.arm_v = [m.jnt_dofadr[m.joint(name).id] for name in arm]
        self.ball_v = [m.jnt_dofadr[m.joint(f"ball_{axis}").id] for axis in "xzy"]
        touch = ("palm", "finger", "thumb", "fingertip", "thumbtip")
        self.touch = [int(m.sensor(f"{name}_touch").adr[0]) for name in touch]
        self.bodies = [m.body(name).id for name in ("hand", "ball", "target_ball")]
        self.ball, self.target = m.site("ball").id, m.site("target_ball").id
        self.qpos, self.qvel, self.s = b.bind("qpos"), b.bind("qvel"), b.bind("sensordata")
        self.xpos, self.xquat, self.site_xpos = b.bind("xpos"), b.bind("xquat"), b.bind("site_xpos")

    def reward(self):
        return scalar_tolerance(_norm(self.site_xpos[:, self.ball] - self.site_xpos[:, self.target]), (0, 0.01), 0.02)

    def observation(self):
        q = self.qpos[:, self.arm_q]
        arm = np.stack([np.sin(q), np.cos(q)], axis=2).reshape(len(q), -1)
        x, quat = self.xpos[:, self.bodies], self.xquat[:, self.bodies]
        hand, ball, target = np.concatenate([x[..., [0, 2]], quat[..., [0, 2]]], axis=2).transpose(1, 0, 2)
        touch, ball_vel = np.log1p(self.s[:, self.touch]), self.qvel[:, self.ball_v]
        parts = [arm, self.qvel[:, self.arm_v], touch, hand, ball, ball_vel, target]
        return np.concatenate(parts, axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {"qpos": self.qpos, "qvel": self.qvel}


class QuadrupedBatch:
    """Quadruped walk batch lanes."""

    def __init__(self, b, models):
        import mujoco

        m, sensor = models[0], mujoco.mjtSensor
        # Gyro, accelerometer, force and torque sensors each have three values.
        sensors = lambda *types: (m.sensor_adr[np.isin(m.sensor_type, types)][:, None] + np.arange(3)).ravel()
        hinges = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.hinge_q, self.hinge_v = m.jnt_qposadr[hinges], m.jnt_dofadr[hinges]
        self.imu = sensors(sensor.mjSENS_GYRO, sensor.mjSENS_ACCELEROMETER)
        self.force_torque = sensors(sensor.mjSENS_FORCE, sensor.mjSENS_TORQUE)
        self.velocimeter = int(m.sensor("velocimeter").adr[0]) + np.arange(3)
        self.torso = m.body("torso").id
        self.qpos, self.qvel, self.act = b.bind("qpos"), b.bind("qvel"), b.bind("act")
        self.xmat, self.s = b.bind("xmat"), b.bind("sensordata")

    def reward(self):
        inf = float("inf")
        upright = tolerance(self.xmat[:, self.torso, ZZ], (1.0, inf), margin=2.0, sigmoid="linear", value_at_margin=0)
        move = tolerance(self.s[:, self.velocimeter[0]], (0.5, inf), margin=0.5, sigmoid="linear", value_at_margin=0.5)
        return upright * move

    def observation(self):
        state = [self.qpos[:, self.hinge_q], self.qvel[:, self.hinge_v], self.act, self.s[:, self.velocimeter]]
        sensors = [self.xmat[:, self.torso, ZZ:], self.s[:, self.imu], np.arcsinh(self.s[:, self.force_torque])]
        return np.concatenate([*state, *sensors], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {
            "qpos": self.qpos,
            "qvel": self.qvel,
            "torso_velocity": self.s[:, self.velocimeter],
        }


class PointMassBatch:
    """Point-mass easy batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.target, self.mass = m.geom("target").id, m.geom("pointmass").id
        self.size = np.array([model.geom_size[self.target, 0] for model in models])
        self.qpos, self.qvel, self.ctrl = b.bind("qpos"), b.bind("qvel"), b.bind("ctrl")
        self.geom_xpos = b.bind("geom_xpos")

    def reward(self):
        distance = _norm(self.geom_xpos[:, self.target] - self.geom_xpos[:, self.mass])
        near_target = scalar_tolerance(distance, bounds=(0, self.size), margin=self.size)
        small_control = (tolerance(self.ctrl, margin=1, value_at_margin=0, sigmoid="quadratic").mean(axis=1) + 4) / 5
        return near_target * small_control

    def observation(self):
        return np.concatenate([self.qpos, self.qvel], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {"qpos": self.qpos, "qvel": self.qvel}


class SwimmerBatch:
    """Swimmer6 batch lanes."""

    def __init__(self, b, models):
        m = models[0]
        self.target, self.nose, self.head = m.geom("target").id, m.geom("nose").id, m.body("head").id
        self.size = np.array([model.geom_size[self.target, 0] for model in models])
        self.qpos, self.qvel, self.geom_xpos = b.bind("qpos"), b.bind("qvel"), b.bind("geom_xpos")
        self.xmat, self.s = b.bind("xmat"), b.bind("sensordata")

    def _to_target(self):
        offset = self.geom_xpos[:, self.target] - self.geom_xpos[:, self.nose]
        return np.matmul(offset[:, None], self.xmat[:, self.head].reshape(-1, 3, 3))[:, 0, :2]

    def reward(self):
        size = self.size
        return scalar_tolerance(_norm(self._to_target()), bounds=(0, size), margin=5 * size, sigmoid="long_tail")

    def observation(self):
        n = len(self.qpos)
        velocities = self.s[:, 12:].reshape(n, -1, 6)[:, :, [0, 1, 5]].reshape(n, -1)
        return np.concatenate([self.qpos[:, 3:], self._to_target(), velocities], axis=1, dtype=np.float32)

    def info(self):
        return np.full(len(self.qpos), np.nan), {"qpos": self.qpos, "qvel": self.qvel}


ALIGNED_BATCH = {
    "ball_in_cup": BallInCupBatch,
    "finger": partial(FingerBatch, spin=False),
    "finger-spin": partial(FingerBatch, spin=True),
    "finger-turn_easy": partial(FingerBatch, spin=False),
    "manipulator": ManipulatorBatch,
    "quadruped": QuadrupedBatch,
    "point_mass-easy": PointMassBatch,
    "swimmer-swimmer6": SwimmerBatch,
}
