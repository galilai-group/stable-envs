"""Registered DMC task outputs evaluated on Torch views of MuJoCo Warp arrays.

Bind names and task constants once from the registered source environment. During
stepping, every calculation stays on the snapshot's device. Snapshot fields use
MuJoCo shapes with a leading world dimension (in particular xmat is [B, nbody, 9]).
"""

from __future__ import annotations

import math
from functools import cache
from numbers import Real
from typing import Protocol

import numpy as np
import torch

_XMAT = {name: i for i, name in enumerate(("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"))}


class Snapshot(Protocol):
    """Read-only batched views; model fields also have a leading world dimension."""

    def data(self, name: str) -> torch.Tensor: ...

    def model(self, name: str) -> torch.Tensor: ...


def _mujoco():
    import mujoco

    return mujoco


def _id(model, kind: str, name: str) -> int:
    mujoco = _mujoco()
    index = mujoco.mj_name2id(model, getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}"), name)
    if index < 0:
        raise KeyError(f"no {kind} named {name!r}")
    return int(index)


def _qpos(model, *joints: str) -> list[int]:
    return [int(model.jnt_qposadr[_id(model, "joint", name)]) for name in joints]


def _dof(model, *joints: str) -> list[int]:
    return [int(model.jnt_dofadr[_id(model, "joint", name)]) for name in joints]


def _sensor_ids(model, ids) -> list[int]:
    return [int(j) for i in ids for j in range(model.sensor_adr[i], model.sensor_adr[i] + model.sensor_dim[i])]


def _sensor(model, *names: str) -> list[int]:
    return _sensor_ids(model, [_id(model, "sensor", name) for name in names])


def _norm(value: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(value, dim=-1)


@cache
def _scalar(value, dtype, device):
    # Immutable task constants. The initial blocking upload finishes before a
    # cached value can be reused by another adapter or CUDA stream. Retain the
    # tensors so their storage cannot be recycled while another stream uses it.
    return torch.tensor(value, dtype=dtype, device=device)


def _tolerance(value, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1):
    """The registered tasks' dm_control tolerance curves, without host reads."""
    value = value.to(torch.float64)

    def parameter(v):
        if isinstance(v, Real):
            return _scalar(float(v), value.dtype, value.device)
        return torch.as_tensor(v, dtype=value.dtype, device=value.device)

    lower, upper = (parameter(v) for v in bounds)
    margin = parameter(margin)
    inside = (lower <= value) & (value <= upper)
    distance = torch.where(value < lower, lower - value, value - upper)
    # Zero margins are hard bounds; a safe denominator also prevents dormant NaNs.
    scaled = distance / torch.where(margin == 0, torch.ones_like(margin), margin)
    if sigmoid == "gaussian":
        curve = torch.exp(-0.5 * (scaled * math.sqrt(-2 * math.log(value_at_margin))).square())
    elif sigmoid == "long_tail":
        curve = 1 / (1 + (scaled * math.sqrt(1 / value_at_margin - 1)).square())
    elif sigmoid == "linear":
        scaled = scaled * (1 - value_at_margin)
        curve = torch.where(scaled.abs() < 1, 1 - scaled, torch.zeros_like(scaled))
    elif sigmoid == "quadratic":
        scaled = scaled * math.sqrt(1 - value_at_margin)
        curve = torch.where(scaled.abs() < 1, 1 - scaled.square(), torch.zeros_like(scaled))
    else:
        raise ValueError(f"unsupported sigmoid {sigmoid!r}")
    return torch.where(inside, torch.ones_like(value), torch.where(margin > 0, curve, torch.zeros_like(value)))


class BatchTask:
    """Device task formulas; observation order and info keys follow the source."""

    def __init__(self, env):
        self.task = env.task
        self.m = env.physics.model.ptr

    def observation(self, s: Snapshot) -> torch.Tensor:
        parts = self._observation(s)
        batch = parts[0].shape[0]
        return torch.cat([part.reshape(batch, -1) for part in parts], dim=1).to(torch.float32)

    def _observation(self, s: Snapshot) -> list[torch.Tensor]:
        raise NotImplementedError

    def reward(self, s: Snapshot) -> torch.Tensor:
        return self._reward(s).to(torch.float64)

    def _reward(self, s: Snapshot) -> torch.Tensor:
        raise NotImplementedError

    def info(self, s: Snapshot):
        return None


def _state_info(s: Snapshot, success=None, **extra):
    qpos = s.data("qpos")
    if success is None:
        success = torch.full((len(qpos),), float("nan"), dtype=torch.float64, device=qpos.device)
    values = {"qpos": qpos, "qvel": s.data("qvel"), **extra}
    return success.to(torch.float64), {
        name: value.to(dtype=torch.float64, copy=True) for name, value in values.items()
    }


class Acrobot(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        m = self.m
        self.arms = [_id(m, "body", "upper_arm"), _id(m, "body", "lower_arm")]
        self.target, self.tip = _id(m, "site", "target"), _id(m, "site", "tip")
        self.margin = 0 if self.task._sparse else 1

    def _observation(self, s):
        xmat = s.data("xmat")[:, self.arms]
        return [xmat[:, :, _XMAT["xz"]], xmat[:, :, _XMAT["zz"]], s.data("qvel")]

    def _reward(self, s):
        site = s.data("site_xpos")
        to_target = _norm(site[:, self.target] - site[:, self.tip])
        radius = s.model("site_size")[:, self.target, 0]
        return _tolerance(to_target, (0, radius), self.margin)


class Pendulum(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        self.pole = _id(self.m, "body", "pole")
        self.hinge = _dof(self.m, "hinge")
        from dm_control.suite import pendulum

        self.cosine_bound = pendulum._COSINE_BOUND

    def _observation(self, s):
        xmat = s.data("xmat")[:, self.pole]
        return [xmat[:, [_XMAT["zz"], _XMAT["xz"]]], s.data("qvel")[:, self.hinge]]

    def _reward(self, s):
        return _tolerance(s.data("xmat")[:, self.pole, _XMAT["zz"]], (self.cosine_bound, 1))


class Cartpole(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        self.slider = _qpos(self.m, "slider")
        self.sparse = self.task._sparse
        self.cart_range = self.task._CART_RANGE
        self.angle_range = self.task._ANGLE_COSINE_RANGE

    def _observation(self, s):
        poles = s.data("xmat")[:, 2:][:, :, [_XMAT["zz"], _XMAT["xz"]]]
        return [s.data("qpos")[:, self.slider], poles.reshape(len(poles), -1), s.data("qvel")]

    def _reward(self, s):

        cart = s.data("qpos")[:, self.slider[0]]
        cosine = s.data("xmat")[:, 2:, _XMAT["zz"]]
        if self.sparse:
            cart_in_bounds = _tolerance(cart, self.cart_range)
            angle_in_bounds = _tolerance(cosine, self.angle_range).prod(dim=1)
            return cart_in_bounds * angle_in_bounds
        upright = (cosine + 1) / 2
        centered = _tolerance(cart, (0.0, 0.0), 2)
        centered = (1 + centered) / 2
        small_control = _tolerance(s.data("ctrl"), margin=1, value_at_margin=0, sigmoid="quadratic")[:, 0]
        small_control = (4 + small_control) / 5
        small_velocity = _tolerance(s.data("qvel")[:, 1:], margin=5).amin(dim=1)
        small_velocity = (1 + small_velocity) / 2
        return upright.mean(dim=1) * small_control * small_velocity * centered


class Reacher(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        self.target, self.finger = _id(self.m, "geom", "target"), _id(self.m, "geom", "finger")

    def _to_target(self, s):
        geom = s.data("geom_xpos")
        return geom[:, self.target, :2] - geom[:, self.finger, :2]

    def _observation(self, s):
        return [s.data("qpos"), self._to_target(s), s.data("qvel")]

    def _reward(self, s):
        size = s.model("geom_size")
        radii = size[:, [self.target, self.finger], 0].sum(dim=1)
        return _tolerance(_norm(self._to_target(s)), (0, radii))


class BallInCup(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        m = self.m
        self.target, self.ball_body, self.ball_geom = (
            _id(m, "site", "target"),
            _id(m, "body", "ball"),
            _id(m, "geom", "ball"),
        )

    def _observation(self, s):
        return [s.data("qpos"), s.data("qvel")]

    def _in_target(self, s):
        target = s.data("site_xpos")[:, self.target][:, [0, 2]]
        ball = s.data("xpos")[:, self.ball_body][:, [0, 2]]
        ball_to_target = abs(target - ball)
        target_size = s.model("site_size")[:, self.target][:, [0, 2]]
        ball_size = s.model("geom_size")[:, self.ball_geom, 0]
        return (ball_to_target < target_size - ball_size[:, None]).all(dim=1).to(torch.float64)

    def _reward(self, s):
        return self._in_target(s)

    def info(self, s):
        return _state_info(s, self._in_target(s))


class Finger(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        from dm_control.suite import finger

        m = self.m
        self.spin = isinstance(self.task, finger.Spin)
        if not self.spin and not isinstance(self.task, finger.Turn):
            raise ValueError(f"unsupported finger task {type(self.task).__name__}")
        self.position = _sensor(m, "proximal", "distal")
        self.velocity = _sensor(m, "proximal_velocity", "distal_velocity", "hinge_velocity")
        self.touch = _sensor(m, "touchtop", "touchbottom")
        self.hinge_velocity = _sensor(m, "hinge_velocity")[0]
        self.tip, self.spinner, self.target = (
            [_sensor(m, n)[i] for i in (0, 2)] for n in ("tip", "spinner", "target")
        )
        self.target_site = _id(m, "site", "target")
        self.spin_velocity = finger._SPIN_VELOCITY

    def _tip(self, sensors):
        return sensors[:, self.tip] - sensors[:, self.spinner]

    def _target(self, sensors):
        return sensors[:, self.target] - sensors[:, self.spinner]

    def _dist_to_target(self, s):
        sensors = s.data("sensordata")
        to_target = self._target(sensors) - self._tip(sensors)
        return _norm(to_target) - s.model("site_size")[:, self.target_site, 0]

    def _observation(self, s):
        sensors = s.data("sensordata")
        obs = [
            torch.hstack((sensors[:, self.position], self._tip(sensors))),
            sensors[:, self.velocity],
            torch.log1p(sensors[:, self.touch]),
        ]
        if not self.spin:
            obs += [self._target(sensors), self._dist_to_target(s)[:, None]]
        return obs

    def _reward(self, s):
        if self.spin:
            return (s.data("sensordata")[:, self.hinge_velocity] <= -self.spin_velocity).to(torch.float64)
        return (self._dist_to_target(s) <= 0).to(torch.float64)

    def info(self, s):
        sensors = s.data("sensordata")
        return _state_info(
            s,
            hinge_velocity=sensors[:, self.hinge_velocity],
            target_position=self._target(sensors),
            tip_position=self._tip(sensors),
        )


class Manipulator(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        from dm_control.suite import manipulator

        task, m = self.task, self.m
        if task._use_peg or task._insert or not task._fully_observable:
            raise ValueError("only the fully observable ball task is batched")
        self.arm_qpos, self.arm_dof = _qpos(m, *manipulator._ARM_JOINTS), _dof(m, *manipulator._ARM_JOINTS)
        self.touch = _sensor(m, *manipulator._TOUCH_SENSORS)
        self.bodies = [_id(m, "body", name) for name in ("hand", task._object, task._target)]
        self.object_dof = _dof(m, *task._object_joints)
        self.ball, self.target = _id(m, "site", "ball"), _id(m, "site", "target_ball")
        self.close = manipulator._CLOSE

    def _observation(self, s):
        joint_pos = s.data("qpos")[:, self.arm_qpos]
        arm_pos = torch.stack([torch.sin(joint_pos), torch.cos(joint_pos)], dim=-1).reshape(len(joint_pos), -1)
        xpos, xquat = s.data("xpos"), s.data("xquat")
        hand, obj, target = (
            torch.hstack([xpos[:, body][:, [0, 2]], xquat[:, body][:, [0, 2]]]) for body in self.bodies
        )
        qvel = s.data("qvel")
        touch = torch.log1p(s.data("sensordata")[:, self.touch])
        return [arm_pos, qvel[:, self.arm_dof], touch, hand, obj, qvel[:, self.object_dof], target]

    def _reward(self, s):

        site = s.data("site_xpos")
        distance = _norm(site[:, self.ball] - site[:, self.target])
        return _tolerance(distance, (0, self.close), self.close * 2)

    def info(self, s):
        return _state_info(s)


class Quadruped(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        mujoco = _mujoco()
        m = self.m
        hinges = np.nonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
        self.hinge_qpos, self.hinge_dof = m.jnt_qposadr[hinges].tolist(), m.jnt_dofadr[hinges].tolist()
        self.torso = _id(m, "body", "torso")
        self.velocimeter = _sensor(m, "velocimeter")
        kinds = mujoco.mjtSensor

        def of_type(*types):
            return _sensor_ids(m, np.where(np.isin(m.sensor_type, [int(t) for t in types]).ravel())[0])

        self.imu = of_type(kinds.mjSENS_GYRO, kinds.mjSENS_ACCELEROMETER)
        self.force_torque = of_type(kinds.mjSENS_FORCE, kinds.mjSENS_TORQUE)
        self.speed = self.task._desired_speed

    def _observation(self, s):
        qpos, qvel, sensors = s.data("qpos"), s.data("qvel"), s.data("sensordata")
        return [
            torch.hstack((qpos[:, self.hinge_qpos], qvel[:, self.hinge_dof], s.data("act"))),
            sensors[:, self.velocimeter],
            s.data("xmat")[:, self.torso, _XMAT["zz"]],
            sensors[:, self.imu],
            torch.asinh(sensors[:, self.force_torque]),
        ]

    def _reward(self, s):

        move_reward = _tolerance(
            s.data("sensordata")[:, self.velocimeter[0]],
            bounds=(self.speed, float("inf")),
            margin=self.speed,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        deviation = 1.0
        upright = _tolerance(
            s.data("xmat")[:, self.torso, _XMAT["zz"]],
            bounds=(deviation, float("inf")),
            sigmoid="linear",
            margin=1 + deviation,
            value_at_margin=0,
        )
        return upright * move_reward

    def info(self, s):
        return _state_info(s, torso_velocity=s.data("sensordata")[:, self.velocimeter])


class PointMass(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        self.target, self.mass = _id(self.m, "geom", "target"), _id(self.m, "geom", "pointmass")

    def _observation(self, s):
        return [s.data("qpos"), s.data("qvel")]

    def _reward(self, s):

        geom = s.data("geom_xpos")
        target_size = s.model("geom_size")[:, self.target, 0]
        near_target = _tolerance(_norm(geom[:, self.target] - geom[:, self.mass]), (0, target_size), target_size)
        control_reward = _tolerance(s.data("ctrl"), margin=1, value_at_margin=0, sigmoid="quadratic").mean(dim=1)
        small_control = (control_reward + 4) / 5
        return near_target * small_control

    def info(self, s):
        return _state_info(s)


class Swimmer(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        m = self.m
        self.target, self.nose, self.head = _id(m, "geom", "target"), _id(m, "geom", "nose"), _id(m, "body", "head")

    def _nose_to_target(self, s):
        geom = s.data("geom_xpos")
        nose_to_target = geom[:, self.target] - geom[:, self.nose]
        head = s.data("xmat")[:, self.head].reshape(-1, 3, 3)
        return torch.bmm(nose_to_target[:, None, :], head)[:, 0, :2]

    def _observation(self, s):
        velocities = s.data("sensordata")[:, 12:].reshape(len(s.data("qpos")), -1, 6)[:, :, [0, 1, 5]]
        return [s.data("qpos")[:, 3:], self._nose_to_target(s), velocities.reshape(len(velocities), -1)]

    def _reward(self, s):
        target_size = s.model("geom_size")[:, self.target, 0]
        return _tolerance(_norm(self._nose_to_target(s)), (0, target_size), 5 * target_size, "long_tail")

    def info(self, s):
        return _state_info(s)


class Cheetah(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        m = self.m
        self.speed = _sensor(m, "torso_subtreelinvel")[0]
        self.torso, self.bfoot, self.ffoot = (_id(m, "body", n) for n in ("torso", "bfoot", "ffoot"))
        from dm_control.suite import cheetah

        self.run_speed = cheetah._RUN_SPEED

    def _observation(self, s):
        return [s.data("qpos")[:, 1:], s.data("qvel")]

    def _reward(self, s):
        return _tolerance(
            s.data("sensordata")[:, self.speed],
            bounds=(self.run_speed, float("inf")),
            margin=self.run_speed,
            value_at_margin=0,
            sigmoid="linear",
        )

    def info(self, s):
        z = s.data("xpos")[:, :, 2]
        return _state_info(
            s,
            speed=s.data("sensordata")[:, self.speed],
            torso_height=z[:, self.torso],
            bfoot_height=z[:, self.bfoot],
            ffoot_height=z[:, self.ffoot],
            angmomentum=s.data("subtree_angmom")[:, self.torso, 1],
        )


class Hopper(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        m = self.m
        self.speed = _sensor(m, "torso_subtreelinvel")[0]
        self.touch = _sensor(m, "touch_toe", "touch_heel")
        self.torso, self.foot = _id(m, "body", "torso"), _id(m, "body", "foot")
        from dm_control.suite import hopper

        self.stand_height, self.hop_speed = hopper._STAND_HEIGHT, hopper._HOP_SPEED
        if not self.task._hopping:
            raise ValueError("only the hopping task is batched")

    def _height(self, s):
        xipos = s.data("xipos")
        return xipos[:, self.torso, 2] - xipos[:, self.foot, 2]

    def _observation(self, s):
        return [s.data("qpos")[:, 1:], s.data("qvel"), torch.log1p(s.data("sensordata")[:, self.touch])]

    def _reward(self, s):
        standing = _tolerance(self._height(s), (self.stand_height, 2))
        hopping = _tolerance(
            s.data("sensordata")[:, self.speed],
            bounds=(self.hop_speed, float("inf")),
            margin=self.hop_speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return standing * hopping

    def info(self, s):
        return _state_info(
            s,
            speed=s.data("sensordata")[:, self.speed],
            height=self._height(s),
            angmomentum=s.data("subtree_angmom")[:, self.torso, 1],
        )


class Walker(BatchTask):
    _HEIGHTS = ("left_foot", "right_foot", "left_thigh", "right_thigh")

    def __init__(self, env):
        super().__init__(env)
        m = self.m
        self.speed = _sensor(m, "torso_subtreelinvel")[0]
        self.torso = _id(m, "body", "torso")
        self.heights = [_id(m, "body", name) for name in self._HEIGHTS]
        self.move_speed = self.task._move_speed
        from dm_control.suite import walker

        self.stand_height = walker._STAND_HEIGHT

    def _observation(self, s):
        orientations = s.data("xmat")[:, 1:][:, :, [_XMAT["xx"], _XMAT["xz"]]]
        return [
            orientations.reshape(len(orientations), -1),
            s.data("xpos")[:, self.torso, 2],
            s.data("qvel"),
        ]

    def _reward(self, s):
        standing = _tolerance(
            s.data("xpos")[:, self.torso, 2], (self.stand_height, float("inf")), self.stand_height / 2
        )
        upright = (1 + s.data("xmat")[:, self.torso, _XMAT["zz"]]) / 2
        stand_reward = (3 * standing + upright) / 4
        if self.move_speed == 0:
            return stand_reward
        move_reward = _tolerance(
            s.data("sensordata")[:, self.speed],
            bounds=(self.move_speed, float("inf")),
            margin=self.move_speed / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return stand_reward * (5 * move_reward + 1) / 6

    def info(self, s):
        z = s.data("xpos")[:, :, 2]
        return _state_info(
            s,
            speed=s.data("sensordata")[:, self.speed],
            torso_height=z[:, self.torso],
            torso_upright=s.data("xmat")[:, self.torso, _XMAT["zz"]],
            **{f"{name}_height": z[:, body] for name, body in zip(self._HEIGHTS, self.heights, strict=True)},
        )


class Humanoid(BatchTask):
    def __init__(self, env):
        super().__init__(env)
        m = self.m
        if self.task._pure_state:
            raise ValueError("only the feature observation is batched")
        self.torso, self.head = _id(m, "body", "torso"), _id(m, "body", "head")
        self.limbs = [_id(m, "body", side + limb) for side in ("left_", "right_") for limb in ("hand", "foot")]
        self.com_velocity = _sensor(m, "torso_subtreelinvel")
        self.move_speed = self.task._move_speed
        from dm_control.suite import humanoid

        self.stand_height = humanoid._STAND_HEIGHT

    def _extremities(self, s):
        xpos = s.data("xpos")
        frames = s.data("xmat")[:, self.torso].reshape(-1, 3, 3)
        torso = xpos[:, self.torso]
        relative = xpos[:, self.limbs] - torso[:, None, :]
        return torch.bmm(relative, frames).flatten(1)

    def _speed(self, s):
        return _norm(s.data("sensordata")[:, self.com_velocity][:, [0, 1]])

    def _observation(self, s):
        xmat = s.data("xmat")[:, self.torso]
        return [
            s.data("qpos")[:, 7:],
            s.data("xpos")[:, self.head, 2],
            self._extremities(s),
            xmat[:, [_XMAT["zx"], _XMAT["zy"], _XMAT["zz"]]],
            s.data("sensordata")[:, self.com_velocity],
            s.data("qvel"),
        ]

    def _reward(self, s):
        standing = _tolerance(
            s.data("xpos")[:, self.head, 2], (self.stand_height, float("inf")), self.stand_height / 4
        )
        upright = _tolerance(
            s.data("xmat")[:, self.torso, _XMAT["zz"]],
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        stand_reward = standing * upright
        small_control = _tolerance(s.data("ctrl"), margin=1, value_at_margin=0, sigmoid="quadratic").mean(dim=1)
        small_control = (4 + small_control) / 5
        if self.move_speed == 0:
            horizontal_velocity = s.data("sensordata")[:, self.com_velocity][:, [0, 1]]
            dont_move = _tolerance(horizontal_velocity, margin=2).mean(dim=1)
            return small_control * stand_reward * dont_move
        move = _tolerance(
            self._speed(s),
            bounds=(self.move_speed, float("inf")),
            margin=self.move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return small_control * stand_reward * move

    def info(self, s):
        return _state_info(
            s,
            speed=self._speed(s),
            torso_upright=s.data("xmat")[:, self.torso, _XMAT["zz"]],
            head_height=s.data("xpos")[:, self.head, 2],
        )


#: Batched task per simulator key.
TASKS: dict[str, type[BatchTask]] = {
    "acrobot": Acrobot,
    "pendulum": Pendulum,
    "cartpole": Cartpole,
    "cartpole-balance": Cartpole,
    "reacher": Reacher,
    "ball_in_cup": BallInCup,
    "finger": Finger,
    "finger-spin": Finger,
    "finger-turn_easy": Finger,
    "manipulator": Manipulator,
    "quadruped": Quadruped,
    "point_mass-easy": PointMass,
    "swimmer-swimmer6": Swimmer,
    "cheetah": Cheetah,
    "hopper": Hopper,
    "walker": Walker,
    "humanoid": Humanoid,
}


def bind_task(key: str, source_env) -> BatchTask:
    """Bind one of the 17 registered simulator keys to its source model and task."""
    try:
        task_type = TASKS[key]
    except KeyError:
        raise ValueError(f"Unsupported DMC MuJoCo Warp task {key!r}") from None
    return task_type(source_env)
