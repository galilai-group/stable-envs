"""Numba port of the PyFlyt 0.29.0 Rocket landing physics and task, in parallel over environments.

One environment is one PyBullet multibody whose links are all fixed, so it moves as one rigid body. The kernel
applies the PyFlyt forces (body drag, four lifting surfaces, gimballed booster with fuel burn) and integrates them
as Bullet's Featherstone solver does for a single body: semi-implicit Euler at 240 Hz, gyroscopic term, world-frame
exponential-map orientation update, velocity clamp of 100 per component. PyBullet rotates link-frame forces with
the link rotation of the previous substep, and reads link velocities through that rotation; the kernel keeps both.
Free flight matches PyBullet to double rounding.

Contacts are approximate. The kernel tests the vertices of the rocket collision shapes against the ground box and
the pad from the pose at the start of the substep, as Bullet does (so contacts show one substep late), with Bullet's
contact threshold of each shape and the 1 mm margin of the convex hulls PyBullet makes of URDF cylinders. A
projected Gauss-Seidel solve with Bullet's multibody contact rows follows (speculative rows for separated points,
``erp2`` push-out for penetrating ones, plane-space friction, no restitution, no warm start). The task logic is
``RocketLandingEnv.compute_term_trunc_reward``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

DT = 1.0 / 240.0
GRAVITY = 9.81
SUBSTEPS_PER_AVIARY_STEP = 2
AVIARY_STEPS_PER_AGENT_STEP = 3
SETTLE_AVIARY_STEPS = 10
#: One thrust noise draw per substep. PyFlyt draws ``np_random.normal(1)``: mean 1, scale 1.
NOISE_PER_STEP = SUBSTEPS_PER_AVIARY_STEP * AVIARY_STEPS_PER_AGENT_STEP
NOISE_PER_SETTLE = SUBSTEPS_PER_AVIARY_STEP * SETTLE_AVIARY_STEPS
OBSERVATION_DIM = 30
START_FUEL = 0.05

FRICTION_PAD = 0.25
FRICTION_GROUND = 0.5
#: PyBullet's solver settings: ``numSolverIterations``, ``erp2`` of contact rows, ``linearSlop``.
SOLVER_ITERATIONS = 50
CONTACT_ERP = 0.08
LINEAR_SLOP = 1e-5
#: Bullet's ``gContactBreakingThreshold``: a pair reports points closer than this factor times the bounding radius of
#: the smaller shape.
BREAKING_THRESHOLD_FACTOR = 0.02
#: ``gUrdfDefaultCollisionMargin`` of the convex hulls PyBullet builds for URDF cylinders.
HULL_MARGIN = 0.001
MAX_CONTACTS = 64
GROUND_HALF_EXTENT = 100.0
PAD_RADIUS = 2.0
PAD_TOP = 0.15
PAD_BOTTOM = 0.05

_BODY_DRAG = 0.5 * 1.225 * np.array([1.16, 1.16, 2.0]) * np.array([1.7649, 1.7649, 0.1075])
#: Origins of the links that PyFlyt's lifting surfaces use (ids 0..3: fuel tank, booster, two fins).
_LINK_R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.35, 0.0, 2.051], [-0.35, 0.0, 2.051]])
_LIFT_U = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
_DRAG_U = np.array([0.0, 0.0, -1.0])
_TORQ_U = np.cross(_LIFT_U, _DRAG_U)

_CLA2D, _CHORD, _SPAN, _FLAP, _ETA = 6.283, 0.5, 0.5, 1.0, 0.65
_ASPECT = _SPAN / _CHORD
_CLA3D = _CLA2D * (_ASPECT / (_ASPECT + ((2.0 * (_ASPECT + 4.0)) / (_ASPECT + 2.0))))
_THETA_F = math.acos(2.0 * _FLAP - 1.0)
_AERO_TAU = 1 - ((_THETA_F - math.sin(_THETA_F)) / math.pi)
_AREA = _CHORD * _SPAN
_STALL_P, _STALL_N, _ALPHA_0 = math.radians(20.0), math.radians(-20.0), 0.0
_CD0, _DEFLECTION = 0.01, 45.0
_FIN_TAU, _GIMBAL_TAU, _THROTTLE_TAU = 0.05, 0.01, 0.01
_GIMBAL_RANGE = math.radians(5.0)
_MIN_THRUST, _MAX_THRUST = 2966.7, 7607.0
_RATIO_MIN = _MIN_THRUST / _MAX_THRUST
_RATIO_THROTTLEABLE = 1.0 - _RATIO_MIN
_FUEL_RATE = 1.451 / 410.9
_NOISE_RATIO = 0.01
_TOTAL_FUEL = 410.9
_PI = 3.1415926535897932384626433832795029


def _hull_vertices():
    """Body-frame contact vertices ``(x, y, z, margin, pad threshold, ground threshold)`` of the rocket shapes.

    Three leg boxes, four fin boxes, and the 32-sided hulls of the booster and body cylinders (vertex ``i`` at
    angle ``2 pi i / 32`` from +y, as PyBullet's URDF importer builds them). Against the pad (a hull), Bullet's
    convex pair reports points closer than ``BREAKING_THRESHOLD_FACTOR`` times the half diagonal of the rocket
    shape's bounding box; the pad and the ground are larger, so the rocket shape sets it. The ground is a box, and
    Bullet's box-box detector reports only penetrating points, so a box vertex touches the ground below zero.
    """
    v = []

    def box(center, yaw, half):
        c, s = math.cos(yaw), math.sin(yaw)
        threshold = BREAKING_THRESHOLD_FACTOR * math.sqrt(sum(h * h for h in half))
        for sx in (-half[0], half[0]):
            for sy in (-half[1], half[1]):
                for sz in (-half[2], half[2]):
                    v.append(
                        (center[0] + c * sx - s * sy, center[1] + s * sx + c * sy, center[2] + sz, 0.0, threshold, 0.0)
                    )

    for yaw, x, y in [(0.0, 0.0, 0.35), (4.188, 0.3031, -0.175), (-4.188, -0.3031, -0.175)]:
        box((x, y, -2.4), yaw, (0.025, 0.25, 0.025))
    for x, y, hx, hy in [(0.35, 0.0, 0.15, 0.015), (-0.35, 0.0, 0.15, 0.015), (0.0, 0.35, 0.015, 0.15),
                         (0.0, -0.35, 0.015, 0.15)]:  # fmt: skip
        box((x, y, 2.051), 0.0, (hx, hy, 0.15))
    for cz, r, hl in [(-2.0, 0.25, 0.25), (0.0, 0.185, 2.385)]:
        rm, hm = r + HULL_MARGIN, hl + HULL_MARGIN
        threshold = BREAKING_THRESHOLD_FACTOR * math.sqrt(2 * rm * rm + hm * hm)
        for k in range(32):
            a = 2 * math.pi * k / 32
            for sz in (-hl, hl):
                v.append((r * math.sin(a), r * math.cos(a), cz + sz, HULL_MARGIN, threshold, threshold))
    return np.array(v)


HULL_VERTICES = _hull_vertices()

_F8, _I8, _B1 = np.dtype(np.float64), np.dtype(np.int64), np.dtype(np.bool_)

#: Per-environment arrays (shape without the batch dimension, dtype). Names after ``task/`` follow the
#: ``RocketLandingEnv`` attributes; ``ignition_state`` and ``throttle_state`` are its auxiliary values
#: (booster ignition and, as in PyFlyt, the fuel ratio).
FIELDS: dict[str, tuple[tuple[int, ...], np.dtype]] = {
    # rigid body, world frame; quaternion is (x, y, z, w)
    "position": ((3,), _F8),
    "quaternion": ((4,), _F8),
    "linear_velocity": ((3,), _F8),
    "angular_velocity": ((3,), _F8),
    # link rotation PyBullet caches from the previous substep (row-major 3x3)
    "link_rotation": ((9,), _F8),
    # body-frame velocities of links 0..3 as PyFlyt reads them after the last substep
    "link_velocity": ((4, 3), _F8),
    # actuators
    "fin": ((4,), _F8),
    "gimbal": ((2,), _F8),
    "throttle": ((), _F8),
    "fuel": ((), _F8),
    "booster_ignition": ((), _F8),
    "command": ((8,), _F8),
    "action": ((7,), _F8),
    "physics_steps": ((), _I8),
    # task; ``state`` is the observation vector the source builds in ``compute_state``
    "state": ((OBSERVATION_DIM,), _F8),
    "ang_vel": ((3,), _F8),
    "ang_pos": ((3,), _F8),
    "lin_vel": ((3,), _F8),
    "lin_pos": ((3,), _F8),
    "ground_lin_vel": ((3,), _F8),
    "previous_ang_vel": ((3,), _F8),
    "previous_lin_vel": ((3,), _F8),
    "previous_lin_pos": ((3,), _F8),
    "previous_ground_lin_vel": ((3,), _F8),
    "ignition_state": ((), _F8),
    "throttle_state": ((), _F8),
    "previous_ignition_state": ((), _F8),
    "previous_throttle_state": ((), _F8),
    "previous_render_ignition_state": ((), _F8),
    "flame_ignited": ((), _B1),
    "landing_pad_contact": ((), _F8),
    "reward": ((), _F8),
    "step_count": ((), _I8),
    "termination": ((), _B1),
    "truncation": ((), _B1),
    "out_of_bounds": ((), _B1),
    "fatal_collision": ((), _B1),
    "env_complete": ((), _B1),
    "start_pos": ((3,), _F8),
    "start_orn": ((3,), _F8),
}
_ORDER = tuple(FIELDS)


def new_state(num_envs: int) -> dict[str, np.ndarray]:
    """Zeroed ``FIELDS`` arrays of shape ``(num_envs, *shape)``; ``start_state`` fills them."""
    return {k: np.zeros((num_envs, *shape), dtype) for k, (shape, dtype) in FIELDS.items()}


def _args(state):
    return tuple(state[k] for k in _ORDER)


# --------------------------------------------------------------------------------------------- small math


@njit(cache=True, inline="always")
def _rotmat(qx, qy, qz, qw):
    return (
        1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
        2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
        2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy),
    )  # fmt: skip


@njit(cache=True, inline="always")
def _matrix_from_quaternion(x, y, z, w):
    """``pybullet.getMatrixFromQuaternion``."""
    s = 2.0 / (x * x + y * y + z * z + w * w)
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    return (1.0 - (yy + zz), xy - wz, xz + wy, xy + wz, 1.0 - (xx + zz), yz - wx, xz - wy, yz + wx, 1.0 - (xx + yy))


@njit(cache=True, inline="always")
def _euler_from_quaternion(x, y, z, w):
    """``pybullet.getEulerFromQuaternion``."""
    sqx, sqy, sqz, squ = x * x, y * y, z * z, w * w
    sarg = -2 * (x * z - w * y)
    if sarg <= -0.99999:
        return 0.0, -0.5 * _PI, 2 * math.atan2(x, -y)
    if sarg >= 0.99999:
        return 0.0, 0.5 * _PI, 2 * math.atan2(-x, y)
    return (
        math.atan2(2 * (y * z + w * x), squ - sqx - sqy + sqz),
        math.asin(sarg),
        math.atan2(2 * (x * y + w * z), squ + sqx - sqy - sqz),
    )


@njit(cache=True, inline="always")
def _quaternion_from_euler(roll, pitch, yaw):
    """``pybullet.getQuaternionFromEuler``."""
    phi, the, psi = roll / 2.0, pitch / 2.0, yaw / 2.0
    x = math.sin(phi) * math.cos(the) * math.cos(psi) - math.cos(phi) * math.sin(the) * math.sin(psi)
    y = math.cos(phi) * math.sin(the) * math.cos(psi) + math.sin(phi) * math.cos(the) * math.sin(psi)
    z = math.cos(phi) * math.cos(the) * math.sin(psi) - math.sin(phi) * math.sin(the) * math.cos(psi)
    w = math.cos(phi) * math.cos(the) * math.cos(psi) + math.sin(phi) * math.sin(the) * math.sin(psi)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return x / n, y / n, z / n, w / n


@njit(cache=True, inline="always")
def _mv(R, x, y, z):
    return (R[0] * x + R[1] * y + R[2] * z, R[3] * x + R[4] * y + R[5] * z, R[6] * x + R[7] * y + R[8] * z)


@njit(cache=True, inline="always")
def _mtv(R, x, y, z):
    return (R[0] * x + R[3] * y + R[6] * z, R[1] * x + R[4] * y + R[7] * z, R[2] * x + R[5] * y + R[8] * z)


@njit(cache=True, inline="always")
def _cross(ax, ay, az, bx, by, bz):
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


@njit(cache=True, inline="always")
def _clamp100(x):
    return min(max(x, -100.0), 100.0)


@njit(cache=True, inline="always")
def _norm2(x, y):
    return math.sqrt(x * x + y * y)


@njit(cache=True, inline="always")
def _norm3(x, y, z):
    return math.sqrt(x * x + y * y + z * z)


@njit(cache=True)
def _aero(alpha, act):
    """Lift, drag and moment coefficients of PyFlyt ``LiftingSurface._compute_aero_data``."""
    defl = math.radians(act * _DEFLECTION)
    dcl = _CLA3D * _AERO_TAU * _ETA * defl
    dclm = _FLAP * dcl
    clmp = _CLA3D * (_STALL_P - _ALPHA_0) + dclm
    clmn = _CLA3D * (_STALL_N - _ALPHA_0) + dclm
    a0 = _ALPHA_0 - (dcl / _CLA3D)
    asp = a0 + (clmp / _CLA3D)
    asn = a0 + (clmn / _CLA3D)
    if asn < alpha and alpha < asp:
        cl = _CLA3D * (alpha - a0)
        ai = cl / (math.pi * _ASPECT)
        ae = alpha - a0 - ai
        ct = _CD0 * math.cos(ae)
        cn = (cl + (ct * math.sin(ae))) / math.cos(ae)
        cd = (cn * math.sin(ae)) + (ct * math.cos(ae))
        cm = -cn * (0.25 - (0.175 * (1.0 - ((2.0 * ae) / math.pi))))
        return cl, cd, cm
    if alpha > 0.0:
        cls = _CLA3D * (asp - a0)
        ais = cls / (math.pi * _ASPECT)
        x0, x1 = asp, math.pi / 2.0
        if alpha <= x0:
            ai = ais
        elif alpha >= x1:
            ai = 0.0
        else:
            ai = ais + (alpha - x0) * (0.0 - ais) / (x1 - x0)
    else:
        cls = _CLA3D * (asn - a0)
        ais = cls / (math.pi * _ASPECT)
        x0, x1 = -math.pi / 2.0, asn
        if alpha <= x0:
            ai = 0.0
        elif alpha >= x1:
            ai = ais
        else:
            ai = 0.0 + (alpha - x0) * (ais - 0.0) / (x1 - x0)
    ae = alpha - a0 - ai
    cd90 = ((-4.26 * (10**-2)) * (defl**2)) + ((2.1 * (10**-1)) * defl) + 1.98
    cn = cd90 * math.sin(ae) * (1.0 / (0.56 + 0.44 * abs(math.sin(ae))) - 0.41 * (1.0 - math.exp(-17.0 / _ASPECT)))
    ct = 0.5 * _CD0 * math.cos(ae)
    cl = (cn * math.cos(ae)) - (ct * math.sin(ae))
    cd = (cn * math.sin(ae)) + (ct * math.cos(ae))
    cm = -cn * (0.25 - (0.175 * (1.0 - ((2.0 * abs(ae)) / math.pi))))
    return cl, cd, cm


# --------------------------------------------------------------------------------------------- one substep


@njit(cache=True)
def _substep(b, z, pos, quat, vel, avel, lrot, lvel, fin, gim, thr_, fuel_, ign_, cmd, verts, crel):
    """One 240 Hz substep of environment ``b`` with standard-normal thrust noise ``z``.

    Return ``(pad contact, ground contact)`` as Bullet reports them after this substep.
    """
    qx, qy, qz, qw = quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3]
    R = _rotmat(qx, qy, qz, qw)
    R0 = R
    # PyBullet rotates link-frame forces with the cached link rotation, which trails the base by one substep.
    RC = (lrot[b, 0], lrot[b, 1], lrot[b, 2], lrot[b, 3], lrot[b, 4], lrot[b, 5], lrot[b, 6], lrot[b, 7], lrot[b, 8])
    Q = (
        R[0] * RC[0] + R[3] * RC[3] + R[6] * RC[6], R[0] * RC[1] + R[3] * RC[4] + R[6] * RC[7],
        R[0] * RC[2] + R[3] * RC[5] + R[6] * RC[8],
        R[1] * RC[0] + R[4] * RC[3] + R[7] * RC[6], R[1] * RC[1] + R[4] * RC[4] + R[7] * RC[7],
        R[1] * RC[2] + R[4] * RC[5] + R[7] * RC[8],
        R[2] * RC[0] + R[5] * RC[3] + R[8] * RC[6], R[2] * RC[1] + R[5] * RC[4] + R[8] * RC[7],
        R[2] * RC[2] + R[5] * RC[5] + R[8] * RC[8],
    )  # fmt: skip
    # Bullet finds contacts at the start of a step, from the pose before the step.
    px0, py0, pz0 = pos[b, 0], pos[b, 1], pos[b, 2]
    nc = 0
    pad_hit = False
    ground_hit = False
    if pz0 < 3.5:
        pad_top = PAD_TOP + HULL_MARGIN
        for i in range(verts.shape[0]):
            rx, ry, rz = _mv(R, verts[i, 0], verts[i, 1], verts[i, 2])
            x = rx + px0
            y = ry + py0
            low = rz + pz0 - verts[i, 3]
            threshold = verts[i, 4]
            gap = 0.0
            mu = -1.0
            if low < verts[i, 5] and abs(x) <= GROUND_HALF_EXTENT and abs(y) <= GROUND_HALF_EXTENT:
                ground_hit = True
                gap = low
                mu = FRICTION_GROUND
            reach = PAD_RADIUS + HULL_MARGIN + verts[i, 3]
            if x * x + y * y <= reach * reach and low - pad_top < threshold and low > PAD_BOTTOM - HULL_MARGIN:
                pad_hit = True
                gap = low - pad_top
                mu = FRICTION_PAD
            if mu >= 0.0 and nc < MAX_CONTACTS:
                crel[b, nc, 0], crel[b, nc, 1], crel[b, nc, 2] = rx, ry, rz
                gap += LINEAR_SLOP
                # target normal speed of the row: speculative when apart, erp2 push-out when penetrating
                crel[b, nc, 3] = -gap / DT if gap > 0.0 else -gap * CONTACT_ERP / DT
                crel[b, nc, 4] = mu
                nc += 1
    Fx = Fy = Fz = 0.0
    Tx = Ty = Tz = 0.0  # torque about the base origin, body frame
    # body drag on link 0
    vx, vy, vz = lvel[b, 0, 0], lvel[b, 0, 1], lvel[b, 0, 2]
    dx_, dy_, dz_ = _mv(
        Q,
        -np.sign(vx) * _BODY_DRAG[0] * vx * vx,
        -np.sign(vy) * _BODY_DRAG[1] * vy * vy,
        -np.sign(vz) * _BODY_DRAG[2] * vz * vz,
    )
    Fx += dx_
    Fy += dy_
    Fz += dz_
    # lifting surfaces on links 0..3
    for k in range(4):
        fin[b, k] += (DT / _FIN_TAU) * (cmd[b, k] - fin[b, k])
        act = fin[b, k]
        lx, ly, lz = lvel[b, k, 0], lvel[b, k, 1], lvel[b, k, 2]
        speed = math.sqrt(lx * lx + ly * ly + lz * lz)
        lifting = lx * _LIFT_U[k, 0] + ly * _LIFT_U[k, 1] + lz * _LIFT_U[k, 2]
        forward = lx * _DRAG_U[0] + ly * _DRAG_U[1] + lz * _DRAG_U[2]
        alpha = math.atan2(-lifting, forward)
        cl, cd, cm = _aero(alpha, act)
        qa = 0.5 * 1.225 * speed * speed * _AREA
        lift = cl * qa
        drag = cd * qa
        fn = (lift * math.cos(alpha)) + (drag * math.sin(alpha))
        fp = (lift * math.sin(alpha)) - (drag * math.cos(alpha))
        fx = _LIFT_U[k, 0] * fn + _DRAG_U[0] * fp
        fy = _LIFT_U[k, 1] * fn + _DRAG_U[1] * fp
        fz = _LIFT_U[k, 2] * fn + _DRAG_U[2] * fp
        tq = qa * cm * _CHORD
        tx, ty, tz = _mv(Q, tq * _TORQ_U[k, 0], tq * _TORQ_U[k, 1], tq * _TORQ_U[k, 2])
        fx, fy, fz = _mv(Q, fx, fy, fz)
        Fx += fx
        Fy += fy
        Fz += fz
        cx_, cy_, cz_ = _cross(_LINK_R[k, 0], _LINK_R[k, 1], _LINK_R[k, 2], fx, fy, fz)
        Tx += cx_ + tx
        Ty += cy_ + ty
        Tz += cz_ + tz
    # gimbal
    for k in range(2):
        gim[b, k] += (DT / _GIMBAL_TAU) * (cmd[b, 6 + k] - gim[b, k])
    a1 = gim[b, 0] * _GIMBAL_RANGE
    a2 = gim[b, 1] * _GIMBAL_RANGE
    s1, h1 = math.sin(a1), 2 * math.sin(a1 / 2.0) ** 2
    s2, h2 = math.sin(a2), 2 * math.sin(a2 / 2.0) ** 2
    ux, uy, uz = s2, -s1 * (1 - h2), (1 - h1) * (1 - h2)
    # booster (reignitable)
    ign = 1.0 if cmd[b, 4] > 0.5 else 0.0
    ign_[b] = ign
    target = ign * (cmd[b, 5] * _RATIO_THROTTLEABLE + _RATIO_MIN)
    thr = thr_[b]
    thr += (DT / _THROTTLE_TAU) * (target - thr)
    thr += (1.0 + z) * thr * _NOISE_RATIO
    fuel = fuel_[b]
    if not (fuel > 0.0):
        thr = 0.0
    fuel -= thr * _FUEL_RATE * DT
    fuel = min(max(fuel, 0.0), 1.0)
    thr_[b] = thr
    fuel_[b] = fuel
    th = thr * _MAX_THRUST
    fx, fy, fz = _mv(Q, th * ux, th * uy, th * uz)
    Fx += fx
    Fy += fy
    Fz += fz
    cx_, cy_, cz_ = _cross(0.0, 0.0, -2.0, fx, fy, fz)
    Tx += cx_
    Ty += cy_
    Tz += cz_
    # composite mass properties (base 91 kg, fuel tank, booster 47 kg at z=-2, four 0.05 kg fins)
    mt = fuel * _TOTAL_FUEL
    M = 91.0 + mt + 47.0 + 4 * 0.05
    cz = (47.0 * (-2.0) + 0.05 * (2.051 * 4)) / M
    Ixx = 372.6 + 91.0 * cz * cz + fuel * 1678.0 + mt * cz * cz
    Iyy = Ixx
    Izz = 1.55 + fuel * 7.01
    dz = -2.0 - cz
    Ixx += 192.43 + 47.0 * dz * dz
    Iyy += 192.43 + 47.0 * dz * dz
    Izz += 0.81
    dz = 2.051 - cz
    Ixx += 2 * 0.05 * (0.0 + dz * dz) + 2 * 0.05 * (0.35 * 0.35 + dz * dz)
    Iyy += 2 * 0.05 * (0.35 * 0.35 + dz * dz) + 2 * 0.05 * (0.0 + dz * dz)
    Izz += 4 * 0.05 * 0.35 * 0.35
    # dynamics about the centre of mass
    tcx, tcy, tcz = Tx + cz * Fy, Ty - cz * Fx, Tz
    wwx, wwy, wwz = avel[b, 0], avel[b, 1], avel[b, 2]
    wbx, wby, wbz = _mtv(R, wwx, wwy, wwz)
    gx, gy, gz = _cross(wbx, wby, wbz, Ixx * wbx, Iyy * wby, Izz * wbz)
    abx, aby, abz = (tcx - gx) / Ixx, (tcy - gy) / Iyy, (tcz - gz) / Izz
    awx, awy, awz = _mv(R, abx, aby, abz)
    fwx, fwy, fwz = _mv(R, Fx, Fy, Fz)
    acx, acy, acz = fwx / M, fwy / M, fwz / M - GRAVITY
    cwx, cwy, cwz = _mv(R, 0.0, 0.0, cz)
    t1 = _cross(awx, awy, awz, cwx, cwy, cwz)
    t2 = _cross(wwx, wwy, wwz, cwx, cwy, cwz)
    t3 = _cross(wwx, wwy, wwz, t2[0], t2[1], t2[2])
    aox, aoy, aoz = acx - t1[0] - t3[0], acy - t1[1] - t3[1], acz - t1[2] - t3[2]
    vx = _clamp100(vel[b, 0] + DT * aox)
    vy = _clamp100(vel[b, 1] + DT * aoy)
    vz = _clamp100(vel[b, 2] + DT * aoz)
    wwx = _clamp100(wwx + DT * awx)
    wwy = _clamp100(wwy + DT * awy)
    wwz = _clamp100(wwz + DT * awz)
    if nc > 0:
        # projected Gauss-Seidel: every normal row, then every friction row (limit from that contact's normal
        # impulse), as btMultiBodyConstraintSolver orders them; impulses accumulate in crel[b, c, 5:8]
        for c in range(nc):
            crel[b, c, 5] = 0.0
            crel[b, c, 6] = 0.0
            crel[b, c, 7] = 0.0
        for _ in range(SOLVER_ITERATIONS):
            for kind in range(3):
                if kind == 0:
                    nx, ny, nz = 0.0, 0.0, 1.0
                elif kind == 1:
                    nx, ny, nz = 0.0, -1.0, 0.0
                else:
                    nx, ny, nz = 1.0, 0.0, 0.0
                for c in range(nc):
                    rx, ry, rz = crel[b, c, 0], crel[b, c, 1], crel[b, c, 2]
                    cx, cy, cz2 = rx - cwx, ry - cwy, rz - cwz
                    pv = _cross(wwx, wwy, wwz, rx, ry, rz)
                    vn = (vx + pv[0]) * nx + (vy + pv[1]) * ny + (vz + pv[2]) * nz
                    rn = _cross(cx, cy, cz2, nx, ny, nz)
                    ib = _mtv(R, rn[0], rn[1], rn[2])
                    ia = _mv(R, ib[0] / Ixx, ib[1] / Iyy, ib[2] / Izz)
                    kk = _cross(ia[0], ia[1], ia[2], cx, cy, cz2)
                    keff = 1.0 / M + kk[0] * nx + kk[1] * ny + kk[2] * nz
                    old = crel[b, c, 5 + kind]
                    if kind == 0:
                        new = max(old + (crel[b, c, 3] - vn) / keff, 0.0)
                    else:
                        lim = crel[b, c, 4] * crel[b, c, 5]
                        new = min(max(old - vn / keff, -lim), lim)
                    dl = new - old
                    crel[b, c, 5 + kind] = new
                    jx, jy, jz = dl * nx, dl * ny, dl * nz
                    dwx, dwy, dwz = ia[0] * dl, ia[1] * dl, ia[2] * dl
                    dv = _cross(dwx, dwy, dwz, cwx, cwy, cwz)
                    vx += jx / M - dv[0]
                    vy += jy / M - dv[1]
                    vz += jz / M - dv[2]
                    wwx += dwx
                    wwy += dwy
                    wwz += dwz
    vel[b, 0], vel[b, 1], vel[b, 2] = vx, vy, vz
    avel[b, 0], avel[b, 1], avel[b, 2] = wwx, wwy, wwz
    pos[b, 0] = px0 + DT * vx
    pos[b, 1] = py0 + DT * vy
    pos[b, 2] = pz0 + DT * vz
    fa = math.sqrt(wwx * wwx + wwy * wwy + wwz * wwz)
    if fa < 0.001:
        sc = 0.5 * DT - (DT * DT * DT) * 0.020833333333 * fa * fa
    else:
        sc = math.sin(0.5 * fa * DT) / fa
    ax_, ay_, az_ = wwx * sc, wwy * sc, wwz * sc
    cw_ = math.cos(fa * DT * 0.5)
    nw = cw_ * qw - ax_ * qx - ay_ * qy - az_ * qz
    nx = cw_ * qx + ax_ * qw + ay_ * qz - az_ * qy
    ny = cw_ * qy - ax_ * qz + ay_ * qw + az_ * qx
    nz = cw_ * qz + ax_ * qy - ay_ * qx + az_ * qw
    nn = math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
    qx, qy, qz, qw = nx / nn, ny / nn, nz / nn, nw / nn
    quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3] = qx, qy, qz, qw
    # body-frame link velocities, read through the link rotation of the previous substep
    R = _rotmat(qx, qy, qz, qw)
    for k in range(4):
        rwx, rwy, rwz = _mv(R, _LINK_R[k, 0], _LINK_R[k, 1], _LINK_R[k, 2])
        c3 = _cross(wwx, wwy, wwz, rwx, rwy, rwz)
        lx, ly, lz = _mtv(R, vx + c3[0], vy + c3[1], vz + c3[2])
        wx_, wy_, wz_ = _mv(R0, lx, ly, lz)
        lx, ly, lz = _mtv(R, wx_, wy_, wz_)
        lvel[b, k, 0], lvel[b, k, 1], lvel[b, k, 2] = lx, ly, lz
    for i in range(9):
        lrot[b, i] = R0[i]
    return pad_hit, ground_hit


# --------------------------------------------------------------------------------------------- task


@njit(cache=True)
def _compute_state(
    b, pos, quat, vel, avel, booster_ignition, fuel, thr, fin, gim, action, pad_contact, obs, ang_vel, ang_pos,
    lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel, ignition_state,
    throttle_state, p_ignition_state, p_throttle_state,
):  # fmt: skip
    """``RocketLandingEnv.compute_state`` (quaternion representation)."""
    for i in range(3):
        p_ang_vel[b, i] = ang_vel[b, i]
        p_lin_vel[b, i] = lin_vel[b, i]
        p_lin_pos[b, i] = lin_pos[b, i]
        p_ground_lin_vel[b, i] = ground_lin_vel[b, i]
    x, y, z, w = quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3]
    R = _matrix_from_quaternion(x, y, z, w)
    ang_vel[b, 0], ang_vel[b, 1], ang_vel[b, 2] = _mtv(R, avel[b, 0], avel[b, 1], avel[b, 2])
    lvx, lvy, lvz = _mtv(R, vel[b, 0], vel[b, 1], vel[b, 2])
    lin_vel[b, 0], lin_vel[b, 1], lin_vel[b, 2] = lvx, lvy, lvz
    roll, pitch, yaw = _euler_from_quaternion(x, y, z, w)
    ang_pos[b, 0], ang_pos[b, 1], ang_pos[b, 2] = roll, pitch, yaw
    for i in range(3):
        lin_pos[b, i] = pos[b, i]
    ex, ey, ez, ew = _quaternion_from_euler(roll, pitch, yaw)
    R2 = _matrix_from_quaternion(ex, ey, ez, ew)
    ground_lin_vel[b, 0], ground_lin_vel[b, 1], ground_lin_vel[b, 2] = _mv(R2, lvx, lvy, lvz)
    p_throttle_state[b] = throttle_state[b]
    p_ignition_state[b] = ignition_state[b]
    throttle_state[b] = fuel[b]
    ignition_state[b] = booster_ignition[b]
    for i in range(3):
        obs[b, i] = ang_vel[b, i]
        obs[b, 7 + i] = lin_vel[b, i]
        obs[b, 10 + i] = lin_pos[b, i]
    obs[b, 3], obs[b, 4], obs[b, 5], obs[b, 6] = ex, ey, ez, ew
    for i in range(7):
        obs[b, 13 + i] = action[b, i]
    for i in range(4):
        obs[b, 20 + i] = fin[b, i]
    obs[b, 24] = booster_ignition[b]
    obs[b, 25] = fuel[b]
    obs[b, 26] = thr[b]
    obs[b, 27] = gim[b, 0]
    obs[b, 28] = gim[b, 1]
    obs[b, 29] = pad_contact[b]


@njit(cache=True)
def _aviary_step(
    b,
    noise,
    k0,
    pos,
    quat,
    vel,
    avel,
    lrot,
    lvel,
    fin,
    gim,
    thr,
    fuel,
    ign,
    cmd,
    action,
    physics_steps,
    verts,
    crel,
):
    """``Aviary.step``: two substeps; return the pad and ground contact flags of the step."""
    pad = False
    ground = False
    for s in range(SUBSTEPS_PER_AVIARY_STEP):
        if physics_steps[b] % 2 == 0:
            sp0, sp1, sp2 = action[b, 0], action[b, 1], action[b, 2]
            cmd[b, 0] = min(max(sp1 + sp2, -1.0), 1.0)
            cmd[b, 1] = min(max(sp1 - sp2, -1.0), 1.0)
            cmd[b, 2] = min(max(sp0 - sp2, -1.0), 1.0)
            cmd[b, 3] = min(max(sp0 + sp2, -1.0), 1.0)
            for k in range(4):
                cmd[b, 4 + k] = action[b, 3 + k]
        p, g = _substep(
            b, noise[b, k0 + s], pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, verts, crel
        )
        pad |= p
        ground |= g
        physics_steps[b] += 1
    return pad, ground


@njit(cache=True, parallel=True)
def _step_kernel(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    actions, noise, active, noise_used, verts, crel, ceiling, max_displacement, max_steps, sparse_reward,
):  # fmt: skip
    for b in prange(pos.shape[0]):
        if not active[b]:
            continue
        for i in range(7):
            action[b, i] = actions[b, i]
        r = 0.0
        used = 0
        for a in range(AVIARY_STEPS_PER_AGENT_STEP):
            if termination[b] or truncation[b]:
                break
            pad, ground = _aviary_step(
                b,
                noise,
                used,
                pos,
                quat,
                vel,
                avel,
                lrot,
                lvel,
                fin,
                gim,
                thr,
                fuel,
                ign,
                cmd,
                action,
                physics_steps,
                verts,
                crel,
            )
            used += SUBSTEPS_PER_AVIARY_STEP
            _compute_state(
                b,
                pos,
                quat,
                vel,
                avel,
                ign,
                fuel,
                thr,
                fin,
                gim,
                action,
                pad_contact,
                obs,
                ang_vel,
                ang_pos,
                lin_vel,
                lin_pos,
                ground_lin_vel,
                p_ang_vel,
                p_lin_vel,
                p_lin_pos,
                p_ground_lin_vel,
                ignition_state,
                throttle_state,
                p_ignition_state,
                p_throttle_state,
            )
            # compute_base_term_trunc_reward
            if step_count[b] > max_steps:
                truncation[b] = True
            if ground or lin_pos[b, 2] < 0.0:
                fatal_collision[b] = True
                termination[b] = True
            if _norm2(lin_pos[b, 0], lin_pos[b, 1]) > max_displacement or lin_pos[b, 2] > ceiling:
                out_of_bounds[b] = True
                termination[b] = True
            if not sparse_reward:
                lateral_progress = _norm2(p_lin_pos[b, 0], p_lin_pos[b, 1]) - _norm2(lin_pos[b, 0], lin_pos[b, 1])
                vertical_progress = p_lin_pos[b, 2] - lin_pos[b, 2]
                lateral_distance = _norm2(lin_pos[b, 0], lin_pos[b, 1]) + 0.1
                deceleration_progress = (
                    (ground_lin_vel[b, 2] - p_ground_lin_vel[b, 2] + 1.0)
                    / math.exp(lin_pos[b, 2])
                    * (1.0 if ground_lin_vel[b, 2] < 0.0 else -1.0)
                )
                r += (
                    -0.3
                    + (0.3 / lateral_distance)
                    + (10.0 * lateral_progress)
                    + (0.2 * vertical_progress)
                    + (4.0 * deceleration_progress)
                    - (1.0 * abs(ang_vel[b, 2]))
                    - (1.0 * _norm2(ang_pos[b, 0], ang_pos[b, 1]))
                )
            if not pad:
                pad_contact[b] = 0.0
                continue
            pad_contact[b] = 1.0
            r += 5.0 - (0.3 * abs(ground_lin_vel[b, 2]))
            pav = _norm3(p_ang_vel[b, 0], p_ang_vel[b, 1], p_ang_vel[b, 2])
            plv = _norm3(p_lin_vel[b, 0], p_lin_vel[b, 1], p_lin_vel[b, 2])
            if pav > 10.0 or plv > 5.0:
                termination[b] = True
                fatal_collision[b] = True
                continue
            if (
                pav < 5.0
                and plv < 4.0
                and _norm2(ang_pos[b, 0], ang_pos[b, 1]) < 0.6
                and _norm2(lin_pos[b, 0], lin_pos[b, 1]) < 12.0
                and lin_pos[b, 2] < 3.0
            ):
                truncation[b] = True
                env_complete[b] = True
                r += 3.0
        step_count[b] += 1
        reward[b] = r
        noise_used[b] = used
        if ignition_state[b] != p_render_ignition[b]:
            p_render_ignition[b] = ignition_state[b]
            flame_ignited[b] = True


@njit(cache=True, parallel=True)
def _settle_kernel(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    noise, mask, verts, crel,
):  # fmt: skip
    for b in prange(pos.shape[0]):
        if not mask[b]:
            continue
        for a in range(SETTLE_AVIARY_STEPS):
            _aviary_step(
                b,
                noise,
                a * SUBSTEPS_PER_AVIARY_STEP,
                pos,
                quat,
                vel,
                avel,
                lrot,
                lvel,
                fin,
                gim,
                thr,
                fuel,
                ign,
                cmd,
                action,
                physics_steps,
                verts,
                crel,
            )
        _compute_state(
            b,
            pos,
            quat,
            vel,
            avel,
            ign,
            fuel,
            thr,
            fin,
            gim,
            action,
            pad_contact,
            obs,
            ang_vel,
            ang_pos,
            lin_vel,
            lin_pos,
            ground_lin_vel,
            p_ang_vel,
            p_lin_vel,
            p_lin_pos,
            p_ground_lin_vel,
            ignition_state,
            throttle_state,
            p_ignition_state,
            p_throttle_state,
        )


@njit(cache=True, parallel=True)
def _start_kernel(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    mask, s_start_pos, s_start_orn, s_position, s_linvel, s_angvel, s_fuel,
):  # fmt: skip
    for b in prange(pos.shape[0]):
        if not mask[b]:
            continue
        x, y, z, w = _quaternion_from_euler(s_start_orn[b, 0], s_start_orn[b, 1], s_start_orn[b, 2])
        quat[b, 0], quat[b, 1], quat[b, 2], quat[b, 3] = x, y, z, w
        R = _rotmat(x, y, z, w)
        for i in range(9):
            lrot[b, i] = R[i]
        for i in range(3):
            start_pos[b, i] = s_start_pos[b, i]
            start_orn[b, i] = s_start_orn[b, i]
            # the drone state PyFlyt reads in the first compute_state: created pose, before the velocity reset
            pos[b, i] = s_start_pos[b, i]
            vel[b, i] = 0.0
            avel[b, i] = 0.0
            ang_vel[b, i] = 0.0
            ang_pos[b, i] = 0.0
            lin_vel[b, i] = 0.0
            lin_pos[b, i] = 0.0
            ground_lin_vel[b, i] = 0.0
        for k in range(4):
            fin[b, k] = 0.0
            for i in range(3):
                lvel[b, k, i] = 0.0
        gim[b, 0] = 0.0
        gim[b, 1] = 0.0
        for i in range(8):
            cmd[b, i] = 0.0
        for i in range(7):
            action[b, i] = 0.0
        thr[b] = 0.0
        fuel[b] = s_fuel[b]
        ign[b] = 0.0
        physics_steps[b] = 0
        ignition_state[b] = 0.0
        throttle_state[b] = 0.0
        p_render_ignition[b] = 0.0
        flame_ignited[b] = False
        pad_contact[b] = 0.0
        reward[b] = 0.0
        step_count[b] = 0
        termination[b] = False
        truncation[b] = False
        out_of_bounds[b] = False
        fatal_collision[b] = False
        env_complete[b] = False
        _compute_state(
            b,
            pos,
            quat,
            vel,
            avel,
            ign,
            fuel,
            thr,
            fin,
            gim,
            action,
            pad_contact,
            obs,
            ang_vel,
            ang_pos,
            lin_vel,
            lin_pos,
            ground_lin_vel,
            p_ang_vel,
            p_lin_vel,
            p_lin_pos,
            p_ground_lin_vel,
            ignition_state,
            throttle_state,
            p_ignition_state,
            p_throttle_state,
        )
        for i in range(3):
            pos[b, i] = s_position[b, i]
            vel[b, i] = s_linvel[b, i]
            avel[b, i] = s_angvel[b, i]


# --------------------------------------------------------------------------------------------- public API


def _scratch(num_envs):
    return np.empty((num_envs, MAX_CONTACTS, 8))


def step(
    state: dict[str, np.ndarray],
    actions: np.ndarray,
    noise: np.ndarray,
    active: np.ndarray,
    *,
    ceiling: float = 500.0,
    max_displacement: float = 200.0,
    max_steps: int = 1200,
    sparse_reward: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """One agent step (3 aviary steps, 6 substeps) of the ``active`` environments, in place on ``state``.

    ``actions`` is ``(B, 7)`` float64 and already valid; ``noise`` is ``(B, NOISE_PER_STEP)`` standard-normal draws
    in the order the source draws them. An episode that ends after an aviary step skips the rest of the step, as
    the source does, and uses fewer draws: ``info["noise_used"]`` counts them. An environment that had already
    ended steps nothing and gets reward 0. ``max_steps`` is ``max_duration_seconds * agent_hz``.

    Return ``(reward, terminated, truncated, info)``; ``info`` holds copies of ``out_of_bounds``,
    ``fatal_collision``, ``env_complete``, ``landing_pad_contact`` and ``noise_used``.
    """
    num_envs = state["position"].shape[0]
    actions = np.ascontiguousarray(actions, dtype=np.float64)
    noise = np.ascontiguousarray(noise, dtype=np.float64)
    active = np.ascontiguousarray(active, dtype=np.bool_)
    if actions.shape != (num_envs, 7) or noise.shape != (num_envs, NOISE_PER_STEP) or active.shape != (num_envs,):
        raise ValueError("Rocket step expects actions (B, 7), noise (B, NOISE_PER_STEP) and active (B,)")
    noise_used = np.zeros(num_envs, np.int64)
    _step_kernel(
        *_args(state), actions, noise, active, noise_used, HULL_VERTICES, _scratch(num_envs),
        float(ceiling), float(max_displacement), int(max_steps), bool(sparse_reward),
    )  # fmt: skip
    info = {
        "out_of_bounds": state["out_of_bounds"].copy(),
        "fatal_collision": state["fatal_collision"].copy(),
        "env_complete": state["env_complete"].copy(),
        "landing_pad_contact": state["landing_pad_contact"].copy(),
        "noise_used": noise_used,
    }
    return state["reward"].copy(), state["termination"].copy(), state["truncation"].copy(), info


def start_state(state: dict[str, np.ndarray], mask: np.ndarray, start: dict[str, np.ndarray]) -> None:
    """Write the sampled start of the ``mask`` environments, as ``RocketLandingEnv.reset`` does before it settles.

    ``start`` holds ``(B, ...)`` float64 arrays:

    - ``start_pos`` (3,): the pose PyFlyt creates the rocket at, ``(offset x, offset y, ceiling * height ratio)``;
      the product is the one the source computes (a float32 height ratio gives a float32 product).
    - ``start_orn`` (3,): roll, pitch, yaw of the start tilt.
    - ``position`` (3,): the pose after the task moves the rocket (horizontal offset clipped to the spawn limit).
    - ``linear_velocity``, ``angular_velocity`` (3,): world-frame start velocities (``(0, 0, -30)`` and zero for
      the default ``accelerate_drop``).
    - ``fuel`` (), optional: starting fuel ratio, ``START_FUEL`` by default.

    All other fields go to their reset values, and the task values take the first ``compute_state`` of the
    source: from the created pose with zero velocity. Call ``settle`` next.
    """
    num_envs = state["position"].shape[0]
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    fuel = start.get("fuel")
    fuel = np.full(num_envs, START_FUEL) if fuel is None else np.ascontiguousarray(fuel, dtype=np.float64)
    columns = [
        np.ascontiguousarray(start[k], dtype=np.float64)
        for k in ("start_pos", "start_orn", "position", "linear_velocity", "angular_velocity")
    ]
    _start_kernel(*_args(state), mask, *columns, fuel)


def settle(state: dict[str, np.ndarray], noise_settle: np.ndarray, mask: np.ndarray) -> None:
    """The 10 aviary steps (20 substeps) of ``RocketBaseEnv.end_reset`` with zero setpoint, then ``compute_state``.

    ``noise_settle`` is ``(B, NOISE_PER_SETTLE)`` standard-normal draws.
    """
    num_envs = state["position"].shape[0]
    noise_settle = np.ascontiguousarray(noise_settle, dtype=np.float64)
    if noise_settle.shape != (num_envs, NOISE_PER_SETTLE):
        raise ValueError("Rocket settle expects noise (B, NOISE_PER_SETTLE)")
    mask = np.ascontiguousarray(mask, dtype=np.bool_)
    _settle_kernel(*_args(state), noise_settle, mask, HULL_VERTICES, _scratch(num_envs))


def observation(state: dict[str, np.ndarray]) -> np.ndarray:
    """A copy of the ``RocketLandingEnv`` state vector, ``(B, OBSERVATION_DIM)`` float64.

    ang_vel, quaternion (from the Euler angles, as the source makes it), lin_vel, lin_pos, action, fin actuation,
    booster ignition, fuel, throttle, gimbal, and the landing pad contact of the aviary step before the last.
    """
    return state["state"].copy()
