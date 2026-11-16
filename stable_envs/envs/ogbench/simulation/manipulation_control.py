"""OGBench's Cube/Scene end-effector controller for a batch of lanes, equal to the source bit for bit."""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np

_MINVAL = mujoco.mjMINVAL
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
_INVERT = np.array([1.0, -1.0, -1.0, -1.0])
#: Lanes per IK batch thread. docs/performance-tuning.md (DM Control threads) gives the hosts it was measured on.
IK_LANES_PER_THREAD = 128


def _normalize4(q):
    """``mju_normalize4`` per row, in place."""
    sq = q * q
    norm = np.sqrt(sq[:, 0] + sq[:, 1] + sq[:, 2] + sq[:, 3])
    small = norm < _MINVAL
    with np.errstate(divide="ignore"):
        inverse = 1 / norm
    scale = (np.abs(norm - 1) > _MINVAL) & ~small
    q[scale] *= inverse[scale, None]
    q[small] = _IDENTITY
    return q


# mju_mat2Quat per case (q0, q1, q2 or q3 largest): signs of mat[0], mat[4], mat[8] under the square root, then
# for the other three components their slots and the terms mat[i] + sign * mat[j] of their numerators.
_M2Q_DIAGONAL = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float64)
_M2Q_SLOTS = np.array([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])
_M2Q_I = np.array([[7, 2, 3], [7, 1, 2], [2, 1, 5], [3, 2, 5]])
_M2Q_J = np.array([[5, 6, 1], [5, 3, 6], [6, 3, 7], [1, 6, 7]])
_M2Q_SIGN = np.array([[-1, -1, -1], [-1, 1, 1], [-1, 1, 1], [-1, 1, 1]], np.float64)


def mat2quat(mat):
    """``mju_mat2Quat`` per row of ``mat`` (n, 9)."""
    n = len(mat)
    rows = np.arange(n)
    m0, m4, m8 = mat[:, 0], mat[:, 4], mat[:, 8]
    case = np.full(n, 3)
    case[m4 > m8] = 2
    case[(m0 > m4) & (m0 > m8)] = 1
    case[m0 + m4 + m8 > 0] = 0
    diagonal = mat[:, [0, 4, 8]] * _M2Q_DIAGONAL[case]
    big = 0.5 * np.sqrt(1 + diagonal[:, 0] + diagonal[:, 1] + diagonal[:, 2])
    numerator = mat[rows[:, None], _M2Q_I[case]] + _M2Q_SIGN[case] * mat[rows[:, None], _M2Q_J[case]]
    q = np.empty((n, 4))
    q[rows, case] = big
    q[rows[:, None], _M2Q_SLOTS[case]] = 0.25 * numerator / big[:, None]
    return _normalize4(q)


def _products(index, sign):
    """Sums of signed products ``a[i] * b[j]``, term by term from the left, as written in the source."""
    flat = (4 * index[..., 0] + index[..., 1]).reshape(-1)
    sign = np.asarray(sign, np.float64)

    def apply(a, b):
        terms = (a[:, :, None] * b[:, None, :]).reshape(len(a), 16)[:, flat].reshape(len(a), 4, 4) * sign
        return terms[:, :, 0] + terms[:, :, 1] + terms[:, :, 2] + terms[:, :, 3]

    return apply


# mju_mulQuat: qa[i] * qb[j] terms of each component.
mju_mul_quat = _products(
    np.array(
        [
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [[0, 1], [1, 0], [2, 3], [3, 2]],
            [[0, 2], [1, 3], [2, 0], [3, 1]],
            [[0, 3], [1, 2], [2, 1], [3, 0]],
        ]
    ),
    [[1, -1, -1, -1], [1, 1, 1, -1], [1, -1, 1, 1], [1, 1, -1, 1]],
)
mju_mul_quat.__doc__ = "``mju_mulQuat`` per row."

# lie.SO3.multiply: terms in (w, x, y, z) order of each component.
so3_multiply = _products(
    np.array(
        [
            [[1, 1], [2, 2], [3, 3], [0, 0]],
            [[1, 0], [2, 3], [3, 2], [0, 1]],
            [[1, 3], [2, 0], [3, 1], [0, 2]],
            [[1, 2], [2, 1], [3, 0], [0, 3]],
        ]
    ),
    [[-1, -1, -1, 1], [1, 1, -1, 1], [-1, 1, 1, 1], [1, -1, 1, 1]],
)
so3_multiply.__doc__ = "``lie.SO3.multiply`` per row."


def quat2vel(q):
    """``mju_quat2Vel`` per row with ``dt`` 1."""
    axis = q[:, 1:].copy()
    norm = np.sqrt(axis[:, 0] * axis[:, 0] + axis[:, 1] * axis[:, 1] + axis[:, 2] * axis[:, 2])
    small = norm < _MINVAL
    with np.errstate(divide="ignore"):
        inverse = 1 / norm
    axis = axis * inverse[:, None]
    axis[small] = (1.0, 0.0, 0.0)
    # MuJoCo calls the C library's atan2, as math.atan2 does.
    speed = np.fromiter(map(math.atan2, norm.tolist(), q[:, 0].tolist()), np.float64, len(q))
    speed = 2 * speed
    speed = np.where(speed > math.pi, speed - 2 * math.pi, speed)
    return axis * (speed / 1.0)[:, None]


def so3_from_z(theta):
    """``lie.SO3.from_z_radians`` per value, with its Taylor branch."""
    zero = np.zeros_like(theta)
    tangent = np.stack([zero, zero, theta], axis=1)
    theta_squared = theta * theta
    theta_pow_4 = theta_squared * theta_squared
    taylor = theta_squared < 1e-10
    safe_theta = np.where(taylor, 1.0, np.sqrt(theta_squared))
    safe_half_theta = 0.5 * safe_theta
    real = np.where(taylor, 1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0, np.cos(safe_half_theta))
    imaginary = np.where(
        taylor, 0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0, np.sin(safe_half_theta) / safe_theta
    )
    return np.concatenate([real[:, None], imaginary[:, None] * tangent], axis=1)


def so3_yaw(q):
    """``lie.SO3.compute_yaw_radians`` per row."""
    q0, q1, q2, q3 = q.T
    # The source squares NumPy scalars, which calls the C library's pow, as math.pow does.
    squares = np.fromiter(map(math.pow, q[:, 2:].ravel().tolist(), [2.0] * (2 * len(q))), np.float64, 2 * len(q))
    squares = squares.reshape(-1, 2)
    return np.arctan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (squares[:, 0] + squares[:, 1]))


def so3_apply(q, v):
    """``lie.SO3.apply`` per row."""
    padded = np.concatenate([np.zeros((len(v), 1)), v], axis=1)
    return so3_multiply(so3_multiply(q, padded), q * _INVERT)[:, 1:]


def _ik_errors_numpy(pos, quat, site_xpos, site_xmat):
    """The IK error rows ``[pos - site_pos, quat2Vel(quat * neg(site_quat))]`` and the norms of their halves."""
    err = np.empty((len(pos), 6))
    err[:, :3] = pos - site_xpos
    err[:, 3:] = quat2vel(mju_mul_quat(quat, mat2quat(site_xmat) * _INVERT))
    sq = err * err
    norms = np.sqrt(np.stack([sq[:, 0] + sq[:, 1] + sq[:, 2], sq[:, 3] + sq[:, 4] + sq[:, 5]], axis=1))
    return err, norms


def _numba_ik_errors():
    """``_ik_errors_numpy`` compiled with Numba: the same operations in the same order, without fused multiply-adds."""
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True)
    def errors(pos, quat, site_xpos, site_xmat):
        n = pos.shape[0]
        err = np.empty((n, 6))
        norms = np.empty((n, 2))
        q = np.empty(4)
        for i in range(n):
            for k in range(3):
                err[i, k] = pos[i, k] - site_xpos[i, k]
            m = site_xmat[i]
            if m[0] + m[4] + m[8] > 0:
                q[0] = 0.5 * math.sqrt(1 + m[0] + m[4] + m[8])
                q[1] = 0.25 * (m[7] - m[5]) / q[0]
                q[2] = 0.25 * (m[2] - m[6]) / q[0]
                q[3] = 0.25 * (m[3] - m[1]) / q[0]
            elif m[0] > m[4] and m[0] > m[8]:
                q[1] = 0.5 * math.sqrt(1 + m[0] - m[4] - m[8])
                q[0] = 0.25 * (m[7] - m[5]) / q[1]
                q[2] = 0.25 * (m[1] + m[3]) / q[1]
                q[3] = 0.25 * (m[2] + m[6]) / q[1]
            elif m[4] > m[8]:
                q[2] = 0.5 * math.sqrt(1 - m[0] + m[4] - m[8])
                q[0] = 0.25 * (m[2] - m[6]) / q[2]
                q[1] = 0.25 * (m[1] + m[3]) / q[2]
                q[3] = 0.25 * (m[5] + m[7]) / q[2]
            else:
                q[3] = 0.5 * math.sqrt(1 - m[0] - m[4] + m[8])
                q[0] = 0.25 * (m[3] - m[1]) / q[3]
                q[1] = 0.25 * (m[2] + m[6]) / q[3]
                q[2] = 0.25 * (m[5] + m[7]) / q[3]
            norm = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
            if norm < _MINVAL:
                q[0], q[1], q[2], q[3] = 1.0, 0.0, 0.0, 0.0
            elif abs(norm - 1) > _MINVAL:
                inverse = 1 / norm
                for k in range(4):
                    q[k] *= inverse
            b0, b1, b2, b3 = q[0], -q[1], -q[2], -q[3]
            a0, a1, a2, a3 = quat[i, 0], quat[i, 1], quat[i, 2], quat[i, 3]
            w = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3
            x = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2
            y = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1
            z = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0
            sin_half = math.sqrt(x * x + y * y + z * z)
            if sin_half < _MINVAL:
                x, y, z = 1.0, 0.0, 0.0
            else:
                inverse = 1 / sin_half
                x, y, z = x * inverse, y * inverse, z * inverse
            speed = 2 * math.atan2(sin_half, w)
            if speed > math.pi:
                speed -= 2 * math.pi
            speed /= 1.0
            err[i, 3], err[i, 4], err[i, 5] = x * speed, y * speed, z * speed
            norms[i, 0] = math.sqrt(err[i, 0] * err[i, 0] + err[i, 1] * err[i, 1] + err[i, 2] * err[i, 2])
            norms[i, 1] = math.sqrt(err[i, 3] * err[i, 3] + err[i, 4] * err[i, 4] + err[i, 5] * err[i, 5])
        return err, norms

    return errors


_NUMBA_IK_ERRORS = _numba_ik_errors()


def _numba_pose():
    """Numba kernels for the arithmetic of ``BatchController.control``, or None without Numba.

    They run the operations of the NumPy functions above in the same order, without fused multiply-adds. ``cos``,
    ``sin``, ``arctan2`` and ``pow`` stay in NumPy and the C library, as the source calls them. Returns
    ``(quat, yaw_terms, half_angles, orientation, target)``.
    """
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True, inline="always")
    def multiply(a0, a1, a2, a3, b0, b1, b2, b3):
        # so3_multiply: each component sums its signed products from the left.
        w = ((-(a1 * b1)) + (-(a2 * b2)) + (-(a3 * b3))) + a0 * b0
        x = ((a1 * b0) + (a2 * b3) + (-(a3 * b2))) + a0 * b1
        y = ((-(a1 * b3)) + (a2 * b0) + (a3 * b1)) + a0 * b2
        z = ((a1 * b2) + (-(a2 * b1)) + (a3 * b0)) + a0 * b3
        return w, x, y, z

    @numba.njit(cache=True, inline="always")
    def from_z(theta, cos_half, sin_half):
        # so3_from_z with its Taylor branch; cos_half and sin_half are of 0.5 * safe_theta.
        theta_squared = theta * theta
        theta_pow_4 = theta_squared * theta_squared
        if theta_squared < 1e-10:
            real = 1.0 - theta_squared / 8.0 + theta_pow_4 / 384.0
            imaginary = 0.5 - theta_squared / 48.0 + theta_pow_4 / 3840.0
        else:
            real = cos_half
            imaginary = sin_half / math.sqrt(theta_squared)
        return real, imaginary * 0.0, imaginary * 0.0, imaginary * theta

    @numba.njit(cache=True)
    def quat(mat):
        """``mat2quat`` of (n, 9) rows."""
        n = mat.shape[0]
        q = np.empty((n, 4))
        for i in range(n):
            m = mat[i]
            if m[0] + m[4] + m[8] > 0:
                q[i, 0] = 0.5 * math.sqrt(1 + m[0] + m[4] + m[8])
                q[i, 1] = 0.25 * (m[7] - m[5]) / q[i, 0]
                q[i, 2] = 0.25 * (m[2] - m[6]) / q[i, 0]
                q[i, 3] = 0.25 * (m[3] - m[1]) / q[i, 0]
            elif m[0] > m[4] and m[0] > m[8]:
                q[i, 1] = 0.5 * math.sqrt(1 + m[0] - m[4] - m[8])
                q[i, 0] = 0.25 * (m[7] - m[5]) / q[i, 1]
                q[i, 2] = 0.25 * (m[1] + m[3]) / q[i, 1]
                q[i, 3] = 0.25 * (m[2] + m[6]) / q[i, 1]
            elif m[4] > m[8]:
                q[i, 2] = 0.5 * math.sqrt(1 - m[0] + m[4] - m[8])
                q[i, 0] = 0.25 * (m[2] - m[6]) / q[i, 2]
                q[i, 1] = 0.25 * (m[1] + m[3]) / q[i, 2]
                q[i, 3] = 0.25 * (m[5] + m[7]) / q[i, 2]
            else:
                q[i, 3] = 0.5 * math.sqrt(1 - m[0] - m[4] + m[8])
                q[i, 0] = 0.25 * (m[3] - m[1]) / q[i, 3]
                q[i, 1] = 0.25 * (m[2] + m[6]) / q[i, 3]
                q[i, 2] = 0.25 * (m[5] + m[7]) / q[i, 3]
            norm = math.sqrt(q[i, 0] * q[i, 0] + q[i, 1] * q[i, 1] + q[i, 2] * q[i, 2] + q[i, 3] * q[i, 3])
            if norm < _MINVAL:
                q[i, 0], q[i, 1], q[i, 2], q[i, 3] = 1.0, 0.0, 0.0, 0.0
            elif abs(norm - 1) > _MINVAL:
                inverse = 1 / norm
                for k in range(4):
                    q[i, k] *= inverse
        return q

    @numba.njit(cache=True)
    def yaw_terms(q, squares):
        """The ``arctan2`` arguments of ``so3_yaw``; ``squares`` are the C library's squares of q2 and q3."""
        n = q.shape[0]
        y, x = np.empty(n), np.empty(n)
        for i in range(n):
            y[i] = 2 * (q[i, 0] * q[i, 3] + q[i, 1] * q[i, 2])
            x[i] = 1 - 2 * (squares[i, 0] + squares[i, 1])
        return y, x

    @numba.njit(cache=True)
    def half_angles(theta):
        """``0.5 * safe_theta`` of ``so3_from_z``, whose cosine and sine it reads."""
        out = np.empty(theta.shape[0])
        for i in range(theta.shape[0]):
            theta_squared = theta[i] * theta[i]
            out[i] = 0.5 * (1.0 if theta_squared < 1e-10 else math.sqrt(theta_squared))
        return out

    @numba.njit(cache=True)
    def orientation(theta, cos_half, sin_half, down):
        """``so3_from_z(a) * so3_from_z(b) * inverse(down)``; ``theta`` holds the n angles a, then the n angles b."""
        n = theta.shape[0] // 2
        out = np.empty((n, 4))
        d0, d1, d2, d3 = down[0] * 1.0, down[1] * -1.0, down[2] * -1.0, down[3] * -1.0
        for i in range(n):
            a0, a1, a2, a3 = from_z(theta[i], cos_half[i], sin_half[i])
            b0, b1, b2, b3 = from_z(theta[n + i], cos_half[n + i], sin_half[n + i])
            c0, c1, c2, c3 = multiply(a0, a1, a2, a3, b0, b1, b2, b3)
            out[i, 0], out[i, 1], out[i, 2], out[i, 3] = multiply(c0, c1, c2, c3, d0, d1, d2, d3)
        return out

    @numba.njit(cache=True)
    def target(yaw, cos_half, sin_half, down, t_pa, translation):
        """The IK target ``(pos, quat)``: ``orientation = so3_from_z(yaw) * down``, then the pinch-to-attach pose."""
        n = yaw.shape[0]
        pos, quat_out = np.empty((n, 3)), np.empty((n, 4))
        for i in range(n):
            a0, a1, a2, a3 = from_z(yaw[i], cos_half[i], sin_half[i])
            o0, o1, o2, o3 = multiply(a0, a1, a2, a3, down[0], down[1], down[2], down[3])
            quat_out[i, 0], quat_out[i, 1], quat_out[i, 2], quat_out[i, 3] = multiply(
                o0, o1, o2, o3, t_pa[0], t_pa[1], t_pa[2], t_pa[3]
            )
            # so3_apply: o * (0, v) * inverse(o), keeping the vector part.
            p0, p1, p2, p3 = multiply(o0, o1, o2, o3, 0.0, t_pa[4], t_pa[5], t_pa[6])
            _, r1, r2, r3 = multiply(p0, p1, p2, p3, o0 * 1.0, o1 * -1.0, o2 * -1.0, o3 * -1.0)
            pos[i, 0] = r1 + translation[i, 0]
            pos[i, 1] = r2 + translation[i, 1]
            pos[i, 2] = r3 + translation[i, 2]
        return pos, quat_out

    return quat, yaw_terms, half_angles, orientation, target


def _numba_ik_steps():
    """Numba kernels for the arithmetic of an iteration of ``BatchController.solve``, or None without Numba.

    Returns ``(reached, advance)``: the rows whose errors are within both thresholds, with the number of rows too close
    to a threshold to decide, and the scaled update of ``qpos``.
    """
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True)
    def reached(norms, pos_thresh, ori_thresh):
        n = norms.shape[0]
        done = np.empty(n, np.bool_)
        close = 0
        for i in range(n):
            done[i] = norms[i, 0] <= pos_thresh and norms[i, 1] <= ori_thresh
            if (
                abs(norms[i, 0] - pos_thresh) <= 1e-9 * pos_thresh
                or abs(norms[i, 1] - ori_thresh) <= 1e-9 * ori_thresh
            ):
                close += 1
        return done, close

    @numba.njit(cache=True)
    def advance(qpos, rows, update, max_angle):
        """``qpos[rows] += update``, each update row scaled down to ``max_angle`` at most, as NumPy's max decides."""
        for r in range(rows.shape[0]):
            peak = abs(update[r, 0])
            for k in range(1, update.shape[1]):
                value = abs(update[r, k])
                if value > peak or value != value:
                    peak = value
            if peak > max_angle:
                scale = max_angle / peak
                for k in range(update.shape[1]):
                    update[r, k] *= scale
            i = rows[r]
            for k in range(update.shape[1]):
                qpos[i, k] = qpos[i, k] + 1.0 * update[r, k]

    return reached, advance


_NUMBA_POSE = _numba_pose()
_NUMBA_IK_STEPS = _numba_ik_steps()


def _squares(q):
    """The C library's squares of the q2 and q3 columns, as the source squares NumPy scalars."""
    n = len(q)
    flat = np.fromiter(map(math.pow, q[:, 2:].ravel().tolist(), [2.0] * (2 * n)), np.float64, 2 * n)
    return flat.reshape(n, 2)


class BatchController:
    """``ManipSpaceEnv.set_control`` for every lane of a batch.

    The differential IK runs OGBench's iterations on a ``mujoco.batch`` of the
    IK model: its kinematics and Jacobians are MuJoCo's own, and the NumPy
    arithmetic repeats the source's operations in the source's order.
    """

    def __init__(self, env, num_envs, threads):
        from ...._execution.mjbatch import NativeBatch

        ik = env._ik
        self._source_ik = ik
        # A forward pass of the arm takes microseconds, so a thread pays off only for many lanes.
        threads = max(1, min(threads, num_envs // IK_LANES_PER_THREAD))
        # The IK reads kinematics only; without contacts and constraints the position stage skips collision.
        model = copy.copy(ik._model)
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT | mujoco.mjtDisableBit.mjDSBL_CONTACT
        self._batch = NativeBatch(model, num_envs, outputs=("site_xpos", "site_xmat"), nthread=threads)
        self._ik = self._batch.b
        self._ik_qpos = self._batch.numpy("qpos")
        self._ik_site_xpos = self._batch.numpy("site_xpos")
        self._ik_site_xmat = self._batch.numpy("site_xmat")
        self._site = int(ik._site_ids[0])
        self._site_body = int(ik._model.site_bodyid[self._site])
        self._damping = ik._damping
        self._max_angle = ik._max_angle_change
        self._arm_qpos = env.model.jnt_qposadr[env._arm_joint_ids]
        self._gripper_qpos = int(env.model.jnt_qposadr[env._gripper_opening_joint_id])
        self._arm_actuators = env._arm_actuator_ids
        self._gripper_actuators = env._gripper_actuator_ids
        self._low, self._high = env.action_low, env.action_high
        self._workspace = env._workspace_bounds
        self._down = env._effector_down_rotation.wxyz
        self._t_pa = env._T_pa.wxyz_xyz
        self._down_rotation, self._pinch_to_attach = env._effector_down_rotation, env._T_pa
        self._norm = np.linalg.norm
        self._errors = _NUMBA_IK_ERRORS or _ik_errors_numpy
        self._pose, self._steps = _NUMBA_POSE, _NUMBA_IK_STEPS
        self._rows = np.arange(num_envs)
        self._down64, self._t_pa64 = np.asarray(self._down, np.float64), np.asarray(self._t_pa, np.float64)

    def control(self, actions, qpos, pinch_pos, pinch_mat, ctrl, ids):
        """Write the controls of lanes ``ids`` into ``ctrl`` from their actions and pinch site poses."""
        n = len(ids)
        action = 0.5 * (np.asarray(actions) + 1) * (self._high - self._low) + self._low
        if self._pose is not None and action.dtype == np.float64:
            pos, quat, target_gripper = self._targets(action, qpos, pinch_pos, pinch_mat, ids)
            if n == 1:
                # One lane solves with OGBench's own controller, which costs less than a batch of one.
                arm = self._source_ik.solve(pos=pos[0], quat=quat[0], curr_qpos=qpos[ids[0]][self._arm_qpos])
                ctrl[ids[0], self._arm_actuators] = arm
                ctrl[ids[0], self._gripper_actuators] = 255.0 * target_gripper[0]
            else:
                ctrl[ids[:, None], self._arm_actuators] = self.solve(pos, quat, qpos[ids][:, self._arm_qpos])
                ctrl[ids[:, None], self._gripper_actuators] = 255.0 * target_gripper
            return
        if n == 1:
            arm, gripper = self._control_one(np.asarray(actions)[0], qpos[ids[0]], pinch_pos[0], pinch_mat[0])
            ctrl[ids[0], self._arm_actuators] = arm
            ctrl[ids[0], self._gripper_actuators] = gripper
            return
        a_pos, a_ori, a_gripper = action[:, :3], action[:, 3], action[:, 4:5]
        effector_pos = np.array(pinch_pos, np.float64)
        effector_yaw = so3_yaw(mat2quat(np.asarray(pinch_mat).reshape(n, 9)))
        gripper_opening = np.clip(qpos[ids, self._gripper_qpos][:, None] / 0.8, 0, 1)
        translation = effector_pos + a_pos
        down = np.broadcast_to(self._down, (n, 4))
        orientation = so3_multiply(so3_multiply(so3_from_z(a_ori), so3_from_z(effector_yaw)), down * _INVERT)
        target_gripper = gripper_opening + a_gripper
        np.clip(translation, *self._workspace, out=translation)
        yaw = np.clip(so3_yaw(orientation), -np.pi, +np.pi)
        orientation = so3_multiply(so3_from_z(yaw), down)
        target_gripper = np.clip(target_gripper, 0.0, 1.0)
        t_pa = np.broadcast_to(self._t_pa, (n, 7))
        quat = so3_multiply(orientation, t_pa[:, :4])
        pos = so3_apply(orientation, t_pa[:, 4:]) + translation
        ctrl[ids[:, None], self._arm_actuators] = self.solve(pos, quat, qpos[ids][:, self._arm_qpos])
        ctrl[ids[:, None], self._gripper_actuators] = 255.0 * target_gripper

    def _targets(self, action, qpos, pinch_pos, pinch_mat, ids):
        """``control``'s IK targets ``(pos, quat)`` and gripper targets, with the Numba kernels."""
        quat, yaw_terms, half_angles, orientation, target = self._pose
        n = len(ids)
        a_pos, a_ori, a_gripper = action[:, :3], action[:, 3], action[:, 4:5]
        gripper_opening = np.clip(qpos[ids, self._gripper_qpos][:, None] / 0.8, 0, 1)
        translation = np.array(pinch_pos, np.float64) + a_pos
        np.clip(translation, *self._workspace, out=translation)
        effector = quat(np.ascontiguousarray(np.asarray(pinch_mat).reshape(n, 9), np.float64))
        theta = np.concatenate([a_ori, np.arctan2(*yaw_terms(effector, _squares(effector)))])
        half = half_angles(theta)
        rotation = orientation(theta, np.cos(half), np.sin(half), self._down64)
        yaw = np.clip(np.arctan2(*yaw_terms(rotation, _squares(rotation))), -np.pi, +np.pi)
        half = half_angles(yaw)
        pos, quat_out = target(yaw, np.cos(half), np.sin(half), self._down64, self._t_pa64, translation)
        return pos, quat_out, np.clip(gripper_opening + a_gripper, 0.0, 1.0)

    def _control_one(self, action, qpos, pinch_pos, pinch_mat):
        """``ManipSpaceEnv.set_control`` of one lane, with OGBench's own pose types and controller.

        Returns the arm and gripper controls.
        """
        from ogbench.manipspace import lie

        action = 0.5 * (action + 1) * (self._high - self._low) + self._low
        a_pos, a_ori, a_gripper = action[:3], action[3], action[4]
        effector_pos = np.array(pinch_pos)
        effector_yaw = lie.SO3.from_matrix(np.array(pinch_mat).reshape(3, 3)).compute_yaw_radians()
        gripper_opening = np.array(np.clip([qpos[self._gripper_qpos] / 0.8], 0, 1))
        translation = effector_pos + a_pos
        down = self._down_rotation
        orientation = lie.SO3.from_z_radians(a_ori) @ lie.SO3.from_z_radians(effector_yaw) @ down.inverse()
        target_gripper = gripper_opening + a_gripper
        np.clip(translation, *self._workspace, out=translation)
        yaw = np.clip(orientation.compute_yaw_radians(), -np.pi, +np.pi)
        orientation = lie.SO3.from_z_radians(yaw) @ down
        target_gripper = np.clip(target_gripper, 0.0, 1.0)
        pose = lie.SE3.from_rotation_and_translation(rotation=orientation, translation=translation)
        attach = pose @ self._pinch_to_attach
        arm = self._source_ik.solve(
            pos=attach.translation(), quat=attach.rotation().wxyz, curr_qpos=qpos[self._arm_qpos]
        )
        return arm, 255.0 * target_gripper

    def solve(self, pos, quat, qpos, max_iters=20, pos_thresh=1e-4, ori_thresh=1e-4):
        """``DiffIKController.solve`` per row, each row stopping at its own iteration.

        While every row is still iterating, the arrays are used whole, without index copies; the arithmetic is
        the same as on the active rows.
        """
        n = len(pos)
        b = self._ik
        qpos = qpos.copy()
        # The IK batch has a row per lane; a call for fewer lanes uses its first rows by index.
        ids = None if n == len(self._ik_qpos) else np.arange(n)
        site_xpos, site_xmat = self._ik_site_xpos[:, self._site], self._ik_site_xmat[:, self._site]
        for _ in range(max_iters):
            if ids is None:
                self._ik_qpos[:] = qpos
                b.forward(None, mujoco.mjtStage.mjSTAGE_POS)
                err, norms = self._errors(pos, quat, site_xpos, site_xmat)
            else:
                self._ik_qpos[ids] = qpos[ids]
                b.forward(ids, mujoco.mjtStage.mjSTAGE_POS)
                err, norms = self._errors(pos[ids], quat[ids], site_xpos[ids], site_xmat[ids])
            done, close = (None, 1) if self._steps is None else self._steps[0](norms, pos_thresh, ori_thresh)
            if close:
                done = self._reached(err[:, :3], norms[:, 0], pos_thresh) & self._reached(
                    err[:, 3:], norms[:, 1], ori_thresh
                )
            if done.any():
                keep = ~done
                ids = (np.arange(n) if ids is None else ids)[keep]
                err = err[keep]
                if not len(ids):
                    break
            jacp, jacr = b.jac(self._site_body, site_xpos if ids is None else site_xpos[ids], ids)
            jac = np.concatenate([jacp, jacr], axis=1)
            h = jac @ jac.transpose(0, 2, 1) + self._damping
            update = (jac.transpose(0, 2, 1) @ np.linalg.solve(h, err[:, :, None]))[:, :, 0]
            if self._steps is not None:
                self._steps[1](qpos, self._rows[:n] if ids is None else ids, update, self._max_angle)
                continue
            update_max = np.max(np.abs(update), axis=1)
            over = update_max > self._max_angle
            update[over] *= (self._max_angle / update_max[over])[:, None]
            if ids is None:
                qpos = qpos + 1.0 * update
            else:
                qpos[ids] = qpos[ids] + 1.0 * update
        return qpos

    def _reached(self, err, norm, threshold):
        """``np.linalg.norm(err) <= threshold`` per row, as the source's (1, 3) norm decides it.

        ``norm`` may differ from NumPy's BLAS norm in the last bit, so rows this close to the threshold use NumPy's.
        """
        result = norm <= threshold
        close = np.flatnonzero(np.abs(norm - threshold) <= 1e-9 * threshold)
        for i in close:
            result[i] = self._norm(err[i : i + 1]) <= threshold
        return result

    def close(self):
        self._batch.close()
