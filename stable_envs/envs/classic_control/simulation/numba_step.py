"""Classic-control steps of every environment in one compiled call, in place on the packed State.

Each kernel repeats the source ``step`` in float64 with the same operation order, one environment after another. NumPy
evaluates the trigonometric functions on the arrays before the call, as the NumPy kernels do, so the values equal
the source's; the kernels do only IEEE arithmetic and comparisons, and Numba compiles them without fast-math, so
nothing is reassociated or fused.
"""

from __future__ import annotations

import math

import numpy as np

from .local import BEYOND, ELAPSED, HAS_BEYOND

try:
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - numba is in the classic extra
    HAVE_NUMBA = False

#: CartPole's ``theta_threshold_radians`` and ``x_threshold``.
_THETA_LIMIT = 12 * 2 * math.pi / 360
_X_LIMIT = 2.4

if HAVE_NUMBA:

    @njit(cache=True, nogil=True)
    def _cartpole(x, x_dot, theta, theta_dot, beyond, has_beyond, elapsed, cos, sin, gravity, masscart, masspole,
                  length, force_mag, tau, action, live, max_steps, reward, terminated, truncated, observation):  # fmt: skip
        for i in range(x.shape[0]):
            if not live[i]:
                reward[i] = 0.0
                terminated[i] = False
                truncated[i] = False
                for j in range(4):
                    observation[i, j] = 0.0
                continue
            total_mass = masspole[i] + masscart[i]
            polemass_length = masspole[i] * length[i]
            force = force_mag[i] if action[i] == 1 else -force_mag[i]
            temp = (force + polemass_length * (theta_dot[i] * theta_dot[i]) * sin[i]) / total_mass
            thetaacc = (gravity[i] * sin[i] - cos[i] * temp) / (
                length[i] * (4.0 / 3.0 - masspole[i] * (cos[i] * cos[i]) / total_mass)
            )
            xacc = temp - polemass_length * thetaacc * cos[i] / total_mass
            step = tau[i]
            nx = x[i] + step * x_dot[i]
            nx_dot = x_dot[i] + step * xacc
            ntheta = theta[i] + step * theta_dot[i]
            ntheta_dot = theta_dot[i] + step * thetaacc
            x[i], x_dot[i], theta[i], theta_dot[i] = nx, nx_dot, ntheta, ntheta_dot
            done = nx < -_X_LIMIT or nx > _X_LIMIT or ntheta < -_THETA_LIMIT or ntheta > _THETA_LIMIT
            again = done and has_beyond[i]
            if again:
                beyond[i] += 1
            elif done:
                beyond[i] = 0
            has_beyond[i] = has_beyond[i] or done
            reward[i] = 0.0 if again else 1.0
            terminated[i] = done
            elapsed[i] += 1
            truncated[i] = elapsed[i] >= max_steps
            observation[i, 0] = np.float32(nx)
            observation[i, 1] = np.float32(nx_dot)
            observation[i, 2] = np.float32(ntheta)
            observation[i, 3] = np.float32(ntheta_dot)


if HAVE_NUMBA:

    @njit(cache=True, nogil=True)
    def _mountain_car(position, velocity, elapsed, cosine, gravity, force, max_speed, action, live, max_steps, reward,
                      terminated, truncated, observation):  # fmt: skip
        for i in range(position.shape[0]):
            if not live[i]:
                reward[i] = 0.0
                terminated[i] = False
                truncated[i] = False
                observation[i, 0] = 0.0
                observation[i, 1] = 0.0
                continue
            v = velocity[i] + ((action[i] - 1) * force[i] + cosine[i] * (-gravity[i]))
            v = min(max(v, -max_speed[i]), max_speed[i])
            p = min(max(position[i] + v, -1.2), 0.6)
            if p == -1.2 and v < 0:
                v = 0.0
            position[i], velocity[i] = p, v
            reward[i] = -1.0
            terminated[i] = p >= 0.5 and v >= 0
            elapsed[i] += 1
            truncated[i] = elapsed[i] >= max_steps
            observation[i, 0] = np.float32(p)
            observation[i, 1] = np.float32(v)


def mountain_car_step(views, params, actions, live, max_steps):
    """Step every MountainCar environment in place; return ``(reward, terminated, truncated, observation)``."""
    n = actions.shape[0]
    position = views["position"][:, 0]
    cosine = np.cos(3 * position)
    reward = np.empty(n, np.float64)
    terminated = np.empty(n, np.bool_)
    truncated = np.empty(n, np.bool_)
    observation = np.empty((n, 2), np.float32)
    _mountain_car(
        position, views["velocity"][:, 0], views[ELAPSED][:, 0], cosine, params["gravity"],
        params["force"], params["max_speed"], actions, live, max_steps, reward, terminated, truncated, observation,
    )  # fmt: skip
    return reward, terminated, truncated, observation


def cartpole_step(views, params, actions, live, max_steps):
    """Step every CartPole environment in place; return ``(reward, terminated, truncated, observation)``."""
    n = actions.shape[0]
    theta = views["theta"][:, 0]
    cos, sin = np.cos(theta), np.sin(theta)
    reward = np.empty(n, np.float64)
    terminated = np.empty(n, np.bool_)
    truncated = np.empty(n, np.bool_)
    observation = np.empty((n, 4), np.float32)
    _cartpole(
        views["x"][:, 0], views["x_dot"][:, 0], theta, views["theta_dot"][:, 0],
        views[BEYOND][:, 0], views[HAS_BEYOND][:, 0],
        views[ELAPSED][:, 0], cos, sin, params["gravity"], params["masscart"],
        params["masspole"], params["length"], params["force_mag"], params["tau"], actions, live, max_steps,
        reward, terminated, truncated, observation,
    )  # fmt: skip
    return reward, terminated, truncated, observation


#: Domains with a compiled step.
STEPS = {"cartpole": cartpole_step, "mountain_car": mountain_car_step} if HAVE_NUMBA else {}
