"""Low-level autopilot used by the planning env: turns a 4-D acceleration command into wheel and jet commands.

The planner (or a world-model policy) commands
    a = [a_fwd, a_lat, a_up, yaw_rate]  in [-1, 1]^4
meaning forward / sideways acceleration in the heading frame, vertical acceleration relative to hover,
and a yaw rate. At 50 Hz this controller
  * on the ground: maps a_fwd to wheel torque and yaw_rate to a bicycle-model steering angle;
    a positive a_up lights the lift jets (jump / take-off);
  * in the air: holds the car level (banking into sideways accelerations), integrates the yaw command,
    and solves a bounded least-squares problem for the nine jets. When the car is inverted (after the
    loop) it first half-rolls upright, keeping its heading.
"""

from __future__ import annotations

import numpy as np

from .model import JETS
from .sim import ACTION_HIGH, ACTION_LOW, AIR_DENSITY, DRAG_CDA, RoadsterSim
from .util import rot_err, rx, rz

G = 9.81
INERTIA = np.array([900.0, 3000.0, 3500.0])
WRENCH_SCALE = np.array([6000.0, 3000.0, 4000.0, 1500.0, 1500.0, 1500.0])
ACC_SCALE = np.array([6.0, 5.0, 6.0])  # m/s^2 for a_fwd, a_lat, a_up at |a| = 1
YAW_RATE_SCALE = 1.2  # rad/s
WHEELBASE = 2.8


class Autopilot:
    """50 Hz controller from the 4-D command to the 11-D raw action; `yaw_cmd` and `roll_cmd` are its state."""

    def __init__(self, env: RoadsterSim):
        self.env = env
        self.mass = float(env.model.body_subtreemass[env.chassis_id])
        self.lo, self.hi = ACTION_LOW[2:], ACTION_HIGH[2:]
        self.yaw_cmd = 0.0
        self.roll_cmd = np.nan  # barrel-roll stunt: target roll angle (rad), NaN = hold level

    # state the planner must carry along with the MuJoCo state
    def get_state(self) -> np.ndarray:
        return np.array([self.yaw_cmd, self.roll_cmd])

    def set_state(self, s: np.ndarray):
        self.yaw_cmd, self.roll_cmd = float(s[0]), float(s[1])

    def reset(self):
        self.yaw_cmd = self.current_yaw()
        self.roll_cmd = np.nan

    def current_yaw(self) -> float:
        h = self.env.rot[:, 0]
        return float(np.arctan2(h[1], h[0]))

    _B = None

    def _alloc_matrix(self) -> np.ndarray:
        """6x9 normalized wrench matrix about the car's center of mass at rest.

        Cached: the CoM barely moves with the suspension.
        """
        if self._B is None:
            com_b = np.zeros(3)
            m = self.env.model
            total = 0.0
            for b in range(self.env.chassis_id, m.nbody):  # chassis subtree, rest pose offsets
                total += m.body_mass[b]
                com_b += m.body_mass[b] * np.asarray(m.body_pos[b] if b != self.env.chassis_id else 0.0)
            com_b /= total
            B = np.zeros((6, len(JETS)))
            for k, jet in enumerate(JETS):
                f = np.asarray(jet.direction, float) * jet.max_force
                B[:3, k] = f
                B[3:, k] = np.cross(np.asarray(jet.pos) - com_b, f)
            Autopilot._B = B / WRENCH_SCALE[:, None]
        return self._B

    def _alloc(self, w: np.ndarray) -> np.ndarray:
        B = self._alloc_matrix()
        w = w / WRENCH_SCALE
        u = np.zeros(B.shape[1])
        free = np.ones(B.shape[1], bool)
        for _ in range(B.shape[1]):
            resid = w - B[:, ~free] @ u[~free]
            u[free] = np.linalg.lstsq(B[:, free], resid, rcond=None)[0]
            sat = free & ((u < self.lo) | (u > self.hi))
            if not sat.any():
                break
            u = np.clip(u, self.lo, self.hi)
            free &= ~sat
            if not free.any():
                break
        return np.clip(u, self.lo, self.hi)

    def _air_jets(self, acc_w: np.ndarray) -> np.ndarray:
        env = self.env
        R, v = env.rot, env.vel
        drag = 0.5 * AIR_DENSITY * DRAG_CDA * np.linalg.norm(v) * v
        F_b = R.T @ (self.mass * (acc_w + np.array([0, 0, G])) + drag)
        Rd = rz(self.yaw_cmd)
        if not np.isnan(self.roll_cmd):  # stunt: follow the commanded roll about the car's long axis
            Rd = Rd @ rx(self.roll_cmd)
        elif R[2, 2] > 0.7:
            a_lat = (Rd.T @ acc_w)[1]
            Rd = Rd @ rx(-np.clip(np.arctan2(a_lat, G + acc_w[2]), -0.4, 0.4))
        wn, zeta = 5.0, 0.9
        tau = INERTIA * (-(wn**2) * rot_err(R, Rd) - 2 * zeta * wn * env.angvel_body)
        return self._alloc(np.r_[F_b, tau])

    def __call__(self, a: np.ndarray, n_wheels: int) -> np.ndarray:
        """4-D command -> 11-D raw action of RoadsterSim."""
        env = self.env
        a = np.clip(np.asarray(a, float), -1.0, 1.0)
        acc = a[:3] * ACC_SCALE
        yaw_rate = a[3] * YAW_RATE_SCALE
        raw = np.zeros(11)
        R, v = env.rot, env.vel
        heading = R[:, 0]
        if n_wheels >= 2:  # rolling: wheels do the work
            v_fwd = float(v @ heading)
            raw[0] = acc[0] / ACC_SCALE[0]
            steer = np.arctan(WHEELBASE * yaw_rate / max(abs(v_fwd), 2.0))
            raw[1] = np.clip(steer / 0.5, -1, 1) * np.sign(v_fwd if abs(v_fwd) > 0.5 else 1.0)
            self.yaw_cmd = self.current_yaw()
            if acc[2] > 0.5:  # lift-off
                psi = self.yaw_cmd
                acc_w = rz(psi) @ np.array([acc[0], acc[1], 0.0]) + np.array([0, 0, acc[2]])
                raw[2:] = self._air_jets(acc_w)
            return raw
        # airborne
        if R[2, 2] < 0.7 and np.isnan(self.roll_cmd):  # inverted / steep: hold heading while rolling upright
            self.yaw_cmd = self.current_yaw()
        else:
            self.yaw_cmd += yaw_rate * env.dt
        acc_w = rz(self.yaw_cmd) @ np.array([acc[0], acc[1], 0.0]) + np.array([0, 0, acc[2]])
        raw[2:] = self._air_jets(acc_w)
        return raw
