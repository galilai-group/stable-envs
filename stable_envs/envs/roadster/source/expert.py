"""Scripted expert for the planning tasks (4-D action space): guidance toward the goal state.

It reads the simulator (RoadsterPlanEnv), so it serves data collection, demos and tests, not as a
policy from observations.
"""

from __future__ import annotations

import numpy as np

from .control import ACC_SCALE, YAW_RATE_SCALE
from .tasks import RoadsterPlanEnv
from .util import rz


def _yaw(env: RoadsterPlanEnv) -> float:
    return env.ap.current_yaw()


class GoalExpert:
    """Velocity-shaping guidance toward env.goal_pos / goal_vel; full throttle for the loop."""

    def __init__(self, v_max: float = 14.0):
        self.v_max = v_max

    def __call__(self, env: RoadsterPlanEnv) -> np.ndarray:
        s = env.sim
        if env.task in ("loop", "mission") and not env.success() and s.status == "drive":
            return np.array([1.0, 0.0, 0.0, 0.0])
        p, v = s.pos, s.vel
        goal = env.goal_pos
        through_ring, final, pass_through = False, True, False
        if env.task in ("traffic", "rooftop", "carhop"):
            wps = env._waypoints()
            goal, final = wps[0][0], len(wps) == 1
            if env._route_touch and env._route_touch[env._route_idx]:
                final = True  # touchdown waypoint: land there like at the goal
            pass_through = not final and np.linalg.norm(wps[0][1]) > 0  # en-route waypoint (vel > 0)
        elif env.task == "mission" and not s.landing_stage:  # follow the rings in order
            goal, through_ring = s.target, True
            if p[0] < env.sim.course.loop_radius + 6 and p[2] < env.sim.course.loop_top + 4:
                goal = np.array([p[0] + 15.0, goal[1], env.sim.course.loop_top + 5.0])  # clear the loop
        dp = goal - p
        dist = np.linalg.norm(dp)
        if through_ring:
            spd = 12.0
            v_des = dp / max(dist, 1e-6) * spd
        elif pass_through:  # en-route waypoint: keep moving through it
            v_des = dp / max(dist, 1e-6) * min(self.v_max, 1.5 * dist + 4.0)
        else:
            spd = min(np.sqrt(2 * 2.5 * dist), 0.8 * dist, self.v_max)
            v_des = dp / max(dist, 1e-6) * spd + env.goal_vel * (dist < 3)
        landing = (env.task == "mission" and s.landing_stage) or (
            env.task in ("traffic", "rooftop", "carhop") and final
        )
        if landing and np.linalg.norm(dp[:2]) > 1.5:
            v_des[2] = np.clip(1.5 * (goal[2] + 3.0 - p[2]), -4, 4)  # approach high, then drop
        elif landing and env.task in ("traffic", "rooftop", "carhop"):
            v_des[2] = min(v_des[2], -2.0)  # settle firmly onto the road / roof
        acc_w = np.array([1.6, 1.6, 2.0]) * (v_des - v)
        acc_h = rz(_yaw(env)).T @ acc_w
        yaw_err = np.angle(np.exp(1j * (env.goal_yaw - _yaw(env))))
        a = np.r_[acc_h / ACC_SCALE, 1.5 * yaw_err / YAW_RATE_SCALE]
        return np.clip(a, -1, 1)
