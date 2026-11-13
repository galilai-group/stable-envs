"""Goal-conditioned planning tasks for the jet Roadster with a 4-D autopilot action.

* 4-D action [a_fwd, a_lat, a_up, yaw_rate] at 10 Hz, executed by a 50 Hz autopilot (control.py).
* Tasks with a start-state distribution and a goal: loop, mission, traffic, rooftop, carhop. Route tasks
  (traffic, rooftop, carhop) follow an ordered list of waypoints ending at the goal.
* get_state / set_state snapshot and restore everything that determines the future.
"""

from __future__ import annotations

import mujoco
import numpy as np

from .control import Autopilot
from .course import Course
from .model import SPAWN_Z
from .scenarios import CAR_SIZE, CITY_Y, HOP_Y, TOWERS, TRAFFIC_Y, place_hop, place_traffic
from .sim import FUEL_CAPACITY, RoadsterSim

TASKS = ("loop", "mission", "traffic", "rooftop", "carhop")
HORIZON = {
    "loop": 60,
    "mission": 450,
    "traffic": 150,
    "rooftop": 200,
    "carhop": 120,
}
ROUTE_TASKS = ("traffic", "rooftop", "carhop")  # tasks defined by an ordered list of waypoints ending at the goal
ROUTE_RADIUS = 3.5  # m: a waypoint counts as reached inside this radius
ROLL_TIME = 1.6  # s: duration of the car-hop barrel roll
REPEAT = 5  # 50 Hz sim steps per 10 Hz action
_STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _quat_from_yaw(psi: float) -> np.ndarray:
    return np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])


class RoadsterPlanEnv:
    """One task of the jet Roadster.

    Args:
        task: one of `TASKS`.
        course: ring and pad layout of the main course.
        template: another env of this world whose compiled model, settled pose and loop goal this one
            shares instead of building its own; the episodes stay independent.

    ``reset(seed, wind, rain)`` samples a start state and a goal under the given weather; ``step(action)``
    returns ``(reward, terminated, truncated, info)``. Reward: decrease of `cost()` (distance to the current
    subgoal plus the path through the remaining ones) per step, +10 on success, -10 on failure. Episodes end
    on success, failure or `HORIZON[task]`.
    """

    def __init__(self, task: str = "traffic", course: Course | None = None, template: RoadsterPlanEnv | None = None):
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; known: {TASKS}")
        self.task = task
        # A template lane shares its compiled model, settled pose and loop goal (all independent of the episode).
        self.sim = RoadsterSim(course=course, template=None if template is None else template.sim)
        self.ap = Autopilot(self.sim)
        self.dt = self.sim.dt * REPEAT
        self._rest_qpos = self._settled_qpos() if template is None else template._rest_qpos.copy()
        self.sim.reset()  # a valid episode state even before the first reset (set_state may come first)
        self._route, self._route_idx, self.jam = [], 0, None
        self._roof_goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "roof_goal")
        self._loop_goal = None if template is None else template._loop_goal
        self.goal_pos, self.goal_yaw, self.goal_vel = np.zeros(3), 0.0, np.zeros(3)
        self._route_touch: list[bool] = []
        self._loop_top_seen, self._ground_hold = False, 3
        self._roll_t0, self._roll_done = None, False
        self._prev_cost, self.steps = 0.0, 0

    @property
    def model(self):
        return self.sim.model

    @property
    def data(self):
        return self.sim.data

    def get_state(self) -> dict:
        """Snapshot of everything that determines the future.

        `phys` is MuJoCo's integration state (time, qpos, qvel, warm start, ctrl, applied forces and the
        mocap bodies: traffic and parked cars); `qpos_pre` is the pose the derived quantities
        (xmat, contacts) refer to after a step (see `RoadsterSim.restore_derived`). The rest is the
        autopilot, the task bookkeeping, the goal and route, the weather (mean wind, gust state and its RNG)
        and the exhaust history the smoke is drawn from (`fx`). Restore it with `set_state` on this env or on
        another one built with the same arguments.
        """
        n = mujoco.mj_stateSize(self.model, _STATE_SPEC)
        phys = np.empty(n)
        mujoco.mj_getState(self.model, self.data, phys, _STATE_SPEC)
        s = self.sim
        return {
            "task": self.task,
            "phys": phys,
            "qpos_pre": s.stale_pose,
            "ap": self.ap.get_state(),
            "fuel": s.fuel,
            "target_idx": s.target_idx,
            "airborne_once": s.airborne_once,
            "status": s.status,
            "steps": self.steps,
            "sim_steps": s.steps,
            "prev_cost": self._prev_cost,
            "loop_top_seen": self._loop_top_seen,
            "ground_hold": self._ground_hold,
            "route_idx": self._route_idx,
            "roll_t0": self._roll_t0,
            "roll_done": self._roll_done,
            "region_y": s.region_y,
            "allow_ground_landing": s.allow_ground_landing,
            "goal_pos": self.goal_pos.copy(),
            "goal_yaw": float(self.goal_yaw),
            "goal_vel": self.goal_vel.copy(),
            "route": [(p.copy(), v.copy(), float(up)) for p, v, up in self._route],
            "route_touch": list(self._route_touch),
            "jam": dict(self.jam) if self.jam else None,
            "weather": s.weather.get_state(),
            "fx": s.fx_state(),
        }

    def set_state(self, st: dict):
        """Restore a `get_state` snapshot (the task may differ from the current one)."""
        self.task = st["task"]
        if self.task == "loop" and self._loop_goal is None:
            self._loop_goal_state()  # reference run: must happen before the physics is restored
        s = self.sim
        s.weather.set_state(st["weather"])
        mujoco.mj_setState(self.model, self.data, st["phys"], _STATE_SPEC)
        s.restore_derived(st["qpos_pre"])
        self.ap.set_state(st["ap"])
        s.fuel, s.target_idx = st["fuel"], st["target_idx"]
        s.airborne_once, s.status = st["airborne_once"], st["status"]
        self.steps, s.steps = st["steps"], st["sim_steps"]
        self._prev_cost = st["prev_cost"]
        self._loop_top_seen, self._ground_hold = st["loop_top_seen"], st["ground_hold"]
        self._route_idx = st["route_idx"]
        self._roll_t0, self._roll_done = st["roll_t0"], st["roll_done"]
        s.region_y, s.allow_ground_landing = st["region_y"], st["allow_ground_landing"]
        self.goal_pos, self.goal_yaw = np.array(st["goal_pos"], float), float(st["goal_yaw"])
        self.goal_vel = np.array(st["goal_vel"], float)
        self._route = [(np.array(p, float), np.array(v, float), float(up)) for p, v, up in st["route"]]
        self._route_touch = list(st["route_touch"])
        self.jam = dict(st["jam"]) if st["jam"] else None
        s.set_fx_state(*st["fx"])

    def _settled_qpos(self) -> np.ndarray:
        """The qpos of the car resting on flat ground at the origin (suspension settled)."""
        m, d = self.model, self.data
        self.sim.weather.apply_grip()
        mujoco.mj_resetData(m, d)
        d.qpos[:3] = (0.0, 0.0, SPAWN_Z)
        d.qpos[0] = self.sim.course.start_x
        for _ in range(600):
            mujoco.mj_step(m, d)
        q = d.qpos.copy()
        q[0] = 0.0
        return q

    def _place(self, pos, yaw=0.0, vel=(0, 0, 0)):
        d = self.data
        d.qpos[:] = self._rest_qpos
        d.qpos[:3] = pos
        d.qpos[3:7] = _quat_from_yaw(yaw)
        d.qvel[:] = 0.0
        d.qvel[:3] = vel
        d.ctrl[:] = 0.0
        self.sim.forward()

    def _sample_task(self, rng: np.random.Generator):
        """Places the car at a start state and returns the goal (pos, yaw, vel)."""
        c, s = self.sim.course, self.sim
        s.status, s.airborne_once = "flight", True
        s.region_y, s.allow_ground_landing = 0.0, False
        self._route, self._route_idx = [], 0
        self._roll_t0, self._roll_done = None, False
        rz = self._rest_qpos[2]
        if self.task == "carhop":
            # a long lane packed with parked cars: start beside one car on its right side, jump sideways
            # over it and land on the left side of the lane
            s.region_y, s.allow_ground_landing = HOP_Y, True
            s.status, s.airborne_once, s.target_idx = "drive", False, len(c.rings)
            cars = place_hop(self.model, self.data, rng)
            k = int(rng.integers(2, len(cars) - 2))
            xc = (cars[k][0] + cars[k][1]) / 2
            y0 = HOP_Y - rng.uniform(4.0, 5.0)
            self._place((xc + rng.uniform(-0.5, 0.5), y0, rz), rng.uniform(-0.05, 0.05))
            alt = 2 * CAR_SIZE[2] + 6.0  # room for the barrel roll above the parked car
            g = np.array([xc, HOP_Y + rng.uniform(4.0, 5.0), rz])
            self._route = [
                (np.array([xc, y0, alt]), np.zeros(3), 1.0),
                (np.array([xc, HOP_Y, alt]), np.array([0.0, 3.0, 0.0]), 1.0),
                (g, np.zeros(3), 1.0),
            ]
            self._route_touch = [False, False, True]  # the goal is a touchdown waypoint
            return g, 0.0, np.zeros(3)
        if self.task == "traffic":
            # stuck behind a jam that blocks both lanes (barriers on both sides): lift off, hop it, land
            s.region_y, s.allow_ground_landing = TRAFFIC_Y, True
            s.status, s.airborne_once, s.target_idx = "drive", False, len(c.rings)
            self.jam = jam = place_traffic(self.model, self.data, rng)
            x0 = jam["x_start"] - rng.uniform(8, 18)
            y0 = TRAFFIC_Y + rng.uniform(-2.5, 2.5)
            self._place((x0, y0, rz), rng.uniform(-0.05, 0.05))
            alt = jam["top"] + 4.0
            g = np.array([jam["x_end"] + rng.uniform(12, 22), TRAFFIC_Y + rng.uniform(-2.5, 2.5), rz])
            self._route = [
                (np.array([x0 + 2.0, y0, alt]), np.zeros(3), 1.0),
                (np.array([jam["x_end"] + 3.0, g[1], alt]), np.array([5.0, 0, 0]), 1.0),
                (g, np.zeros(3), 1.0),
            ]
            self._route_touch = [False] * len(self._route)
            return g, 0.0, np.zeros(3)
        if self.task == "rooftop":
            # parked on a tower roof; a taller tower blocks the straight line to the helipad roof
            s.region_y = CITY_Y
            s.status, s.airborne_once, s.target_idx = "drive", False, len(c.rings)
            (_, sx, sy, _, _, sh), (_, gx, gy, _, _, gh), (_, bx, by, _, bhy, _) = TOWERS
            p0 = np.array([sx - 5 + rng.uniform(-2, 2), CITY_Y + sy + rng.uniform(-4, 4), sh + rz])
            self._place(p0, rng.uniform(-0.3, 0.3))
            cruise = gh + 8.0
            side = -1.0 if rng.random() < 0.5 else 1.0  # go around the blocker on either side
            g = np.array([gx, CITY_Y + gy, gh + rz])
            self._route = [
                (np.array([p0[0] + 3, p0[1], cruise]), np.zeros(3), 1.0),
                (np.array([bx, CITY_Y + by + side * (bhy + 9.0), cruise]), np.array([6.0, 0, 0]), 1.0),
                (np.array([gx, CITY_Y + gy, cruise]), np.zeros(3), 1.0),
                (g, np.zeros(3), 1.0),
            ]
            self._route_touch = [False] * len(self._route)
            return g, 0.0, np.zeros(3)
        self._route_touch = []
        if self.task == "loop":
            g = self._loop_goal_state()  # reference run first: it moves the car
            x0 = rng.uniform(-70, -35)
            self._place((x0, rng.uniform(-0.5, 0.5), self._rest_qpos[2]), 0.0, (rng.uniform(4, 16), 0, 0))
            s.status, s.airborne_once, s.target_idx = "drive", False, 0
            return g["pos"], g["yaw"], g["vel"]
        # mission: the full course from the start line
        self._place((c.start_x, 0.0, self._rest_qpos[2]))
        s.status, s.airborne_once, s.target_idx = "drive", False, 0
        return c.pad_top + np.array([0, 0, self._rest_qpos[2]]), 0.0, np.zeros(3)

    def _loop_goal_state(self) -> dict:
        """Upright, airborne and heading back past the loop.

        Taken from a full-throttle run in calm weather (computed once per env), with the state at the top of
        the loop as the intermediate subgoal.
        """
        if self._loop_goal is None:
            weather = self.sim.weather.get_state()
            self.sim.weather.set()
            self._place((-40.0, 0.0, self._rest_qpos[2]), 0.0, (20.0, 0, 0))
            self.sim.status, self.sim.airborne_once, self.sim.target_idx = "drive", False, 0
            self.ap.reset()
            self._loop_top_seen = False
            self._ground_hold = 3
            top = None
            for _ in range(80):
                self._step_raw(np.array([1.0, 0, 0.3 * self._loop_top_seen, 0]))
                if self._loop_top_seen and top is None:
                    top = self.sim.pos.copy(), self.sim.vel.copy()
                if self._loop_done():
                    break
            assert self._loop_done(), "reference loop run failed"
            self._loop_goal = {
                "pos": self.sim.pos.copy(),
                "yaw": self.ap.current_yaw(),
                "vel": self.sim.vel.copy(),
                "top_pos": top[0],
                "top_vel": top[1],
            }
            self.sim.weather.set_state(weather)
            self.sim.clear_fx()  # no smoke from the reference run in the episode
        return self._loop_goal

    def _loop_done(self) -> bool:
        s = self.sim
        return (
            self._loop_top_seen
            and s.rot[2, 2] > 0.9
            and s.pos[0] < -3
            and s.pos[2] > 5
            and not self.sim.contacts()[0]["any"].any()
        )

    def reset(self, seed: int | None = None, wind: float = 0.0, rain: float = 0.0):
        """Start an episode of the task.

        `seed` draws the start state, the goal, the wind direction and the gusts; `wind` is the mean wind
        speed (m/s) and `rain` the rain strength (0..1).
        """
        rng = np.random.default_rng(seed)
        self.sim.weather.set(seed or 0, wind, rain)
        self.sim.reset()
        self._loop_top_seen = False
        self._ground_hold = 3
        self.goal_pos, self.goal_yaw, self.goal_vel = self._sample_task(rng)
        self._loop_top_seen = False  # the loop-goal reference run may have set it
        self._ground_hold = 3
        self.sim.fuel = FUEL_CAPACITY
        self.ap.reset()
        self.steps = 0
        self._prev_cost = self.cost()

    def _waypoints(self) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """(position, velocity, target up_z) of the remaining subgoals, ending with the goal."""
        goal = (self.goal_pos, self.goal_vel, 1.0)
        if self.task in ROUTE_TASKS:
            return self._route[self._route_idx :]
        if self.task == "loop" and not self._loop_top_seen:
            lg = self._loop_goal
            return [(lg["top_pos"], lg["top_vel"], -1.0), goal]
        if self.task == "mission" and not self.sim.landing_stage:
            rings = [np.asarray(r.center, float) for r in self.sim.course.rings[self.sim.target_idx :]]
            return [(c, np.array([12.0, 0, 0]), 1.0) for c in rings] + [goal]
        return [goal]

    def cost(self) -> float:
        """Distance to the current subgoal (position, velocity, attitude) plus the path through the others.

        Making progress through the subgoals always lowers the cost.
        """
        s = self.sim
        wps = self._waypoints()
        p, v, up = wps[0]
        c = np.linalg.norm(s.pos - p) + 0.3 * np.linalg.norm(s.vel - v) + 5.0 * abs(up - s.rot[2, 2])
        if len(wps) == 1 and up > 0:
            yaw_err = np.angle(np.exp(1j * (self.ap.current_yaw() - self.goal_yaw)))
            c += 2.0 * abs(yaw_err) * (s.rot[2, 2] > 0)
        for (a, _, _), (b, _, _) in zip(wps[:-1], wps[1:]):
            c += np.linalg.norm(b - a)
        return float(c)

    def _barrel_roll(self, n_wheels: int):
        """Car hop stunt: a full barrel roll toward the landing side.

        Once the car is airborne high over the lane, the autopilot rolls it a full turn (smooth 0 -> -2 pi
        profile over ROLL_TIME).
        """
        s = self.sim
        t = s.data.time
        if self._roll_t0 is None and not self._roll_done:
            if n_wheels == 0 and s.pos[2] > 2 * CAR_SIZE[2] + 4.5 and s.pos[1] > HOP_Y - 2.5:
                self._roll_t0 = t
        if self._roll_t0 is not None and not self._roll_done:
            u = min((t - self._roll_t0) / ROLL_TIME, 1.0)
            self.ap.roll_cmd = -2 * np.pi * (u - np.sin(2 * np.pi * u) / (2 * np.pi))
            if u >= 1.0:
                self._roll_done, self.ap.roll_cmd = True, np.nan

    def _step_raw(self, a: np.ndarray):
        term = False
        for _ in range(REPEAT):
            n = int(self.sim.contacts()[0]["any"].sum())
            # hysteresis: wheels bounce off the loop surface for a frame or two at speed
            self._ground_hold = 3 if n >= 1 else max(self._ground_hold - 1, 0)
            n_wheels = 4 if self._ground_hold > 0 else 0
            if self.task == "carhop":
                self._barrel_roll(n_wheels)
            term = self.sim.step(self.ap(a, n_wheels))
            if self.sim.pos[2] > self.sim.course.loop_top - 3 and self.sim.rot[2, 2] < -0.5 and self.task == "loop":
                self._loop_top_seen = True
            if self._route and self._route_idx < len(self._route) - 1:
                wp = self._route[self._route_idx][0]
                if self._route_touch[self._route_idx]:
                    # touchdown waypoint: wheels on the ground close to it
                    if n >= 3 and np.linalg.norm(self.sim.pos[:2] - wp[:2]) < 3.0:
                        self._route_idx += 1
                elif np.linalg.norm(self.sim.pos - wp) < ROUTE_RADIUS:
                    self._route_idx += 1
            if term:
                break
        return term

    def success(self) -> bool:
        s = self.sim
        if self.task == "loop":
            return self._loop_done()
        if self.task in ROUTE_TASKS:
            wheels, _ = s.contacts()
            if self.task == "traffic":
                parked = wheels["ground"].sum() >= 3 and s.pos[0] > self.jam["x_end"] + 2
            elif self.task == "carhop":
                parked = wheels["ground"].sum() >= 3 and s.pos[1] > HOP_Y + 2.5 and self._roll_done
            else:
                parked = self._wheels_on(self._roof_goal) >= 3
            return bool(
                parked
                and np.linalg.norm(s.vel) < 1.0
                and s.rot[2, 2] > 0.9
                and self._route_idx == len(self._route) - 1
            )
        return s.status == "success"  # mission: parked on the pad

    def _wheels_on(self, geom: int) -> int:
        d = self.data
        g = d.contact.geom[: d.ncon]
        tires = self.sim.tire_ids
        hit = (np.isin(g[:, 0], tires) & (g[:, 1] == geom)) | (np.isin(g[:, 1], tires) & (g[:, 0] == geom))
        return int(len(set(np.where(np.isin(g[hit, 0], tires), g[hit, 0], g[hit, 1]).tolist())))

    def failed(self) -> bool:
        s = self.sim
        if s.status.startswith("crash"):
            return True
        return s.pos[2] > 75

    def step(self, action):
        """One 10 Hz action.

        Returns ``(reward, terminated, truncated, info)`` with info ``success`` / ``failed`` / ``cost`` /
        ``status``.
        """
        self._step_raw(np.asarray(action, float))
        self.steps += 1
        c = self.cost()
        reward = self._prev_cost - c
        self._prev_cost = c
        succ, fail = self.success(), self.failed()
        reward += 10.0 * succ - 10.0 * fail
        truncated = self.steps >= HORIZON[self.task] and not (succ or fail)
        info = {"success": bool(succ), "failed": bool(fail), "cost": c, "status": self.sim.status}
        return float(reward), bool(succ or fail), truncated, info
