"""What the batched Roadster backends share: the physics model, its indices and the episode sampler.

The physics model is the source's world (``source.model.build_xml``) plus MuJoCo contact sensors, one per
contact question the source asks of ``data.contact``: which tyres touch anything, the pad, the ground or the
goal roof, whether a hull geom touches anything, and whether the car touches each obstacle. Sensors do not
act on the dynamics, so the model steps as the source's. Episodes are sampled per environment with the
source's own random draws (``source.tasks.RoadsterPlanEnv.reset``), without MuJoCo calls.
"""

from __future__ import annotations

import functools
import threading
from types import SimpleNamespace

import mujoco
import numpy as np

from ..source import model as source_model
from ..source.course import Course
from ..source.model import JETS, WHEELS
from ..source.scenarios import CAR_SIZE, CITY_Y, HOP_Y, TOWERS, TRAFFIC_Y, place_hop, place_traffic
from ..source.sim import FRAME_SKIP
from ..source.tasks import TASKS, RoadsterPlanEnv
from ..source.textures import assets
from .adapter import MAX_ROUTE, STATUSES

#: Contact sensor groups, in sensordata order: (name, count).
SENSOR_GROUPS = (
    ("tyre_any", 4),
    ("tyre_pad", 4),
    ("tyre_ground", 4),
    ("tyre_roof", 4),
    ("hull", 2),
)
STATUS = {name: k for k, name in enumerate(STATUSES)}
CRASHES = ("crash: hit obstacle", "crash: body impact", "crash: ditched", "crash: out of bounds")
_LOCK = threading.Lock()


def _sensor_xml(model: mujoco.MjModel) -> str:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    rows = []
    for w in WHEELS:
        rows.append(f'<contact geom1="tire_{w}" num="1" data="found"/>')
    for other in ("pad", "ground", "roof_goal"):
        for w in WHEELS:
            rows.append(f'<contact geom1="tire_{w}" geom2="{other}" num="1" data="found"/>')
    for hull in ("chassis_col", "roof_col"):
        rows.append(f'<contact geom1="{hull}" num="1" data="found"/>')
    for name in names:
        if name.startswith("obs_"):
            rows.append(f'<contact subtree1="chassis" geom2="{name}" num="1" data="found"/>')
    return "<sensor>\n" + "\n".join(rows) + "\n</sensor>\n"


@functools.cache
def physics_model() -> mujoco.MjModel:
    """The source's world with the contact sensors, without the visual geoms outside the car; compiled once.

    Geoms that collide with nothing and do not belong to the car's bodies (scenery decor, rain, smoke) only
    cost kinematics; the car's bodies keep every geom, so their masses and inertias are the source's.
    """
    xml = source_model.build_xml(Course())
    files = assets()
    plain = mujoco.MjModel.from_xml_string(xml, files)
    end = xml.rindex("</mujoco>")
    spec = mujoco.MjSpec.from_string(xml[:end] + _sensor_xml(plain) + xml[end:], files)
    car = plain.body_rootid[mujoco.mj_name2id(plain, mujoco.mjtObj.mjOBJ_BODY, "chassis")]
    car_bodies = {
        mujoco.mj_id2name(plain, mujoco.mjtObj.mjOBJ_BODY, b)
        for b in range(plain.nbody)
        if plain.body_rootid[b] == car
    }
    for geom in list(spec.geoms):
        if geom.contype == 0 and geom.conaffinity == 0 and geom.parent.name not in car_bodies:
            spec.delete(geom)
    spec.assets = files
    return spec.compile()


class Index:
    """Ids and constants of the physics model that the task logic reads."""

    def __init__(self, model: mujoco.MjModel):
        aid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        gid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.drive = np.array([aid(f"drive_{w}") for w in WHEELS])
        self.steer = np.array([aid("steer_fl"), aid("steer_fr")])
        self.jets = np.array([aid(j.name) for j in JETS])
        self.jet_force = np.array([j.max_force for j in JETS])
        self.tires = np.array([gid(f"tire_{w}") for w in WHEELS])
        self.mass = float(model.body_subtreemass[self.chassis])
        self.nobs = sum(
            (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("obs_")
            for i in range(model.ngeom)
        )
        offset, self.sensors = 0, {}
        for name, count in (*SENSOR_GROUPS, ("obstacle", self.nobs)):
            self.sensors[name] = slice(offset, offset + count)
            offset += count
        if offset != model.nsensordata:
            raise RuntimeError("Roadster contact sensors do not match the physics model")
        self.timestep = float(model.opt.timestep)
        self.dt = model.opt.timestep * FRAME_SKIP  # the source's RoadsterSim.dt
        self.course = Course()
        self.ring_x = np.array([r.center[0] for r in self.course.rings])
        self.rings = np.array([r.center for r in self.course.rings], dtype=float)
        self.nrings = len(self.course.rings)
        self.loop_top = self.course.loop_top
        self.loop_radius = self.course.loop_radius
        self.hull_z = 2 * CAR_SIZE[2]
        self.hop_y = HOP_Y


@functools.cache
def allocation_matrix() -> np.ndarray:
    """The autopilot's normalised 6x9 jet wrench matrix (``Autopilot._alloc_matrix``)."""
    return reference("loop").ap._alloc_matrix().copy()


@functools.cache
def allocation_table() -> np.ndarray:
    """``(512, 9, 6)``: for each set of free jets (bit k: jet k free), the minimum-norm least-squares solve.

    ``table[mask] @ r`` is ``lstsq(B[:, free], r)`` scattered to the free jets, zero elsewhere.
    """
    B = allocation_matrix()
    n = B.shape[1]
    table = np.zeros((1 << n, n, B.shape[0]))
    for mask in range(1, 1 << n):
        free = np.array([(mask >> k) & 1 for k in range(n)], bool)
        table[mask][free] = np.linalg.pinv(B[:, free])
    return table


_REFERENCES: dict[str, RoadsterPlanEnv] = {}


def reference(task: str) -> RoadsterPlanEnv:
    """A source env of ``task`` that holds the settled pose and the loop goal; built once per process."""
    with _LOCK:
        env = _REFERENCES.get(task)
        if env is None:
            env = RoadsterPlanEnv(task=task)
            if task == "loop":
                env._loop_goal_state()
            _REFERENCES[task] = env
        return env


def _quat_from_yaw(psi: float) -> np.ndarray:
    return np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])


class Episode:
    """One environment's reset: the source's draws for the task, as plain arrays (no MuJoCo calls).

    The fields follow ``RoadsterPlanEnv.reset`` and ``Weather.set``: the start pose and velocity, the mocap
    cars, the goal, the route, the jam and the weather.
    """

    def __init__(self, task: str, seed: int | None, wind: float, rain: float, base: SimpleNamespace):
        rng = np.random.default_rng(seed)
        self.weather(seed or 0, wind, rain)
        self.mocap_pos, self.mocap_quat = base.mocap_pos.copy(), base.mocap_quat.copy()
        self.qpos, self.qvel = base.qpos.copy(), np.zeros(base.nv)
        self.region_y, self.allow_ground_landing = 0.0, False
        self.route = np.zeros((MAX_ROUTE, 3))
        self.route_vel = np.zeros((MAX_ROUTE, 3))
        self.route_up = np.ones(MAX_ROUTE)
        self.route_touch = np.zeros(MAX_ROUTE, bool)
        self.route_count = 0
        self.jam = None
        self.target_idx, self.status, self.airborne_once = 0, "flight", True
        rest = base.rest_qpos
        rz = rest[2]
        data = SimpleNamespace(mocap_pos=self.mocap_pos, mocap_quat=self.mocap_quat)
        if task == "carhop":
            self.region_y, self.allow_ground_landing = HOP_Y, True
            self.status, self.airborne_once, self.target_idx = "drive", False, base.nrings
            cars = place_hop(base.model, data, rng)
            k = int(rng.integers(2, len(cars) - 2))
            xc = (cars[k][0] + cars[k][1]) / 2
            y0 = HOP_Y - rng.uniform(4.0, 5.0)
            self.place(rest, (xc + rng.uniform(-0.5, 0.5), y0, rz), rng.uniform(-0.05, 0.05))
            alt = 2 * CAR_SIZE[2] + 6.0
            g = np.array([xc, HOP_Y + rng.uniform(4.0, 5.0), rz])
            self.set_route(
                [
                    (np.array([xc, y0, alt]), np.zeros(3), 1.0),
                    (np.array([xc, HOP_Y, alt]), np.array([0.0, 3.0, 0.0]), 1.0),
                    (g, np.zeros(3), 1.0),
                ],
                [False, False, True],
            )
            self.goal = (g, 0.0, np.zeros(3))
        elif task == "traffic":
            self.region_y, self.allow_ground_landing = TRAFFIC_Y, True
            self.status, self.airborne_once, self.target_idx = "drive", False, base.nrings
            self.jam = jam = place_traffic(base.model, data, rng)
            x0 = jam["x_start"] - rng.uniform(8, 18)
            y0 = TRAFFIC_Y + rng.uniform(-2.5, 2.5)
            self.place(rest, (x0, y0, rz), rng.uniform(-0.05, 0.05))
            alt = jam["top"] + 4.0
            g = np.array([jam["x_end"] + rng.uniform(12, 22), TRAFFIC_Y + rng.uniform(-2.5, 2.5), rz])
            self.set_route(
                [
                    (np.array([x0 + 2.0, y0, alt]), np.zeros(3), 1.0),
                    (np.array([jam["x_end"] + 3.0, g[1], alt]), np.array([5.0, 0, 0]), 1.0),
                    (g, np.zeros(3), 1.0),
                ],
                [False] * 3,
            )
            self.goal = (g, 0.0, np.zeros(3))
        elif task == "rooftop":
            self.region_y = CITY_Y
            self.status, self.airborne_once, self.target_idx = "drive", False, base.nrings
            (_, sx, sy, _, _, sh), (_, gx, gy, _, _, gh), (_, bx, by, _, bhy, _) = TOWERS
            p0 = np.array([sx - 5 + rng.uniform(-2, 2), CITY_Y + sy + rng.uniform(-4, 4), sh + rz])
            self.place(rest, p0, rng.uniform(-0.3, 0.3))
            cruise = gh + 8.0
            side = -1.0 if rng.random() < 0.5 else 1.0
            g = np.array([gx, CITY_Y + gy, gh + rz])
            self.set_route(
                [
                    (np.array([p0[0] + 3, p0[1], cruise]), np.zeros(3), 1.0),
                    (np.array([bx, CITY_Y + by + side * (bhy + 9.0), cruise]), np.array([6.0, 0, 0]), 1.0),
                    (np.array([gx, CITY_Y + gy, cruise]), np.zeros(3), 1.0),
                    (g, np.zeros(3), 1.0),
                ],
                [False] * 4,
            )
            self.goal = (g, 0.0, np.zeros(3))
        elif task == "loop":
            x0 = rng.uniform(-70, -35)
            self.place(rest, (x0, rng.uniform(-0.5, 0.5), rest[2]), 0.0, (rng.uniform(4, 16), 0, 0))
            self.status, self.airborne_once, self.target_idx = "drive", False, 0
            lg = base.loop_goal
            self.goal = (lg["pos"], lg["yaw"], lg["vel"])
        else:
            self.place(rest, (base.start_x, 0.0, rest[2]))
            self.status, self.airborne_once, self.target_idx = "drive", False, 0
            self.goal = (base.pad_top + np.array([0, 0, rest[2]]), 0.0, np.zeros(3))

    def weather(self, seed: int, wind: float, rain: float) -> None:
        """``Weather.set`` for a mean wind speed: grip, mean wind, gust size and the gust RNG."""
        self.seed = int(seed)
        self.rain = float(np.clip(rain, 0, 1))
        self.grip = 1.0 - 0.45 * self.rain if self.rain > 0 else 1.0
        ang = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        self.gust_std = 0.3 * float(wind)
        self.wind_mean = float(wind) * np.array([np.cos(ang), np.sin(ang), 0.0])
        self.wind_rng = np.random.default_rng(seed + 1)

    def place(self, rest, pos, yaw=0.0, vel=(0, 0, 0)) -> None:
        self.qpos[:] = rest
        self.qpos[:3] = pos
        self.qpos[3:7] = _quat_from_yaw(yaw)
        self.qvel[:] = 0.0
        self.qvel[:3] = vel

    def set_route(self, route, touch) -> None:
        self.route_count = len(route)
        for k, (p, v, up) in enumerate(route):
            self.route[k], self.route_vel[k], self.route_up[k] = p, v, up
        self.route_touch[: len(touch)] = touch


@functools.cache
def episode_base(task: str) -> SimpleNamespace:
    """What every reset of ``task`` starts from: the reset MjData's mocap bodies and the settled pose."""
    src = reference(task)
    model = physics_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    course = src.sim.course
    return SimpleNamespace(
        model=model,
        nv=model.nv,
        qpos=data.qpos.copy(),
        mocap_pos=data.mocap_pos.copy(),
        mocap_quat=data.mocap_quat.copy(),
        rest_qpos=src._rest_qpos.copy(),
        loop_goal=src._loop_goal if task == "loop" else None,
        nrings=len(course.rings),
        start_x=course.start_x,
        pad_top=course.pad_top,
    )


def check_task(task: str) -> None:
    if task not in TASKS:
        raise ValueError(f"Unsupported Roadster task {task!r}; known: {TASKS}")
