"""The Roadster planning tasks as a batch of CPU simulators with complete continuation State.

Each lane is one ``source.RoadsterPlanEnv`` (MuJoCo, 10 Hz autopilot action). State is the env's own
``get_state`` snapshot laid out as typed columns: MuJoCo's integration state split into its
components (time, qpos, qvel, warm start, ctrl, applied forces, mocap bodies), the pose the derived
quantities refer to after a step, the autopilot, the task bookkeeping, the goal and route, the traffic
jam, the weather with its gust RNG, the exhaust history the smoke is drawn from, and the elapsed steps.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.physics import PhysicsLayout, state_key
from ...._execution.protocol import SimulatorAdapter
from ..source.fx import ENTRY_DIM, HISTORY
from ..source.tasks import TASKS, RoadsterPlanEnv

#: Episode status strings of the source env; ``("task", "status")`` indexes this tuple (``("task", "id")``
#: indexes the source's ``TASKS``).
STATUSES = (
    "drive",
    "flight",
    "landed",
    "success",
    "crash: hit obstacle",
    "crash: body impact",
    "crash: ditched",
    "crash: out of bounds",
)
#: Longest route a task defines (rooftop: four waypoints).
MAX_ROUTE = 4
#: The weather seed travels as two 24-bit halves, each exact in a float32 render row.
SEED_BITS = 24
SEED_MASK = (1 << SEED_BITS) - 1
MAX_SEED = (1 << 2 * SEED_BITS) - 1
#: Bounds of the ``weather.wind`` leaf (mean speed, m/s) and the ``weather.rain`` leaf (strength).
WIND_MAX = 20.0
RAIN_MAX = 1.0
#: The physics Variation leaves, their row layout and the source's defaults (calm, dry).
PHYSICS_LEAVES = ("weather.wind", "weather.rain")
PHYSICS_LAYOUT = PhysicsLayout(leaves=PHYSICS_LEAVES, widths=(1, 1), offsets=(0, 1), nominal=(0.0, 0.0))
MAX_TARGETS = 4  # three rings and the pad: ("task", "target_idx") is the next one
#: Exhaust history: the source keeps the last FX_HISTORY steps of (step, time, pose, velocity, jet commands,
#: wind), FX_ENTRY values each, and draws the smoke from them.
FX_HISTORY, FX_ENTRY = HISTORY, ENTRY_DIM

#: MuJoCo integration-state components in the order ``mj_getState`` writes them.
_COMPONENTS = (
    ("time", "mjSTATE_TIME"),
    ("qpos", "mjSTATE_QPOS"),
    ("qvel", "mjSTATE_QVEL"),
    ("act", "mjSTATE_ACT"),
    ("warmstart", "mjSTATE_WARMSTART"),
    ("ctrl", "mjSTATE_CTRL"),
    ("qfrc_applied", "mjSTATE_QFRC_APPLIED"),
    ("xfrc_applied", "mjSTATE_XFRC_APPLIED"),
    ("eq_active", "mjSTATE_EQ_ACTIVE"),
    ("mocap_pos", "mjSTATE_MOCAP_POS"),
    ("mocap_quat", "mjSTATE_MOCAP_QUAT"),
    ("userdata", "mjSTATE_USERDATA"),
    ("plugin", "mjSTATE_PLUGIN"),
)

_NUMPY_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}

#: State fields that only reset, restore and the process methods change; a step leaves them as they are,
#: so a process batch keeps them cached (``process_static_leaves``).
STATIC_FIELDS = (
    ("sim", "mocap_pos"),  # the traffic and parked cars, placed at reset
    ("sim", "mocap_quat"),
    ("task", "id"),
    ("task", "region_y"),
    ("task", "allow_ground_landing"),
    ("goal", "position"),
    ("goal", "yaw"),
    ("goal", "velocity"),
    ("route", "count"),
    ("route", "positions"),
    ("route", "velocities"),
    ("route", "up"),
    ("route", "touch"),
    ("jam", "present"),
    ("jam", "extent"),
    ("weather", "seed_hi"),
    ("weather", "seed_lo"),
    ("weather", "rain"),
    ("weather", "wind_mean"),
    ("weather", "gust_std"),
    ("weather", "grip"),
    *(state_key(leaf) for leaf in PHYSICS_LEAVES),
)
#: State fields a render row is packed from (``render.row_layout``): what a frame needs and nothing more.
RENDER_FIELDS = (
    ("sim", "time"),
    ("sim", "qpos"),
    ("sim", "qpos_pre"),
    ("sim", "has_qpos_pre"),
    ("sim", "ctrl"),
    ("sim", "mocap_pos"),
    ("sim", "mocap_quat"),
    ("weather", "seed_hi"),
    ("weather", "seed_lo"),
    ("weather", "rain"),
    ("weather", "wind_mean"),
    ("fx", "count"),
    ("fx", "history"),
)


def integration_layout(model) -> dict[str, tuple[int, int]]:
    """``{component: (offset, width)}`` of the nonempty components of the integration state vector."""
    import mujoco

    out, offset = {}, 0
    for name, flag in _COMPONENTS:
        width = int(mujoco.mj_stateSize(model, getattr(mujoco.mjtState, flag)))
        if width:
            out[name] = (offset, width)
            offset += width
    assert offset == int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    return out


def state_fields(model) -> dict[tuple[str, str], tuple[int, torch.dtype]]:
    """The State schema for ``model``: ``{(group, name): (width, dtype)}``."""
    nq = int(model.nq)
    f64, i64, b = torch.float64, torch.int64, torch.bool
    fields: dict = {("sim", name): (width, f64) for name, (_, width) in integration_layout(model).items()}
    fields.update(
        {
            # Pose that xpos/xmat/contacts refer to after a step (one substep behind qpos), when it differs.
            ("sim", "qpos_pre"): (nq, f64),
            ("sim", "has_qpos_pre"): (1, b),
            ("autopilot", "yaw_cmd"): (1, f64),
            # The barrel-roll target; NaN upstream means "hold level", carried here as has_roll_cmd=False.
            ("autopilot", "roll_cmd"): (1, f64),
            ("autopilot", "has_roll_cmd"): (1, b),
            ("task", "id"): (1, i64),
            ("task", "fuel"): (1, f64),
            ("task", "target_idx"): (1, i64),
            ("task", "airborne_once"): (1, b),
            ("task", "status"): (1, i64),
            ("task", "sim_steps"): (1, i64),
            ("task", "prev_cost"): (1, f64),
            ("task", "loop_top_seen"): (1, b),
            ("task", "ground_hold"): (1, i64),
            ("task", "route_idx"): (1, i64),
            ("task", "roll_t0"): (1, f64),
            ("task", "has_roll_t0"): (1, b),
            ("task", "roll_done"): (1, b),
            ("task", "region_y"): (1, f64),
            ("task", "allow_ground_landing"): (1, b),
            ("goal", "position"): (3, f64),
            ("goal", "yaw"): (1, f64),
            ("goal", "velocity"): (3, f64),
            ("route", "count"): (1, i64),
            ("route", "positions"): (3 * MAX_ROUTE, f64),
            ("route", "velocities"): (3 * MAX_ROUTE, f64),
            ("route", "up"): (MAX_ROUTE, f64),
            ("route", "touch"): (MAX_ROUTE, b),  # touchdown waypoints
            ("jam", "present"): (1, b),
            ("jam", "extent"): (3, f64),  # x_start, x_end, top
            # The weather seed in two 24-bit halves, each exact in a float32 render row (seed < 2**48).
            ("weather", "seed_hi"): (1, i64),
            ("weather", "seed_lo"): (1, i64),
            ("weather", "rain"): (1, f64),
            ("weather", "wind_mean"): (3, f64),
            ("weather", "gust"): (3, f64),
            ("weather", "gust_std"): (1, f64),
            ("weather", "grip"): (1, f64),
            ("weather", "rng_state"): (4, i64),
            ("weather", "rng_increment"): (4, i64),
            ("weather", "rng_has_uint32"): (1, i64),
            ("weather", "rng_uinteger"): (1, i64),
            ("fx", "count"): (1, i64),
            ("fx", "history"): (FX_HISTORY * FX_ENTRY, f64),
            ("lifecycle", "elapsed_steps"): (1, i64),
            # The physics leaves the episode runs under (installed at its reset; float64, exact).
            **{state_key(leaf): (1, f64) for leaf in PHYSICS_LEAVES},
        }
    )
    return fields


def _limbs(value: int) -> np.ndarray:
    """Split a PCG64 unsigned 128-bit integer into four uint32 limbs, least significant first."""
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], dtype=np.int64)


def _from_limbs(value) -> int:
    parts = np.asarray(value, dtype=np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(parts))


def physics_rows(values, num_envs: int) -> np.ndarray:
    """Return ``values`` as owned ``(num_envs, 2)`` float64 ``(wind speed, rain)`` rows, or raise."""
    rows = np.array(values, np.float64)
    if rows.shape != (num_envs, 2) or not np.isfinite(rows).all():
        raise ValueError(f"Roadster physics rows must be finite with shape ({num_envs}, 2)")
    if (rows < 0).any() or (rows[:, 0] > WIND_MAX).any() or (rows[:, 1] > RAIN_MAX).any():
        raise ValueError(f"Roadster weather.wind lies in [0, {WIND_MAX}] and weather.rain in [0, {RAIN_MAX}]")
    return rows


def host_actions(actions, num_envs: int) -> np.ndarray:
    """Return ``actions`` as a finite host array, rejecting any shape other than ``(num_envs, 4)``."""
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions)
    if actions.shape != (num_envs, 4):
        raise ValueError(f"Roadster actions require shape ({num_envs}, 4), got {actions.shape}")
    if actions.dtype.kind != "f":
        raise TypeError(f"Roadster actions must be floating point, got {actions.dtype}")
    if not np.isfinite(actions).all():
        raise ValueError("Roadster actions must contain only finite values")
    return actions


class RoadsterAdapter(SimulatorAdapter):
    """``num_envs`` independent ``RoadsterPlanEnv`` instances on one task, with no rendering responsibility.

    The lanes share one compiled MuJoCo model of the world (the tasks live in separate regions of it); each
    writes its own tyre grip into it before it steps. Actions are clipped to [-1, 1] by the source
    autopilot. Wind and rain are installed with ``configure_physics`` and apply at the next ``reset``, as the
    source sets the weather at reset. ``simulation.process.RoadsterProcessAdapter`` runs these lanes in
    worker processes.
    """

    #: Run on every worker of a process batch, each with its own rows.
    process_methods = {"configure_physics": "per_lane"}
    #: Copied by a process batch from its one-lane prototype.
    process_attributes = ("task", "model", "layout", "fields", "physics_layout")
    #: Fields a step leaves unchanged; a process batch keeps them cached and workers write only the rest.
    process_static_leaves = STATIC_FIELDS

    def __init__(self, task: str, num_envs: int, device="cpu"):
        if task not in TASKS:
            raise ValueError(f"Unsupported Roadster task {task!r}; known: {TASKS}")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.task = task
        self.num_envs = num_envs
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Roadster runs its physics on CPU and stores State on CPU or CUDA")
        self.envs: list = []
        try:
            first = RoadsterPlanEnv(task=task)
            if task == "loop":
                first._loop_goal_state()  # computed once, before the other lanes share it
            self.envs.append(first)
            for _ in range(num_envs - 1):
                self.envs.append(RoadsterPlanEnv(task=task, template=first))
        except BaseException:
            self.close()
            raise
        self.model = self.envs[0].model
        self.layout = integration_layout(self.model)
        self.fields = state_fields(self.model)
        self.physics_layout = PHYSICS_LAYOUT
        self._physics = np.zeros((num_envs, 2))  # installed for the next reset: wind speed, rain
        self._episode = np.zeros((num_envs, 2))  # the values the current episode was reset with
        self._closed = False
        self.reset(0)  # every lane in a valid episode, so a step before the first reset is defined

    # ------------------------------------------------------------------ lifecycle
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("RoadsterAdapter is closed")

    def configure_physics(self, values) -> None:
        """Install per-lane ``(wind speed, rain)`` float64 rows for the next reset."""
        self._check_open()
        self._physics = physics_rows(values, self.num_envs)

    def reset(self, seed: int | None = None) -> TensorDict:
        """Reset every lane with ``seed + lane`` (the source seeds the task draw and the gusts with it)."""
        self._check_open()
        if seed is not None and not 0 <= int(seed) <= MAX_SEED - self.num_envs + 1:
            raise ValueError(f"Roadster seeds lie in [0, {MAX_SEED}]; lane i resets with seed + i")
        for i, env in enumerate(self.envs):
            wind, rain = self._physics[i]
            env.reset(seed=None if seed is None else int(seed) + i, wind=float(wind), rain=float(rain))
        self._episode = self._physics.copy()
        return self.get_state()

    def close(self) -> None:
        self.envs = []
        self._closed = True

    # ------------------------------------------------------------------ State
    def _row(self, env) -> dict:
        st = env.get_state()
        values: dict = {}
        phys = st["phys"]
        for name, (off, width) in self.layout.items():
            values[("sim", name)] = phys[off : off + width]
        pre = st["qpos_pre"]
        values[("sim", "qpos_pre")] = (
            phys[self.layout["qpos"][0] : self.layout["qpos"][0] + self.model.nq] if pre is None else pre
        )
        values[("sim", "has_qpos_pre")] = [pre is not None]
        yaw_cmd, roll_cmd = st["ap"]
        values[("autopilot", "yaw_cmd")] = [yaw_cmd]
        values[("autopilot", "roll_cmd")] = [0.0 if np.isnan(roll_cmd) else roll_cmd]
        values[("autopilot", "has_roll_cmd")] = [not np.isnan(roll_cmd)]
        values[("task", "id")] = [TASKS.index(st["task"])]
        for key in (
            "fuel",
            "target_idx",
            "airborne_once",
            "sim_steps",
            "prev_cost",
            "loop_top_seen",
            "ground_hold",
            "route_idx",
            "roll_done",
            "region_y",
            "allow_ground_landing",
        ):
            values[("task", key)] = [st[key]]
        values[("task", "status")] = [STATUSES.index(st["status"])]
        values[("task", "roll_t0")] = [0.0 if st["roll_t0"] is None else st["roll_t0"]]
        values[("task", "has_roll_t0")] = [st["roll_t0"] is not None]
        values[("goal", "position")] = st["goal_pos"]
        values[("goal", "yaw")] = [st["goal_yaw"]]
        values[("goal", "velocity")] = st["goal_vel"]
        route = st["route"]
        positions, velocities = np.zeros((MAX_ROUTE, 3)), np.zeros((MAX_ROUTE, 3))
        up, touch = np.ones(MAX_ROUTE), np.zeros(MAX_ROUTE, bool)
        for k, (p, v, u) in enumerate(route):
            positions[k], velocities[k], up[k] = p, v, u
        touch[: len(st["route_touch"])] = st["route_touch"]  # one entry per waypoint on route tasks, none otherwise
        values[("route", "count")] = [len(route)]
        values[("route", "positions")] = positions.reshape(-1)
        values[("route", "velocities")] = velocities.reshape(-1)
        values[("route", "up")] = up
        values[("route", "touch")] = touch
        jam = st["jam"]
        values[("jam", "present")] = [jam is not None]
        values[("jam", "extent")] = [0.0, 0.0, 0.0] if jam is None else [jam["x_start"], jam["x_end"], jam["top"]]
        w = st["weather"]
        rng = w["rng"]
        if rng["bit_generator"] != "PCG64":
            raise RuntimeError(f"Expected PCG64, got {rng['bit_generator']!r}")
        values[("weather", "seed_hi")] = [w["seed"] >> SEED_BITS]
        values[("weather", "seed_lo")] = [w["seed"] & SEED_MASK]
        values[("weather", "rain")] = [w["rain"]]
        values[("weather", "wind_mean")] = w["wind_mean"]
        values[("weather", "gust")] = w["gust"]
        values[("weather", "gust_std")] = [w["gust_std"]]
        values[("weather", "grip")] = [w["grip"]]
        values[("weather", "rng_state")] = _limbs(rng["state"]["state"])
        values[("weather", "rng_increment")] = _limbs(rng["state"]["inc"])
        values[("weather", "rng_has_uint32")] = [rng["has_uint32"]]
        values[("weather", "rng_uinteger")] = [rng["uinteger"]]
        count, hist = st["fx"]
        values[("fx", "count")] = [count]
        values[("fx", "history")] = hist.reshape(-1)
        values[("lifecycle", "elapsed_steps")] = [st["steps"]]
        return values

    def _columns(self, keys) -> dict:
        """Host ``(num_envs, width)`` arrays of the State fields ``keys`` of the live lanes."""
        keys = tuple(keys)
        columns = {
            key: np.empty((self.num_envs, self.fields[key][0]), _NUMPY_DTYPES[self.fields[key][1]]) for key in keys
        }
        lane_keys = [key for key in keys if key[0] != "physics"]
        for index, env in enumerate(self.envs):
            row = self._row(env)
            for key in lane_keys:
                columns[key][index] = row[key]
        for j, leaf in enumerate(PHYSICS_LEAVES):
            if state_key(leaf) in columns:
                columns[state_key(leaf)][:, 0] = self._episode[:, j]
        return columns

    def _to_device(self, columns: dict) -> dict:
        """The host columns as tensors on ``device``, moved with one copy per dtype rather than one per field."""
        if self.device.type == "cpu":
            return {key: torch.from_numpy(value) for key, value in columns.items()}
        out = {}
        for dtype in {value.dtype for value in columns.values()}:
            keys = [key for key, value in columns.items() if value.dtype == dtype]
            flat = torch.from_numpy(np.concatenate([columns[key].reshape(-1) for key in keys])).to(self.device)
            offset = 0
            for key in keys:
                size = columns[key].size
                out[key] = flat[offset : offset + size].view(columns[key].shape)
                offset += size
        return out

    def get_state(self) -> TensorDict:
        self._check_open()
        tensors = self._to_device(self._columns(self.fields))
        return TensorDict({key: tensors[key] for key in self.fields}, [self.num_envs], device=self.device)

    def process_step_state(self) -> dict:
        """Host arrays of the State fields a step can change (all but ``process_static_leaves``)."""
        self._check_open()
        return self._columns(key for key in self.fields if key not in STATIC_FIELDS)

    def render_fields(self) -> dict:
        """Host arrays of the State fields a render row is packed from, for the live lanes; no snapshot."""
        self._check_open()
        return self._columns(RENDER_FIELDS)

    def validate(self, state, *, batch: int | None = None) -> None:
        """Require the complete, typed resume schema with finite values and consistent task fields.

        Compact render rows and partial TensorDicts fail. Nothing is changed.
        """
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
        have, want = set(state.keys(True, True)), set(self.fields)
        if have != want:
            missing, extra = sorted(want - have, key=str), sorted(have - want, key=str)
            raise ValueError(
                "Full State keys do not match the Roadster resume schema"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        n = int(state.batch_size[0])
        for key, (width, dtype) in self.fields.items():
            value = state[key]
            if value.shape != (n, width):
                raise ValueError(f"State field {key!r} has invalid shape {tuple(value.shape)}")
            if value.dtype != dtype:
                raise ValueError(f"State field {key!r} has dtype {value.dtype}; expected {dtype}")
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"State field {key!r} contains non-finite values")
        host = state.cpu()

        def within(key, lo, hi):
            v = host[key]
            if bool((v < lo).any() or (v > hi).any()):
                raise ValueError(f"State field {key!r} must lie in [{lo}, {hi}]")

        task = TASKS.index(self.task)
        within(("task", "id"), task, task)
        within(("task", "status"), 0, len(STATUSES) - 1)
        within(("task", "target_idx"), 0, MAX_TARGETS)
        within(("task", "ground_hold"), 0, 3)
        within(("task", "fuel"), 0.0, np.inf)
        within(("task", "sim_steps"), 0, np.iinfo(np.int64).max)
        within(("lifecycle", "elapsed_steps"), 0, np.iinfo(np.int64).max)
        within(("route", "count"), 0, MAX_ROUTE)
        count = host["route", "count"].reshape(-1)
        idx = host["task", "route_idx"].reshape(-1)
        if bool((idx < 0).any() or (idx > torch.clamp(count - 1, min=0)).any()):
            raise ValueError("State field ('task', 'route_idx') must index the route")
        within(("weather", "rain"), 0.0, RAIN_MAX)
        within(("weather", "seed_hi"), 0, SEED_MASK)
        within(("weather", "seed_lo"), 0, SEED_MASK)
        within(("weather", "grip"), 0.0, 1.0)
        within(("weather", "gust_std"), 0.0, np.inf)
        within(("weather", "rng_state"), 0, 0xFFFF_FFFF)
        within(("weather", "rng_increment"), 0, 0xFFFF_FFFF)
        within(("weather", "rng_has_uint32"), 0, 1)
        within(("weather", "rng_uinteger"), 0, 0xFFFF_FFFF)
        within(state_key("weather.wind"), 0.0, WIND_MAX)
        within(state_key("weather.rain"), 0.0, RAIN_MAX)
        within(("fx", "count"), 0, FX_HISTORY)
        hist = host["fx", "history"].reshape(n, FX_HISTORY, FX_ENTRY)
        for lane in range(n):
            c = int(host["fx", "count"][lane].item())
            steps = hist[lane, :c, 0]
            if c and (bool((steps < 0).any()) or bool((steps[1:] <= steps[:-1]).any())):
                raise ValueError("State field ('fx', 'history') must list steps in increasing order")

    def _restore(self, env, row) -> None:
        get = lambda key: row[key].numpy()
        phys = np.concatenate([get(("sim", name)) for name in self.layout])
        route_n = int(row["route", "count"].item())
        positions = get(("route", "positions")).reshape(MAX_ROUTE, 3)
        velocities = get(("route", "velocities")).reshape(MAX_ROUTE, 3)
        up, touch = get(("route", "up")), get(("route", "touch"))
        jam_present = bool(row["jam", "present"].item())
        x_start, x_end, top = get(("jam", "extent")).tolist()
        st = {
            "task": TASKS[int(row["task", "id"].item())],
            "phys": phys,
            "qpos_pre": get(("sim", "qpos_pre")).copy() if bool(row["sim", "has_qpos_pre"].item()) else None,
            "ap": np.array(
                [
                    row["autopilot", "yaw_cmd"].item(),
                    row["autopilot", "roll_cmd"].item() if bool(row["autopilot", "has_roll_cmd"].item()) else np.nan,
                ]
            ),
            "fuel": row["task", "fuel"].item(),
            "target_idx": int(row["task", "target_idx"].item()),
            "airborne_once": bool(row["task", "airborne_once"].item()),
            "status": STATUSES[int(row["task", "status"].item())],
            "steps": int(row["lifecycle", "elapsed_steps"].item()),
            "sim_steps": int(row["task", "sim_steps"].item()),
            "prev_cost": row["task", "prev_cost"].item(),
            "loop_top_seen": bool(row["task", "loop_top_seen"].item()),
            "ground_hold": int(row["task", "ground_hold"].item()),
            "route_idx": int(row["task", "route_idx"].item()),
            "roll_t0": row["task", "roll_t0"].item() if bool(row["task", "has_roll_t0"].item()) else None,
            "roll_done": bool(row["task", "roll_done"].item()),
            "region_y": row["task", "region_y"].item(),
            "allow_ground_landing": bool(row["task", "allow_ground_landing"].item()),
            "goal_pos": get(("goal", "position")).copy(),
            "goal_yaw": row["goal", "yaw"].item(),
            "goal_vel": get(("goal", "velocity")).copy(),
            "route": [(positions[k].copy(), velocities[k].copy(), float(up[k])) for k in range(route_n)],
            "route_touch": [bool(t) for t in touch[:route_n]],
            "jam": {"x_start": x_start, "x_end": x_end, "top": top} if jam_present else None,
            "fx": (int(row["fx", "count"].item()), get(("fx", "history")).reshape(FX_HISTORY, FX_ENTRY).copy()),
            "weather": {
                "seed": (int(row["weather", "seed_hi"].item()) << SEED_BITS) | int(row["weather", "seed_lo"].item()),
                "rain": row["weather", "rain"].item(),
                "wind_mean": get(("weather", "wind_mean")).copy(),
                "gust": get(("weather", "gust")).copy(),
                "gust_std": row["weather", "gust_std"].item(),
                "grip": row["weather", "grip"].item(),
                "rng": {
                    "bit_generator": "PCG64",
                    "state": {
                        "state": _from_limbs(get(("weather", "rng_state"))),
                        "inc": _from_limbs(get(("weather", "rng_increment"))),
                    },
                    "has_uint32": int(row["weather", "rng_has_uint32"].item()),
                    "uinteger": int(row["weather", "rng_uinteger"].item()),
                },
            },
        }
        env.set_state(st)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        self.validate(state, batch=self.num_envs)
        host = state.cpu()
        for i, env in enumerate(self.envs):
            self._restore(env, host[i])
        # The restored episode's physics leaves become the installed ones, as a restored classic-control
        # State installs its physics; the next reset returns to the Variation defaults.
        for j, leaf in enumerate(PHYSICS_LEAVES):
            self._physics[:, j] = host[state_key(leaf)].numpy().reshape(-1)
        self._episode = self._physics.copy()

    # ------------------------------------------------------------------ stepping
    @staticmethod
    def _idle_output():
        return 0.0, False, False, {"success": False, "failed": False, "cost": 0.0}

    def check_step(self, actions, active):
        """Return host ``(actions, active)``; reads only ``num_envs``, so a process batch can call it."""
        actions = host_actions(actions, self.num_envs)
        if active is not None:
            active = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active, dtype=bool)
            if active.shape != (self.num_envs,):
                raise ValueError(f"Roadster active mask requires shape ({self.num_envs},), got {active.shape}")
        return actions, active

    def probe_actions(self) -> np.ndarray:
        return np.zeros((1, 4))

    def step(self, actions, *, active=None, capture_state: bool = True) -> TensorDict:
        """Advance one action per lane; an inactive lane holds its state and reports zero outputs.

        ``capture_state=False`` leaves the State snapshot out of the result.
        """
        self._check_open()
        actions, active = self.check_step(actions, active)
        outputs = []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            if active is None or active[i]:
                reward, terminated, truncated, info = env.step(action)
                outputs.append((reward, terminated, truncated, info))
            else:
                outputs.append(self._idle_output())
        dev = self.device
        result = TensorDict(
            {
                "reward": torch.tensor([o[0] for o in outputs], dtype=torch.float64, device=dev),
                "success": torch.tensor([o[3]["success"] for o in outputs], dtype=torch.bool, device=dev),
                "failed": torch.tensor([o[3]["failed"] for o in outputs], dtype=torch.bool, device=dev),
                "terminated": torch.tensor([o[1] for o in outputs], dtype=torch.bool, device=dev),
                "truncated": torch.tensor([o[2] for o in outputs], dtype=torch.bool, device=dev),
                "cost": torch.tensor([o[3]["cost"] for o in outputs], dtype=torch.float64, device=dev),
            },
            batch_size=[self.num_envs],
            device=dev,
        )
        if capture_state:
            result["state"] = self.get_state()
        return result
