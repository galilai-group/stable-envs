"""The Roadster task logic on NumPy arrays of environments, with MuJoCo in one native ``mujoco.batch``.

Each operation of the source (``source.tasks``, ``source.control``, ``source.sim``) runs on stacked arrays
with the same float64 operations in the same order: NumPy dispatches a stack of small matrix products,
dot products and least-squares solves to the same BLAS and LAPACK routines as one product, so every
environment computes the bits the source computes. The physics batch steps each substep with the drag
and wind forces of that substep, as the source does.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.mjbatch import MAX_NATIVE_THREADS, NativeBatch, native_threads
from ..source.control import ACC_SCALE, INERTIA, WHEELBASE, WRENCH_SCALE, YAW_RATE_SCALE, G
from ..source.sim import ACTION_HIGH, ACTION_LOW, AIR_DENSITY, DRAG_CDA, FRAME_SKIP
from ..source.tasks import HORIZON, REPEAT, ROLL_TIME, ROUTE_RADIUS, ROUTE_TASKS
from .base import DATA_NAMES, RoadsterBatch
from .world import STATUS, allocation_matrix

try:  # the gufunc behind np.linalg.lstsq, which also solves stacks of problems
    from numpy.linalg import _umath_linalg
except ImportError:  # pragma: no cover
    _umath_linalg = None

_DRIVE, _FLIGHT, _LANDED, _SUCCESS = (STATUS[s] for s in ("drive", "flight", "landed", "success"))
_CRASH = STATUS["crash: hit obstacle"]
_ARM = np.array([-0.35, 0.0, 0.12])
_AREA = np.array([DRAG_CDA, 2.4, 4.0])
_GUST_SCALE = np.array([1, 1, 0.3])


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise ``a[i] @ b[i]`` of ``(n, k)`` vectors."""
    return np.matmul(a[:, None, :], b[:, :, None])[:, 0, 0]


def _norm(v: np.ndarray) -> np.ndarray:
    """Row-wise ``np.linalg.norm`` of ``(n, k)`` vectors."""
    return np.sqrt(_dot(v, v))


def _mv(m: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise ``m[i] @ v[i]``."""
    return np.matmul(m, v[:, :, None])[:, :, 0]


def _rz(psi: np.ndarray) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    out = np.zeros((len(psi), 3, 3))
    out[:, 0, 0], out[:, 0, 1], out[:, 1, 0], out[:, 1, 1], out[:, 2, 2] = c, -s, s, c, 1.0
    return out


def _rx(phi: np.ndarray) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    out = np.zeros((len(phi), 3, 3))
    out[:, 0, 0], out[:, 1, 1], out[:, 1, 2], out[:, 2, 1], out[:, 2, 2] = 1.0, c, -s, s, c
    return out


def _rot_err(R: np.ndarray, Rd: np.ndarray) -> np.ndarray:
    """``source.util.rot_err`` for each row."""
    E = np.matmul(Rd.transpose(0, 2, 1), R)
    trace = E[:, 0, 0] + E[:, 1, 1] + E[:, 2, 2]
    ang = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    axis = np.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], 1)
    flip = np.flatnonzero(ang > np.pi - 1e-3)
    if flip.size:
        M = E[flip] + np.eye(3)
        col = np.argmax(np.sqrt(np.add.reduce(M * M, axis=1)), axis=1)
        axis[flip] = M[np.arange(len(flip)), :, col]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = axis / _norm(axis)[:, None] * ang[:, None]
    out[ang < 1e-6] = 0.0
    return out


def _lstsq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``np.linalg.lstsq(a[i], b[i], rcond=None)[0]`` for a stack of ``(m, k)`` problems."""
    n, m, k = a.shape
    if _umath_linalg is None:  # pragma: no cover
        return np.stack([np.linalg.lstsq(a[i], b[i], rcond=None)[0] for i in range(n)])
    rcond = np.finfo(np.float64).eps * max(m, k)
    return _umath_linalg.lstsq(a, b[:, :, None], rcond, signature="ddd->ddid")[0][:, :, 0]


def _columns(BT: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """``B[:, cols[i]]`` for each row of ``cols``, as a C-contiguous stack."""
    return np.ascontiguousarray(BT[cols].transpose(0, 2, 1))


#: Batches of at most this many wrenches allocate row by row, which costs fewer NumPy calls.
_ROW_ALLOCATION = 24


def _allocate_row(B: np.ndarray, w: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """``Autopilot._alloc`` of one normalised wrench, as the source computes it."""
    u = np.zeros(B.shape[1])
    free = np.ones(B.shape[1], bool)
    for _ in range(B.shape[1]):
        resid = w - B[:, ~free] @ u[~free]
        u[free] = _lstsq(B[None, :, free], resid[None])[0]
        sat = free & ((u < lo) | (u > hi))
        if not sat.any():
            break
        u = np.clip(u, lo, hi)
        free &= ~sat
        if not free.any():
            break
    return np.clip(u, lo, hi)


def allocate(w: np.ndarray) -> np.ndarray:
    """``Autopilot._alloc`` for each row of the ``(n, 6)`` wrenches: the bounded least-squares jet commands.

    Rows with the same number of free jets solve in one stacked call.
    """
    B = allocation_matrix()
    BT = B.T
    lo, hi = ACTION_LOW[2:], ACTION_HIGH[2:]
    njet = B.shape[1]
    w = w / WRENCH_SCALE
    if len(w) <= _ROW_ALLOCATION:
        return np.stack([_allocate_row(B, row, lo, hi) for row in w]) if len(w) else np.zeros((0, njet))
    u = np.zeros((len(w), njet))
    free = np.ones((len(w), njet), bool)
    going = np.ones(len(w), bool)
    bits = 1 << np.arange(njet)
    for _ in range(njet):
        rows = np.flatnonzero(going)
        if not rows.size:
            break
        nfree = free[rows].sum(1)
        for k in np.unique(nfree):
            g = rows[nfree == k]
            f = free[g]
            resid = w[g]
            if k < njet:
                # B[:, ~free] @ u[~free]: one 2-D matrix per set of fixed jets, as the source multiplies it.
                resid = resid.copy()
                masks = f @ bits
                for mask in np.unique(masks):
                    sel = masks == mask
                    fixed = ~f[np.argmax(sel)]
                    resid[sel] -= np.matmul(B[:, fixed], u[g[sel]][:, fixed][:, :, None])[:, :, 0]
            cols = np.nonzero(f)[1].reshape(len(g), k)
            ug = u[g]
            np.put_along_axis(ug, cols, _lstsq(_columns(BT, cols), resid), 1)
            sat = f & ((ug < lo) | (ug > hi))
            hit = sat.any(1)
            ug[hit] = np.clip(ug[hit], lo, hi)
            u[g] = ug
            left = f & ~sat
            free[g[hit]] = left[hit]
            going[g[~hit]] = False
            going[g[hit & ~left.any(1)]] = False
    return np.clip(u, lo, hi)


class RoadsterNativeAdapter(RoadsterBatch):
    """The Roadster task on CPU: NumPy task logic, MuJoCo on a native ``mujoco.batch``.

    Steps bit for bit as the source env, environment by environment: the physics model is the source's
    with contact sensors and without the visual geoms outside the car, which do not act on the dynamics.
    """

    backend = "native"

    def _build(self) -> None:
        names = tuple(DATA_NAMES.get(n, n) for n in self.layout)
        outputs = ("xmat", "xpos", "xquat", "sensordata", *names)
        n = self.num_envs
        # A substep is short: waking a thread per environment costs more than it saves below 64 environments.
        threads = min(native_threads(n), 1 if n < 4 else 2 if n < 8 else 4 if n < 64 else MAX_NATIVE_THREADS)
        self.physics = p = NativeBatch(self.model, self.num_envs, "cpu", outputs=outputs, nthread=threads)
        self._alloc(torch.device("cpu"))
        self.qpos, self.qvel, self.time = p.numpy("qpos"), p.numpy("qvel"), p.numpy("time")
        self.ctrl, self.xfrc = p.numpy("ctrl"), p.numpy("xfrc_applied")
        self.xmat, self.xpos, self.xquat = p.numpy("xmat"), p.numpy("xpos"), p.numpy("xquat")
        self.sensors = p.numpy("sensordata")
        self.friction = p.model_field("geom_friction").numpy()
        self._tire_friction = self.model.geom_friction[self.index.tires].copy()
        col = lambda g, n: self._col(g, n).numpy()  # noqa: E731
        self.yaw_cmd, self.roll_cmd = col("autopilot", "yaw_cmd"), col("autopilot", "roll_cmd")
        self.has_roll_cmd = col("autopilot", "has_roll_cmd")
        self.fuel, self.target_idx = col("task", "fuel"), col("task", "target_idx")
        self.airborne_once, self.status = col("task", "airborne_once"), col("task", "status")
        self.sim_steps, self.prev_cost = col("task", "sim_steps"), col("task", "prev_cost")
        self.loop_top_seen, self.ground_hold = col("task", "loop_top_seen"), col("task", "ground_hold")
        self.route_idx, self.roll_t0 = col("task", "route_idx"), col("task", "roll_t0")
        self.has_roll_t0, self.roll_done = col("task", "has_roll_t0"), col("task", "roll_done")
        self.region_y, self.allow_ground = col("task", "region_y"), col("task", "allow_ground_landing")
        self.goal_pos, self.goal_yaw, self.goal_vel = (
            col("goal", "position"),
            col("goal", "yaw"),
            col("goal", "velocity"),
        )
        self.route_count = col("route", "count")
        self.route = col("route", "positions").reshape(self.num_envs, -1, 3)
        self.route_vel = col("route", "velocities").reshape(self.num_envs, -1, 3)
        self.route_up, self.route_touch = col("route", "up"), col("route", "touch")
        self.jam_extent = col("jam", "extent")
        self.wind_mean, self.gust = col("weather", "wind_mean"), col("weather", "gust")
        self.gust_std, self.grip = col("weather", "gust_std"), col("weather", "grip")
        self.fx_count, self.elapsed = col("fx", "count"), col("lifecycle", "elapsed_steps")
        self.qpos_pre, self.has_pre = self._s["sim", "qpos_pre"].numpy(), col("sim", "has_qpos_pre")
        self.ring, self.head = self._ring.numpy(), self._head.numpy()
        self.gust_decay = np.exp(-self.index.timestep / 1.5)
        self.gust_gain = np.sqrt(1 - self.gust_decay * self.gust_decay)

    def _sim(self, name: str) -> torch.Tensor:
        return self.physics.data(DATA_NAMES.get(name, name))

    def _close_physics(self) -> None:
        self.physics.close()

    def _apply_grip(self) -> None:
        # Calm everywhere: no gusts and no wind, so drag needs no airspeed.
        self._windy = bool(self.wind_mean.any() or self.gust_std.any() or self.gust.any())
        tires = self.index.tires
        self.friction[:, tires] = (
            self._tire_friction[None]
            * np.stack([self.grip, np.ones_like(self.grip), np.ones_like(self.grip)], -1)[:, None, :]
        )

    def _write_physics(self, values, qpos_pre, has_pre) -> None:
        p = self.physics
        for name in self.layout:
            if name not in values:
                values[name] = torch.zeros_like(self._sim(name))
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            if name in self.layout:
                self._sim(name).copy_(values[name].reshape(self._sim(name).shape))
        has = has_pre.cpu().numpy().astype(bool)
        qpos = values["qpos"].cpu().numpy().reshape(self.qpos.shape).copy()
        state = {DATA_NAMES.get(n, n): values[n] for n in self.layout if DATA_NAMES.get(n, n) in p.fields}
        state["qpos"] = np.where(has[:, None], qpos_pre.cpu().numpy(), qpos)
        # The derived quantities after a step describe the pose before its last substep: forward there.
        p.set_physics(state)
        self.qpos[:] = qpos
        self.qpos_pre[:] = np.where(has[:, None], qpos_pre.cpu().numpy(), qpos)

    def _after_reset(self) -> None:
        ids = np.arange(self.num_envs)
        R = self.xmat[:, self.index.chassis].reshape(-1, 3, 3)
        self.yaw_cmd[:] = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        self.prev_cost[:] = self._cost(ids)

    # ------------------------------------------------------------------ autopilot (source.control)
    def _rot(self, ids) -> np.ndarray:
        return self.xmat[ids, self.index.chassis].reshape(-1, 3, 3)

    def _air_jets(self, ids, R, v, acc_w, yaw) -> np.ndarray:
        drag = (0.5 * AIR_DENSITY * DRAG_CDA * _norm(v))[:, None] * v
        F_b = _mv(R.transpose(0, 2, 1), self.index.mass * (acc_w + np.array([0, 0, G])) + drag)
        Rd = _rz(yaw)
        roll = self.has_roll_cmd[ids]
        if roll.any():
            Rd[roll] = np.matmul(Rd[roll], _rx(self.roll_cmd[ids][roll]))
        level = ~roll & (R[:, 2, 2] > 0.7)
        if level.any():
            a_lat = _mv(Rd[level].transpose(0, 2, 1), acc_w[level])[:, 1]
            Rd[level] = np.matmul(Rd[level], _rx(-np.clip(np.arctan2(a_lat, G + acc_w[level, 2]), -0.4, 0.4)))
        wn, zeta = 5.0, 0.9
        tau = INERTIA * (-(wn**2) * _rot_err(R, Rd) - 2 * zeta * wn * self.qvel[ids, 3:6])
        return allocate(np.concatenate([F_b, tau], 1))

    def _autopilot(self, ids, a, n_wheels) -> np.ndarray:
        """``Autopilot.__call__``: the 4-D command to the 11-D raw action; updates the yaw command."""
        a = np.clip(a, -1.0, 1.0)
        acc = a[:, :3] * ACC_SCALE
        yaw_rate = a[:, 3] * YAW_RATE_SCALE
        raw = np.zeros((len(ids), 11))
        R, v = self._rot(ids), self.qvel[ids, :3]
        current = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        ground = n_wheels >= 2
        v_fwd = _dot(v, R[:, :, 0])
        steer = np.arctan(WHEELBASE * yaw_rate / np.maximum(np.abs(v_fwd), 2.0))
        raw[ground, 0] = acc[ground, 0] / ACC_SCALE[0]
        sign = np.sign(np.where(np.abs(v_fwd) > 0.5, v_fwd, 1.0))
        raw[ground, 1] = (np.clip(steer / 0.5, -1, 1) * sign)[ground]
        steep = ~ground & (R[:, 2, 2] < 0.7) & ~self.has_roll_cmd[ids]
        yaw = np.where(ground | steep, current, self.yaw_cmd[ids] + yaw_rate * self.index.dt)
        self.yaw_cmd[ids] = yaw
        jets = np.flatnonzero(~ground | (acc[:, 2] > 0.5))
        if jets.size:
            n = len(jets)
            planar = np.stack([acc[jets, 0], acc[jets, 1], np.zeros(n)], 1)
            lift = np.stack([np.zeros(n), np.zeros(n), acc[jets, 2]], 1)
            acc_w = _mv(_rz(yaw[jets]), planar) + lift
            raw[jets, 2:] = self._air_jets(ids[jets], R[jets], v[jets], acc_w, yaw[jets])
        return raw

    # ------------------------------------------------------------------ simulator (source.sim)
    def _contacts(self, ids) -> dict[str, np.ndarray]:
        s = self.sensors[ids] > 0
        return {name: s[:, sl] for name, sl in self.index.sensors.items()}

    def _physics_step(self, ids, raw, noise, substep0, everyone) -> np.ndarray:
        """``RoadsterSim.step`` for ``ids``; returns which terminated."""
        ix = self.index
        a = np.clip(raw, ACTION_LOW, ACTION_HIGH)
        self.ctrl[ids[:, None], ix.drive] = a[:, :1]
        self.ctrl[ids[:, None], ix.steer] = 0.5 * a[:, 1:2]
        jets = np.where((self.fuel[ids] > 0)[:, None], a[:, 2:], 0.0)
        self.ctrl[ids[:, None], ix.jets] = jets
        burn = np.sum(np.abs(jets) * ix.jet_force, axis=1) / 1000.0
        prev_x = self.qpos[ids, 0].copy()
        windy_ids = ids[self.gust_std[ids] > 0]
        active = None if everyone else ids
        for k in range(FRAME_SKIP):
            if windy_ids.size:
                draw = noise[windy_ids, substep0 + k] * _GUST_SCALE
                self.gust[windy_ids] = (
                    self.gust_decay * self.gust[windy_ids]
                    + (self.gust_gain * self.gust_std[windy_ids])[:, None] * draw
                )
            self.xfrc[ids, ix.chassis] = self._aero(ids)
            if k == FRAME_SKIP - 1:
                self.qpos_pre[ids] = self.qpos[ids]
            self.physics.step(active=active)
        self.has_pre[ids] = True
        self.sim_steps[ids] += 1
        self.fuel[ids] = np.maximum(self.fuel[ids] - burn * ix.dt, 0.0)
        self._fx_record(ids)

        c = self._contacts(ids)
        pos, vel, R = self.qpos[ids, :3], self.qvel[ids, :3], self._rot(ids)
        airborne = self.airborne_once[ids] | (~c["tyre_any"].any(1) & (pos[:, 2] > 1.5))
        self.airborne_once[ids] = airborne
        status = np.where((self.status[ids] == _DRIVE) & airborne, _FLIGHT, self.status[ids])
        target = self.target_idx[ids]
        ahead = target < ix.nrings
        ring_x = ix.ring_x[np.minimum(target, ix.nrings - 1)]
        target = target + (ahead & (prev_x < ring_x) & (ring_x <= pos[:, 0]))
        self.target_idx[ids] = target
        landed = (target >= ix.nrings) & (c["tyre_pad"].sum(1) >= 3)
        status = np.where(landed, _LANDED, status)
        success = landed & (_norm(vel) < 1.0) & (R[:, 2, 2] > 0.9)
        status = np.where(success, _SUCCESS, status)
        crash = np.select(
            [
                c["obstacle"].any(1),
                c["hull"].any(1),
                airborne & c["tyre_ground"].any(1) & ~self.allow_ground[ids],
                (pos[:, 2] < -2) | (np.abs(pos[:, 1] - self.region_y[ids]) > 150) | (pos[:, 0] > 400),
            ],
            [_CRASH, _CRASH + 1, _CRASH + 2, _CRASH + 3],
            -1,
        )
        self.status[ids] = np.where(crash >= 0, crash, status)
        return success | (crash >= 0)

    def _aero(self, ids) -> np.ndarray:
        """``RoadsterSim._aero`` for each environment: drag, and with wind the airspeed force and torque."""
        v = self.qvel[ids, :3]
        if not self._windy:
            out = np.zeros((len(ids), 6))
            out[:, :3] = (-0.5 * AIR_DENSITY * DRAG_CDA * _norm(v))[:, None] * v
            return out
        w = self.wind_mean[ids] + self.gust[ids]
        out = np.zeros((len(ids), 6))
        windy = w.any(1)
        calm = ~windy
        if calm.any():
            vc = v[calm]
            out[calm, :3] = (-0.5 * AIR_DENSITY * DRAG_CDA * _norm(vc))[:, None] * vc
        if windy.any():
            R = self._rot(ids[windy])
            rel_b = _mv(R.transpose(0, 2, 1), v[windy] - w[windy])
            f_b = (-0.5 * AIR_DENSITY * _norm(rel_b))[:, None] * _AREA * rel_b
            f = _mv(R, f_b)
            arm = _mv(R, np.broadcast_to(_ARM, f.shape))
            out[windy, :3], out[windy, 3:] = f, np.cross(arm, f)
        return out

    def _fx_record(self, ids) -> None:
        ix = self.index
        count, head = self.fx_count[ids], self.head[ids]
        full = count >= self.ring.shape[1]
        slot = np.where(full, head, count)
        row = self.ring[ids, slot]
        row[:, 0], row[:, 1] = self.sim_steps[ids], self.time[ids]
        row[:, 2:5] = self.xpos[ids, ix.chassis]
        row[:, 5:9] = self.xquat[ids, ix.chassis]
        row[:, 9:12] = self.qvel[ids, :3]
        row[:, 12:21] = self.ctrl[ids[:, None], ix.jets]
        row[:, 21:24] = self.wind_mean[ids] + self.gust[ids]
        self.ring[ids, slot] = row
        self.fx_count[ids] = np.where(full, count, count + 1)
        self.head[ids] = np.where(full, (head + 1) % self.ring.shape[1], head)

    # ------------------------------------------------------------------ task (source.tasks)
    def _barrel_roll(self, ids, n_wheels) -> None:
        t, pos = self.time[ids], self.qpos[ids, :3]
        idle = ~self.has_roll_t0[ids] & ~self.roll_done[ids]
        start = idle & (n_wheels == 0) & (pos[:, 2] > self.index.hull_z + 4.5) & (pos[:, 1] > self.index.hop_y - 2.5)
        self.roll_t0[ids] = np.where(start, t, self.roll_t0[ids])
        self.has_roll_t0[ids] |= start
        rolling = self.has_roll_t0[ids] & ~self.roll_done[ids]
        if not rolling.any():
            return
        r = ids[rolling]
        u = np.minimum((self.time[r] - self.roll_t0[r]) / ROLL_TIME, 1.0)
        done = u >= 1.0
        self.roll_cmd[r] = np.where(done, 0.0, -2 * np.pi * (u - np.sin(2 * np.pi * u) / (2 * np.pi)))
        self.has_roll_cmd[r] = ~done
        self.roll_done[r] |= done

    def _sim_step(self, ids, a, noise, substep0, everyone) -> np.ndarray:
        """One 50 Hz step of ``RoadsterPlanEnv._step_raw`` for ``ids``; returns which terminated."""
        ix = self.index
        n = self._contacts(ids)["tyre_any"].sum(1)
        hold = np.where(n >= 1, 3, np.maximum(self.ground_hold[ids] - 1, 0))
        self.ground_hold[ids] = hold
        n_wheels = np.where(hold > 0, 4, 0)
        if self.task == "carhop":
            self._barrel_roll(ids, n_wheels)
        term = self._physics_step(ids, self._autopilot(ids, a, n_wheels), noise, substep0, everyone)
        pos = self.qpos[ids, :3]
        if self.task == "loop":
            R = self._rot(ids)
            self.loop_top_seen[ids] |= (pos[:, 2] > ix.loop_top - 3) & (R[:, 2, 2] < -0.5)
        if self.task in ROUTE_TASKS:
            idx = self.route_idx[ids]
            wp = self.route[ids, idx]
            on_way = idx < self.route_count[ids] - 1
            touch = self.route_touch[ids, idx]
            near_ground = (n >= 3) & (_norm(pos[:, :2] - wp[:, :2]) < 3.0)
            near = _norm(pos - wp) < ROUTE_RADIUS
            self.route_idx[ids] = idx + (on_way & np.where(touch, near_ground, near))
        return term

    def _waypoints(self, ids):
        """First waypoint (position, velocity, up), all positions ``(n, K, 3)`` and the count of each row."""
        n = len(ids)
        if self.task in ROUTE_TASKS:
            idx = self.route_idx[ids]
            count = self.route_count[ids] - idx
            k = np.minimum(idx[:, None] + np.arange(self.route.shape[1]), self.route.shape[1] - 1)
            points = self.route[ids[:, None], k]
            return points[:, 0], self.route_vel[ids, idx], self.route_up[ids, idx], points, count
        goal = self.goal_pos[ids]
        if self.task == "loop":
            lg = self._base.loop_goal
            top = ~self.loop_top_seen[ids]
            points = np.stack([np.where(top[:, None], lg["top_pos"], goal), goal], 1)
            vel = np.where(top[:, None], lg["top_vel"], self.goal_vel[ids])
            return points[:, 0], vel, np.where(top, -1.0, 1.0), points, np.where(top, 2, 1)
        if self.task == "mission":
            ix = self.index
            target = self.target_idx[ids]
            rings = target < ix.nrings
            k = target[:, None] + np.arange(ix.nrings + 1)
            stops = np.concatenate([np.broadcast_to(ix.rings, (n, ix.nrings, 3)), goal[:, None]], 1)
            points = stops[np.arange(n)[:, None], np.minimum(k, ix.nrings)]
            vel = np.where(rings[:, None], np.array([12.0, 0, 0]), self.goal_vel[ids])
            return points[:, 0], vel, np.ones(n), points, np.where(rings, ix.nrings - target + 1, 1)
        return goal, self.goal_vel[ids], np.ones(n), goal[:, None], np.ones(n, np.int64)

    def _cost(self, ids) -> np.ndarray:
        """``RoadsterPlanEnv.cost`` for ``ids``."""
        pos, vel, R = self.qpos[ids, :3], self.qvel[ids, :3], self._rot(ids)
        p, v, up, points, count = self._waypoints(ids)
        up_z = R[:, 2, 2]
        c = _norm(pos - p) + 0.3 * _norm(vel - v) + 5.0 * np.abs(up - up_z)
        current = np.arctan2(R[:, 1, 0], R[:, 0, 0])
        yaw_err = np.angle(np.exp(1j * (current - self.goal_yaw[ids])))
        c = np.where((count == 1) & (up > 0), c + 2.0 * np.abs(yaw_err) * (up_z > 0), c)
        for k in range(points.shape[1] - 1):
            c = np.where(k + 1 < count, c + _norm(points[:, k + 1] - points[:, k]), c)
        return c

    def _success(self, ids) -> np.ndarray:
        pos, vel, up_z = self.qpos[ids, :3], self.qvel[ids, :3], self._rot(ids)[:, 2, 2]
        c = self._contacts(ids)
        if self.task == "loop":
            return self.loop_top_seen[ids] & (up_z > 0.9) & (pos[:, 0] < -3) & (pos[:, 2] > 5) & ~c["tyre_any"].any(1)
        if self.task in ROUTE_TASKS:
            on_ground = c["tyre_ground"].sum(1) >= 3
            if self.task == "traffic":
                parked = on_ground & (pos[:, 0] > self.jam_extent[ids, 1] + 2)
            elif self.task == "carhop":
                parked = on_ground & (pos[:, 1] > self.index.hop_y + 2.5) & self.roll_done[ids]
            else:
                parked = c["tyre_roof"].sum(1) >= 3
            last = self.route_idx[ids] == self.route_count[ids] - 1
            return parked & (_norm(vel) < 1.0) & (up_z > 0.9) & last
        return self.status[ids] == _SUCCESS

    # ------------------------------------------------------------------ stepping
    def step(self, actions, *, active=None, capture_state: bool = True) -> TensorDict:
        """Advance one action per environment; an inactive environment holds its state and reports zeros.

        ``capture_state=False`` leaves the State snapshot out of the result.
        """
        self._check_open()
        actions, active = self.check_step(actions, active)
        a = np.asarray(actions, np.float64)
        n = self.num_envs
        run = np.ones(n, bool) if active is None else active.copy()
        stepped = np.flatnonzero(run)
        noise, saved = self._gust_rows(np.flatnonzero(run & (self.gust_std > 0)), REPEAT * FRAME_SKIP)
        substeps = np.zeros(n, np.int64)
        for j in range(REPEAT):
            ids = np.flatnonzero(run)
            if not ids.size:
                break
            term = self._sim_step(ids, a[ids], noise, j * FRAME_SKIP, ids.size == n)
            substeps[ids] += FRAME_SKIP
            run[ids[term]] = False
        self._rewind_gusts(saved, substeps, REPEAT * FRAME_SKIP)
        reward, success, failed = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
        terminated, truncated, cost = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n)
        if stepped.size:
            ids = stepped
            self.elapsed[ids] += 1
            c = self._cost(ids)
            r = self.prev_cost[ids] - c
            self.prev_cost[ids] = c
            succ = self._success(ids)
            fail = (self.status[ids] >= _CRASH) | (self.qpos[ids, 2] > 75)
            reward[ids] = r + (10.0 * succ - 10.0 * fail)
            success[ids], failed[ids], cost[ids] = succ, fail, c
            terminated[ids] = succ | fail
            truncated[ids] = (self.elapsed[ids] >= HORIZON[self.task]) & ~(succ | fail)
        dev = self.device
        result = TensorDict(
            {
                "reward": torch.from_numpy(reward),
                "success": torch.from_numpy(success),
                "failed": torch.from_numpy(failed),
                "terminated": torch.from_numpy(terminated),
                "truncated": torch.from_numpy(truncated),
                "cost": torch.from_numpy(cost),
            },
            batch_size=[n],
        ).to(dev)
        if capture_state:
            result["state"] = self.get_state()
        return result
