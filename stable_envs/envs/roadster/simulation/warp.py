"""The Roadster task logic on CUDA tensors of environments, with MuJoCo Warp physics.

The source's operations (``source.tasks``, ``source.control``, ``source.sim``) run on every environment at
once in float64 Torch, without branches that read the device: per-environment conditions select results with
masks. Physics is float32 MuJoCo Warp, so trajectories follow the source approximately. The autopilot's
bounded least-squares jet allocation solves each active set with a precomputed pseudo-inverse.

Gusts come from each environment's NumPy generator as in the source, drawn on the host for a whole action;
an environment that terminates within an action has drawn the gusts of the whole action.
"""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.mjbatch import WarpBatch
from ..source.control import ACC_SCALE, INERTIA, WHEELBASE, WRENCH_SCALE, YAW_RATE_SCALE, G
from ..source.sim import ACTION_HIGH, ACTION_LOW, AIR_DENSITY, DRAG_CDA, FRAME_SKIP
from ..source.tasks import HORIZON, REPEAT, ROLL_TIME, ROUTE_RADIUS, ROUTE_TASKS
from .base import RoadsterBatch
from .world import STATUS, allocation_matrix, allocation_table

_DRIVE, _FLIGHT, _LANDED, _SUCCESS = (STATUS[s] for s in ("drive", "flight", "landed", "success"))
_CRASH = STATUS["crash: hit obstacle"]
#: Contacts per world and constraint rows per world that MuJoCo Warp reserves.
NACON_PER_WORLD = 64
NJMAX = 160
#: Fields a step changes that an environment that stops within an action holds.
_HELD = ("qpos", "qvel", "qacc_warmstart", "ctrl", "xfrc_applied", "xmat", "xpos", "xquat", "sensordata")


def _norm(v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(v, dim=-1)


def _mv(m: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.matmul(m, v.unsqueeze(-1)).squeeze(-1)


def _rz(psi: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(psi), torch.sin(psi)
    z, o = torch.zeros_like(psi), torch.ones_like(psi)
    return torch.stack([c, -s, z, s, c, z, z, z, o], -1).reshape(-1, 3, 3)


def _rx(phi: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(phi), torch.sin(phi)
    z, o = torch.zeros_like(phi), torch.ones_like(phi)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(-1, 3, 3)


def _rot_err(R: torch.Tensor, Rd: torch.Tensor) -> torch.Tensor:
    """``source.util.rot_err`` for each row."""
    E = torch.matmul(Rd.transpose(1, 2), R)
    trace = E[:, 0, 0] + E[:, 1, 1] + E[:, 2, 2]
    ang = torch.arccos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
    axis = torch.stack([E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]], 1)
    M = E + torch.eye(3, dtype=E.dtype, device=E.device)
    col = torch.argmax(_norm(M.transpose(1, 2)), 1)
    flipped = torch.gather(M, 2, col[:, None, None].expand(-1, 3, 1))[:, :, 0]
    axis = torch.where((ang > np.pi - 1e-3)[:, None], flipped, axis)
    out = axis / _norm(axis).clamp_min(1e-300)[:, None] * ang[:, None]
    return torch.where((ang < 1e-6)[:, None], torch.zeros_like(out), out)


class RoadsterWarpAdapter(RoadsterBatch):
    """The Roadster task on CUDA: Torch task logic, MuJoCo Warp physics; approximately the source."""

    backend = "mjwarp"

    def _build(self) -> None:
        n = self.num_envs
        if self.device.type != "cuda":
            raise ValueError("The MuJoCo Warp Roadster backend requires a CUDA device")
        self.physics = p = WarpBatch(
            self.model,
            n,
            self.device,
            outputs=("xmat", "xpos", "xquat", "sensordata", "ctrl", "xfrc_applied", "qfrc_applied"),
            batched_fields=("geom_friction",),
            naconmax=NACON_PER_WORLD * n,
            njmax=NJMAX,
        )
        dev = self.device
        self._alloc(dev)
        f64 = dict(dtype=torch.float64, device=dev)
        self._time = torch.zeros(n, **f64)
        self.friction = p.model_field("geom_friction")
        self._tire_friction = torch.as_tensor(self.model.geom_friction[self.index.tires], **f64)
        ix = self.index
        self._drive = torch.as_tensor(ix.drive, device=dev)
        self._steer = torch.as_tensor(ix.steer, device=dev)
        self._jets = torch.as_tensor(ix.jets, device=dev)
        self._jet_force = torch.as_tensor(ix.jet_force, **f64)
        self._lo, self._hi = torch.as_tensor(ACTION_LOW, **f64), torch.as_tensor(ACTION_HIGH, **f64)
        self._B = torch.as_tensor(allocation_matrix(), **f64)
        self._P = torch.as_tensor(allocation_table(), **f64)
        self._bits = 1 << torch.arange(self._B.shape[1], device=dev)
        self._acc_scale = torch.as_tensor(ACC_SCALE, **f64)
        self._wrench = torch.as_tensor(WRENCH_SCALE, **f64)
        self._inertia = torch.as_tensor(INERTIA, **f64)
        self._area = torch.as_tensor([DRAG_CDA, 2.4, 4.0], **f64)
        self._arm = torch.as_tensor([-0.35, 0.0, 0.12], **f64)
        self._gust_scale = torch.as_tensor([1.0, 1.0, 0.3], **f64)
        self._g = torch.as_tensor([0.0, 0.0, G], **f64)
        self._ring_x = torch.as_tensor(ix.ring_x, **f64)
        self._stops = torch.as_tensor(ix.rings, **f64)
        self._ring_vel = torch.as_tensor([12.0, 0.0, 0.0], **f64)
        lg = self._base.loop_goal
        if lg is not None:
            self._top_pos, self._top_vel = (torch.as_tensor(lg[k], **f64) for k in ("top_pos", "top_vel"))
        self._sens = {name: (sl.start, sl.stop) for name, sl in ix.sensors.items()}
        self._noise = torch.zeros((n, REPEAT * FRAME_SKIP, 3), **f64)
        self._noise_host = torch.zeros((n, REPEAT * FRAME_SKIP, 3), dtype=torch.float64, pin_memory=True)
        self._gust_decay = float(np.exp(-ix.timestep / 1.5))
        self._gust_gain = float(np.sqrt(1 - self._gust_decay * self._gust_decay))
        self._windy_host = np.zeros(n, bool)

    def _close_physics(self) -> None:
        self.physics.close()

    # ------------------------------------------------------------------ physics state
    def _sim(self, name: str) -> torch.Tensor:
        if name == "time":
            return self._time
        return self.physics.data({"warmstart": "qacc_warmstart"}.get(name, name))

    def _apply_grip(self) -> None:
        grip = self._col("weather", "grip")
        scale = torch.stack([grip, torch.ones_like(grip), torch.ones_like(grip)], -1)
        tires = torch.as_tensor(self.index.tires, device=self.device)
        self.friction[:, tires] = (self._tire_friction[None] * scale[:, None, :]).to(self.friction.dtype)
        self._windy_host = (self._col("weather", "gust_std") > 0).cpu().numpy()

    def _write_physics(self, values, qpos_pre, has_pre) -> None:
        p = self.physics
        dev = self.device
        values = {k: torch.as_tensor(v).to(dev) for k, v in values.items()}
        has = torch.as_tensor(has_pre).to(dev, torch.bool)
        qpos = values["qpos"].to(torch.float64).reshape(self.num_envs, -1)
        pre = torch.as_tensor(qpos_pre).to(dev, torch.float64).reshape(qpos.shape)
        pre = torch.where(has[:, None], pre, qpos)
        for name in ("ctrl", "qfrc_applied", "xfrc_applied"):
            target = p.data(name)
            value = values.get(name)
            target.copy_(torch.zeros_like(target) if value is None else value.reshape(target.shape).to(target.dtype))
        self._time.copy_(values["time"].reshape(-1) if "time" in values else torch.zeros_like(self._time))
        state = {"qpos": pre, "time": self._time}
        for name in ("qvel", "mocap_pos", "mocap_quat"):
            state[name] = values[name]
        state["qacc_warmstart"] = values.get("warmstart", torch.zeros_like(values["qvel"]))
        # Derived quantities after a step describe the pose before its last substep: forward there.
        p.set_physics(state)
        p.data("qpos").copy_(qpos)
        self._s["sim", "qpos_pre"].copy_(pre)
        self._s["sim", "has_qpos_pre"].copy_(has[:, None])

    def _after_reset(self) -> None:
        R = self._rot()
        self._col("autopilot", "yaw_cmd").copy_(torch.atan2(R[:, 1, 0], R[:, 0, 0]))
        self._col("task", "prev_cost").copy_(self._cost())

    # ------------------------------------------------------------------ views
    def _rot(self) -> torch.Tensor:
        return self.physics.data("xmat")[:, self.index.chassis].to(torch.float64).reshape(-1, 3, 3)

    def _pos(self) -> torch.Tensor:
        return self.physics.data("qpos")[:, :3].to(torch.float64)

    def _vel(self) -> torch.Tensor:
        return self.physics.data("qvel")[:, :3].to(torch.float64)

    def _contacts(self) -> dict[str, torch.Tensor]:
        s = self.physics.data("sensordata") > 0
        return {name: s[:, a:b] for name, (a, b) in self._sens.items()}

    # ------------------------------------------------------------------ autopilot (source.control)
    def _allocate(self, w: torch.Tensor) -> torch.Tensor:
        """``Autopilot._alloc`` for each row: the bounded least-squares jet commands."""
        lo, hi = self._lo[2:], self._hi[2:]
        w = w / self._wrench
        n, njet = len(w), self._B.shape[1]
        u = torch.zeros((n, njet), dtype=w.dtype, device=w.device)
        free = torch.ones((n, njet), dtype=torch.bool, device=w.device)
        going = torch.ones(n, dtype=torch.bool, device=w.device)
        for _ in range(njet):
            resid = w - _mv(self._B, torch.where(free, 0.0, u))
            x = _mv(self._P[(free * self._bits).sum(1)], resid)
            new = torch.where(free, x, u)
            sat = free & ((new < lo) | (new > hi))
            hit = sat.any(1)
            new = torch.where(hit[:, None], torch.clamp(new, lo, hi), new)
            u = torch.where(going[:, None], new, u)
            free = torch.where((going & hit)[:, None], free & ~sat, free)
            going = going & hit & free.any(1)
        return torch.clamp(u, lo, hi)

    def _air_jets(self, R, v, acc_w, yaw) -> torch.Tensor:
        drag = (0.5 * AIR_DENSITY * DRAG_CDA * _norm(v))[:, None] * v
        F_b = _mv(R.transpose(1, 2), self.index.mass * (acc_w + self._g) + drag)
        Rd = _rz(yaw)
        roll = self._col("autopilot", "has_roll_cmd")
        Rd_roll = torch.matmul(Rd, _rx(self._col("autopilot", "roll_cmd")))
        a_lat = _mv(Rd.transpose(1, 2), acc_w)[:, 1]
        Rd_level = torch.matmul(Rd, _rx(-torch.clamp(torch.atan2(a_lat, G + acc_w[:, 2]), -0.4, 0.4)))
        Rd = torch.where(roll[:, None, None], Rd_roll, torch.where((R[:, 2, 2] > 0.7)[:, None, None], Rd_level, Rd))
        wn, zeta = 5.0, 0.9
        angvel = self.physics.data("qvel")[:, 3:6].to(torch.float64)
        tau = self._inertia * (-(wn**2) * _rot_err(R, Rd) - 2 * zeta * wn * angvel)
        return self._allocate(torch.cat([F_b, tau], 1))

    def _autopilot(self, a, n_wheels, run) -> torch.Tensor:
        """``Autopilot.__call__``: the 4-D command to the 11-D raw action; updates the yaw command."""
        a = torch.clamp(a, -1.0, 1.0)
        acc = a[:, :3] * self._acc_scale
        yaw_rate = a[:, 3] * YAW_RATE_SCALE
        R, v = self._rot(), self._vel()
        current = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        ground = n_wheels >= 2
        v_fwd = (v * R[:, :, 0]).sum(1)
        steer = torch.arctan(WHEELBASE * yaw_rate / torch.clamp(v_fwd.abs(), min=2.0))
        sign = torch.sign(torch.where(v_fwd.abs() > 0.5, v_fwd, torch.ones_like(v_fwd)))
        drive = torch.where(ground, acc[:, 0] / ACC_SCALE[0], 0.0)
        steering = torch.where(ground, torch.clamp(steer / 0.5, -1, 1) * sign, 0.0)
        yaw_cmd = self._col("autopilot", "yaw_cmd")
        steep = ~ground & (R[:, 2, 2] < 0.7) & ~self._col("autopilot", "has_roll_cmd")
        yaw = torch.where(ground | steep, current, yaw_cmd + yaw_rate * self.index.dt)
        yaw_cmd.copy_(torch.where(run, yaw, yaw_cmd))
        zero = torch.zeros_like(yaw)
        acc_w = _mv(_rz(yaw), torch.stack([acc[:, 0], acc[:, 1], zero], 1)) + torch.stack([zero, zero, acc[:, 2]], 1)
        jets = self._air_jets(R, v, acc_w, yaw)
        jets = torch.where((~ground | (acc[:, 2] > 0.5))[:, None], jets, 0.0)
        return torch.cat([drive[:, None], steering[:, None], jets], 1)

    # ------------------------------------------------------------------ simulator (source.sim)
    def _aero(self) -> torch.Tensor:
        v = self._vel()
        w = self._col("weather", "wind_mean") + self._col("weather", "gust")
        calm = (-0.5 * AIR_DENSITY * DRAG_CDA * _norm(v))[:, None] * v
        R = self._rot()
        rel_b = _mv(R.transpose(1, 2), v - w)
        f = _mv(R, (-0.5 * AIR_DENSITY * _norm(rel_b))[:, None] * self._area * rel_b)
        torque = torch.linalg.cross(_mv(R, self._arm.expand_as(f)), f)
        windy = (w != 0).any(1)[:, None]
        return torch.where(windy, torch.cat([f, torque], 1), torch.cat([calm, torch.zeros_like(calm)], 1))

    def _physics_step(self, raw, run, substep0) -> torch.Tensor:
        """``RoadsterSim.step`` for the running environments; returns which terminated."""
        ix, p = self.index, self.physics
        a = torch.clamp(raw, self._lo, self._hi)
        fuel = self._col("task", "fuel")
        jets = torch.where((fuel > 0)[:, None], a[:, 2:], 0.0)
        ctrl = p.data("ctrl")
        new_ctrl = ctrl.clone()
        new_ctrl[:, self._drive] = a[:, :1].to(ctrl.dtype)
        new_ctrl[:, self._steer] = (0.5 * a[:, 1:2]).to(ctrl.dtype)
        new_ctrl[:, self._jets] = jets.to(ctrl.dtype)
        burn = (jets.abs() * self._jet_force).sum(1) / 1000.0
        held = {name: p.data(name).clone() for name in _HELD}
        held_time = self._time.clone()
        ctrl.copy_(new_ctrl)
        prev_x = self._pos()[:, 0]
        gust, std = self._col("weather", "gust"), self._col("weather", "gust_std")
        gust_before = gust.clone()
        windy = (std > 0)[:, None]
        xfrc = p.data("xfrc_applied")
        qpos_pre = self._s["sim", "qpos_pre"]
        for k in range(FRAME_SKIP):
            draw = self._noise[:, substep0 + k] * self._gust_scale
            gust.copy_(torch.where(windy, self._gust_decay * gust + (self._gust_gain * std)[:, None] * draw, gust))
            xfrc[:, ix.chassis] = self._aero().to(xfrc.dtype)
            if k == FRAME_SKIP - 1:
                qpos_pre.copy_(torch.where(run[:, None], p.data("qpos").to(torch.float64), qpos_pre))
            p.step()
            self._time += ix.timestep
        r = run[:, None]
        for name, old in held.items():
            new = p.data(name)
            new.copy_(torch.where(r.reshape(-1, *([1] * (new.ndim - 1))), new, old))
        self._time.copy_(torch.where(run, self._time, held_time))
        gust.copy_(torch.where(r, gust, gust_before))
        has_pre = self._col("sim", "has_qpos_pre")
        has_pre |= run
        steps = self._col("task", "sim_steps")
        steps += run.long()
        fuel.copy_(torch.where(run, torch.clamp(fuel - burn * ix.dt, min=0.0), fuel))
        self._fx_record(run)

        c = self._contacts()
        pos, vel, R = self._pos(), self._vel(), self._rot()
        airborne_once = self._col("task", "airborne_once")
        airborne = airborne_once | (~c["tyre_any"].any(1) & (pos[:, 2] > 1.5))
        status = self._col("task", "status")
        st = torch.where((status == _DRIVE) & airborne, _FLIGHT, status)
        target = self._col("task", "target_idx")
        ahead = target < ix.nrings
        ring_x = self._ring_x[torch.clamp(target, max=ix.nrings - 1)]
        new_target = target + (ahead & (prev_x < ring_x) & (ring_x <= pos[:, 0])).long()
        landed = (new_target >= ix.nrings) & (c["tyre_pad"].sum(1) >= 3)
        st = torch.where(landed, _LANDED, st)
        success = landed & (_norm(vel) < 1.0) & (R[:, 2, 2] > 0.9)
        st = torch.where(success, _SUCCESS, st)
        region = self._col("task", "region_y")
        out = (pos[:, 2] < -2) | ((pos[:, 1] - region).abs() > 150) | (pos[:, 0] > 400)
        ditched = airborne & c["tyre_ground"].any(1) & ~self._col("task", "allow_ground_landing")
        crash = torch.where(
            c["obstacle"].any(1),
            _CRASH,
            torch.where(
                c["hull"].any(1),
                _CRASH + 1,
                torch.where(ditched, _CRASH + 2, torch.where(out, _CRASH + 3, torch.full_like(status, -1))),
            ),
        )
        st = torch.where(crash >= 0, crash, st)
        airborne_once.copy_(torch.where(run, airborne, airborne_once))
        target.copy_(torch.where(run, new_target, target))
        status.copy_(torch.where(run, st, status))
        return run & (success | (crash >= 0))

    def _fx_record(self, run) -> None:
        ix, p = self.index, self.physics
        count = self._col("fx", "count")
        head = self._head
        size = self._ring.shape[1]
        full = count >= size
        slot = torch.where(full, head, count)
        row = torch.cat(
            [
                self._col("task", "sim_steps").to(torch.float64)[:, None],
                self._time[:, None],
                p.data("xpos")[:, ix.chassis].to(torch.float64),
                p.data("xquat")[:, ix.chassis].to(torch.float64),
                self._vel(),
                p.data("ctrl")[:, self._jets].to(torch.float64),
                self._col("weather", "wind_mean") + self._col("weather", "gust"),
            ],
            1,
        )
        rows = torch.arange(self.num_envs, device=self.device)
        self._ring[rows, slot] = torch.where(run[:, None], row, self._ring[rows, slot])
        count.copy_(torch.where(run & ~full, count + 1, count))
        head.copy_(torch.where(run & full, (head + 1) % size, head))

    # ------------------------------------------------------------------ task (source.tasks)
    def _barrel_roll(self, n_wheels, run) -> None:
        t, pos = self._time, self._pos()
        has_t0, t0 = self._col("task", "has_roll_t0"), self._col("task", "roll_t0")
        done = self._col("task", "roll_done")
        start = run & ~has_t0 & ~done & (n_wheels == 0)
        start &= (pos[:, 2] > self.index.hull_z + 4.5) & (pos[:, 1] > self.index.hop_y - 2.5)
        t0.copy_(torch.where(start, t, t0))
        has_t0 |= start
        rolling = run & has_t0 & ~done
        u = torch.clamp((t - t0) / ROLL_TIME, max=1.0)
        finished = u >= 1.0
        roll = -2 * np.pi * (u - torch.sin(2 * np.pi * u) / (2 * np.pi))
        cmd, has_cmd = self._col("autopilot", "roll_cmd"), self._col("autopilot", "has_roll_cmd")
        cmd.copy_(torch.where(rolling, torch.where(finished, 0.0, roll), cmd))
        has_cmd.copy_(torch.where(rolling, ~finished, has_cmd))
        done |= rolling & finished

    def _sim_step(self, a, run, substep0) -> torch.Tensor:
        """One 50 Hz step of ``RoadsterPlanEnv._step_raw``; returns which terminated."""
        ix = self.index
        n = self._contacts()["tyre_any"].sum(1)
        hold = self._col("task", "ground_hold")
        new_hold = torch.where(n >= 1, 3, torch.clamp(hold - 1, min=0))
        hold.copy_(torch.where(run, new_hold, hold))
        n_wheels = (hold > 0).long() * 4
        if self.task == "carhop":
            self._barrel_roll(n_wheels, run)
        term = self._physics_step(self._autopilot(a, n_wheels, run), run, substep0)
        pos = self._pos()
        if self.task == "loop":
            top = self._col("task", "loop_top_seen")
            top |= run & (pos[:, 2] > ix.loop_top - 3) & (self._rot()[:, 2, 2] < -0.5)
        if self.task in ROUTE_TASKS:
            idx = self._col("task", "route_idx")
            rows = torch.arange(self.num_envs, device=self.device)
            wp = self._s["route", "positions"].reshape(self.num_envs, -1, 3)[rows, idx]
            touch = self._s["route", "touch"][rows, idx]
            on_way = idx < self._col("route", "count") - 1
            near_ground = (n >= 3) & (_norm(pos[:, :2] - wp[:, :2]) < 3.0)
            near = _norm(pos - wp) < ROUTE_RADIUS
            idx += (run & on_way & torch.where(touch, near_ground, near)).long()
        return term

    def _waypoints(self):
        n = self.num_envs
        rows = torch.arange(n, device=self.device)
        goal, goal_vel = self._col("goal", "position"), self._col("goal", "velocity")
        if self.task in ROUTE_TASKS:
            idx = self._col("task", "route_idx")
            route = self._s["route", "positions"].reshape(n, -1, 3)
            size = route.shape[1]
            k = torch.clamp(idx[:, None] + torch.arange(size, device=self.device), max=size - 1)
            points = torch.gather(route, 1, k[:, :, None].expand(-1, -1, 3))
            vel = self._s["route", "velocities"].reshape(n, -1, 3)[rows, idx]
            up = self._s["route", "up"][rows, idx]
            return points[:, 0], vel, up, points, self._col("route", "count") - idx
        if self.task == "loop":
            top = ~self._col("task", "loop_top_seen")
            points = torch.stack([torch.where(top[:, None], self._top_pos, goal), goal], 1)
            vel = torch.where(top[:, None], self._top_vel, goal_vel)
            up = 1.0 - 2.0 * top.double()
            return points[:, 0], vel, up, points, 1 + top.long()
        if self.task == "mission":
            ix = self.index
            target = self._col("task", "target_idx")
            rings = target < ix.nrings
            stops = torch.cat([self._stops.expand(n, -1, -1), goal[:, None]], 1)
            k = torch.clamp(target[:, None] + torch.arange(ix.nrings + 1, device=self.device), max=ix.nrings)
            points = torch.gather(stops, 1, k[:, :, None].expand(-1, -1, 3))
            vel = torch.where(rings[:, None], self._ring_vel, goal_vel)
            count = torch.where(rings, ix.nrings - target + 1, 1)
            return points[:, 0], vel, torch.ones_like(goal[:, 0]), points, count
        return goal, goal_vel, torch.ones_like(goal[:, 0]), goal[:, None], torch.ones_like(rows)

    def _cost(self) -> torch.Tensor:
        pos, vel, R = self._pos(), self._vel(), self._rot()
        p, v, up, points, count = self._waypoints()
        up_z = R[:, 2, 2]
        c = _norm(pos - p) + 0.3 * _norm(vel - v) + 5.0 * (up - up_z).abs()
        current = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        d = current - self._col("goal", "yaw")
        yaw_err = torch.atan2(torch.sin(d), torch.cos(d))
        c = torch.where((count == 1) & (up > 0), c + 2.0 * yaw_err.abs() * (up_z > 0), c)
        for k in range(points.shape[1] - 1):
            c = torch.where(k + 1 < count, c + _norm(points[:, k + 1] - points[:, k]), c)
        return c

    def _success(self) -> torch.Tensor:
        pos, vel, up_z = self._pos(), self._vel(), self._rot()[:, 2, 2]
        c = self._contacts()
        if self.task == "loop":
            top = self._col("task", "loop_top_seen")
            return top & (up_z > 0.9) & (pos[:, 0] < -3) & (pos[:, 2] > 5) & ~c["tyre_any"].any(1)
        if self.task in ROUTE_TASKS:
            on_ground = c["tyre_ground"].sum(1) >= 3
            if self.task == "traffic":
                parked = on_ground & (pos[:, 0] > self._s["jam", "extent"][:, 1] + 2)
            elif self.task == "carhop":
                parked = on_ground & (pos[:, 1] > self.index.hop_y + 2.5) & self._col("task", "roll_done")
            else:
                parked = c["tyre_roof"].sum(1) >= 3
            last = self._col("task", "route_idx") == self._col("route", "count") - 1
            return parked & (_norm(vel) < 1.0) & (up_z > 0.9) & last
        return self._col("task", "status") == _SUCCESS

    # ------------------------------------------------------------------ stepping
    def check_step(self, actions, active):
        """Return ``(actions, active)`` for the step, or raise before any environment moves.

        A tensor is checked for shape and dtype only: a finite check would read the device.
        """
        if not torch.is_tensor(actions):
            return super().check_step(actions, active)
        if actions.shape != (self.num_envs, 4):
            raise ValueError(f"Roadster actions require shape ({self.num_envs}, 4), got {tuple(actions.shape)}")
        if not actions.is_floating_point():
            raise TypeError(f"Roadster actions must be floating point, got {actions.dtype}")
        if active is not None:
            active = torch.as_tensor(active, device=self.device)
            if active.shape != (self.num_envs,):
                raise ValueError(f"Roadster active mask requires shape ({self.num_envs},), got {tuple(active.shape)}")
        return actions, active

    def _draw_gusts(self, active) -> None:
        if not self._windy_host.any():
            return
        run = np.ones(self.num_envs, bool) if active is None else np.asarray(torch.as_tensor(active).cpu(), bool)
        noise, saved = self._gust_rows(np.flatnonzero(run & self._windy_host), REPEAT * FRAME_SKIP)
        self._rewind_gusts(saved, np.full(self.num_envs, REPEAT * FRAME_SKIP), REPEAT * FRAME_SKIP)
        self._noise_host.copy_(torch.from_numpy(noise))
        self._noise.copy_(self._noise_host, non_blocking=True)

    def step(self, actions, *, active=None, capture_state: bool = True) -> TensorDict:
        """Advance one action per environment; an inactive environment holds its state and reports zeros.

        ``capture_state=False`` leaves the State snapshot out of the result.
        """
        self._check_open()
        actions, active = self.check_step(actions, active)
        dev, n = self.device, self.num_envs
        a = torch.as_tensor(actions).to(dev, torch.float64)
        stepped = (
            torch.ones(n, dtype=torch.bool, device=dev)
            if active is None
            else torch.as_tensor(active).to(dev, torch.bool)
        )
        self._draw_gusts(active)
        run = stepped.clone()
        for j in range(REPEAT):
            term = self._sim_step(a, run, j * FRAME_SKIP)
            run = run & ~term
        elapsed = self._col("lifecycle", "elapsed_steps")
        elapsed += stepped.long()
        c = self._cost()
        prev = self._col("task", "prev_cost")
        reward = prev - c
        prev.copy_(torch.where(stepped, c, prev))
        succ = self._success()
        fail = (self._col("task", "status") >= _CRASH) | (self._pos()[:, 2] > 75)
        reward = reward + (10.0 * succ - 10.0 * fail.double())
        done = succ | fail
        zero = torch.zeros(n, dtype=torch.float64, device=dev)
        result = TensorDict(
            {
                "reward": torch.where(stepped, reward, zero),
                "success": succ & stepped,
                "failed": fail & stepped,
                "terminated": done & stepped,
                "truncated": stepped & (elapsed >= HORIZON[self.task]) & ~done,
                "cost": torch.where(stepped, c, zero),
            },
            batch_size=[n],
            device=dev,
        )
        if capture_state:
            result["state"] = self.get_state()
        return result
