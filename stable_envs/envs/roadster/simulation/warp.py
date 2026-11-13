"""The Roadster task logic on CUDA tensors of environments, with MuJoCo Warp physics.

The source's operations (``source.tasks``, ``source.control``, ``source.sim``) run on every environment at
once in float64 Torch, without branches that read the device: per-environment conditions select results with
masks. Physics is float32 MuJoCo Warp, so trajectories follow the source approximately. The autopilot's
bounded least-squares jet allocation solves each active set with a precomputed pseudo-inverse.

A 50 Hz step runs as three CUDA graph launches: the Torch phase before the substeps (contacts, autopilot,
controls), one MuJoCo Warp graph of the 10 substeps with a Warp kernel for the drag and wind force before
each, and the Torch phase after them (bookkeeping, crash checks). One more Torch graph computes the outputs.

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


_KERNELS = None


def _kernels():
    """The Warp kernels a substep runs before ``mj_step``: gusts with drag and wind, and the pre-substep pose."""
    global _KERNELS
    if _KERNELS is None:
        import warp as wp

        f64 = wp.float64

        @wp.kernel
        def aero(
            qvel: wp.array2d(dtype=float),
            xmat: wp.array2d(dtype=wp.mat33),
            wind: wp.array2d(dtype=f64),
            gust: wp.array2d(dtype=f64),
            std: wp.array(dtype=f64),
            noise: wp.array3d(dtype=f64),
            offset: wp.array(dtype=int),
            k: int,
            decay: f64,
            gain: f64,
            drag: f64,
            pressure: f64,
            chassis: int,
            xfrc: wp.array2d(dtype=wp.spatial_vector),
        ):
            w = wp.tid()
            s = std[w]
            sub = offset[0] + k
            if s > f64(0.0):
                gust[w, 0] = decay * gust[w, 0] + gain * s * noise[w, sub, 0]
                gust[w, 1] = decay * gust[w, 1] + gain * s * noise[w, sub, 1]
                gust[w, 2] = decay * gust[w, 2] + gain * s * (noise[w, sub, 2] * f64(0.3))
            v = wp.vec3d(f64(qvel[w, 0]), f64(qvel[w, 1]), f64(qvel[w, 2]))
            air = wp.vec3d(wind[w, 0] + gust[w, 0], wind[w, 1] + gust[w, 1], wind[w, 2] + gust[w, 2])
            r = xmat[w, chassis]
            R = wp.mat33d(
                f64(r[0, 0]), f64(r[0, 1]), f64(r[0, 2]),
                f64(r[1, 0]), f64(r[1, 1]), f64(r[1, 2]),
                f64(r[2, 0]), f64(r[2, 1]), f64(r[2, 2]),
            )  # fmt: skip
            force = drag * wp.length(v) * v
            torque = wp.vec3d(f64(0.0), f64(0.0), f64(0.0))
            if air[0] != f64(0.0) or air[1] != f64(0.0) or air[2] != f64(0.0):
                rel = wp.transpose(R) @ (v - air)
                scale = pressure * wp.length(rel)
                body = wp.vec3d(scale * f64(DRAG_CDA) * rel[0], scale * f64(2.4) * rel[1], scale * f64(4.0) * rel[2])
                force = R @ body
                torque = wp.cross(R @ wp.vec3d(f64(-0.35), f64(0.0), f64(0.12)), force)
            xfrc[w, chassis] = wp.spatial_vector(
                float(force[0]), float(force[1]), float(force[2]), float(torque[0]), float(torque[1]), float(torque[2])
            )

        @wp.kernel
        def keep_pose(qpos: wp.array2d(dtype=float), out: wp.array2d(dtype=f64)):
            w, i = wp.tid()
            out[w, i] = f64(qpos[w, i])

        _KERNELS = aero, keep_pose
    return _KERNELS


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
            frame_skip=FRAME_SKIP,
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
        self._noise_ready = torch.cuda.Event()
        self._windy_host = np.zeros(n, bool)
        # Inputs and carried values of the captured phases of a step (``_phases``).
        self._a = torch.zeros((n, 4), **f64)
        self._run = torch.zeros(n, dtype=torch.bool, device=dev)
        self._stepped = torch.zeros(n, dtype=torch.bool, device=dev)
        self._offset = torch.zeros(1, dtype=torch.int32, device=dev)
        self._carry: dict[str, torch.Tensor] = {}
        self._out: dict[str, torch.Tensor] = {}
        self._graphs: tuple | None = None
        import warp as wp

        cols = {key: self._s[key] for key in (("weather", "wind_mean"), ("weather", "gust"))}
        self._aero_args = [
            wp.from_torch(p.data("qvel")),
            wp.from_torch(p.data("xmat"), dtype=wp.mat33),
            wp.from_torch(cols["weather", "wind_mean"]),
            wp.from_torch(cols["weather", "gust"]),
            wp.from_torch(self._col("weather", "gust_std")),
            wp.from_torch(self._noise),
            wp.from_torch(self._offset),
        ]
        self._xfrc = wp.from_torch(p.data("xfrc_applied"), dtype=wp.spatial_vector)
        self._pose_args = [wp.from_torch(p.data("qpos")), wp.from_torch(self._s["sim", "qpos_pre"])]
        self._substeps = p.capture_step(self._before_substep)

    def _before_substep(self, k: int) -> None:
        import warp as wp

        aero, keep_pose = _kernels()
        n, ix = self.num_envs, self.index
        if k == FRAME_SKIP - 1:  # the derived quantities after the step describe this pose
            wp.launch(keep_pose, dim=(n, self.model.nq), inputs=self._pose_args)
        f64 = wp.float64
        decay, gain = f64(self._gust_decay), f64(self._gust_gain)
        drag, pressure = f64(-0.5 * AIR_DENSITY * DRAG_CDA), f64(-0.5 * AIR_DENSITY)
        wp.launch(aero, dim=n, inputs=[*self._aero_args, k, decay, gain, drag, pressure, ix.chassis, self._xfrc])

    def _close_physics(self) -> None:
        self.physics.close()

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

    def _rot(self) -> torch.Tensor:
        return self.physics.data("xmat")[:, self.index.chassis].to(torch.float64).reshape(-1, 3, 3)

    def _pos(self) -> torch.Tensor:
        return self.physics.data("qpos")[:, :3].to(torch.float64)

    def _vel(self) -> torch.Tensor:
        return self.physics.data("qvel")[:, :3].to(torch.float64)

    def _contacts(self) -> dict[str, torch.Tensor]:
        s = self.physics.data("sensordata") > 0
        return {name: s[:, a:b] for name, (a, b) in self._sens.items()}

    # Autopilot, as in source.control.
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

    # Simulator, as in source.sim.
    def _physics_pre(self, raw) -> None:
        """``RoadsterSim.step`` up to the substeps: the controls, and what a stopped environment holds."""
        p, carry = self.physics, self._carry
        a = torch.clamp(raw, self._lo, self._hi)
        jets = torch.where((self._col("task", "fuel") > 0)[:, None], a[:, 2:], 0.0)
        for name in _HELD:
            carry[name] = p.data(name).clone()
        carry["time"], carry["qpos_pre"] = self._time.clone(), self._s["sim", "qpos_pre"].clone()
        carry["gust"] = self._col("weather", "gust").clone()
        carry["burn"] = (jets.abs() * self._jet_force).sum(1) / 1000.0
        carry["prev_x"] = self._pos()[:, 0]
        ctrl = p.data("ctrl")
        new_ctrl = ctrl.clone()
        new_ctrl[:, self._drive] = a[:, :1].to(ctrl.dtype)
        new_ctrl[:, self._steer] = (0.5 * a[:, 1:2]).to(ctrl.dtype)
        new_ctrl[:, self._jets] = jets.to(ctrl.dtype)
        ctrl.copy_(new_ctrl)

    def _physics_post(self) -> torch.Tensor:
        """``RoadsterSim.step`` after the substeps; returns which environments terminated."""
        ix, p, carry, run = self.index, self.physics, self._carry, self._run
        for _ in range(FRAME_SKIP):
            self._time += ix.timestep
        r = run[:, None]
        for name in _HELD:
            new = p.data(name)
            new.copy_(torch.where(r.reshape(-1, *([1] * (new.ndim - 1))), new, carry[name]))
        self._time.copy_(torch.where(run, self._time, carry["time"]))
        qpos_pre = self._s["sim", "qpos_pre"]
        qpos_pre.copy_(torch.where(r, qpos_pre, carry["qpos_pre"]))
        gust = self._col("weather", "gust")
        gust.copy_(torch.where(r, gust, carry["gust"]))
        has_pre = self._col("sim", "has_qpos_pre")
        has_pre |= run
        steps = self._col("task", "sim_steps")
        steps += run.long()
        fuel = self._col("task", "fuel")
        fuel.copy_(torch.where(run, torch.clamp(fuel - carry["burn"] * ix.dt, min=0.0), fuel))
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
        new_target = target + (ahead & (carry["prev_x"] < ring_x) & (ring_x <= pos[:, 0])).long()
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

    # Task, as in source.tasks.
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

    def _pre(self) -> None:
        """A 50 Hz step of ``RoadsterPlanEnv._step_raw`` up to the substeps, for the running environments."""
        run = self._run
        n = self._contacts()["tyre_any"].sum(1)
        self._carry["n"] = n
        hold = self._col("task", "ground_hold")
        hold.copy_(torch.where(run, torch.where(n >= 1, 3, torch.clamp(hold - 1, min=0)), hold))
        n_wheels = (hold > 0).long() * 4
        if self.task == "carhop":
            self._barrel_roll(n_wheels, run)
        self._physics_pre(self._autopilot(self._a, n_wheels, run))

    def _post(self) -> None:
        """The rest of the 50 Hz step after the substeps; environments that terminate stop running."""
        ix, run, n = self.index, self._run, self._carry["n"]
        term = self._physics_post()
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
        run &= ~term

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
        self._noise_ready.synchronize()  # the previous action's copy has left the pinned buffer
        self._noise_host.copy_(torch.from_numpy(noise))
        self._noise.copy_(self._noise_host, non_blocking=True)
        self._noise_ready.record()

    def _final(self) -> None:
        """The outputs of the action: ``RoadsterPlanEnv.step`` after its 50 Hz steps."""
        stepped, out = self._stepped, self._out
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
        out["reward"] = torch.where(stepped, reward, 0.0)
        out["success"] = succ & stepped
        out["failed"] = fail & stepped
        out["terminated"] = done & stepped
        out["truncated"] = stepped & (elapsed >= HORIZON[self.task]) & ~done
        out["cost"] = torch.where(stepped, c, 0.0)

    def _phases(self) -> tuple:
        """The before-substeps, after-substeps and output phases of a step, each in one captured CUDA graph.

        Captured at the first step from warm-up runs, after which the State is restored.
        """
        if self._graphs is None:
            saved, physics = self.get_state(), self._physics.copy()
            self._stepped.fill_(True)
            self._run.fill_(True)
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self._pre()
                    self._post()
                    self._final()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            pool = torch.cuda.graph_pool_handle()
            graphs = []
            for phase in (self._pre, self._post, self._final):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    phase()
                graphs.append(graph)
            self.set_state(saved)
            self._physics = physics
            self._graphs = tuple(graphs)
        return self._graphs

    def step(self, actions, *, active=None, capture_state: bool = True) -> TensorDict:
        """Advance one action per environment; an inactive environment holds its state and reports zeros.

        ``capture_state=False`` leaves the State snapshot out of the result.
        """
        self._check_open()
        actions, active = self.check_step(actions, active)
        pre, post, final = self._phases()
        self._a.copy_(torch.as_tensor(actions))
        if active is None:
            self._stepped.fill_(True)
        else:
            self._stepped.copy_(torch.as_tensor(active))
        self._run.copy_(self._stepped)
        self._draw_gusts(active)
        for j in range(REPEAT):
            pre.replay()
            self._offset.fill_(j * FRAME_SKIP)
            self.physics.launch_step(self._substeps)
            post.replay()
        final.replay()
        result = TensorDict(
            {key: value.clone() for key, value in self._out.items()}, batch_size=[self.num_envs], device=self.device
        )
        if capture_state:
            result["state"] = self.get_state()
        return result
