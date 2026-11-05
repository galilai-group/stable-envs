"""The jet Roadster in MuJoCo.

Raw 11-D control (drive, steer, nine jets), aerodynamics, the course bookkeeping (next ring, fuel, crash
detection) and the visual effects (flames, glow, smoke, rain, shadow).
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import model as world
from .course import Course
from .fx import ENTRY_DIM, HISTORY, Smoke, _quat_to_mat
from .model import JETS, WHEELS, build_xml, flame_geoms
from .textures import assets
from .weather import Weather

AIR_DENSITY = 1.225
DRAG_CDA = 0.6  # Cd * frontal area [m^2]
FUEL_CAPACITY = 800.0  # total jet impulse budget [kN*s]
FRAME_SKIP = 10  # physics substeps (2 ms) per control step
#: Bounds of the 11-D raw action: drive, steer, then one command per jet.
ACTION_LOW = np.array([-1.0, -1.0] + [(-1.0 if j.bidirectional else 0.0) for j in JETS])
ACTION_HIGH = np.ones_like(ACTION_LOW)


class RoadsterSim:
    """Raw control of the car at 50 Hz.

    ``step(action)`` takes drive torque [-1, 1], steering [-1, 1], then one command per jet (lift_* / boost_*
    / retro in [0, 1], lat_* in [-1, 1]) and returns ``terminated``.

    ``xml_patch`` rewrites the model XML before compiling (``cinematic.patch_space``); visual only.
    ``template`` (another sim) shares its compiled model instead of compiling one: the only physics a lane
    writes into the model is its tyre grip, which it applies before each step.
    """

    def __init__(self, course: Course | None = None, xml_patch=None, template: RoadsterSim | None = None):
        if template is not None:
            self.course, self.model = template.course, template.model
        else:
            self.course = course or Course()
            xml = build_xml(self.course)
            self.model = mujoco.MjModel.from_xml_string(xml_patch(xml) if xml_patch else xml, assets())
        self.weather = Weather(self.model, base=None if template is None else template.weather)
        self.smoke = Smoke(self.model)
        self.region_y = 0.0  # centre line of the active scenario (out-of-bounds check)
        self.allow_ground_landing = False  # route scenarios: the road is the landing target
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep * FRAME_SKIP

        m = self.model
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)  # noqa: E731
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        self.chassis_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.drive_ids = np.array([aid(f"drive_{w}") for w in WHEELS])
        self.steer_ids = np.array([aid("steer_fl"), aid("steer_fr")])
        self.jet_ids = np.array([aid(j.name) for j in JETS])
        self.jet_force = np.array([j.max_force for j in JETS])
        self.tire_ids = np.array([gid(f"tire_{w}") for w in WHEELS])
        self._body_col_arr = np.array(sorted({gid("chassis_col"), gid("roof_col")}))
        self.ground_id, self.pad_id = gid("ground"), gid("pad")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self._obstacle_arr = np.array([i for i, n in enumerate(names) if n.startswith("obs_")], dtype=int)
        self._car_geoms = np.array(sorted(set(self._body_col_arr.tolist()) | set(self.tire_ids.tolist())))
        self.obstacle_hit = False
        self._engine_glow = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "engine_glow")
        self._shadow = gid("car_shadow")
        self._ray_groups = np.array([1, 0, 0, 1, 0, 0], np.uint8)  # scenery + collision hulls, not fx

        # visual flames: (geom id, halo id, glow id, jet index, sign: 0 unidirectional / +1 / -1, nozzle, exhaust)
        self._flames = []
        for k, jet in enumerate(JETS):
            for name, p, ex in flame_geoms(jet):
                sign = 0 if not jet.bidirectional else (1 if name.endswith("_pos") else -1)
                self._flames.append((gid(name), gid(name + "_halo"), gid("glow_" + name[6:]), k, sign, p, ex))

        # geoms the effects move by writing the model (flames, halos, blob shadow, rain): their world poses
        # are refreshed right away so a frame shows the current effects, not the last step's
        fx = {g for g, halo, _, *_ in self._flames for g in (g, halo) if g >= 0} | {self._shadow}
        fx |= set(self.weather.rain_ids.tolist())
        self._fx_geoms = np.array(sorted(g for g in fx if g >= 0), dtype=int)
        self._fx_bodies = m.geom_bodyid[self._fx_geoms]
        self.smoke.nozzles = [(k, sign, p, ex) for _, _, _, k, sign, p, ex in self._flames]
        self.smoke.tint = np.asarray(world.PLUME_CORE, float)
        # exhaust history (one entry per step, oldest first): everything the smoke is drawn from
        self._fx_hist = np.zeros((HISTORY, ENTRY_DIM))
        self._fx_count = 0

        self.targets = self.course.targets()
        # After mj_step the derived quantities (xpos, xmat, contacts) belong to the pose before the last
        # substep, and the controllers read them like that. `_stale_pose` keeps that pose so that a restored
        # state can reproduce the live data exactly; None means they match qpos (after `forward`).
        self._pose_buf = np.zeros(m.nq)
        self._stale_pose: np.ndarray | None = None
        self.reset()

    # ------------------------------------------------------------------ helpers
    @property
    def pos(self) -> np.ndarray:
        return self.data.qpos[:3]

    @property
    def rot(self) -> np.ndarray:
        return self.data.xmat[self.chassis_id].reshape(3, 3)

    @property
    def vel(self) -> np.ndarray:
        return self.data.qvel[:3]

    @property
    def angvel_body(self) -> np.ndarray:
        return self.data.qvel[3:6]  # free-joint angular velocity is expressed in the body frame

    @property
    def target(self) -> np.ndarray:
        return self.targets[min(self.target_idx, len(self.targets) - 1)]

    @property
    def landing_stage(self) -> bool:
        return self.target_idx >= len(self.course.rings)

    def forward(self):
        """mj_forward at the current qpos: xpos, xmat and contacts now match it."""
        self.weather.apply_grip()
        mujoco.mj_forward(self.model, self.data)
        self._stale_pose = None

    @property
    def stale_pose(self) -> np.ndarray | None:
        """The qpos the derived quantities refer to when it differs from `data.qpos` (see `restore_derived`)."""
        return None if self._stale_pose is None else self._stale_pose.copy()

    def restore_derived(self, pose: np.ndarray | None):
        """Recompute xpos, xmat and contacts at `pose` (None: the current qpos).

        qpos, qvel and the solver warm start stay unchanged, so the data looks exactly as it does after
        `mj_step`.
        """
        if pose is None:
            self.forward()
            return
        m, d = self.model, self.data
        self.weather.apply_grip()
        keep = mujoco.mjtState.mjSTATE_QPOS | mujoco.mjtState.mjSTATE_WARMSTART
        saved = np.empty(mujoco.mj_stateSize(m, keep))
        mujoco.mj_getState(m, d, saved, keep)
        d.qpos[:] = pose
        mujoco.mj_forward(m, d)
        mujoco.mj_setState(m, d, saved, keep)
        self._pose_buf[:] = pose
        self._stale_pose = self._pose_buf

    def contacts(self):
        """Wheel contacts (any / on the pad / on the ground) and the geom the hull hit, if any."""
        wheels = {k: np.zeros(4, bool) for k in ("any", "pad", "ground")}
        n = self.data.ncon
        self.obstacle_hit = False
        if n == 0:
            return wheels, None
        g = self.data.contact.geom[:n]  # (n, 2), vectorised: iterating MjContact structs is slow
        if self._obstacle_arr.size:
            self.obstacle_hit = bool(
                (np.isin(g[:, 0], self._car_geoms) & np.isin(g[:, 1], self._obstacle_arr)).any()
                or (np.isin(g[:, 1], self._car_geoms) & np.isin(g[:, 0], self._obstacle_arr)).any()
            )
        body_hit = None
        for a, b in ((g[:, 0], g[:, 1]), (g[:, 1], g[:, 0])):
            col = np.isin(a, self._body_col_arr) & ~np.isin(b, self._body_col_arr)
            if col.any():
                body_hit = int(b[col][-1])
            for k, tid in enumerate(self.tire_ids):
                m = a == tid
                if m.any():
                    wheels["any"][k] = True
                    wheels["pad"][k] |= bool((b[m] == self.pad_id).any())
                    wheels["ground"][k] |= bool((b[m] == self.ground_id).any())
        return wheels, body_hit

    def _aero(self, v: np.ndarray) -> np.ndarray:
        """Aerodynamic force and torque.

        Without wind: isotropic drag. With wind: drag on the airspeed (velocity minus wind) with per-axis
        areas (front 0.6, side 2.4, top 4.0 m^2 Cd*A), acting at a centre of pressure behind and above the
        CoM, so a crosswind pushes the car sideways and turns its nose, and gusts rock it.
        """
        w = self.weather.wind
        if not w.any():
            return np.r_[-0.5 * AIR_DENSITY * DRAG_CDA * np.linalg.norm(v) * v, 0.0, 0.0, 0.0]
        R = self.rot
        rel_b = R.T @ (v - w)
        f_b = -0.5 * AIR_DENSITY * np.linalg.norm(rel_b) * np.array([DRAG_CDA, 2.4, 4.0]) * rel_b
        f = R @ f_b
        arm = R @ np.array([-0.35, 0.0, 0.12])
        return np.r_[f, np.cross(arm, f)]

    # ------------------------------------------------------------------ exhaust history
    def _fx_record(self):
        """Append this step to the exhaust history.

        Step, time, chassis pose (as drawn), velocity, jet commands and wind. Cheap, and always on: the smoke
        is a function of this history and the state must be complete whether or not frames are drawn.
        """
        d = self.data
        if self._fx_count < HISTORY:
            row = self._fx_hist[self._fx_count]
            self._fx_count += 1
        else:
            self._fx_hist[:-1] = self._fx_hist[1:]
            row = self._fx_hist[-1]
        row[0], row[1] = self.steps, d.time
        row[2:5] = d.xpos[self.chassis_id]
        row[5:9] = d.xquat[self.chassis_id]
        row[9:12] = self.vel
        row[12:21] = d.ctrl[self.jet_ids]
        row[21:24] = self.weather.wind

    def fx_state(self) -> tuple[int, np.ndarray]:
        """(valid count, (HISTORY, ENTRY_DIM) copy) of the exhaust history."""
        return self._fx_count, self._fx_hist.copy()

    def set_fx_state(self, count: int, hist: np.ndarray):
        self._fx_count = int(count)
        self._fx_hist[:] = hist

    def clear_fx(self):
        """Forget the exhaust history (no smoke until the jets fire again)."""
        self._fx_count = 0
        self._fx_hist[:] = 0.0

    # ------------------------------------------------------------------ visual effects
    def _refresh_fx_geoms(self):
        """World poses of the effect geoms from the model values just written (what mj_kinematics would do)."""
        m, d, g, b = self.model, self.data, self._fx_geoms, self._fx_bodies
        R = d.xmat[b].reshape(-1, 3, 3)
        d.geom_xpos[g] = d.xpos[b] + np.einsum("nij,nj->ni", R, m.geom_pos[g])
        d.geom_xmat[g] = (R @ _quat_to_mat(m.geom_quat[g])).reshape(-1, 9)

    def draw_effects(self, rng: np.random.Generator):
        """Write the visual effects of the current state into the model.

        Nozzle glow and short hot cores (flicker drawn from `rng`), smoke puffs, under-car engine light, rain
        and the blob shadow.
        """
        m, ctrl = self.model, self.data.ctrl
        lift = 0.0
        for g, halo, glow, k, sign, p, ex in self._flames:
            u = ctrl[self.jet_ids[k]]
            u = max(u, 0.0) if sign == 0 else max(sign * u, 0.0)
            if k < 4:
                lift += u / 4
            on = u > 0.02
            flicker = 1.0 + 0.12 * rng.standard_normal() if on else 1.0
            half = (0.04 + 0.32 * u) * flicker  # short hot core at the nozzle; the smoke carries the plume
            rad = 0.04 + 0.035 * u
            m.geom_size[g, :2] = (rad, half)
            m.geom_pos[g] = p + ex * (0.03 + half + rad)
            m.geom_rgba[g] = (*world.PLUME_CORE, 0.0 if not on else 0.55 + 0.4 * u)
            if halo >= 0:
                hh, hr = half * 0.9 + 0.05, 0.07 + 0.08 * u
                m.geom_size[halo, :2] = (hr, hh)
                m.geom_pos[halo] = p + ex * (0.03 + hh + hr * 0.6)
                m.geom_rgba[halo] = (*world.PLUME_HALO, 0.0 if not on else 0.12 + 0.2 * u)
            if glow >= 0:
                b = 0.04 + 0.96 * u  # dark when idle, bright when firing
                a = 1.0 if sign == 0 else float(on)  # side ports: hidden until they fire
                m.geom_rgba[glow] = (*(np.asarray(world.GLOW) * b), a)
        if self._engine_glow >= 0:
            m.light_diffuse[self._engine_glow] = np.asarray(world.PLUME_HALO) * 1.5 * lift
        self.weather.update(self.data.time, self.pos)
        self.smoke.draw(self._fx_hist[: self._fx_count], self.data.time, self.data)
        if self._shadow >= 0:  # blob shadow: soft dark ellipse on whatever surface is under the car
            gid_out = np.zeros(1, np.int32)
            dist = mujoco.mj_ray(
                m,
                self.data,
                self.pos + np.array([0, 0, -0.3]),
                np.array([0.0, 0.0, -1.0]),
                self._ray_groups,
                1,
                self.chassis_id,
                gid_out,
            )
            if dist >= 0 and gid_out[0] != self._shadow:
                h = dist + 0.3
                yaw = np.arctan2(self.rot[1, 0], self.rot[0, 0])
                m.geom_pos[self._shadow] = self.pos + np.array([0, 0, -h + 0.03])
                m.geom_quat[self._shadow] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
                grow = 1.0 + 0.04 * h
                m.geom_size[self._shadow] = (2.3 * grow, 1.2 * grow, 0.01)
                m.geom_rgba[self._shadow, 3] = 0.5 / (1.0 + 0.08 * h)
            else:
                m.geom_pos[self._shadow] = (0, 0, -50)
        self._refresh_fx_geoms()

    # ------------------------------------------------------------------ control
    def reset(self):
        """The car on the start line of the main course, at rest, with a full tank and no exhaust history."""
        mujoco.mj_resetData(self.model, self.data)
        self.clear_fx()
        self.target_idx = 0
        self.airborne_once = False
        self.status = "drive"
        self.forward()
        self.fuel = FUEL_CAPACITY
        self.steps = 0

    def step(self, action) -> bool:
        """Advance one control step (FRAME_SKIP substeps).

        Returns whether the episode terminated (landed on the pad and stopped, or crashed).
        """
        a = np.clip(np.asarray(action, dtype=np.float64), ACTION_LOW, ACTION_HIGH)
        d = self.data
        d.ctrl[self.drive_ids] = a[0]
        d.ctrl[self.steer_ids] = 0.5 * a[1]
        jets = a[2:] if self.fuel > 0 else np.zeros(len(JETS))
        d.ctrl[self.jet_ids] = jets
        burn = float(np.sum(np.abs(jets) * self.jet_force)) / 1000.0  # kN
        prev_x = float(self.pos[0])
        self.weather.apply_grip()  # the model may be shared with other lanes
        for _ in range(FRAME_SKIP):
            v = self.vel
            self.weather.step_wind(self.model.opt.timestep)
            d.xfrc_applied[self.chassis_id] = self._aero(v)
            self._pose_buf[:] = d.qpos
            mujoco.mj_step(self.model, d)
        self._stale_pose = self._pose_buf
        self.steps += 1
        self.fuel = max(self.fuel - burn * self.dt, 0.0)
        self._fx_record()

        terminated = False
        wheels, body_hit = self.contacts()
        if not wheels["any"].any() and self.pos[2] > 1.5:
            self.airborne_once = True
        if self.status == "drive" and self.airborne_once:
            self.status = "flight"

        if not self.landing_stage and prev_x < self.course.rings[self.target_idx].center[0] <= self.pos[0]:
            self.target_idx += 1  # past the plane of the next ring: the one after it is the target

        up_z = self.rot[2, 2]
        if self.landing_stage and wheels["pad"].sum() >= 3:
            self.status = "landed"
            if np.linalg.norm(self.vel) < 1.0 and up_z > 0.9:
                self.status = "success"
                terminated = True
        crash = None
        if self.obstacle_hit:
            crash = "hit obstacle"
        elif body_hit is not None:
            crash = "body impact"
        elif self.airborne_once and wheels["ground"].any() and not self.allow_ground_landing:
            crash = "ditched"
        elif self.pos[2] < -2 or abs(self.pos[1] - self.region_y) > 150 or self.pos[0] > 400:
            crash = "out of bounds"
        if crash:
            self.status = f"crash: {crash}"
            terminated = True
        return terminated
