"""OGBench Cube and Scene lanes on MuJoCo Warp: float32 physics, the controller in one float64 kernel."""

from __future__ import annotations

import copy
import math

import mujoco
import numpy as np
import torch
from ogbench.manipspace import lie

from ...._core.tensordicts import build_tensordict
from ...._execution.mjbatch import WarpBatch
from .compatibility import COMPILED_FIELDS, compiled_fields
from .manipulation import site_ids
from .manipulation_batch import LaneModels, OGBManipulationBatchAdapter, Snapshot, make_lanes, metadata_env
from .pcg import pcg_decode, pcg_encode

_KERNELS = None
#: Contacts per environment and constraints per environment of the MuJoCo Warp buffers.
NACON_PER_ENV = 64
NJMAX = 512


def contact_exclusions(model) -> list[tuple[int, int]]:
    """Geom pairs that MuJoCo Warp does not test for contact: the links of the Robotiq gripper with each other.

    The linkage moves with one degree of freedom, and the pads touch before any two links do, so these pairs
    make no contacts; their bounding volumes overlap on every substep, which costs a mesh-mesh GJK/EPA each.
    The pads keep their contacts with each other.
    """

    def body(g):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or ""

    gripper = [
        g
        for g in range(model.ngeom)
        if (model.geom_contype[g] or model.geom_conaffinity[g]) and body(g).startswith("ur5e/robotiq/")
    ]
    pads = {g for g in gripper if body(g).endswith("_pad")}
    return [(a, b) for i, a in enumerate(gripper) for b in gripper[i + 1 :] if not (a in pads and b in pads)]


def _batchable(names):
    """The names that are MuJoCo Warp model arrays with a per-world dimension."""
    import dataclasses

    import mujoco_warp as mjw

    fields = {f.name: f for f in dataclasses.fields(mjw.Model)}
    return tuple(n for n in names if n in fields and (getattr(fields[n].type, "shape", ()) or ("",))[0] == "*")


def _kernels():
    """Build the Warp controller kernel once."""
    global _KERNELS
    if _KERNELS is not None:
        return _KERNELS
    import warp as wp

    vec6d = wp.types.vector(length=6, dtype=wp.float64)
    mat66d = wp.types.matrix(shape=(6, 6), dtype=wp.float64)

    @wp.func
    def qmul(a: wp.vec4d, b: wp.vec4d):
        return wp.vec4d(
            a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
        )

    @wp.func
    def qconj(q: wp.vec4d):
        return wp.vec4d(q[0], -q[1], -q[2], -q[3])

    @wp.func
    def qrot(q: wp.vec4d, v: wp.vec3d):
        u = wp.vec3d(q[1], q[2], q[3])
        t = wp.float64(2.0) * wp.cross(u, v)
        return v + q[0] * t + wp.cross(u, t)

    @wp.func
    def qnorm(q: wp.vec4d):
        n = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        if n < wp.float64(1e-15):
            return wp.vec4d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0), wp.float64(0.0))
        return q / n

    @wp.func
    def zquat(theta: wp.float64):
        h = wp.float64(0.5) * theta
        return wp.vec4d(wp.cos(h), wp.float64(0.0), wp.float64(0.0), wp.sin(h))

    @wp.func
    def yaw_of(q: wp.vec4d):
        return wp.atan2(
            wp.float64(2.0) * (q[0] * q[3] + q[1] * q[2]),
            wp.float64(1.0) - wp.float64(2.0) * (q[2] * q[2] + q[3] * q[3]),
        )

    @wp.func
    def quat2vel(q: wp.vec4d):
        v = wp.vec3d(q[1], q[2], q[3])
        s = wp.length(v)
        axis = wp.vec3d(wp.float64(1.0), wp.float64(0.0), wp.float64(0.0))
        if s >= wp.float64(1e-15):
            axis = v / s
        speed = wp.float64(2.0) * wp.atan2(s, q[0])
        if speed > wp.float64(math.pi):
            speed = speed - wp.float64(2.0 * math.pi)
        return axis * speed

    @wp.kernel
    def control(
        actions: wp.array2d(dtype=wp.float32),
        active: wp.array(dtype=wp.bool),
        qpos: wp.array2d(dtype=wp.float32),
        pinch_pos: wp.array2d(dtype=wp.float64),
        pinch_mat: wp.array2d(dtype=wp.float64),
        scale: wp.array(dtype=wp.float64),
        low: wp.vec3d,
        high: wp.vec3d,
        down: wp.vec4d,
        tpa_quat: wp.vec4d,
        tpa_pos: wp.vec3d,
        arm_qpos: wp.array(dtype=wp.int32),
        gripper_qpos: int,
        arm_ctrl: wp.array(dtype=wp.int32),
        gripper_ctrl: wp.array(dtype=wp.int32),
        body_pos: wp.array(dtype=wp.vec3d),
        body_quat: wp.array(dtype=wp.vec4d),
        body_joint: wp.array(dtype=wp.int32),
        joint_pos: wp.array(dtype=wp.vec3d),
        joint_axis: wp.array(dtype=wp.vec3d),
        joint_qpos0: wp.array(dtype=wp.float64),
        site_pos: wp.vec3d,
        site_quat: wp.vec4d,
        max_angle: wp.float64,
        ctrl: wp.array2d(dtype=wp.float32),
    ):
        w = wp.tid()
        if not active[w]:
            return
        zero = wp.float64(0.0)
        one = wp.float64(1.0)
        a = vec6d()
        for k in range(5):
            a[k] = wp.float64(actions[w, k]) * scale[k]
        eff = wp.vec3d(pinch_pos[w, 0], pinch_pos[w, 1], pinch_pos[w, 2])
        eff_yaw = wp.atan2(pinch_mat[w, 3], pinch_mat[w, 0])
        opening = wp.clamp(wp.float64(qpos[w, gripper_qpos]) / wp.float64(0.8), zero, one)
        target = eff + wp.vec3d(a[0], a[1], a[2])
        target = wp.vec3d(
            wp.clamp(target[0], low[0], high[0]),
            wp.clamp(target[1], low[1], high[1]),
            wp.clamp(target[2], low[2], high[2]),
        )
        orientation = qmul(qmul(zquat(a[3]), zquat(eff_yaw)), qconj(down))
        yaw = wp.clamp(yaw_of(orientation), wp.float64(-math.pi), wp.float64(math.pi))
        quat = qmul(qmul(zquat(yaw), down), tpa_quat)
        pos = qrot(qmul(zquat(yaw), down), tpa_pos) + target
        grip = wp.clamp(opening + a[4], zero, one)

        nb = body_pos.shape[0]
        q = vec6d()
        for j in range(6):
            q[j] = wp.float64(qpos[w, arm_qpos[j]])
        for _ in range(20):
            # Forward kinematics of the serial arm, then the attachment site and the hinge axes and anchors.
            xpos = wp.vec3d(zero, zero, zero)
            xquat = wp.vec4d(one, zero, zero, zero)
            axes = mat66d()
            anchors = mat66d()
            for b in range(nb):
                xpos = xpos + qrot(xquat, body_pos[b])
                xquat = qmul(xquat, body_quat[b])
                j = body_joint[b]
                if j >= 0:
                    axis = qrot(xquat, joint_axis[j])
                    anchor = qrot(xquat, joint_pos[j]) + xpos
                    half = wp.float64(0.5) * (q[j] - joint_qpos0[j])
                    s = wp.sin(half)
                    local = wp.vec4d(wp.cos(half), joint_axis[j][0] * s, joint_axis[j][1] * s, joint_axis[j][2] * s)
                    xquat = qnorm(qmul(xquat, local))
                    xpos = anchor - qrot(xquat, joint_pos[j])
                    for k in range(3):
                        axes[j, k] = axis[k]
                        anchors[j, k] = anchor[k]
            spos = xpos + qrot(xquat, site_pos)
            squat = qmul(xquat, site_quat)
            err = vec6d()
            perr = pos - spos
            rerr = quat2vel(qmul(quat, qconj(squat)))
            for k in range(3):
                err[k] = perr[k]
                err[k + 3] = rerr[k]
            if wp.length(perr) <= wp.float64(1e-4) and wp.length(rerr) <= wp.float64(1e-4):
                break
            jac = mat66d()
            for j in range(6):
                axis = wp.vec3d(axes[j, 0], axes[j, 1], axes[j, 2])
                lin = wp.cross(axis, spos - wp.vec3d(anchors[j, 0], anchors[j, 1], anchors[j, 2]))
                for k in range(3):
                    jac[k, j] = lin[k]
                    jac[k + 3, j] = axis[k]
            # Damped least squares, H x = err with H = J J^T + 1e-12 I, by Gaussian elimination.
            h = mat66d()
            for r in range(6):
                for c in range(6):
                    acc = zero
                    for k in range(6):
                        acc = acc + jac[r, k] * jac[c, k]
                    h[r, c] = acc
                h[r, r] = h[r, r] + wp.float64(1e-12)
            y = err
            for c in range(6):
                pivot = c
                for r in range(c + 1, 6):
                    if wp.abs(h[r, c]) > wp.abs(h[pivot, c]):
                        pivot = r
                if pivot != c:
                    for k in range(6):
                        t = h[c, k]
                        h[c, k] = h[pivot, k]
                        h[pivot, k] = t
                    t = y[c]
                    y[c] = y[pivot]
                    y[pivot] = t
                for r in range(c + 1, 6):
                    f = h[r, c] / h[c, c]
                    for k in range(c, 6):
                        h[r, k] = h[r, k] - f * h[c, k]
                    y[r] = y[r] - f * y[c]
            for c in range(5, -1, -1):
                acc = y[c]
                for k in range(c + 1, 6):
                    acc = acc - h[c, k] * y[k]
                y[c] = acc / h[c, c]
            update = vec6d()
            biggest = zero
            for j in range(6):
                acc = zero
                for k in range(6):
                    acc = acc + jac[k, j] * y[k]
                update[j] = acc
                biggest = wp.max(biggest, wp.abs(acc))
            if biggest > max_angle:
                update = update * (max_angle / biggest)
            q = q + update
        for j in range(6):
            ctrl[w, arm_ctrl[j]] = wp.float32(q[j])
        for k in range(gripper_ctrl.shape[0]):
            ctrl[w, gripper_ctrl[k]] = wp.float32(wp.float64(255.0) * grip)

    _KERNELS = control
    return _KERNELS


def _orthonormal(rotations):
    """The nearest-by-Gram-Schmidt rotations in float64 of float32 rotation matrices (..., 3, 3), by rows."""
    r0 = torch.nn.functional.normalize(rotations[..., 0, :], dim=-1)
    r1 = rotations[..., 1, :]
    r1 = torch.nn.functional.normalize(r1 - (r1 * r0).sum(-1, keepdim=True) * r0, dim=-1)
    return torch.stack([r0, r1, torch.linalg.cross(r0, r1)], dim=-2)


class WarpController:
    """``ManipSpaceEnv.set_control`` for every lane in one MuJoCo Warp kernel, in float64.

    The kernel runs OGBench's differential IK, each lane to its own stop, on the arm's kinematics written out for
    its serial chain of hinges.
    """

    def __init__(self, env, device):
        import warp as wp

        self._wp = wp
        self.kernel = _kernels()
        ik = env._ik._model
        site = int(env._ik._site_ids[0])
        chain, body = [], int(ik.site_bodyid[site])
        while body > 0:
            chain.append(body)
            body = int(ik.body_parentid[body])
        chain.reverse()
        if any(ik.body_jntnum[b] > 1 for b in chain) or any(t != mujoco.mjtJoint.mjJNT_HINGE for t in ik.jnt_type):
            raise ValueError("The Warp controller needs an arm of hinges, one per body")
        dev = str(device)

        def arr(values, dtype):
            return wp.array(values, dtype=dtype, device=dev)

        self._args = {
            "scale": arr(np.asarray(env.action_high, np.float64), wp.float64),
            "low": wp.vec3d(*env._workspace_bounds[0]),
            "high": wp.vec3d(*env._workspace_bounds[1]),
            "down": wp.vec4d(*env._effector_down_rotation.wxyz),
            "tpa_quat": wp.vec4d(*env._T_pa.wxyz_xyz[:4]),
            "tpa_pos": wp.vec3d(*env._T_pa.wxyz_xyz[4:]),
            "arm_qpos": arr(env.model.jnt_qposadr[env._arm_joint_ids].astype(np.int32), wp.int32),
            "gripper_qpos": int(env.model.jnt_qposadr[env._gripper_opening_joint_id]),
            "arm_ctrl": arr(np.asarray(env._arm_actuator_ids, np.int32), wp.int32),
            "gripper_ctrl": arr(np.asarray(env._gripper_actuator_ids, np.int32), wp.int32),
            "body_pos": arr(ik.body_pos[chain], wp.vec3d),
            "body_quat": arr(ik.body_quat[chain], wp.vec4d),
            "body_joint": arr(
                np.array([ik.body_jntadr[b] if ik.body_jntnum[b] else -1 for b in chain], np.int32), wp.int32
            ),
            "joint_pos": arr(ik.jnt_pos, wp.vec3d),
            "joint_axis": arr(ik.jnt_axis, wp.vec3d),
            "joint_qpos0": arr(ik.qpos0[ik.jnt_qposadr], wp.float64),
            "site_pos": wp.vec3d(*ik.site_pos[site]),
            "site_quat": wp.vec4d(*ik.site_quat[site]),
            "max_angle": float(env._ik._max_angle_change),
        }
        self.device = dev

    def close(self):
        self._args = {}

    def control(self, actions, active, qpos, pinch_pos, pinch_mat, ctrl):
        wp = self._wp
        a = self._args
        wp.launch(
            self.kernel,
            dim=len(actions),
            inputs=[
                wp.from_torch(actions),
                wp.from_torch(active),
                wp.from_torch(qpos),
                wp.from_torch(pinch_pos),
                wp.from_torch(pinch_mat),
                a["scale"],
                a["low"],
                a["high"],
                a["down"],
                a["tpa_quat"],
                a["tpa_pos"],
                a["arm_qpos"],
                a["gripper_qpos"],
                a["arm_ctrl"],
                a["gripper_ctrl"],
                a["body_pos"],
                a["body_quat"],
                a["body_joint"],
                a["joint_pos"],
                a["joint_axis"],
                a["joint_qpos0"],
                a["site_pos"],
                a["site_quat"],
                a["max_angle"],
                wp.from_torch(ctrl),
            ],
            device=self.device,
            stream=wp.stream_from_torch(torch.cuda.current_stream(torch.device(self.device))),
        )


#: Integration State leaves that MuJoCo Warp steps; the others keep their loaded values.
_WARP_PHYSICS = {"time": "time", "qpos": "qpos", "qvel": "qvel", "warmstart": "qacc_warmstart", "ctrl": "ctrl"}
_WARP_LOADED = {"mocap_pos": "mocap_pos", "mocap_quat": "mocap_quat", "qfrc_applied": "qfrc_applied"}


class OGBManipulationWarpAdapter(OGBManipulationBatchAdapter):
    """Cube/Scene lanes on MuJoCo Warp (float32 physics) with OGBench's resets, restores and Variations.

    The step follows ``ManipSpaceEnv.step``: the controller (one float64 Warp kernel), ``frame_skip`` substeps,
    ``rne_postconstraint`` and the family's ``post_step`` in Torch, without a host synchronization (Scene
    synchronizes once to detect a cube that left the workspace). State has the leaves of the CPU State; its
    physics leaves are the float32 values in float64.
    """

    execution_mode = "NATIVE_MJWARP"

    def __init__(
        self, family, num_envs=1, device="cuda", seed=0, reward_task_id=None, *, lanes=None, exclude_contacts=True
    ):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("The MuJoCo Warp Cube/Scene backend requires a CUDA device")
        self.family, self.num_envs, self.device = family, num_envs, device
        self._closed = False
        self.physics = self._control = None
        self.lanes = make_lanes(family, num_envs, seed, reward_task_id) if lanes is None else lanes
        try:
            self._meta = env = metadata_env(family)
            self.fields, self.nq = self.lanes.fields, self.lanes.nq
            self._order = tuple(self.fields)
            self._snapshot = Snapshot(self.fields, num_envs)
            # Variations recompile these fields per lane, and Scene's locks and targets set damping and sites.
            batched = _batchable((*COMPILED_FIELDS, "dof_damping", "site_pos"))
            self._models = LaneModels(env, ())
            self._variants = {}
            state = self.lanes.get_state()
            self.physics = WarpBatch(
                self._lane_models({key: state[key].numpy() for key in self._order if key[0] == "model"}),
                num_envs,
                device,
                frame_skip=int(env._n_steps),
                rne_postconstraint=True,
                outputs=("site_xpos", "site_xmat", "cfrc_ext"),
                batched_fields=batched,
                # Grasps, the drawer, the window and the buttons add contacts and constraints beyond the
                # sizes that MuJoCo Warp derives from the reset state.
                naconmax=NACON_PER_ENV * num_envs,
                njmax=NJMAX,
                exclude_pairs=contact_exclusions(env.model) if exclude_contacts else (),
            )
            self._control = WarpController(env, device)
        except BaseException:
            self.close()
            raise
        p = self.physics
        self._qpos, self._qvel, self._ctrl = p.data("qpos"), p.data("qvel"), p.data("ctrl")
        self._mocap_pos = p.data("mocap_pos")
        self._damping = p._wp.to_torch(p._m.dof_damping)
        self._site_pos = p._wp.to_torch(p._m.site_pos)
        self._pos_sites = torch.as_tensor(site_ids(env)[0], device=device)
        self._rot_sites = torch.as_tensor(site_ids(env)[1], device=device)
        self._pad = int(env.model.body("ur5e/robotiq/right_pad").id)
        self._cube = int(env.model.joint("object_joint_0").qposadr[0])
        self._cube_dof = int(env.model.joint("object_joint_0").dofadr[0])
        self._target = int(env._cube_target_mocap_ids[0])
        self._workspace = torch.as_tensor(env._workspace_bounds, device=device)
        self._object_bounds = env._object_sampling_bounds
        self._reward_task = reward_task_id is not None
        if family == "Scene":
            self._buttons = torch.as_tensor(
                [int(env.model.joint(f"buttonbox_joint_{i}").qposadr[0]) for i in range(env._num_buttons)],
                device=device,
            )
            self._drawer = int(env.model.joint("drawer_slide").qposadr[0])
            self._window = int(env.model.joint("window_slide").qposadr[0])
            self._lock_dofs = torch.as_tensor(
                [int(env.model.joint(name).dofadr[0]) for name in ("drawer_slide", "window_slide")], device=device
            )
            self._num_button_states = int(env._num_button_states)
        self._dirty = False
        self._load(state)

    @property
    def envs(self):
        """The in-process OGBench objects as of the last reset or restore, or the metadata object."""
        return getattr(self.lanes, "envs", [self._meta])

    def _lane_models(self, columns):
        """One MjModel per lane: the metadata model, or a copy with the compiled fields of the lane's Variation."""
        base = self._meta.model
        models = [base] * self.num_envs
        for key, (values, lanes) in self._models.variants(columns).items():
            model = self._variants.get(key)
            if model is None:
                model = self._variants[key] = copy.copy(base)
                fields, _ = compiled_fields(self.family, values)
                for name, value in fields.items():
                    if name == "tex_data":
                        start = int(model.tex_adr[model.texture("grid").id])
                        model.tex_data[start : start + len(value)] = value
                    else:
                        getattr(model, name)[:] = value
            for i in lanes:
                models[i] = model
        return models

    # Loading lanes from State.

    def _load(self, state):
        state = state.to(self.device)
        # MuJoCo Warp steps the physics leaves in _WARP_PHYSICS; the others keep their loaded values.
        self._columns = {
            key: state[key].clone() for key in self._order if key[0] != "physics" or key[1] not in _WARP_PHYSICS
        }
        host = {key: state[key].cpu().numpy() for key in self._order if key[0] in ("model", "scene")}
        models = self._lane_models(host)
        ids = tuple(id(m) for m in models)
        if ids != getattr(self, "_model_ids", None):
            self.physics.set_models(models)
            self._model_ids = ids
        physics = {
            field: state["physics", name]
            for name, field in {**_WARP_PHYSICS, **_WARP_LOADED}.items()
            if ("physics", name) in state.keys(True, True)
        }
        for name in ("ctrl", "qfrc_applied"):
            if name in physics:
                target = self.physics.data(name)
                target.copy_(physics[name].to(target.dtype).reshape(target.shape))
        if self.family == "Scene":
            fields = self._models.fields(host)
            self._damping.copy_(torch.as_tensor(fields["dof_damping"], device=self.device, dtype=self._damping.dtype))
            self._site_pos.copy_(
                torch.as_tensor(fields["site_pos"], device=self.device, dtype=self._site_pos.dtype).reshape(
                    self._site_pos.shape
                )
            )
        writes = {k: (v[:, 0] if k == "time" else v) for k, v in physics.items() if k in self.physics.fields}
        self.physics.set_physics(writes, forward=True)
        success = self._columns["task", "success"][:, 0]
        self._outputs = {
            "reward": torch.where(success, 1.0, 0.0).double()
            if not self._reward_task
            else torch.zeros(self.num_envs, dtype=torch.float64, device=self.device),
            "success": success.clone(),
            "terminated": success.clone(),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }

    def _load_models(self, columns):
        host = {
            k: (v.cpu().numpy() if torch.is_tensor(v) else v) for k, v in columns.items() if k[0] in ("model", "scene")
        }
        models = self._lane_models(host)
        ids = tuple(id(m) for m in models)
        if ids != getattr(self, "_model_ids", None):
            self.physics.set_models(models)
            self._model_ids = ids

    # State.

    def get_state(self):
        self._check_open()
        columns = dict(self._columns)
        for name, field in _WARP_PHYSICS.items():
            if ("physics", name) in self.fields:
                columns["physics", name] = self.physics.data(field).to(torch.float64)
        return self._snapshot.build(columns, self.device)

    def configure(self, values):
        self._check_open()
        self._push()
        self.lanes.configure(values)
        current = self.lanes.get_state().to(self.device)
        for key in self._order:
            if key[0] in ("variation", "model"):
                self._columns[key] = current[key].clone()
        self._load_models(self._columns)

    def check_step(self, actions, active):
        actions = torch.as_tensor(actions, device=self.device)
        if actions.shape != (self.num_envs, 5) or actions.dtype not in (torch.float32, torch.float64):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        if active is not None:
            active = torch.as_tensor(active, device=self.device)
            if active.shape != (self.num_envs,) or active.dtype != torch.bool:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        actions, active = self.check_step(actions, active)
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device) if active is None else active
        self._advance(actions.to(torch.float32).contiguous(), mask, active is None)
        result = {key: value.clone() for key, value in self._outputs.items()}
        if capture_state:
            result["state"] = self.get_state()
        return build_tensordict(result, [self.num_envs], self.device)

    def _advance(self, actions, mask, everyone):
        c = self._columns
        m = mask[:, None]
        site_xpos, site_xmat = c["controller", "site_xpos"], c["controller", "site_xmat"]
        self._control.control(
            actions, mask, self._qpos, site_xpos[:, 0].contiguous(), site_xmat[:, 0].contiguous(), self._ctrl
        )
        scene = self.family == "Scene"
        if scene:
            c["scene", "prev_button_states"] = torch.where(
                m, c["scene", "cur_button_states"], c["scene", "prev_button_states"]
            )
            button_pos = self._qpos[:, self._buttons].to(torch.float64)
        c["output", "prev_qpos"] = torch.where(m, self._qpos.to(torch.float64), c["output", "prev_qpos"])
        c["output", "prev_qvel"] = torch.where(m, self._qvel.to(torch.float64), c["output", "prev_qvel"])
        self.physics.step(active=None if everyone else mask)
        force = self.physics.data("cfrc_ext")[:, self._pad].to(torch.float64)
        c["output", "gripper_force"] = torch.where(m, force, c["output", "gripper_force"])
        if scene:
            self._scene_post_step(mask, button_pos)
        out_pos = self.physics.data("site_xpos")[:, self._pos_sites].to(torch.float64)
        out_mat = _orthonormal(self.physics.data("site_xmat")[:, self._rot_sites].to(torch.float64))
        out_mat = out_mat.reshape(self.num_envs, -1, 9)
        c["controller", "site_xpos"] = torch.where(m[:, :, None], out_pos, site_xpos)
        c["controller", "site_xmat"] = torch.where(m[:, :, None], out_mat, site_xmat)
        qpos = self._qpos.to(torch.float64)
        delta = qpos[:, self._cube : self._cube + 3] - self._mocap_pos[:, self._target].to(torch.float64)
        cube = delta.norm(dim=-1) <= 0.04
        if scene:
            buttons = c["scene", "cur_button_states"] == c["scene", "target_button_states"]
            drawer = (qpos[:, self._drawer] - c["scene", "target_drawer_pos"][:, 0]).abs() <= 0.04
            window = (qpos[:, self._window] - c["scene", "target_window_pos"][:, 0]).abs() <= 0.04
            parts = torch.cat([cube[:, None], buttons, drawer[:, None], window[:, None]], dim=1)
        else:
            parts = cube[:, None]
        success = parts.all(dim=1)
        c["task", "success"] = torch.where(m, success[:, None], c["task", "success"])
        if self._reward_task:
            reward = parts.sum(dim=1).to(torch.float64) - parts.shape[1]
        else:
            reward = success.to(torch.float64)
        out = self._outputs
        for key, value in (
            ("reward", reward),
            ("success", success),
            ("terminated", success),
            ("truncated", torch.zeros_like(success)),
        ):
            out[key] = torch.where(mask, value, out[key])

    def _scene_post_step(self, mask, button_pos):
        c = self._columns
        cube = self._qpos[:, self._cube : self._cube + 3].to(torch.float64)
        unhealthy = mask & ((cube <= self._workspace[0] - 0.2) | (cube >= self._workspace[1] + 0.2)).any(dim=1)
        if bool(unhealthy.any()):
            self._repair(torch.nonzero(unhealthy).flatten().cpu().numpy())
        current = self._qpos[:, self._buttons].to(torch.float64)
        pressed = mask[:, None] & (button_pos > -0.02) & (current <= -0.02)
        states = c["scene", "cur_button_states"]
        states = torch.where(pressed, (states + 1) % self._num_button_states, states)
        c["scene", "cur_button_states"] = states
        self._damping[:, self._lock_dofs] = torch.where(states == 0, 1e6, 2.0).to(self._damping.dtype)
        self.physics.forward()

    def _repair(self, ids):
        """OGBench's reset of a cube that left the workspace, drawing from the lane's environment generator."""
        rng = np.random.Generator(np.random.PCG64())
        columns = self._columns
        for i in ids:
            print("Numerical instability detected. Resetting cube positions.", flush=True)
            rng.bit_generator.state = pcg_decode(
                {
                    k: columns["rng", "environment", k][i].cpu().numpy()
                    for k in ("state", "increment", "has_uint32", "uinteger")
                }
            )
            xy = rng.uniform(*self._object_bounds)
            yaw = rng.uniform(0, 2 * np.pi)
            values = [*xy, 0.02, *lie.SO3.from_z_radians(yaw).wxyz.tolist()]
            self._qpos[i, self._cube : self._cube + 7] = torch.as_tensor(
                values, dtype=self._qpos.dtype, device=self.device
            )
            self._qvel[i, self._cube_dof : self._cube_dof + 6] = 0
            for key, value in pcg_encode(rng).items():
                columns["rng", "environment", key][i] = torch.as_tensor(value, device=self.device)
        self._dirty = True
