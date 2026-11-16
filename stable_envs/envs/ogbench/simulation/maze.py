from __future__ import annotations

from functools import cache

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.state_schema import Field, StateSchema
from ...._execution.protocol import SimulatorAdapter
from ...._execution.state_columns import owned_columns
from .pcg import bytes_uint128, pcg_state
from .pcg import uint128_bytes as _uint128_bytes

#: body -> (qpos fields, qvel fields, action width). Legged qpos splits at the free joint.
BODIES = {
    "point": ((("root/position", 2),), (("root/velocity", 2),), 2),
    "ant": (
        (("root/position", 3), ("root/orientation", 4), ("joints/position", 8)),
        (("root/velocity", 6), ("joints/velocity", 8)),
        8,
    ),
    "humanoid": (
        (("root/position", 3), ("root/orientation", 4), ("joints/position", 21)),
        (("root/velocity", 6), ("joints/velocity", 21)),
        21,
    ),
}

#: State groups. ``physics`` changes on every step; ``host`` holds the task and the RNGs, which change at reset,
#: restore and teleport.
PHYSICS = "maze/physics/float64"
HOST_FLOAT, HOST_INT, HOST_BYTES, HOST_BOOL = (
    "maze/host/float64",
    "maze/host/int64",
    "maze/host/uint8",
    "maze/host/bool",
)
HOST_GROUPS = (HOST_FLOAT, HOST_INT, HOST_BYTES, HOST_BOOL)


@cache
def maze_schema(loco: str) -> StateSchema:
    """The State layout of a maze body: one leaf per dtype and role."""
    qpos, qvel, _ = BODIES[loco]
    nv = sum(width for _, width in qvel)
    f64, i64 = torch.float64, torch.int64
    return StateSchema(
        f"OGBMaze {loco}",
        [
            *(Field(name, (width,), f64, PHYSICS) for name, width in qpos + qvel),
            Field("simulator/time", (1,), f64, PHYSICS),
            Field("simulator/warmstart", (nv,), f64, PHYSICS),
            Field("task/goal_pos", (3,), f64, HOST_FLOAT),
            Field("task/id", (1,), i64, HOST_INT),
            Field("rng/env_state", (16,), torch.uint8, HOST_BYTES),
            Field("rng/env_inc", (16,), torch.uint8, HOST_BYTES),
            Field("rng/env_has_uint32", (1,), i64, HOST_INT),
            Field("rng/env_uinteger", (1,), i64, HOST_INT),
            Field("rng/global_keys", (624,), i64, HOST_INT),
            Field("rng/global_position", (1,), i64, HOST_INT),
            Field("rng/global_has_gauss", (1,), torch.bool, HOST_BOOL),
            Field("rng/global_cached_gaussian", (1,), f64, HOST_FLOAT),
        ],
    )


def numpy_dtype(dtype: torch.dtype) -> np.dtype:
    return torch.empty((), dtype=dtype).numpy().dtype


def group_path(key: str) -> tuple[str, ...]:
    return tuple(key.split("/"))


def _split(fields, values):
    out, start = {}, 0
    for key, width in fields:
        out[key] = values[start : start + width].copy()
        start += width
    return out


def _make_maze(loco, maze):
    from ogbench.locomaze.maze import make_maze_env

    return make_maze_env(
        loco_env_type=loco,
        maze_env_type="maze",
        maze_type=maze,
        ob_type="states",
        width=224,
        height=224,
    )


# Clearance in world units between the ball and every wall below which an environment takes a full MuJoCo step.
_REST_CLEARANCE = 1e-3


def _rest_walls(model):
    """Return the wall boxes ``(lo, hi)``, ball radius and ball offset of a point maze, or None.

    A step sets the ball velocity to zero. With gravity normal to both slide joints, no joint damping,
    stiffness, friction or limits, and only the floor in contact, MuJoCo then leaves the ball where it
    is. None means that the model does not meet these conditions.
    """
    import mujoco

    m = model
    slide = int(mujoco.mjtJoint.mjJNT_SLIDE)
    if m.nv != 2 or m.njnt != 2 or any(int(t) != slide for t in m.jnt_type):
        return None
    if not np.array_equal(m.jnt_axis, np.eye(3)[:2]) or m.jnt_limited.any() or m.jnt_stiffness.any():
        return None
    if (
        m.dof_damping.any()
        or m.dof_frictionloss.any()
        or m.neq
        or m.ntendon
        or m.npair
        or m.nexclude
        or m.nmocap
        or m.na
    ):
        return None
    if m.opt.gravity[:2].any() or m.opt.wind.any() or m.opt.density or m.opt.viscosity:
        return None
    body = int(m.jnt_bodyid[0])
    own = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == body and (m.geom_contype[g] or m.geom_conaffinity[g])]
    if len(own) != 1 or int(m.geom_type[own[0]]) != int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return None
    ball = own[0]
    radius = float(m.geom_size[ball][0])
    offset = np.asarray(m.geom_pos[ball], np.float64)
    data = mujoco.MjData(m)
    mujoco.mj_kinematics(m, data)
    boxes, floors = [], 0
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) == body:
            continue
        if not (m.geom_contype[g] & m.geom_conaffinity[ball] or m.geom_conaffinity[g] & m.geom_contype[ball]):
            continue
        if m.geom_margin[g] or m.geom_gap[g] or not np.array_equal(data.geom_xmat[g], np.eye(3).ravel()):
            return None
        kind = int(m.geom_type[g])
        if kind == int(mujoco.mjtGeom.mjGEOM_PLANE) and data.geom_xpos[g][2] == 0 and offset[2] == radius:
            floors += 1
        elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
            boxes.append((data.geom_xpos[g] - m.geom_size[g], data.geom_xpos[g] + m.geom_size[g]))
        else:
            return None
    if floors != 1 or not boxes or m.geom_margin[ball] or m.geom_gap[ball]:
        return None
    lo, hi = (np.stack(side) for side in zip(*boxes, strict=True))
    return lo, hi, radius, offset


def _resting_instances(rest, envs, qpos):
    """Environments whose step leaves the ball where it is, so only the clock advances.

    This is exact: the step sets the velocity to zero, ``_rest_walls`` checked that no joint, gravity
    or fluid term then makes a force, and here the environment has no control or applied force and its ball
    keeps a clearance from every wall. The only contact is the floor, whose normal is orthogonal to
    both joints, so MuJoCo computes zero acceleration and the warmstart it stores is zero.
    """
    if rest is None:
        return np.zeros(len(qpos), bool)
    lo, hi, radius, offset = rest
    centre = np.zeros((len(qpos), 3))
    centre[:, :2] = qpos
    centre += offset
    near = np.clip(centre[:, None], lo, hi) - centre[:, None]
    clear = (near * near).sum(-1).min(1) > (radius + _REST_CLEARANCE) ** 2
    idle = [not (e.data.ctrl.any() or e.data.qfrc_applied.any() or e.data.xfrc_applied.any()) for e in envs]
    return clear & np.asarray(idle, bool)


def tree(groups: dict) -> dict:
    """Nest ``{"a/b/c": value}`` as ``{"a": {"b": {"c": value}}}``."""
    out: dict = {}
    for key, value in groups.items():
        *path, leaf = key.split("/")
        node = out
        for part in path:
            node = node.setdefault(part, {})
        node[leaf] = value
    return out


def decode_state(state, num_envs, schema, num_tasks):
    """Return ``(host State, field views, [(task id, PCG64 state, legacy MT19937 state)] per environment)``, or raise."""
    if not isinstance(state, TensorDict) or state.batch_size != torch.Size([num_envs]):
        raise TypeError("OGBMaze requires full TensorDict State with batch [num_envs]")
    schema.validate(state, finite=True)
    host = state.cpu()
    v = schema.views(host, numpy=True)
    decoded = []
    for i in range(num_envs):
        task_id = int(v["task/id"][i, 0])
        if not 0 <= task_id <= num_tasks:
            raise ValueError(f"Invalid OGBench task id {task_id}")
        pcg = pcg_state(
            bytes_uint128(v["rng/env_state"][i]),
            bytes_uint128(v["rng/env_inc"][i]),
            int(v["rng/env_has_uint32"][i, 0]),
            int(v["rng/env_uinteger"][i, 0]),
        )
        np.random.PCG64().state = pcg
        legacy = (
            "MT19937",
            v["rng/global_keys"][i].astype(np.uint32),
            int(v["rng/global_position"][i, 0]),
            int(v["rng/global_has_gauss"][i, 0]),
            float(v["rng/global_cached_gaussian"][i, 0]),
        )
        np.random.RandomState().set_state(legacy)
        decoded.append((task_id, pcg, legacy))
    return v, decoded


class OGBMazeAdapter(SimulatorAdapter):
    """Independent CPU locomotion-maze simulators with owned state snapshots.

    ``device`` selects output placement. Each environment owns both its environment PCG64
    and its own copy of NumPy's legacy global RNG (``np.random`` MT19937), which
    upstream draws from; calls must be serialized.
    """

    process_attributes = ("action_dim",)

    def __init__(self, num_envs, device="cpu", seed=0, *, loco="point", maze="large"):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.loco, self.maze = loco, maze
        self.schema = maze_schema(loco)
        self._dtypes = {key: numpy_dtype(group.dtype) for key, group in self.schema.groups.items()}
        self._qpos, self._qvel, self.action_dim = BODIES[loco]
        self.envs = [_make_maze(loco, maze) for _ in range(self.num_envs)]
        self._global_rng = [np.random.RandomState(int(seed) + i).get_state() for i in range(self.num_envs)]
        # A point step outside the teleport maze draws no random numbers, so the environments step in one loop.
        self._point = loco == "point" and all(env._teleport_info is None for env in self.envs)
        self._rest = _rest_walls(self.envs[0].model) if self._point else None
        self._static_keys = tuple(group_path(key) for key in HOST_GROUPS)
        self._closed = False
        self.reset(seed=seed)

    @property
    def process_static_leaves(self):
        """State groups that only reset and restore change, for point environments."""
        return self._static_keys if self._point else ()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OGBMazeAdapter is closed")

    def reset(self, seed=None):
        self._check_open()
        for i, env in enumerate(self.envs):
            if seed is not None:
                self._global_rng[i] = np.random.RandomState(int(seed) + i).get_state()
            self._in_legacy_rng(i, env.reset, seed=None if seed is None else int(seed) + i)
        return self.get_state()

    def _in_legacy_rng(self, instance, function, *args, **kwargs):
        """Install an environment's NumPy legacy global RNG state for one call, restoring the caller's on exit."""
        outside = np.random.get_state()
        np.random.set_state(self._global_rng[instance])
        try:
            return function(*args, **kwargs)
        finally:
            self._global_rng[instance] = np.random.get_state()
            np.random.set_state(outside)

    def _values(self, instance, env):
        algorithm, keys, position, has_gauss, cached = self._global_rng[instance]
        assert algorithm == "MT19937"
        pcg = env.np_random.bit_generator.state
        assert pcg["bit_generator"] == "PCG64"
        values = {
            **_split(self._qpos, env.data.qpos),
            **_split(self._qvel, env.data.qvel),
            "simulator/time": [env.data.time],
            "simulator/warmstart": env.data.qacc_warmstart.copy(),
            "task/goal_pos": env.model.geom("target").pos.copy(),
            "task/id": [0 if env.cur_task_id is None else env.cur_task_id],
            "rng/env_state": _uint128_bytes(pcg["state"]["state"]),
            "rng/env_inc": _uint128_bytes(pcg["state"]["inc"]),
            "rng/env_has_uint32": [pcg["has_uint32"]],
            "rng/env_uinteger": [pcg["uinteger"]],
            "rng/global_keys": keys.astype(np.int64),
            "rng/global_position": [position],
            "rng/global_has_gauss": [has_gauss],
            "rng/global_cached_gaussian": [cached],
        }
        return {
            group_path(key): np.concatenate([np.asarray(values[f.name]) for f in group.fields]).astype(
                self._dtypes[key]
            )
            for key, group in self.schema.groups.items()
        }

    def get_state(self):
        self._check_open()
        return owned_columns((self._values(i, env) for i, env in enumerate(self.envs)), self.num_envs, self.device)

    def process_step_state(self):
        """Return ``{key: (num_envs, width) array}`` for the state leaves a step changes."""
        data = [env.data for env in self.envs]
        time = np.array([[d.time] for d in data], np.float64)
        parts = [np.stack([d.qpos for d in data]), np.stack([d.qvel for d in data]), time]
        parts.append(np.stack([d.qacc_warmstart for d in data]))
        return {group_path(PHYSICS): np.concatenate(parts, 1)}

    def validate(self, state):
        self._check_open()
        return decode_state(state, self.num_envs, self.schema, self.envs[0].num_tasks)

    def set_state(self, state):
        v, decoded = self.validate(state)
        import mujoco

        for i, (env, (task_id, pcg, legacy)) in enumerate(zip(self.envs, decoded, strict=True)):
            env.model.geom("target").pos[:] = v["task/goal_pos"][i]
            env.cur_goal_xy = v["task/goal_pos"][i, :2].copy()
            env.cur_task_id = task_id or None
            env.cur_task_info = None if task_id == 0 else env.task_infos[task_id - 1]
            env.data.qpos[:] = np.concatenate([v[name][i] for name, _ in self._qpos])
            env.data.qvel[:] = np.concatenate([v[name][i] for name, _ in self._qvel])
            env.data.time = float(v["simulator/time"][i, 0])
            mujoco.mj_forward(env.model, env.data)
            env.data.qacc_warmstart[:] = v["simulator/warmstart"][i]
            env.np_random.bit_generator.state = pcg
            self._global_rng[i] = legacy

    def check_step(self, actions, active):
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"OGBMaze action must have finite shape (num_envs, {self.action_dim})")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"OGBMaze action must have finite shape (num_envs, {self.action_dim})")
        if self._point:
            return self._step_point(actions, active, capture_state)
        outputs = [
            self._in_legacy_rng(i, env.step, action) if active is None or active[i] else self._last_outputs[i]
            for i, (env, action) in enumerate(zip(self.envs, actions, strict=True))
        ]
        self._last_outputs = outputs
        infos = [o[4] for o in outputs]
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.tensor([o[1] for o in outputs], dtype=torch.float64, device=self.device),
                "terminated": torch.tensor([o[2] for o in outputs], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([o[3] for o in outputs], dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(np.stack([o[0] for o in outputs]), device=self.device),
                "success": torch.tensor([info["success"] for info in infos], dtype=torch.float64, device=self.device),
                "info": TensorDict(
                    {
                        key: torch.as_tensor(np.stack([info[key] for info in infos]), device=self.device)
                        for key in ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel")
                    },
                    batch_size=[self.num_envs],
                    device=self.device,
                ),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _success(self, xy, goal):
        """``norm(xy - goal) <= tol`` per environment, with the source's norm wherever rounding could flip it."""
        tol = self.envs[0]._goal_tol
        diff = xy - goal
        dist = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1])
        for i in np.flatnonzero(np.abs(dist - tol) <= 1e-9 * max(1.0, tol)):
            dist[i] = np.linalg.norm(diff[i])
        return dist <= tol

    def _step_point(self, actions, active, capture_state):
        """Step point environments as ``PointEnv.step`` and ``MazeEnv.step`` do, with task outputs for the whole batch."""
        import mujoco

        envs = self.envs
        first = envs[0]
        prev_qpos = np.stack([env.data.qpos for env in envs])
        prev_qvel = np.stack([env.data.qvel for env in envs])
        goal = np.stack([env.cur_goal_xy for env in envs])
        if first._success_timing == "pre":
            success = self._success(prev_qpos, goal)
        target = prev_qpos + 0.2 * actions
        resting = _resting_instances(self._rest, envs, target)
        for i in range(self.num_envs) if active is None else np.flatnonzero(active):
            env = envs[i]
            data = env.data
            data.qpos[:] = target[i]
            data.qvel[:] = 0.0
            if resting[i]:
                time, h = data.time, env.model.opt.timestep
                for _ in range(env.frame_skip):
                    time += h
                data.time = time
                data.qacc_warmstart[:] = 0.0
            else:
                mujoco.mj_step(env.model, data, nstep=env.frame_skip)
        qpos = np.stack([env.data.qpos for env in envs])
        qvel = np.stack([env.data.qvel for env in envs])
        if first._success_timing == "post":
            success = self._success(qpos, goal)
        reward = np.where(success, 1.0, 0.0)
        if first._reward_task_id is not None:
            reward = reward - 1.0
        values = {
            "reward": reward,
            "terminated": success & bool(first._terminate_at_goal),
            "truncated": np.zeros(self.num_envs, bool),
            "observation": qpos,
            "success": np.where(success, 1.0, 0.0),
            "xy": qpos,
            "prev_qpos": prev_qpos,
            "prev_qvel": prev_qvel,
            "qpos": qpos,
            "qvel": qvel,
        }
        if active is not None:
            idle = ~active
            values = {
                key: np.where(idle.reshape(-1, *[1] * (v.ndim - 1)), self._last_point[key], v)
                for key, v in values.items()
            }
        self._last_point = values
        out = {key: torch.tensor(value, device=self.device) for key, value in values.items()}
        info = TensorDict(
            {key: out.pop(key) for key in ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel")},
            [self.num_envs],
            device=self.device,
        )
        return TensorDict(
            {**({"state": self.get_state()} if capture_state else {}), **out, "info": info},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
