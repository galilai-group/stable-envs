"""Fetch lanes stepped together: gymnasium-robotics' step and reset on lane arrays, MuJoCo in one batched call."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.cpus import cpu_limit
from ...._core.state_layout import StateLayout
from ...._core.tensordicts import build_tensordict
from ...._execution.protocol import SimulatorAdapter
from .adapter import (
    _VISUAL_GROUPS,
    VISUAL_LEAVES,
    _integer,
    _words,
    episode_limits,
    fetch_schema,
    host_actions,
    host_active,
)

#: Most physics threads of one batch.
MAX_THREADS = 8
#: Batch fields that hold MuJoCo integration state; the others are outputs of the last call.
_STATE_BITS = (
    ("time", "mjSTATE_TIME"),
    ("qpos", "mjSTATE_QPOS"),
    ("qvel", "mjSTATE_QVEL"),
    ("act", "mjSTATE_ACT"),
    ("qacc_warmstart", "mjSTATE_WARMSTART"),
    ("ctrl", "mjSTATE_CTRL"),
    ("qfrc_applied", "mjSTATE_QFRC_APPLIED"),
    ("xfrc_applied", "mjSTATE_XFRC_APPLIED"),
    ("eq_active", "mjSTATE_EQ_ACTIVE"),
    ("mocap_pos", "mjSTATE_MOCAP_POS"),
    ("mocap_quat", "mjSTATE_MOCAP_QUAT"),
)
_OUTPUTS = ("xpos", "xquat", "site_xpos")


def physics_threads(num_envs: int) -> int:
    cpus = cpu_limit()
    return max(1, min(MAX_THREADS, cpus, int(num_envs)))


class LaneBatch:
    """The part of ``mujoco.batch.Batch`` that Fetch uses, for MuJoCo builds without it.

    Each lane's integration state is a row of one array, as in ``mujoco.batch``. Worker threads load a row into
    their own MjData, call ``mj_step`` or ``mj_forward`` (which release the GIL), and store the row and the
    outputs back. A loaded state steps exactly as the MjData it was read from.
    """

    def __init__(self, model, nsim: int, nthread: int = 1):
        import mujoco

        self._mj, self._model, self.nsim = mujoco, model, int(nsim)
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION
        self.state = np.zeros((self.nsim, mujoco.mj_stateSize(model, self._spec)))
        self._views, offset = {}, 0
        probe = mujoco.MjData(model)
        for name, bit in _STATE_BITS:
            width = mujoco.mj_stateSize(model, getattr(mujoco.mjtState, bit))
            shape = np.shape(getattr(probe, name))
            view = self.state[:, offset : offset + width]
            self._views[name] = view.reshape(self.nsim, *shape) if shape else view[:, 0]
            assert view.size == 0 or np.shares_memory(self._views[name], self.state)
            offset += width
        self._outputs = {name: np.zeros((self.nsim, *np.shape(getattr(probe, name)))) for name in _OUTPUTS}
        self._threads = max(1, min(int(nthread), self.nsim))
        self._datas = [mujoco.MjData(model) for _ in range(self._threads)]
        self._pool = ThreadPoolExecutor(self._threads) if self._threads > 1 else None
        self._mass = None
        self._models = {}
        self.reset()

    def bind(self, name):
        return self._views[name] if name in self._views else self._outputs[name]

    def expand(self, name):
        if name != "body_mass":
            raise ValueError("LaneBatch varies body_mass only")
        if self._mass is None:
            self._mass = np.repeat(self._model.body_mass[None], self.nsim, 0)
        return self._mass

    def _lane_model(self, lane):
        if self._mass is None or np.array_equal(self._mass[lane], self._model.body_mass):
            return self._model
        key = self._mass[lane].tobytes()
        model = self._models.get(key)
        if model is None:
            import copy

            model = self._models[key] = copy.deepcopy(self._model)
            model.body_mass[:] = self._mass[lane]
        return model

    def _run(self, ids, work):
        ids = np.arange(self.nsim) if ids is None else np.asarray(ids, np.int64).reshape(-1)
        if self._pool is None or len(ids) < 2:
            work(self._datas[0], ids)
            return
        # The threads take lanes from one queue, so a thread whose lanes step faster takes more of them. The
        # calling thread takes lanes too instead of waiting idle.
        lanes = iter(ids.tolist())
        count = min(self._threads, len(ids))
        futures = [self._pool.submit(work, d, lanes) for d in self._datas[: count - 1]]
        try:
            work(self._datas[count - 1], lanes)
        finally:
            for future in futures:
                future.result()

    def _store(self, data, lane):
        self._mj.mj_getState(self._model, data, self.state[lane], self._spec)
        for name, out in self._outputs.items():
            out[lane] = getattr(data, name)

    def step(self, ids=None, nstep: int = 1, stage: int = 0, zero_qpos=None):
        """``mj_step``; with ``zero_qpos``, then zero those coordinates and ``mj_forward``, in one pass."""
        mj, spec = self._mj, self._spec

        def work(data, lanes):
            for lane in lanes:
                model = self._lane_model(lane)
                mj.mj_setState(model, data, self.state[lane], spec)
                mj.mj_step(model, data, nstep=nstep)
                if zero_qpos is not None:
                    data.qpos[zero_qpos] = 0.0
                    mj.mj_forward(model, data)
                self._store(data, lane)

        self._run(ids, work)

    def forward(self, ids=None, stage: int = 3):
        mj, spec = self._mj, self._spec

        def work(data, lanes):
            for lane in lanes:
                model = self._lane_model(lane)
                mj.mj_setState(model, data, self.state[lane], spec)
                mj.mj_forward(model, data)
                self._store(data, lane)

        self._run(ids, work)

    def reset(self, ids=None, keyframe=-1, stage: int = 3):
        mj, spec = self._mj, self._spec

        def work(data, lanes):
            for lane in lanes:
                model = self._lane_model(lane)
                mj.mj_resetData(model, data)
                mj.mj_getState(model, data, self.state[lane], spec)

        self._run(ids, work)
        if stage:
            self.forward(ids)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None


def make_batch(model, nsim: int, nthread: int):
    """``mujoco.batch.Batch`` when this MuJoCo has it, else ``LaneBatch``; both step as ``mj_step``."""
    try:
        from mujoco.batch import Batch
    except ImportError:
        return LaneBatch(model, nsim, nthread)
    return Batch(model, nsim, nthread=nthread)


class FetchTask:
    """Constants of one gymnasium-robotics Fetch task, read from a source environment."""

    def __init__(self, task: str, reward_type: str):
        import mujoco

        from .observations import TASK_CLASSES

        source = TASK_CLASSES[task](reward_type=reward_type)
        self.source, self.xml = source, source.fullpath
        m = source.model
        self.model, self.nq, self.nv = m, int(m.nq), int(m.nv)
        self.has_object, self.block_gripper = bool(source.has_object), bool(source.block_gripper)
        self.n_substeps = int(source.n_substeps)
        self.reward_type = source.reward_type
        self.distance_threshold = source.distance_threshold
        self.obj_range, self.target_range = source.obj_range, source.target_range
        self.target_offset, self.target_in_the_air = source.target_offset, source.target_in_the_air
        self.height_offset = getattr(source, "height_offset", None)
        self.initial_time = source.initial_time
        self.initial_qpos, self.initial_qvel = source.initial_qpos.copy(), source.initial_qvel.copy()
        self.anchor = source.initial_gripper_xpos.copy()
        self.low, self.high = source.action_space.low, source.action_space.high
        self.gripper = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link")
        self.grip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        self.goal_site = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_SITE, "object0" if self.has_object else "robot0:grip"
        )
        self.object = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object0")
        joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object0:joint")
        self.object_qpos = int(m.jnt_qposadr[joint]) if joint >= 0 else -1
        self.fingers = [
            int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"robot0:{side}_gripper_finger_joint")])
            for side in ("l", "r")
        ]
        # mujoco_utils.ctrl_set_action: position actuators move relative to their joint's current position.
        # Only PickAndPlace has gripper actuators.
        if m.nu not in (0, 2):
            raise ValueError("Fetch expects no actuators or two gripper actuators")
        self.relative = [int(m.actuator_biastype[i]) != 0 for i in range(m.nu)]
        self.ctrl_qpos = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in range(m.nu)]
        # mujoco_utils.reset_mocap2body_xpos: each weld puts its mocap on the body it holds.
        self.welds = []
        for kind, a, b in zip(m.eq_type, m.eq_obj1id, m.eq_obj2id):
            if kind != mujoco.mjtEq.mjEQ_WELD:
                continue
            mocap = int(m.body_mocapid[a])
            self.welds.append((mocap, int(b)) if mocap != -1 else (int(m.body_mocapid[b]), int(a)))
        if m.nmocap != 1 or len(self.welds) != 1:
            raise ValueError("Fetch expects one mocap weld")

    def close(self):
        self.source.close()


def lane_rng(seed):
    from gymnasium.utils import seeding

    return seeding.np_random(seed)[0]


def sample_reset(task: FetchTask, rng, anchor):
    """gymnasium-robotics' reset draws for one lane: the object's xy (or None) and the goal."""
    object_xy = None
    if task.has_object:
        object_xy = anchor[:2]
        while np.linalg.norm(object_xy - anchor[:2]) < 0.1:
            object_xy = anchor[:2] + rng.uniform(-task.obj_range, task.obj_range, size=2)
        goal = anchor[:3] + rng.uniform(-task.target_range, task.target_range, size=3)
        goal += task.target_offset
        goal[2] = task.height_offset
        if task.target_in_the_air and rng.uniform() < 0.5:
            goal[2] += rng.uniform(0, 0.45)
    else:
        goal = anchor[:3] + rng.uniform(-task.target_range, task.target_range, size=3)
    return object_xy, goal.copy()


def visual_colors(model, values):
    """Colors that ``FetchAdapter.reset_visual_colors`` reports, without changing ``model``."""
    import mujoco

    groups = {
        name: [
            g
            for g in range(model.ngeom)
            if model.geom_matid[g] >= 0
            and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, int(model.geom_matid[g])) in materials
        ]
        for name, (_, materials) in _VISUAL_GROUPS.items()
    }
    result = []
    for selected in values:
        colors = {}
        if VISUAL_LEAVES & selected.keys():
            for name, geoms in groups.items():
                if geoms:
                    source = _VISUAL_GROUPS[name][0]
                    rgba = model.geom_rgba[geoms[0], :3].copy()
                    if source in selected:
                        rgba[:] = selected[source]
                    colors[name] = rgba
        result.append(colors)
    return result


def rng_words(rngs):
    """PCG64 words of each lane's generator: state, increment, has_uint32, uinteger, as ``(B, 10)`` int64."""
    words = np.empty((len(rngs), 10), np.int64)
    for lane, rng in enumerate(rngs):
        r = rng.bit_generator.state
        if r["bit_generator"] != "PCG64":
            raise ValueError("Fetch requires PCG64 RNG")
        words[lane] = [*_words(r["state"]["state"]), *_words(r["state"]["inc"]), r["has_uint32"], r["uinteger"]]
    return words


class FetchLanes(SimulatorAdapter):
    """Host bookkeeping that every batched Fetch backend shares: RNG, goals, reset anchors, masses and limits."""

    action_dim = 4

    def _init_lanes(self, num_envs, device, task, reward_type):
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self.task = t = FetchTask(task, reward_type)
        self.schema = fetch_schema(t.nq, t.nv)
        self.has_object = t.has_object
        self._closed = False
        B = self.num_envs
        self._registered_mass = float(t.model.body_mass[t.object]) if t.has_object else 0.0
        self._mass = np.full(B, self._registered_mass)
        self._limits = np.full(B, 50, np.int64)
        self._goal = np.zeros((B, 3))
        self._anchor = np.repeat(t.anchor[None], B, 0)
        self._rngs = [None] * B
        self._words = None

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def reference_model(self):
        self._check_open()
        return self.task.model

    @property
    def object_mass(self) -> np.ndarray:
        self._check_open()
        return self._mass.copy()

    def configure_object_mass(self, values):
        self._check_open()
        # As in FetchAdapter, a new mass takes effect at the next reset.
        self._mass = np.asarray(values, np.float64).reshape(self.num_envs).copy()

    def _set_episode_limits(self, limits):
        self._check_open()
        self._limits = episode_limits(limits, self.num_envs).astype(np.int64)

    def reset_visual_colors(self, values):
        self._check_open()
        return visual_colors(self.task.model, values)

    def _draw_resets(self, seed, variations, qpos):
        """Seed, sample and vary every lane as gymnasium-robotics' reset and ``FetchAdapter.reset`` do.

        Write each lane's start configuration into ``qpos`` and its goal, reset anchor and mass into the lane arrays.
        """
        t = self.task
        qpos[:] = t.initial_qpos
        for lane in range(self.num_envs):
            values = {} if variations is None else variations[lane]
            if seed is not None or self._rngs[lane] is None:
                self._rngs[lane] = lane_rng(None if seed is None else int(seed) + lane)
            anchor = self._anchor[lane]
            anchor[:] = t.anchor
            if "agent.start_position" in values:
                anchor[:2] = values["agent.start_position"]
            object_xy, self._goal[lane] = sample_reset(t, self._rngs[lane], anchor)
            if object_xy is not None:
                qpos[lane, t.object_qpos : t.object_qpos + 2] = object_xy
            if "block.start_position" in values:
                qpos[lane, 15:17] = values["block.start_position"]
            if "block.angle" in values:
                angle = float(np.asarray(values["block.angle"]).reshape(-1)[0])
                qpos[lane, 18:22] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
            if "goal.start_position" in values:
                self._goal[lane] = np.asarray(values["goal.start_position"], np.float64)
            if "block.mass" in values:
                self._mass[lane] = float(np.asarray(values["block.mass"]).reshape(-1)[0])
        self._words = None

    def _rng_words(self):
        """PCG64 words of every lane; they change only when a lane resets or restores."""
        if self._words is None:
            self._words = rng_words(self._rngs)
        return self._words

    def _restore_rngs(self, v):
        state_words, inc_words = v["rng/state_words"], v["rng/increment_words"]
        has_uint32, uinteger = v["rng/has_uint32"][:, 0], v["rng/uinteger"][:, 0]
        for lane, rng in enumerate(self._rngs):
            rng.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {"state": _integer(state_words[lane].tolist()), "inc": _integer(inc_words[lane].tolist())},
                "has_uint32": int(has_uint32[lane]),
                "uinteger": int(uinteger[lane]),
            }
        self._words = None

    def validate(self, state: TensorDict) -> None:
        from .adapter import FetchAdapter

        FetchAdapter._validate_state(self, state)

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        out = self.advance(actions, active=active)
        if capture_state:
            out["state"] = self.get_state()
        return out


class FetchBatchAdapter(FetchLanes):
    """CPU Fetch lanes in one batch: gymnasium-robotics' step and reset, bit for bit, on lane arrays."""

    execution_mode = "DELEGATED_CPU"

    def __init__(self, num_envs, device="cpu", seed=0, task="PickAndPlace", reward_type="sparse", threads=None):
        self._init_lanes(num_envs, device, task, reward_type)
        t, B = self.task, self.num_envs
        self.b = make_batch(t.model, B, physics_threads(B) if threads is None else int(threads))
        for name in ("qpos", "qvel", "time", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat"):
            setattr(self, "_" + name, self.b.bind(name))
        self._qfrc, self._xfrc = self.b.bind("qfrc_applied"), self.b.bind("xfrc_applied")
        # Batch outputs are copied only for fields bound before a call.
        self._xpos, self._xquat, self._site_xpos = (self.b.bind(name) for name in _OUTPUTS)
        self._body_mass = None
        self._layouts: dict[str, StateLayout] = {}
        self._elapsed = np.zeros(B, np.int64)
        self._grip_pos, self._grip_quat = np.zeros((B, 3)), np.zeros((B, 4))
        self._reward_dtype = np.float32 if t.reward_type == "sparse" else np.float64
        self.reset(seed=seed)

    def _install_mass(self, ids):
        if not self.has_object:
            return
        if self._body_mass is None:
            if np.all(self._mass[ids] == self._registered_mass):
                return
            self._body_mass = self.b.expand("body_mass")
        self._body_mass[ids, self.task.object] = self._mass[ids]

    def _read_controller(self, ids=None):
        body = self.task.welds[0][1]
        rows = slice(None) if ids is None else ids
        self._grip_pos[rows] = self._xpos[rows, body]
        self._grip_quat[rows] = self._xquat[rows, body]

    def reset(self, seed: int | None = None, variations=None) -> TensorDict:
        self._check_open()
        t, B = self.task, self.num_envs
        self.b.reset(None, stage=0)
        self._time[:] = t.initial_time
        self._qvel[:] = t.initial_qvel
        self._draw_resets(seed, variations, self._qpos)
        self._install_mass(np.arange(B))
        self.b.forward()
        self._read_controller()
        self._elapsed[:] = 0
        self._outputs = {
            "reward": np.zeros(B, self._reward_dtype),
            "terminated": np.zeros(B, bool),
            "truncated": np.zeros(B, bool),
            "success": np.zeros(B, np.float32),
        }
        return self.get_state()

    def get_state(self) -> TensorDict:
        self._check_open()
        return self._export(self._state_groups(), "state", owned=self.schema.groups)

    def _state_groups(self) -> dict[str, np.ndarray]:
        """The State groups, packed from the live arrays into new arrays."""
        words = self._rng_words()
        return self.schema.pack(
            {
                "qpos": self._qpos,
                "qvel": self._qvel,
                "time": self._time[:, None],
                "task/initial_gripper_xpos": self._anchor,
                "warmstart": self._qacc_warmstart,
                "controller/position": self._grip_pos,
                "controller/quaternion": self._grip_quat,
                "goal": self._goal,
                "lifecycle/elapsed_steps": self._elapsed[:, None],
                "rng/state_words": words[:, 0:4],
                "rng/increment_words": words[:, 4:8],
                "rng/has_uint32": words[:, 8:9],
                "rng/uinteger": words[:, 9:10],
            }
        )

    def _export(self, values: dict[str, np.ndarray], name: str, owned=()) -> TensorDict:
        """A TensorDict of ``values`` on the env device; ``owned`` values are new arrays already.

        On the host each value becomes a tensor. On an accelerator the values pack into one copy per dtype, with the
        layout built once.
        """
        if self.device.type == "cpu":
            tree: dict = {}
            for key, value in values.items():
                *path, leaf = key.split("/")
                node = tree
                for part in path:
                    node = node.setdefault(part, {})
                node[leaf] = torch.from_numpy(value if key in owned else np.array(value))
            return build_tensordict(tree, [self.num_envs], self.device)
        layout = self._layouts.get(name)
        if layout is None:
            leaves = [(key, value.shape[1:], torch.from_numpy(value[:0]).dtype) for key, value in values.items()]
            layout = self._layouts[name] = StateLayout(leaves, self.num_envs)
        # The export copies every value, so the result does not change with later steps.
        return layout.export(values, self.device)

    def set_state(self, state: TensorDict) -> None:
        self.validate(state)
        v = self.schema.views(state.cpu(), numpy=True)
        self._qpos[:] = v["qpos"]
        self._qvel[:] = v["qvel"]
        self._goal[:] = v["goal"]
        self._time[:] = v["time"][:, 0]
        self._anchor[:] = v["task/initial_gripper_xpos"]
        self._qfrc[:] = 0
        self._xfrc[:] = 0
        self.b.forward()
        self._qacc_warmstart[:] = v["warmstart"]
        self._grip_pos[:] = v["controller/position"]
        self._grip_quat[:] = v["controller/quaternion"]
        self._elapsed[:] = v["lifecycle/elapsed_steps"][:, 0]
        self._restore_rngs(v)

    def check_step(self, actions, active):
        return host_actions(actions, self.num_envs), host_active(active, self.num_envs)

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        out = self._advance(actions, active)
        if not capture_state:
            return self._export(out, "step")
        self._check_open()
        groups = {f"state/{key}": value for key, value in self._state_groups().items()}
        return self._export({**out, **groups}, "step+state", owned=groups)

    def advance(self, actions, *, active=None) -> TensorDict:
        return self._export(self._advance(actions, active), "step")

    def _advance(self, actions, active) -> dict[str, np.ndarray]:
        """robot_env.step, _set_action, _step_callback, compute_reward and TimeLimit on the active lanes."""
        self._check_open()
        t = self.task
        active = host_active(active, self.num_envs)
        actions = host_actions(actions, self.num_envs)
        ids = np.arange(self.num_envs) if active is None else np.flatnonzero(active)
        if len(ids):
            action = np.clip(actions[ids], t.low, t.high)
            pos_ctrl = action[:, :3] * 0.05
            gripper = action[:, 3]
            if t.block_gripper:
                gripper = np.zeros_like(gripper)
            # _set_action concatenates with a float64 rotation, so both parts reach MuJoCo as float64.
            pos_ctrl, gripper = pos_ctrl.astype(np.float64), gripper.astype(np.float64)
            for i, (relative, adr) in enumerate(zip(t.relative, t.ctrl_qpos)):
                self._ctrl[ids, i] = self._qpos[ids, adr] + gripper if relative else gripper
            mocap = t.welds[0][0]
            self._mocap_pos[ids, mocap] = self._grip_pos[ids] + pos_ctrl
            self._mocap_quat[ids, mocap] = self._grip_quat[ids] + np.array([1.0, 0.0, 1.0, 0.0])
            if isinstance(self.b, LaneBatch):
                self.b.step(ids, t.n_substeps, zero_qpos=t.fingers if t.block_gripper else None)
            else:
                self.b.step(ids, t.n_substeps)
                if t.block_gripper:
                    self._qpos[np.ix_(ids, t.fingers)] = 0.0
                    self.b.forward(ids)
            self._read_controller(ids)
            achieved = self._site_xpos[ids, t.goal_site]
            d = np.linalg.norm(achieved - self._goal[ids], axis=-1)
            out = self._outputs
            out["success"][ids] = (d < t.distance_threshold).astype(np.float32)
            out["reward"][ids] = -(d > t.distance_threshold).astype(np.float32) if t.reward_type == "sparse" else -d
            self._elapsed[ids] += 1
            out["truncated"][ids] = self._elapsed[ids] >= self._limits[ids]
        return dict(self._outputs)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            close = getattr(self.b, "close", None)
            if close is not None:
                close()
            self.task.close()
            self.b = None
