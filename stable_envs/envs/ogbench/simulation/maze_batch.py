"""OGBench maze simulations stepped together: ``mujoco.batch`` on CPU, MuJoCo Warp on CUDA."""

from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.mjbatch import NativeBatch, WarpBatch, native_threads
from ...._execution.protocol import SimulatorAdapter
from .maze import BODIES, _make_maze, _rest_walls, decode_state, maze_fields
from .pcg import uint128_bytes

INFO_KEYS = ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel")
#: ``HumanoidEnv.get_ob`` limb bodies (left hand, left foot, right hand, right foot) relative to the torso.
_LIMBS = (16, 10, 13, 7)
#: Teleport destinations drawn ahead per lane on CUDA; the host refills them at least this often.
TELEPORT_DRAWS = 32


class MazeTask:
    """The source maze's static task description, read from one OGBench env."""

    def __init__(self, loco: str, maze: str):
        env = _make_maze(loco, maze)
        try:
            self.loco, self.maze = loco, maze
            self.model = env.model
            self.frame_skip = int(env.frame_skip)
            self.init_qpos = env.init_qpos.copy()
            self.init_qvel = env.init_qvel.copy()
            self.qpos0 = env.model.qpos0.copy()
            self.num_tasks = int(env.num_tasks)
            self.tasks = [(info["init_ij"], info["goal_ij"]) for info in env.task_infos]
            self.maze_unit, self.offset = env._maze_unit, (env._offset_x, env._offset_y)
            self.noise, self.goal_tol = env._noise, env._goal_tol
            self.success_timing, self.terminate_at_goal = env._success_timing, bool(env._terminate_at_goal)
            self.reward_task_id, self.add_noise_to_goal = env._reward_task_id, env._add_noise_to_goal
            self.target_z = float(env.model.geom("target").pos[2])
            self.teleport = env._teleport_info
            self.jnt_range = env.model.jnt_range.copy()
        finally:
            # The model outlives the env; only the renderer is released.
            env.close()

    def ij_to_xy(self, ij):
        i, j = ij
        return j * self.maze_unit - self.offset[0], i * self.maze_unit - self.offset[1]

    def add_noise(self, xy, mt):
        random_x = mt.uniform(low=-self.noise, high=self.noise) * self.maze_unit / 4
        random_y = mt.uniform(low=-self.noise, high=self.noise) * self.maze_unit / 4
        return xy[0] + random_x, xy[1] + random_y


def norm_within(diff, limit):
    """``np.linalg.norm(diff[i]) <= limit`` per row, with the source's norm wherever rounding could flip it."""
    dist = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1])
    for i in np.flatnonzero(np.abs(dist - limit) <= 1e-9 * max(1.0, limit)):
        dist[i] = np.linalg.norm(diff[i])
    return dist <= limit


class MazeRandom:
    """Each lane's OGBench RNGs: the env's PCG64 ``np_random`` and its own copy of NumPy's global MT19937.

    The State columns follow the generators and change only at reset, restore and teleport.
    """

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.pcg = [np.random.Generator(np.random.PCG64(0)) for _ in range(num_envs)]
        self.mt = [np.random.RandomState(0) for _ in range(num_envs)]
        fields = maze_fields("point")
        self.columns = {
            key: np.zeros((num_envs, width), dtype=np.dtype(str(dtype).removeprefix("torch.")))
            for key, (width, dtype) in fields.items()
            if key[0] == "rng"
        }

    def seed(self, ids, seed):
        for i in ids:
            self.pcg[i] = np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed) + int(i))))
            self.mt[i] = np.random.RandomState(int(seed) + int(i))

    def sync(self, ids):
        """Write the generators of lanes ``ids`` into the State columns."""
        c = self.columns
        for i in ids:
            pcg = self.pcg[i].bit_generator.state
            c["rng", "env_state"][i] = uint128_bytes(pcg["state"]["state"])
            c["rng", "env_inc"][i] = uint128_bytes(pcg["state"]["inc"])
            c["rng", "env_has_uint32"][i] = pcg["has_uint32"]
            c["rng", "env_uinteger"][i] = pcg["uinteger"]
            _, keys, position, has_gauss, cached = self.mt[i].get_state()
            c["rng", "global_keys"][i] = keys
            c["rng", "global_position"][i] = position
            c["rng", "global_has_gauss"][i] = has_gauss
            c["rng", "global_cached_gaussian"][i] = cached

    def load(self, decoded):
        """Install generators decoded from State by ``decode_state``."""
        for i, (_, pcg, legacy) in enumerate(decoded):
            self.pcg[i].bit_generator.state = pcg
            self.mt[i].set_state(legacy)
        self.sync(range(self.num_envs))


class MazeReset:
    """OGBench ``MazeEnv.reset`` for many lanes, sampled from each lane's own RNGs.

    The settling steps between the two inner resets draw from the action space's own unseeded RNG and are
    discarded by the second reset, so they do not change the reset State and are not run. Humanoid starts
    are rejection-sampled with a position-stage forward of ``rb``, a ``mujoco.batch`` of the lanes.
    """

    def __init__(self, task: MazeTask, num_envs: int, rb=None):
        self.task, self.num_envs, self.rb = task, num_envs, rb
        m = task.model
        self.nq, self.nv = m.nq, m.nv
        # A bound output is copied out only by calls after it is bound.
        self._ncon = rb.bind("ncon") if task.loco == "humanoid" else None

    def _reset_model(self, ids, rand: MazeRandom, qpos, qvel):
        t = self.task
        if t.loco in ("point", "ant"):
            for i in ids:
                pcg = rand.pcg[i]
                qpos[i] = t.init_qpos + pcg.uniform(low=-0.1, high=0.1, size=self.nq)
                qvel[i] = t.init_qvel + 0.1 * pcg.standard_normal(self.nv)
            return
        pending = np.asarray(ids, np.int64)
        lo, hi = t.jnt_range[1:, 0], t.jnt_range[1:, 1]
        while pending.size:
            for i in pending:
                pcg = rand.pcg[i]
                quat = pcg.uniform(size=4)
                quat /= np.linalg.norm(quat)
                qpos[i, 3:7] = quat
                qvel[i] = 0.1 * pcg.standard_normal(self.nv)
                for joint, (low, high) in enumerate(zip(lo, hi, strict=True), 1):
                    qpos[i, 6 + joint] = pcg.uniform(low, high)
            rb = self.rb
            rb.reset(pending, stage=mujoco.mjtStage.mjSTAGE_NONE)
            rb.qpos[pending] = qpos[pending]
            rb.qvel[pending] = qvel[pending]
            rb.forward(pending, stage=mujoco.mjtStage.mjSTAGE_POS)
            pending = pending[self._ncon[pending].reshape(-1) > 0]

    def sample(self, ids, rand: MazeRandom, seeded: bool):
        """Return ``(qpos, qvel, goal_pos, task_id)`` rows for lanes ``ids`` and advance their RNGs."""
        t = self.task
        n = len(ids)
        goal = np.zeros((n, 3))
        goal[:, 2] = t.target_z
        task_id = np.zeros((n, 1), np.int64)
        init = np.zeros((n, 2))
        for row, i in enumerate(ids):
            mt = rand.mt[i]
            if t.reward_task_id is not None:
                task_id[row] = t.reward_task_id
            else:
                task_id[row] = mt.randint(1, t.num_tasks + 1)
            init_ij, goal_ij = t.tasks[int(task_id[row, 0]) - 1]
            init[row] = t.add_noise(t.ij_to_xy(init_ij), mt)
            goal_xy = t.ij_to_xy(goal_ij)
            if t.add_noise_to_goal:
                goal_xy = t.add_noise(goal_xy, mt)
            goal[row, :2] = goal_xy
        qpos = np.tile(t.qpos0, (self.num_envs, 1))
        qvel = np.zeros((self.num_envs, self.nv))
        # A seeded inner reset reseeds before each of the two draws, so they repeat each other.
        for _ in range(1 if seeded else 2):
            qpos[ids] = t.qpos0
            self._reset_model(ids, rand, qpos, qvel)
        qpos, qvel = qpos[ids], qvel[ids]
        qpos[:, :2] = init
        return qpos, qvel, goal, task_id


class MazeOutputs:
    """``MazeEnv.step`` task outputs for every lane from the step's qpos and qvel."""

    def __init__(self, task: MazeTask):
        self.task = task
        tp = task.teleport
        if tp is not None:
            self.teleport_in = [np.array([x, y]) for x, y in tp["teleport_in_xys"]]
            self.teleport_out = [np.array(xy) for xy in tp["teleport_out_xys"]]
            self.teleport_reach = tp["teleport_radius"] * 1.5

    def success(self, xy, goal):
        return norm_within(xy - goal, self.task.goal_tol)

    def finish(self, success):
        t = self.task
        reward = np.where(success, 1.0, 0.0)
        if t.reward_task_id is not None:
            reward = reward - 1.0
        return {
            "reward": reward,
            "terminated": success & t.terminate_at_goal,
            "truncated": np.zeros(len(success), bool),
            "success": np.where(success, 1.0, 0.0),
        }

    def teleports(self, xy, ids, rand: MazeRandom):
        """Return ``(lanes, destinations)`` that the source teleports after this step, drawing their RNGs."""
        if self.task.teleport is None or not len(ids):
            return np.zeros(0, np.int64), np.zeros((0, 2))
        near = np.zeros(len(xy), bool)
        for centre in self.teleport_in:
            near |= norm_within(xy - centre, self.teleport_reach)
        lanes = np.asarray(ids)[near[ids]]
        count = len(self.teleport_out)
        return lanes, np.array([self.teleport_out[rand.mt[i].randint(count)] for i in lanes]).reshape(-1, 2)


def humanoid_observation(qpos, qvel, xpos, xmat, sensordata):
    """``HumanoidEnv.get_ob`` for every lane, in the source's operation order (NumPy or Torch rows)."""
    torso_frame = xmat[:, 1].reshape(-1, 3, 3)
    torso_pos = xpos[:, 1]
    # A stacked matmul rounds as the source's per-lane ``ndarray.dot``; einsum sums in another order.
    positions = [((xpos[:, idx] - torso_pos)[:, None] @ torso_frame)[:, 0] for idx in _LIMBS]
    cat = np.concatenate if isinstance(qpos, np.ndarray) else torch.cat
    return cat(
        [qpos[:, :2], qpos[:, 7:], xpos[:, 2, 2:3], *positions, xmat[:, 1, [6, 7, 8]], sensordata[:, 0:3], qvel],
        1,
    )


class _MazeLanes(SimulatorAdapter):
    """Host task bookkeeping shared by the batched maze adapters."""

    def _setup(self, num_envs, device, loco, maze, task):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.loco, self.maze = loco, maze
        self.fields = maze_fields(loco)
        self._qpos, self._qvel, self.action_dim = BODIES[loco]
        self.task = MazeTask(loco, maze) if task is None else task
        self._random = MazeRandom(self.num_envs)
        self._outputs = MazeOutputs(self.task)
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _sample_reset(self, seed):
        ids = np.arange(self.num_envs)
        if seed is not None:
            # Check the complete lane range before any lane resets.
            np.random.RandomState(int(seed))
            np.random.RandomState(int(seed) + self.num_envs - 1)
            self._random.seed(ids, seed)
        sample = self._sampler.sample(ids, self._random, seed is not None)
        self._random.sync(ids)
        return sample

    def _split(self, qpos, qvel):
        out = {}
        for fields, values in ((self._qpos, qpos), (self._qvel, qvel)):
            start = 0
            for key, width in fields:
                out[key] = values[:, start : start + width]
                start += width
        return out

    def _join(self, host, fields):
        return np.concatenate([host[key].numpy() for key, _ in fields], 1)

    def validate(self, state):
        self._check_open()
        return decode_state(state, self.num_envs, self.fields, self.task.num_tasks)

    def check_step(self, actions, active):
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"OGBMaze action must have finite shape (num_envs, {self.action_dim})")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active


class OGBMazeBatchAdapter(_MazeLanes):
    """OGBench maze lanes stepped together in one float64 ``mujoco.batch`` (``NativeBatch``).

    Physics, observations, task outputs, resets and RNG streams equal the source's bit for bit. ``device``
    selects output placement. The State is the format of ``OGBMazeAdapter``.
    """

    execution_mode = "DELEGATED_CPU"

    def __init__(self, num_envs, device="cpu", seed=0, *, loco="point", maze="large", task=None):
        self._setup(num_envs, device, loco, maze, task)
        outputs = ("xpos", "xmat", "sensordata", "ncon") if loco == "humanoid" else ()
        self.sim = NativeBatch(self.task.model, self.num_envs, "cpu", frame_skip=self.task.frame_skip, outputs=outputs)
        # The task steps a subset of lanes, and the humanoid observation needs the velocity stage after the step,
        # as ``HumanoidEnv`` runs ``mj_step1``: these use the batch directly.
        self.b = self.sim.b
        # Live arrays over the batch's rows; looking one up on the batch costs more than the point step.
        self._live = SimpleNamespace(
            **{f: getattr(self.b, f) for f in ("qpos", "qvel", "time", "qacc_warmstart", "ctrl")}
        )
        self._sampler = MazeReset(self.task, self.num_envs, self.b)
        self._kinematic = loco == "humanoid"
        if self._kinematic:
            self._xpos, self._xmat, self._sensordata = (self.sim.numpy(f) for f in ("xpos", "xmat", "sensordata"))
        self._rest = _rest_walls(self.task.model) if loco == "point" else None
        self._goal = np.zeros((self.num_envs, 3))
        self._task_id = np.zeros((self.num_envs, 1), np.int64)
        self._last = None
        self.reset(seed=seed)

    def reset(self, seed=None):
        self._check_open()
        qpos, qvel, goal, task_id = self._sample_reset(seed)
        self.sim.reset()
        self.sim.set_physics({"qpos": qpos, "qvel": qvel})
        self._goal[:], self._task_id[:] = goal, task_id
        self._last = None
        return self.get_state()

    def _host_state(self):
        b = self._live
        columns = {
            **self._split(b.qpos, b.qvel),
            ("simulator", "time"): b.time[:, None],
            ("simulator", "warmstart"): b.qacc_warmstart,
            ("task", "goal_pos"): self._goal,
            ("task", "id"): self._task_id,
            **self._random.columns,
        }
        return TensorDict({key: torch.from_numpy(np.array(value)) for key, value in columns.items()}, [self.num_envs])

    def get_state(self):
        self._check_open()
        return self._host_state().to(self.device)

    def set_state(self, state):
        host, decoded = self.validate(state)
        self.sim.set_physics(
            {
                "qpos": self._join(host, self._qpos),
                "qvel": self._join(host, self._qvel),
                "time": host["simulator", "time"].numpy()[:, 0],
                "qacc_warmstart": host["simulator", "warmstart"].numpy(),
            }
        )
        self._goal[:] = host["task", "goal_pos"].numpy()
        self._task_id[:] = host["task", "id"].numpy()
        self._random.load(decoded)
        self._last = None

    def _resting(self, target, ids):
        """Point lanes whose step leaves the ball where it is (see ``_resting_lanes``)."""
        lo, hi, radius, offset = self._rest
        centre = np.zeros((len(ids), 3))
        centre[:, :2] = target[ids]
        centre += offset
        near = np.clip(centre[:, None], lo, hi) - centre[:, None]
        return (near * near).sum(-1).min(1) > (radius + 1e-3) ** 2

    def _advance(self, actions, ids):
        b, task = self._live, self.task
        if self.loco == "point":
            target = b.qpos + 0.2 * actions
            b.qpos[ids] = target[ids]
            b.qvel[ids] = 0.0
            if self._rest is not None:
                resting = self._resting(target, ids)
                idle, ids = ids[resting], ids[~resting]
                if idle.size:
                    h = task.model.opt.timestep
                    time = b.time[idle]
                    for _ in range(task.frame_skip):
                        time = time + h
                    b.time[idle] = time
                    b.qacc_warmstart[idle] = 0.0
        else:
            b.ctrl[ids] = actions[ids]
        if ids.size:
            stage = mujoco.mjtStage.mjSTAGE_VEL if self._kinematic else mujoco.mjtStage.mjSTAGE_NONE
            self.b.step(ids, task.frame_skip, stage=stage)

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        actions, active = self.check_step(actions, active)
        if active is not None and self._last is None:
            raise RuntimeError("An active mask needs a previous step")
        ids = np.arange(self.num_envs) if active is None else np.flatnonzero(active)
        b, out = self._live, self._outputs
        prev_qpos, prev_qvel = b.qpos.copy(), b.qvel.copy()
        goal = self._goal[:, :2]
        if self.task.success_timing == "pre":
            success = out.success(prev_qpos[:, :2], goal)
        self._advance(actions, ids)
        qpos, qvel = b.qpos.copy(), b.qvel.copy()
        if self.task.success_timing == "post":
            success = out.success(qpos[:, :2], goal)
        if self.loco == "point":
            observation = qpos
        elif self.loco == "ant":
            observation = np.concatenate([qpos, qvel], axis=1)
        else:
            observation = humanoid_observation(qpos, qvel, self._xpos, self._xmat, self._sensordata)
        values = {
            **out.finish(success),
            "observation": observation,
            "xy": qpos[:, :2],
            "prev_qpos": prev_qpos,
            "prev_qvel": prev_qvel,
            "qpos": qpos,
            "qvel": qvel,
        }
        lanes, dest = out.teleports(qpos[:, :2], ids, self._random)
        if lanes.size:
            b.qpos[lanes, :2] = dest
            self._random.sync(lanes)
        if active is not None:
            idle = ~active
            values = {
                key: np.where(idle.reshape(-1, *[1] * (v.ndim - 1)), self._last[key], v) for key, v in values.items()
            }
        self._last = values
        tensors = {key: torch.from_numpy(np.ascontiguousarray(v)) for key, v in values.items()}
        info = TensorDict({key: tensors.pop(key) for key in INFO_KEYS}, [self.num_envs])
        result = TensorDict({**tensors, "info": info}, [self.num_envs])
        if capture_state:
            result["state"] = self._host_state()
        return result.to(self.device)

    def close(self):
        if not self._closed:
            self._closed = True
            self.sim.close()
            self.b = self._live = self._xpos = self._xmat = self._sensordata = None


class OGBMazeWarpAdapter(_MazeLanes):
    """OGBench maze lanes as MuJoCo Warp worlds on CUDA (``WarpBatch``): float32 physics, no host sync per step.

    Task outputs follow the source's step on the device. Resets sample the source's start and goal from each
    lane's own RNGs on the host. Teleport destinations are drawn ahead from each lane's RNG, so the step needs
    no host draw; the host advances the RNGs by the draws used when it returns State or refills them.
    """

    execution_mode = "MJWARP"

    def __init__(self, num_envs, device="cuda", seed=0, *, loco="point", maze="large", task=None, **warp_options):
        self._setup(num_envs, device, loco, maze, task)
        task = self.task
        self.sim = WarpBatch(task.model, self.num_envs, self.device, frame_skip=task.frame_skip, **warp_options)
        rb = None
        if loco == "humanoid":
            from mujoco.batch import Batch

            rb = Batch(task.model, self.num_envs, nthread=native_threads(self.num_envs))
            self._step1 = self._capture(lambda: self.sim._mjw.step1(self.sim._m, self.sim._d))
        self._sampler = MazeReset(task, self.num_envs, rb)
        f64 = dict(device=self.device, dtype=torch.float64)
        self._goal = torch.zeros(self.num_envs, 3, **f64)
        self._task_id = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.int64)
        self._tol = float(task.goal_tol)
        self._qpos_v, self._qvel_v = self.sim.data("qpos"), self.sim.data("qvel")
        self._teleport = task.teleport is not None
        if self._teleport:
            out = self._outputs
            self._tp_in = torch.as_tensor(np.stack(out.teleport_in), **f64)
            self._tp_out = torch.as_tensor(np.stack(out.teleport_out), device=self.device, dtype=self._qpos_v.dtype)
            self._draws = torch.zeros(self.num_envs, TELEPORT_DRAWS, device=self.device, dtype=torch.int64)
            self._used = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
            self._steps_since_refill = 0
        self._rng_columns = {}
        self._last = None
        self.reset(seed=seed)

    def _capture(self, body):
        sim, wp = self.sim, self.sim._wp
        with wp.ScopedDevice(sim._wp_device):
            body()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                body()
            wp.synchronize()
        return capture.graph

    # RNG columns and teleport draws ------------------------------------------------------------------------

    def _upload_rng(self, ids=None):
        cols = self._random.columns
        if ids is None or not self._rng_columns:
            self._rng_columns = {key: torch.from_numpy(value.copy()).to(self.device) for key, value in cols.items()}
        elif len(ids):
            index = torch.as_tensor(ids, device=self.device)
            for key, value in cols.items():
                self._rng_columns[key][index] = torch.from_numpy(value[ids]).to(self.device)
        if self._teleport:
            self._refill(np.arange(self.num_envs) if ids is None else ids)

    def _refill(self, ids):
        """Draw the next teleport destinations of lanes ``ids`` from copies of their RNGs."""
        if not len(ids):
            return
        count = len(self._outputs.teleport_out)
        probe = np.random.RandomState()
        draws = np.zeros((len(ids), TELEPORT_DRAWS), np.int64)
        for row, i in enumerate(ids):
            probe.set_state(self._random.mt[i].get_state())
            draws[row] = [probe.randint(count) for _ in range(TELEPORT_DRAWS)]
        index = torch.as_tensor(ids, device=self.device)
        self._draws[index] = torch.from_numpy(draws).to(self.device)
        self._used[index] = 0

    def _settle_teleports(self):
        """Advance each lane's RNG by the teleport draws its steps used, then draw ahead again."""
        if not self._teleport:
            return
        used = self._used.cpu().numpy()
        ids = np.flatnonzero(used)
        count = len(self._outputs.teleport_out)
        for i in ids:
            mt = self._random.mt[i]
            for _ in range(int(used[i])):
                mt.randint(count)
        self._random.sync(ids)
        self._upload_rng(ids)
        self._steps_since_refill = 0

    # Adapter interface --------------------------------------------------------------------------------------

    def reset(self, seed=None):
        self._check_open()
        qpos, qvel, goal, task_id = self._sample_reset(seed)
        self.sim.reset()
        self.sim.set_physics({"qpos": qpos, "qvel": qvel})
        self._goal.copy_(torch.from_numpy(goal))
        self._task_id.copy_(torch.from_numpy(task_id))
        self._upload_rng()
        self._last = None
        return self.get_state()

    def get_state(self):
        self._check_open()
        if self._teleport:
            self._settle_teleports()
        d = self.sim.data
        columns = {
            **self._split(self._qpos_v, self._qvel_v),
            ("simulator", "time"): d("time")[:, None],
            ("simulator", "warmstart"): d("qacc_warmstart"),
            ("task", "goal_pos"): self._goal,
            ("task", "id"): self._task_id,
            **self._rng_columns,
        }
        return TensorDict(
            {key: value.to(self.fields[key][1]).clone() for key, value in columns.items()},
            [self.num_envs],
            device=self.device,
        )

    def set_state(self, state):
        host, decoded = self.validate(state)
        self.sim.set_physics(
            {
                "qpos": torch.from_numpy(self._join(host, self._qpos)),
                "qvel": torch.from_numpy(self._join(host, self._qvel)),
                "time": host["simulator", "time"][:, 0],
                "qacc_warmstart": host["simulator", "warmstart"],
            }
        )
        self._goal.copy_(host["task", "goal_pos"])
        self._task_id.copy_(host["task", "id"])
        self._random.load(decoded)
        self._upload_rng()
        self._last = None

    def check_step(self, actions, active):
        if not torch.is_tensor(actions):
            return super().check_step(actions, active)
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"OGBMaze action must have finite shape (num_envs, {self.action_dim})")
        if active is not None and (active.shape != (self.num_envs,) or active.dtype != torch.bool):
            raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active

    def _success(self, xy):
        diff = xy - self._goal[:, :2]
        return torch.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]) <= self._tol

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        actions, active = self.check_step(actions, active)
        if active is not None and self._last is None:
            raise RuntimeError("An active mask needs a previous step")
        sim = self.sim
        a = torch.as_tensor(actions, device=self.device)
        if active is not None:
            active = torch.as_tensor(active, device=self.device)
            saved = {name: sim.data(name).clone() for name in sim.fields}
        qpos_v, qvel_v = self._qpos_v, self._qvel_v
        prev_qpos, prev_qvel = qpos_v.to(torch.float64), qvel_v.to(torch.float64)
        if self.task.success_timing == "pre":
            success = self._success(prev_qpos[:, :2])
        if self.loco == "point":
            qpos_v.add_(0.2 * a.to(qpos_v.dtype))
            qvel_v.zero_()
            sim.step()
        else:
            sim.step(ctrl=a)
        if self.loco == "humanoid":
            sim._launch(self._step1)
        if active is not None:
            idle = ~active
            for name, value in saved.items():
                live = sim.data(name)
                live.copy_(torch.where(idle.view(-1, *[1] * (live.dim() - 1)), value, live))
        qpos, qvel = qpos_v.to(torch.float64), qvel_v.to(torch.float64)
        if self.task.success_timing == "post":
            success = self._success(qpos[:, :2])
        if self.loco == "point":
            observation = qpos
        elif self.loco == "ant":
            observation = torch.cat([qpos, qvel], 1)
        else:
            d = sim.data
            observation = humanoid_observation(
                qpos,
                qvel,
                d("xpos").to(torch.float64),
                d("xmat").reshape(self.num_envs, -1, 9).to(torch.float64),
                d("sensordata").to(torch.float64),
            )
        reward = success.to(torch.float64)
        values = {
            "reward": reward - 1.0 if self.task.reward_task_id is not None else reward,
            "terminated": success & self.task.terminate_at_goal,
            "truncated": torch.zeros_like(success),
            "success": success.to(torch.float64),
            "observation": observation,
            "xy": qpos[:, :2],
            "prev_qpos": prev_qpos,
            "prev_qvel": prev_qvel,
            "qpos": qpos,
            "qvel": qvel,
        }
        if self._teleport:
            self._teleport_lanes(qpos[:, :2], active)
        if active is not None:
            idle = ~active
            values = {
                key: torch.where(idle.view(-1, *[1] * (v.dim() - 1)), self._last[key], v) for key, v in values.items()
            }
        self._last = values
        info = TensorDict({key: values[key] for key in INFO_KEYS}, [self.num_envs], device=self.device)
        result = TensorDict(
            {key: v for key, v in values.items() if key not in INFO_KEYS} | {"info": info},
            [self.num_envs],
            device=self.device,
        )
        if capture_state:
            result["state"] = self.get_state()
        return result

    def _teleport_lanes(self, xy, active):
        """Move lanes within reach of an inbound teleport to their next drawn outbound teleport."""
        diff = xy[:, None] - self._tp_in
        near = (
            torch.sqrt(diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1]) <= self._outputs.teleport_reach
        ).any(1)
        if active is not None:
            near &= active
        draw = self._draws.gather(1, self._used.clamp(max=TELEPORT_DRAWS - 1)[:, None])[:, 0]
        qpos = self._qpos_v
        qpos[:, :2] = torch.where(near[:, None], self._tp_out[draw], qpos[:, :2])
        self._used += near
        self._steps_since_refill += 1
        if self._steps_since_refill >= TELEPORT_DRAWS:
            self._settle_teleports()

    def close(self):
        if not self._closed:
            self._closed = True
            self.sim.close()
            self._step1 = self._last = None
