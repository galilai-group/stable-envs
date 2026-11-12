"""Step every DMC lane in one mujoco.batch and compute task outputs for all lanes at once."""

from __future__ import annotations

import contextlib
import copy
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from ...._core.tensordicts import build_tensordict
from ...._execution.protocol import SimulatorAdapter
from .state import export_columns

XX, XZ, ZX, ZY, ZZ = 0, 2, 6, 7, 8  # MuJoCo's row-major xmat entries
_pow = np.frompyfunc(math.pow, 2, 1)
#: Domains whose dm_control reset is mostly long MuJoCo calls: CheetahRun settles for 200 steps in one call. Their
#: lanes reset on the batch threads. The resets of other domains are faster in one thread: Quadruped's height search
#: makes many short MuJoCo calls, and threads that take the GIL back after each call wait for each other.
THREADED_RESETS = frozenset({"cheetah"})
#: MuJoCo work, in seconds, that each batch thread gets per step call. Waking a thread can cost about this much.
THREAD_WORK = 1e-4
#: Fewest lanes per batch thread: each thread that a call waits for can be the late one.
THREAD_LANES = 2


def tolerance(x, *args, **kwargs):
    """dm_control's ``rewards.tolerance`` on lane arrays."""
    from dm_control.utils import rewards

    return rewards.tolerance(x, *args, **kwargs)


def scalar_tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1):
    """``rewards.tolerance`` of each lane's scalar, for the gaussian and long_tail sigmoids."""
    lower, upper = bounds
    d = np.where(x < lower, lower - x, x - upper) / margin
    # dm_control squares a scalar with libm pow, which can round differently from x * x.
    if sigmoid == "gaussian":
        value = np.exp(-0.5 * _pow(d * np.sqrt(-2 * np.log(value_at_margin)), 2.0).astype(np.float64))
    elif sigmoid == "long_tail":
        value = 1 / (_pow(d * np.sqrt(1 / value_at_margin - 1), 2.0).astype(np.float64) + 1)
    else:
        raise ValueError(f"Unsupported scalar sigmoid {sigmoid!r}")
    return np.where((lower <= x) & (x <= upper), 1.0, value)


def _put(tree, key, value):
    """Set ``key``, a name or a tuple of names, in nested dicts."""
    if isinstance(key, str):
        tree[key] = value
        return
    for part in key[:-1]:
        tree = tree.setdefault(part, {})
    tree[key[-1]] = value


def step_threads(physics, nsub, lanes, most):
    """Threads for a step call: one per ``THREAD_WORK`` of the lanes' MuJoCo work, timed on a copy of ``physics``."""
    most = min(most, -(-lanes // THREAD_LANES))
    if most <= 1:
        return 1
    import mujoco

    m, d = physics.model.ptr, copy.copy(physics.data.ptr)
    reps, start = 0, time.perf_counter()
    while (elapsed := time.perf_counter() - start) < 1e-3 or reps < 2:
        mujoco.mj_step(m, d, nsub)
        reps += 1
    return max(1, min(most, math.ceil(lanes * elapsed / reps / THREAD_WORK)))


#: State fields read from the batch: ``State key -> MjData field``.
_LIVE = {"qpos": "qpos", "qvel": "qvel", "time": "time", "warmstart": "qacc_warmstart", "activation": "act"}


class DMCBatchAdapter(SimulatorAdapter):
    """DMC lanes stepped together in one mujoco.batch; dm_control resets, restores and varies them."""

    #: Each step returns fresh output tensors.
    outputs_owned = True

    def __init__(self, lanes, task, threads, *, fit_threads=False):
        """Step on ``threads`` threads, or on at most that many by ``step_threads`` with ``fit_threads``."""
        import mujoco
        from dm_control.rl.control import PhysicsError
        from mujoco.batch import Batch

        self.lanes, self._task, self._cumulative = lanes, task, lanes._cumulative
        self.key, self.num_envs, self.device = lanes.key, lanes.num_envs, lanes.device
        self.action_dim, self.fields = lanes.action_dim, lanes.fields
        self.physics_layout, self.physics_dim = lanes.physics_layout, lanes.physics_dim
        self._models = [env.physics.model.ptr for env in lanes.envs]
        self._transform = lanes.envs[0]._transform
        self._nsub, self._limit = lanes.envs[0]._env._n_sub_steps, lanes.envs[0]._env._step_limit
        self._get_state, self._spec = mujoco.mj_getState, mujoco.mjtState.mjSTATE_INTEGRATION
        self._vel, self._error = mujoco.mjtStage.mjSTAGE_VEL, PhysicsError
        if fit_threads:
            nthread = step_threads(lanes.envs[0].physics, self._nsub, self.num_envs, threads)
        else:
            nthread = threads
        # A persistent batch keeps each lane's MjData, so a step continues from the stages the last call computed,
        # as dm_control's step does.
        self.b = Batch(self._models[0], self.num_envs, nthread=nthread, persistent=True)
        # These resets spend most of their time in MuJoCo calls, which release the GIL.
        self._pool = ThreadPoolExecutor(threads) if threads > 1 and lanes.key in THREADED_RESETS else None
        self.b.set_models(self._models)
        self.task = task(self.b, self._models)
        self._live = {key: self.b.bind(name) for key, name in _LIVE.items() if key in self.fields}
        self._count, self._reset_next = np.zeros(self.num_envs, np.int64), np.zeros(self.num_envs, bool)
        self._load(np.arange(self.num_envs))

    def _reset_lanes(self, ids, seed=None):
        """Reset lanes ``ids`` with dm_control, ``seed + i`` seeding lane ``i``; lanes share no state.

        This is dm_control's ``Environment.reset`` without its first observation, which the batch task computes.
        """
        envs = self.lanes.envs

        def reset(i):
            env = envs[i]._env
            if seed is not None:
                env.task._random = np.random.RandomState(int(seed) + int(i))
            env._reset_next_step, env._step_count = False, 0
            with env.physics.reset_context():
                env.task.initialize_episode(env.physics)

        if self._pool is None or len(ids) < 2:
            for i in ids:
                reset(i)
        else:
            list(self._pool.map(reset, ids))

    def _load(self, ids):
        """Copy lanes from dm_control into the batch and forward them as dm_control's reset does."""
        envs = [self.lanes.envs[i] for i in ids]
        self._warnings = self.b.warning[:, :, 1].copy()
        with contextlib.ExitStack() as stack:
            for i, env in zip(ids, envs, strict=True):
                self._get_state(env.physics.model.ptr, env.physics.data.ptr, self.b.state[i], self._spec)
                self._count[i], self._reset_next[i] = env._env._step_count, env._env._reset_next_step
                stack.enter_context(env.physics.model.disable("actuation"))
            self.b.forward(ids)
        columns = export_columns(envs, self._cumulative[ids], self.lanes._base_fields)
        if len(ids) == self.num_envs:
            self._columns = columns
        else:
            for key, column in columns.items():
                self._columns[key][ids] = column

    def configure_physics(self, values):
        self.lanes.configure_physics(values)
        self.b.set_models(self._models)
        self.b.forward(stage=self._vel)
        self.task = self._task(self.b, self._models)

    def reset(self, seed=None):
        self.lanes._check_open()
        self._reset_lanes(range(self.num_envs), seed)
        self._cumulative.fill(0)
        self._load(np.arange(self.num_envs))
        return self.get_state()

    def get_state(self):
        self.lanes._check_open()
        live = {
            **self._live,
            "time": self._live["time"][:, None],
            ("lifecycle", "step_count"): self._count[:, None],
            ("lifecycle", "reset_next_step"): self._reset_next[:, None],
            ("task", "cumulative_reward"): self._cumulative[:, None],
        }
        # Fresh host copies: later steps and resets cannot change the returned snapshot.
        state = {}
        for key, column in self._columns.items():
            _put(state, key, torch.from_numpy(np.array(live.get(key, column))))
        for key, value in ({} if self.lanes.physics is None else self.lanes.physics.export()).items():
            _put(state, key, torch.from_numpy(value))
        state = build_tensordict(state, [self.num_envs], "cpu")
        return state if self.device.type == "cpu" else state.to(self.device)

    def validate(self, state):
        return self.lanes.validate(state)

    def set_state(self, state):
        self.lanes.set_state(state)
        self.b.set_models(self._models)
        self.task = self._task(self.b, self._models)
        self._load(np.arange(self.num_envs))

    def check_step(self, actions, active):
        return self.lanes.check_step(actions, active)

    def step(self, actions, *, active=None, capture_state=True):
        self.lanes._check_open()
        actions, _ = self.lanes.check_step(actions, active)
        ctrl, reward = self._transform(actions), np.zeros(self.num_envs)
        for _ in range(2):
            reset = self._reset_next.copy()
            if reset.any():
                ids = np.flatnonzero(reset)
                self._reset_lanes(ids)
                self._load(ids)
            self.b.ctrl[:] = ctrl
            self.b.step(~reset, self._nsub, stage=self._vel)
            if (bad := np.flatnonzero((self.b.warning[:, :, 1] > self._warnings).any(axis=1))).size:
                raise self._error(f"Physics state is invalid in lanes {bad.tolist()}")
            reward += np.where(reset, 0.0, self.task.reward())
            self._count += ~reset
            self._reset_next = self._count >= self._limit
        self._cumulative += reward
        out = {
            **({"state": self.get_state()} if capture_state else {}),
            "reward": torch.from_numpy(reward),
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool),
            "observation": torch.from_numpy(self.task.observation()),
            "score": torch.from_numpy(self._cumulative / 1000.0),
        }
        if (info := self.task.info()) is not None:
            out["success"] = torch.from_numpy(np.array(info[0], np.float64))
            out["info"] = {k: torch.from_numpy(np.array(v)) for k, v in info[1].items()}
        out = build_tensordict(out, [self.num_envs], "cpu")
        return out if self.device.type == "cpu" else out.to(self.device)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
        self.lanes.close()
        self.b = self.task = None
