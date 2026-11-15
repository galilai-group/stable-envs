"""Step every DMC lane in one mujoco.batch and compute task outputs for all lanes at once."""

from __future__ import annotations

import ctypes
import math

import numpy as np
import torch

from ...._core.state_layout import StateLayout
from ...._core.tensordicts import build_tensordict, leaf_tree, to_device
from ...._execution.protocol import SimulatorAdapter
from .state import group_views, model_columns, new_groups, write_task_row

XX, XZ, ZX, ZY, ZZ = 0, 2, 6, 7, 8  # MuJoCo's row-major xmat entries
_pow = np.frompyfunc(math.pow, 2, 1)
#: Domains whose dm_control reset is mostly long MuJoCo calls: CheetahRun settles for 200 steps in one call. Their
#: MuJoCo Warp reset owners reset on threads. The resets of other domains are faster in one thread: Quadruped's
#: height search makes many short MuJoCo calls, and threads that take the GIL back after each call wait for each other.
THREADED_RESETS = frozenset({"cheetah"})

_CPU = torch.device("cpu")


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


#: State fields read from the batch: ``State key -> MjData field``.
_LIVE = {"qpos": "qpos", "qvel": "qvel", "time": "time", "warmstart": "qacc_warmstart", "activation": "act"}
_COUNT, _RESET = ("lifecycle", "step_count"), ("lifecycle", "reset_next_step")
_CUMULATIVE = ("task", "cumulative_reward")


class DMCBatchAdapter(SimulatorAdapter):
    """DMC lanes stepped together in one mujoco.batch; dm_control resets, restores and varies them."""

    #: Each step returns fresh output tensors.
    outputs_owned = True

    def __init__(self, lanes, task, threads):
        """Step on ``threads`` threads."""
        import mujoco
        from dm_control.rl.control import PhysicsError
        from mujoco.batch import Batch

        self.lanes, self._task, self._cumulative = lanes, task, lanes._cumulative
        self.key, self.num_envs, self.device = lanes.key, lanes.num_envs, lanes.device
        self.action_dim, self.fields, self.schema = lanes.action_dim, lanes.fields, lanes.schema
        self.physics_layout, self.physics_dim = lanes.physics_layout, lanes.physics_dim
        self._models = [env.physics.model.ptr for env in lanes.envs]
        self._transform = lanes.envs[0]._transform
        self._nsub, self._limit = lanes.envs[0]._env._n_sub_steps, lanes.envs[0]._env._step_limit
        self._get_state, self._spec = mujoco.mj_getState, mujoco.mjtState.mjSTATE_INTEGRATION
        self._vel, self._error = mujoco.mjtStage.mjSTAGE_VEL, PhysicsError
        self._actuation = mujoco.mjtDisableBit.mjDSBL_ACTUATION
        # A persistent batch keeps each lane's MjData, so a step continues from the stages the last call computed,
        # as dm_control's step does.
        self.b = Batch(self._models[0], self.num_envs, nthread=threads, persistent=True)
        self.b.set_models(self._models)
        self.task = task(self.b, self._models)
        # The State lives in host group arrays; the step copies the batch's physics and counters into their views.
        self._groups = new_groups(self.schema, self.num_envs)
        self._columns = group_views(self.schema, self._groups, self.fields)
        self._layout = StateLayout(self.schema.leaves(), self.num_envs)
        self._live = [
            (self._columns[key], bound[:, None] if bound.ndim == 1 else bound)
            for key, name in _LIVE.items()
            if key in self.fields
            for bound in (self.b.bind(name),)
        ]
        # Bound once: ``Batch.ctrl`` makes a new view at each access.
        self._ctrl, self._warning = self.b.bind("ctrl"), self.b.warning[:, :, 1]
        self._count, self._reset_next = np.zeros(self.num_envs, np.int64), np.zeros(self.num_envs, bool)
        #: Substeps that every lane has before its step limit; 0 until a step computes it.
        self._margin = 0.0
        self._rng_views = {}
        self._load(np.arange(self.num_envs))

    def _reset_lanes(self, ids, seed=None):
        """Reset lanes ``ids`` with dm_control, ``seed + i`` seeding lane ``i``; lanes share no state.

        This is dm_control's ``Environment.reset`` without its first observation, which the batch task computes. A
        batch task with ``begin_reset`` runs the MuJoCo work of the reset for all lanes in the batch, on its threads:
        ``begin_reset`` does each lane's reset up to that work, ``batch_reset`` runs it and ``end_reset`` finishes it.
        """
        envs = self.lanes.envs
        begin = getattr(self.task, "begin_reset", None)
        begun = []
        for i in ids:
            env = envs[i]._env
            if seed is not None:
                # Legacy seeding of the lane's RandomState gives the state of np.random.RandomState(seed + i).
                env.task.random.seed(int(seed) + int(i))
            env._reset_next_step, env._step_count = False, 0
            if begin is None:
                with env.physics.reset_context():
                    env.task.initialize_episode(env.physics)
            else:
                begun.append(begin(env, i))
        if begin is None or not len(ids):
            return
        ids = np.asarray(ids)
        for i, value in zip(ids, self.task.batch_reset(ids, begun), strict=True):
            self.task.end_reset(envs[i]._env, i, value)

    def _load(self, ids, seeded=False):
        """Copy lanes from dm_control into the batch and forward them as dm_control's reset does.

        ``seeded`` tells that a seeded reset just drew the lanes.
        """
        ids = np.asarray(ids)
        envs = [self.lanes.envs[i] for i in ids]
        self._margin = 0.0
        self._warnings = self.b.warning[:, :, 1].copy()
        models = [self._models[i] for i in ids]
        flags = [m.opt.disableflags for m in models]
        try:
            for i, env, m in zip(ids, envs, models, strict=True):
                self._get_state(m, env.physics.data.ptr, self.b.state[i], self._spec)
                self._count[i], self._reset_next[i] = env._env._step_count, env._env._reset_next_step
                m.opt.disableflags |= self._actuation
            self.b.forward(ids)
        finally:
            for m, flag in zip(models, flags, strict=True):
                m.opt.disableflags = flag
        self._export(ids, envs, seeded and not getattr(self.task, "RESET_DRAWS_NORMALS", True))
        if len(ids) == self.num_envs:
            self._export_physics()

    def _export(self, ids, envs, gauss_free=False):
        """Write the task RNG and model entries of lanes ``ids`` from dm_control into the State columns.

        With ``gauss_free``, the task RNGs hold no cached gaussian, and their keys and positions are read in place.
        """
        columns = self._columns
        model_fields = model_columns(columns)
        if not gauss_free or model_fields:
            for i, env in zip(ids, envs, strict=True):
                write_task_row(columns, i, env, model_fields)
            return
        keys, position = columns["rng", "keys"], columns["rng", "position"]
        for i, env in zip(ids, envs, strict=True):
            keys[i], position[i] = self._rng_view(i, env.task.random)
        columns["rng", "has_gauss"][ids] = False
        columns["rng", "cached_gaussian"][ids] = 0.0

    def _export_physics(self):
        """Write the installed physics Variation of every lane into the State columns."""
        if self.lanes.physics is not None:
            for key, value in self.lanes.physics.export().items():
                self._columns[key][:] = value

    def _rng_view(self, lane, random):
        """Views of the MT19937 keys and position of ``random``, the task RNG of lane ``lane``."""
        view = self._rng_views.get(lane)
        if view is None or view[0] is not random:
            # numpy's mt19937_state: 624 uint32 keys, then the int position.
            address = random._bit_generator.ctypes.state_address
            keys = np.ctypeslib.as_array((ctypes.c_uint32 * 624).from_address(address))
            position = np.ctypeslib.as_array((ctypes.c_int * 1).from_address(address + keys.nbytes))
            view = self._rng_views[lane] = (random, keys, position)
        return view[1], view[2]

    def configure_physics(self, values):
        self.lanes.configure_physics(values)
        self._export_physics()
        self.b.set_models(self._models)
        self.b.forward(stage=self._vel)
        self.task = self._task(self.b, self._models)

    def reset(self, seed=None):
        self.lanes._check_open()
        self._reset_lanes(range(self.num_envs), seed)
        self._cumulative.fill(0)
        self._load(np.arange(self.num_envs), seeded=seed is not None)
        return self.get_state()

    def get_state(self):
        self.lanes._check_open()
        return self._state_on(self.device)

    def _state_on(self, device):
        # The export copies every group: later steps and resets cannot change the returned snapshot.
        return self._layout.export(self._state_groups(), device)

    def _state_groups(self):
        """The State group arrays, with the batch's physics and the step counters copied in."""
        for column, bound in self._live:
            column[:] = bound
        columns = self._columns
        columns[_COUNT][:, 0] = self._count
        columns[_RESET][:, 0] = self._reset_next
        columns[_CUMULATIVE][:, 0] = self._cumulative
        return self._groups

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
        out = leaf_tree(self._advance(actions, active))
        if capture_state:
            out = {"state": self._state_on(_CPU), **out}
        out = build_tensordict(out, [self.num_envs], _CPU)
        # The State and the outputs move to the device in one copy.
        return out if self.device.type == "cpu" else to_device(out, self.device)

    def step_leaves(self, actions):
        """Advance every lane one step; the State and the outputs as host arrays by key path, for one device copy.

        The arrays may be live buffers: the caller copies them before the next call.
        """
        outputs = self._advance(actions, None)
        return {**self._layout.host_leaves(self._state_groups(), ("state",)), **outputs}

    def _advance(self, actions, active):
        """Advance the lanes; the task outputs as ``{key path: array}``."""
        self.lanes._check_open()
        actions, _ = self.lanes.check_step(actions, active)
        ctrl, reward = self._transform(actions), np.zeros(self.num_envs)
        for _ in range(2):
            # While every lane is more than one substep from its limit, no lane resets or reaches it.
            quiet = self._margin > 1
            reset = self._reset_next
            resetting = not quiet and reset.any()
            if resetting:
                reset = reset.copy()
                ids = np.flatnonzero(reset)
                self._reset_lanes(ids)
                self._load(ids)
            self._ctrl[:] = ctrl
            self.b.step(~reset if resetting else None, self._nsub, stage=self._vel)
            if (self._warning > self._warnings).any():
                bad = np.flatnonzero((self._warning > self._warnings).any(axis=1))
                raise self._error(f"Physics state is invalid in lanes {bad.tolist()}")
            if resetting:
                reward += np.where(reset, 0.0, self.task.reward())
                self._count += ~reset
            else:
                reward += self.task.reward()
                self._count += 1
            if quiet:
                self._margin -= 1
            else:
                self._reset_next = self._count >= self._limit
                self._margin = float((self._limit - self._count).min())
        self._cumulative += reward
        out = {
            ("reward",): reward,
            ("terminated",): np.zeros(self.num_envs, np.bool_),
            ("truncated",): np.zeros(self.num_envs, np.bool_),
            ("observation",): self.task.observation(),
            ("score",): self._cumulative / 1000.0,
        }
        if (info := self.task.info()) is not None:
            out["success",] = np.array(info[0], np.float64)
            out.update({("info", k): np.array(v) for k, v in info[1].items()})
        return out

    def close(self):
        self.lanes.close()
        self.b = self.task = None
