"""Step every DMC environment in one mujoco.batch and compute task outputs for all environments at once."""

from __future__ import annotations

import ctypes
import functools
import itertools
import math

import numpy as np
import torch

from ...._core.state_layout import StateLayout
from ...._core.tensordicts import build_tensordict, leaf_tree, to_device
from ...._execution.protocol import SimulatorAdapter
from .state import group_views, model_columns, new_groups, write_task_row

XX, XZ, ZX, ZY, ZZ = 0, 2, 6, 7, 8  # MuJoCo's row-major xmat entries
#: Domains whose dm_control reset is mostly long MuJoCo calls: CheetahRun settles for 200 steps in one call. Their
#: MuJoCo Warp reset owners reset on threads. The resets of other domains are faster in one thread: Quadruped's
#: height search makes many short MuJoCo calls, and threads that take the GIL back after each call wait for each other.
THREADED_RESETS = frozenset({"cheetah"})

_CPU = torch.device("cpu")


@functools.cache
def _scale(sigmoid, value_at_margin):
    """The scale that dm_control's ``rewards._sigmoids`` computes for ``sigmoid``, by the same operations."""
    if sigmoid == "gaussian":
        return np.sqrt(-2 * np.log(value_at_margin))
    if sigmoid == "long_tail":
        return np.sqrt(1 / value_at_margin - 1)
    if sigmoid == "linear":
        return 1 - value_at_margin
    return np.sqrt(1 - value_at_margin)


_ARRAY_SIGMOIDS = frozenset({"gaussian", "linear", "quadratic"})


def _outside(x, lower, upper, margin):
    """dm_control's distance of ``x`` outside ``[lower, upper]`` over ``margin``, and the in-bounds mask.

    With both bounds 0 the distance is ``abs(x) / margin``, equal in magnitude, and the mask is None: every sigmoid
    is exactly 1 at distance 0, so the in-bounds values need no mask.
    """
    if type(lower) in (int, float) and type(upper) in (int, float) and lower == 0 and upper == 0:
        return np.abs(x) / margin, None
    return np.where(x < lower, lower - x, x - upper) / margin, np.logical_and(lower <= x, x <= upper)


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1):
    """dm_control's ``rewards.tolerance`` on environment arrays.

    The gaussian, linear and quadratic sigmoids run dm_control's array operations in its order, without its
    argument checks; the other sigmoids call dm_control.
    """
    low = 0 < value_at_margin if sigmoid == "gaussian" else 0 <= value_at_margin
    if sigmoid not in _ARRAY_SIGMOIDS or not (low and value_at_margin < 1):
        from dm_control.utils import rewards

        return rewards.tolerance(x, bounds, margin, sigmoid, value_at_margin)
    lower, upper = bounds
    if margin == 0:
        return np.where(np.logical_and(lower <= x, x <= upper), 1.0, 0.0)
    distance, in_bounds = _outside(x, lower, upper, margin)
    scaled = distance * _scale(sigmoid, value_at_margin)
    if sigmoid == "gaussian":
        value = np.exp(-0.5 * scaled**2)
    elif sigmoid == "linear":
        value = np.where(abs(scaled) < 1, 1 - scaled, 0.0)
    else:
        value = np.where(abs(scaled) < 1, 1 - scaled**2, 0.0)
    return value if in_bounds is None else np.where(in_bounds, 1.0, value)


def _libm_square(x):
    """``pow(x, 2.0)`` of each element with the C library, as dm_control squares a scalar."""
    return np.fromiter(map(math.pow, x.ravel().tolist(), itertools.repeat(2.0)), np.float64, x.size).reshape(x.shape)


def scalar_tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1):
    """``rewards.tolerance`` of each environment's scalar, for the gaussian and long_tail sigmoids."""
    if sigmoid not in ("gaussian", "long_tail"):
        raise ValueError(f"Unsupported scalar sigmoid {sigmoid!r}")
    distance, in_bounds = _outside(x, *bounds, margin)
    # dm_control squares a scalar with libm pow, which can round differently from x * x.
    square = _libm_square(distance * _scale(sigmoid, value_at_margin))
    value = np.exp(-0.5 * square) if sigmoid == "gaussian" else 1 / (square + 1)
    return value if in_bounds is None else np.where(in_bounds, 1.0, value)


#: State fields read from the batch: ``State key -> MjData field``.
_LIVE = {"qpos": "qpos", "qvel": "qvel", "time": "time", "warmstart": "qacc_warmstart", "activation": "act"}
_COUNT, _RESET = ("lifecycle", "step_count"), ("lifecycle", "reset_next_step")
_CUMULATIVE = ("task", "cumulative_reward")


class DMCBatchAdapter(SimulatorAdapter):
    """DMC environments stepped together in one mujoco.batch; dm_control resets, restores and varies them."""

    #: Each step returns fresh output tensors.
    outputs_owned = True

    def __init__(self, instances, task, threads):
        """Step on ``threads`` threads."""
        import mujoco
        from dm_control.rl.control import PhysicsError
        from mujoco.batch import Batch

        self.instances, self._task, self._cumulative = instances, task, instances._cumulative
        self.key, self.num_envs, self.device = instances.key, instances.num_envs, instances.device
        self.action_dim, self.fields, self.schema = instances.action_dim, instances.fields, instances.schema
        self.physics_layout, self.physics_dim = instances.physics_layout, instances.physics_dim
        self._models = [env.physics.model.ptr for env in instances.envs]
        self._transform = instances.envs[0]._transform
        self._nsub, self._limit = instances.envs[0]._env._n_sub_steps, instances.envs[0]._env._step_limit
        self._get_state, self._spec = mujoco.mj_getState, mujoco.mjtState.mjSTATE_INTEGRATION
        self._vel, self._error = mujoco.mjtStage.mjSTAGE_VEL, PhysicsError
        self._actuation = mujoco.mjtDisableBit.mjDSBL_ACTUATION
        # A persistent batch keeps each environment's MjData, so a step continues from the stages the last call computed,
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
        #: Substeps that every environment has before its step limit; 0 until a step computes it.
        self._margin = 0.0
        self._rng_views = {}
        self._load(np.arange(self.num_envs))

    def _reset_instances(self, ids, seed=None):
        """Reset environments ``ids`` with dm_control, ``seed + i`` seeding environment ``i``; environments share no state.

        This is dm_control's ``Environment.reset`` without its first observation, which the batch task computes. A
        batch task with ``begin_reset`` runs the MuJoCo work of the reset for all environments in the batch, on its threads:
        ``begin_reset`` does each environment's reset up to that work, ``batch_reset`` runs it and ``end_reset`` finishes it.
        """
        envs = self.instances.envs
        begin = getattr(self.task, "begin_reset", None)
        begun = []
        for i in ids:
            env = envs[i]._env
            if seed is not None:
                # Legacy seeding of the environment's RandomState gives the state of np.random.RandomState(seed + i).
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
        """Copy environments from dm_control into the batch and forward them as dm_control's reset does.

        ``seeded`` tells that a seeded reset just drew the environments.
        """
        ids = np.asarray(ids)
        envs = [self.instances.envs[i] for i in ids]
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
        """Write the task RNG and model entries of environments ``ids`` from dm_control into the State columns.

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
        """Write the installed physics Variation of every environment into the State columns."""
        if self.instances.physics is not None:
            for key, value in self.instances.physics.export().items():
                self._columns[key][:] = value

    def _rng_view(self, instance, random):
        """Views of the MT19937 keys and position of ``random``, the task RNG of environment ``instance``."""
        view = self._rng_views.get(instance)
        if view is None or view[0] is not random:
            # numpy's mt19937_state: 624 uint32 keys, then the int position.
            address = random._bit_generator.ctypes.state_address
            keys = np.ctypeslib.as_array((ctypes.c_uint32 * 624).from_address(address))
            position = np.ctypeslib.as_array((ctypes.c_int * 1).from_address(address + keys.nbytes))
            view = self._rng_views[instance] = (random, keys, position)
        return view[1], view[2]

    def configure_physics(self, values):
        self.instances.configure_physics(values)
        self._export_physics()
        self.b.set_models(self._models)
        self.b.forward(stage=self._vel)
        self.task = self._task(self.b, self._models)

    def reset(self, seed=None):
        self.instances._check_open()
        self._reset_instances(range(self.num_envs), seed)
        self._cumulative.fill(0)
        self._load(np.arange(self.num_envs), seeded=seed is not None)
        return self.get_state()

    def get_state(self):
        self.instances._check_open()
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
        return self.instances.validate(state)

    def set_state(self, state):
        self.instances.set_state(state)
        self.b.set_models(self._models)
        self.task = self._task(self.b, self._models)
        self._load(np.arange(self.num_envs))

    def check_step(self, actions, active):
        return self.instances.check_step(actions, active)

    def step(self, actions, *, active=None, capture_state=True):
        out = self.step_host(actions, active=active, capture_state=capture_state)
        # The State and the outputs move to the device in one copy.
        return out if self.device.type == "cpu" else to_device(out, self.device)

    def step_host(self, actions, *, active=None, capture_state=True):
        """``step`` with the result on the CPU, for a caller that moves several steps to the device at once."""
        out = leaf_tree(self._advance(actions, active))
        if capture_state:
            out = {"state": self._state_on(_CPU), **out}
        return build_tensordict(out, [self.num_envs], _CPU)

    def state_leaves(self):
        """The State as host arrays by key path under ``"state"``. They may be live buffers: the caller copies them."""
        return self._layout.host_leaves(self._state_groups(), ("state",))

    def step_leaves(self, actions, *, state=True):
        """Advance every environment one step; the State and the outputs as host arrays by key path, for one device copy.

        With ``state`` False, the State is left out. The arrays may be live buffers: the caller copies them before
        the next call.
        """
        outputs = self._advance(actions, None)
        if not state:
            return outputs
        return {**self.state_leaves(), **outputs}

    def _advance(self, actions, active):
        """Advance the environments; the task outputs as ``{key path: array}``."""
        self.instances._check_open()
        actions, _ = self.instances.check_step(actions, active)
        ctrl, reward = self._transform(actions), np.zeros(self.num_envs)
        for _ in range(2):
            # While every environment is more than one substep from its limit, no environment resets or reaches it.
            quiet = self._margin > 1
            reset = self._reset_next
            resetting = not quiet and reset.any()
            if resetting:
                reset = reset.copy()
                ids = np.flatnonzero(reset)
                self._reset_instances(ids)
                self._load(ids)
            self._ctrl[:] = ctrl
            self.b.step(~reset if resetting else None, self._nsub, stage=self._vel)
            if (self._warning > self._warnings).any():
                bad = np.flatnonzero((self._warning > self._warnings).any(axis=1))
                raise self._error(f"Physics state is invalid in environments {bad.tolist()}")
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
        self.instances.close()
        self.b = self.task = None
