"""Native reset owners with compact per-lane model and RNG checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from functools import cache

import numpy as np
import torch

from ...._execution.state_columns import require_columns
from .batch import THREADED_RESETS
from .factory import batch_threads, make_local
from .state import MODEL_FIELDS, columns_to_state, model_columns, new_columns, write_row


@cache
def warp_batch_fields() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The MuJoCo Warp model and option fields that can hold one value per world."""
    import mujoco_warp as mjw

    def batched(cls):
        return tuple(f.name for f in fields(cls) if getattr(f.type, "shape", (None,))[0] == "*")

    return batched(mjw.Model), batched(mjw.Option)


class _ModelView:
    """Read-only model attributes; large static assets belong to one prototype."""

    def __init__(self, batch, lane, section=None):
        self._batch, self._lane, self._section = batch, lane, section

    def __getattr__(self, name):
        batch = self._batch
        if name in {"opt", "stat"} and self._section is None:
            return _ModelView(batch, self._lane, section=name)
        columns = {"opt": batch.options, "stat": batch.statistics}.get(self._section, batch.columns)
        if name in columns:
            return columns[name][self._lane]
        owner = getattr(batch.reference, self._section) if self._section else batch.reference
        return getattr(owner, name)


class WarpModelBatch:
    """Native model columns needed by Warp, without duplicating native assets."""

    def __init__(self, reference, count, model_fields, option_fields):
        self.reference = reference
        self.count = count
        self._bound = {}
        # Native reset/variation also uses constants which Warp does not store.
        # Keep these small arrays, not native texture/mesh assets or MjData.
        names = dict.fromkeys(
            (*model_fields, "site_size", "geom_gap", "pair_gap", "bvh_aabb", "dof_M0", "dof_length", "geom_sameframe")
        )
        self.columns = {
            name: np.empty(
                (count, *np.shape(getattr(reference, name))), dtype=np.asarray(getattr(reference, name)).dtype
            )
            for name in names
        }
        options = dict.fromkeys("impratio" if name == "impratio_invsqrt" else name for name in option_fields)
        self.options = {
            name: np.empty(
                (count, *np.shape(getattr(reference.opt, name))), dtype=np.asarray(getattr(reference.opt, name)).dtype
            )
            for name in options
        }
        self.statistics = {name: np.empty(count) for name in ("meaninertia", "meanmass")}
        body = reference.geom_bodyid
        fixed = (reference.body_weldid[body] == 0) & (reference.body_mocapid[reference.body_rootid[body]] == -1)
        self.fixed_indices = np.flatnonzero(fixed)
        self.fixed_positions = np.empty((count, len(self.fixed_indices), 3))
        self.fixed_rotations = np.empty((count, len(self.fixed_indices), 3, 3))

    def __len__(self):
        return self.count

    def __iter__(self):
        return (_ModelView(self, i) for i in range(self.count))

    def __getitem__(self, lane):
        if not 0 <= lane < self.count:
            raise IndexError(lane)
        return _ModelView(self, lane)

    def select(self, lanes):
        selected = copy.copy(self)
        selected.count = len(lanes)
        selected.columns = {name: values[lanes].copy() for name, values in self.columns.items()}
        selected.options = {name: values[lanes].copy() for name, values in self.options.items()}
        selected.statistics = {name: values[lanes].copy() for name, values in self.statistics.items()}
        selected.fixed_positions = self.fixed_positions[lanes].copy()
        selected.fixed_rotations = self.fixed_rotations[lanes].copy()
        return selected

    def clone(self):
        return self.select(np.arange(self.count))

    def _arrays(self, model):
        """The model's column arrays, which keep their memory for the life of the model."""
        entry = self._bound.get(id(model))
        if entry is None:
            entry = self._bound[id(model)] = (model, {name: getattr(model, name) for name in self.columns})
        return entry[1]

    def capture(self, lane, physics):
        model, data = physics.model.ptr, physics.data.ptr
        arrays = self._arrays(model)
        for name, values in self.columns.items():
            values[lane] = arrays[name]
        for name, values in self.options.items():
            values[lane] = getattr(model.opt, name)
        for name, values in self.statistics.items():
            values[lane] = getattr(model.stat, name)
        self.fixed_positions[lane] = data.geom_xpos[self.fixed_indices]
        self.fixed_rotations[lane] = data.geom_xmat[self.fixed_indices].reshape(-1, 3, 3)

    def restore(self, lane, physics):
        """Select a lane without recomputing its previously derived constants."""
        model = physics.model.ptr
        arrays = self._arrays(model)
        for name, values in self.columns.items():
            arrays[name][:] = values[lane]
        for name, values in self.options.items():
            if values.ndim == 1:
                setattr(model.opt, name, values[lane].item())
            else:
                getattr(model.opt, name)[:] = values[lane]
        for name, values in self.statistics.items():
            setattr(model.stat, name, values[lane].item())


@dataclass
class _Prepared:
    state: object
    models: WarpModelBatch
    physics_values: np.ndarray | None


class WarpResetPool:
    """Native DMC resets with independent per-lane saved state.

    Domains whose reset is mostly MuJoCo work use one owner per batch thread, each
    resetting a shard of the lanes. Other domains use one owner. Transient native
    work never changes the installed lane records. Model preparation can fail
    without committing another lane's partial updates.
    """

    def __init__(self, key, num_envs, seed, model_fields, option_fields):
        self.key, self.num_envs = key, int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self._closed = False
        self._owners, self._pool = [], None
        try:
            count = batch_threads(self.num_envs) if key in THREADED_RESETS else 1
            self._owners = [make_local(key, 1, "cpu", seed) for _ in range(count)]
            self._pool = ThreadPoolExecutor(count) if count > 1 else None
            self._owner = self._owners[0]
            self.envs = self._owner.envs
            self.fields, self.action_dim = self._owner.fields, self._owner.action_dim
            reference = copy.copy(self.envs[0].physics.model.ptr)
            self.models = WarpModelBatch(reference, self.num_envs, model_fields, option_fields)
            # This holder uses the native physics layout/column validation, but
            # only the owners' PhysicsLanes ever apply native changes.
            self.physics = None if self._owner.physics is None else copy.copy(self._owner.physics)
            if self.physics is not None:
                self.physics.models = self.models
                self.physics._data = None
                self.physics.values = np.tile(self.physics.spec.nominal, (self.num_envs, 1))
            self._state = None
            self.commit(self.prepare_reset(seed, initial=True))
        except BaseException:
            self.close()
            raise

    @property
    def physics_layout(self):
        return self._owner.physics_layout

    @property
    def physics_dim(self):
        return self._owner.physics_dim

    def _check_open(self):
        if self._closed:
            raise RuntimeError("WarpResetPool is closed")

    def _select_owner(self, owner, state, lane, values):
        env = owner.envs[0]
        if self._state is not None:
            self.models.restore(lane, env.physics)
            if self.physics is not None:
                # Compare proposed values with this lane, not the previously
                # selected owner. Reapplying unchanged variation can recompute
                # constants from a later episode's target or rounded geometry.
                owner.physics.values = self.physics.values[lane : lane + 1].copy()
        if values is not None:
            owner.configure_physics(values[lane : lane + 1])
        if state is not None:
            for key in self.fields:
                if key in MODEL_FIELDS:
                    name, index = MODEL_FIELDS[key]
                    getattr(env.physics.named.model, name)[index] = state[key][lane].numpy()
            env.task.random.set_state(
                (
                    "MT19937",
                    state["rng", "keys"][lane].numpy().astype(np.uint32),
                    state["rng", "position"][lane].item(),
                    int(state["rng", "has_gauss"][lane].item()),
                    state["rng", "cached_gaussian"][lane].item(),
                )
            )
        return env

    def prepare_reset(self, seed=None, lanes=None, *, initial=False):
        self._check_open()
        ids = np.arange(self.num_envs) if lanes is None else np.asarray(lanes, dtype=np.int64)
        values = None if self.physics is None else self.physics.values.copy()
        models = self.models if initial else self.models.clone()
        cumulative = (
            np.zeros(len(ids)) if lanes is None else self._state["task", "cumulative_reward"][ids, 0].numpy().copy()
        )

        columns = new_columns(len(ids), self.fields)
        model_fields = model_columns(columns)

        def reset_shard(owner, positions):
            for position in positions:
                lane = ids[position]
                env = self._select_owner(owner, self._state, lane, values)
                # make_local performed the first lane's registered initial
                # reset already. Reuse it without consuming a second RNG draw.
                if not (initial and lane == 0):
                    base = env._env
                    if seed is not None:
                        base.task._random = np.random.RandomState(int(seed) + int(lane))
                    base._reset_next_step, base._step_count = False, 0
                    with env.physics.reset_context():
                        base.task.initialize_episode(env.physics)
                models.capture(lane, env.physics)
                write_row(columns, position, env, cumulative[position], model_fields)

        shards = [shard for shard in np.array_split(np.arange(len(ids)), len(self._owners)) if len(shard)]
        if self._pool is None or len(shards) < 2:
            reset_shard(self._owners[0], range(len(ids)))
        else:
            list(self._pool.map(lambda job: reset_shard(*job), zip(self._owners, shards, strict=False)))
        extra = None if self.physics is None else {key: value[ids] for key, value in self.physics.export().items()}
        reset = columns_to_state(columns, "cpu", extra)
        state = reset if lanes is None else self._state.clone()
        if lanes is not None:
            state[ids] = reset
        return _Prepared(state, models, values)

    def validate(self, state):
        self._check_open()
        from tensordict import TensorDict

        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError(f"{self.key} requires full TensorDict State with batch [num_envs]")
        rows = None if self.physics is None else self.physics.rows_from_state(state)
        require_columns(state, {key: ((self.num_envs, width), dtype) for key, (width, dtype) in self.fields.items()})
        return rows

    def prepare_state(self, state):
        values = self.validate(state)
        host = state.cpu().clone()
        changed = np.zeros(self.num_envs, dtype=bool)
        if values is not None:
            changed |= np.any(values != self.physics.values, axis=1)
        for key in self.fields:
            if key in MODEL_FIELDS:
                changed |= np.any(host[key].numpy() != self._state[key].numpy(), axis=1)
        models = self.models.clone() if changed.any() else self.models
        for lane in np.flatnonzero(changed):
            env = self._select_owner(self._owner, host, lane, values)
            env.physics.forward()
            models.capture(lane, env.physics)
        return _Prepared(host, models, values)

    def prepare_physics(self, values):
        self._check_open()
        if self.physics is None:
            raise ValueError(f"{self.key} declares no physics Variation")
        spec = self.physics.spec
        if isinstance(values, Mapping):
            rows = self.physics.values.copy()
            for leaf, value in values.items():
                index = spec.leaves.index(leaf)
                offset, width = spec.offsets[index], spec.widths[index]
                rows[:, offset : offset + width] = np.broadcast_to(
                    np.asarray(value, np.float64).reshape(-1, width), (len(rows), width)
                )
        else:
            rows = np.broadcast_to(
                np.asarray(values, np.float64).reshape(-1, spec.dim), self.physics.values.shape
            ).copy()
        proposed = copy.copy(self.physics)
        proposed.values = rows
        state = self._state.clone()
        for key, value in proposed.export().items():
            state[key] = torch.from_numpy(value)
        return self.prepare_state(state)

    def commit(self, prepared):
        self._state, self.models = prepared.state, prepared.models
        if self.physics is not None:
            self.physics.models = self.models
            self.physics.values = prepared.physics_values

    def get_state(self):
        self._check_open()
        return self._state.clone()

    def reset(self, seed=None):
        self.commit(self.prepare_reset(seed))
        return self.get_state()

    def set_state(self, state):
        self.commit(self.prepare_state(state))

    def configure_physics(self, values):
        self.commit(self.prepare_physics(values))

    def close(self):
        if not self._closed:
            if self._pool is not None:
                self._pool.shutdown()
            for owner in self._owners:
                owner.close()
            self._closed = True
