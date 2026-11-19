"""Native reset owners with compact per-environment model and RNG checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from functools import cache

import numpy as np

from ...._core.state_schema import unfold
from .batch import THREADED_RESETS
from .factory import BATCH_TASKS, batch_threads, make_local
from .state import (
    MODEL_FIELDS,
    Columns,
    columns_to_state,
    field_arrays,
    field_name,
    model_columns,
    validate_state,
    write_row,
)


@cache
def warp_batch_fields() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The MuJoCo Warp model and option fields that can hold one value per world."""
    import mujoco_warp as mjw

    def batched(cls):
        return tuple(f.name for f in fields(cls) if getattr(f.type, "shape", (None,))[0] == "*")

    return batched(mjw.Model), batched(mjw.Option)


class _ModelView:
    """Read-only model attributes; large static assets belong to one prototype."""

    def __init__(self, batch, instance, section=None):
        self._batch, self._instance, self._section = batch, instance, section

    def __getattr__(self, name):
        batch = self._batch
        if name in {"opt", "stat"} and self._section is None:
            return _ModelView(batch, self._instance, section=name)
        columns = {"opt": batch.options, "stat": batch.statistics}.get(self._section, batch.columns)
        if name in columns:
            return columns[name][self._instance]
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
        self._varying = None

    def __len__(self):
        return self.count

    def __iter__(self):
        return (_ModelView(self, i) for i in range(self.count))

    def __getitem__(self, instance):
        if not 0 <= instance < self.count:
            raise IndexError(instance)
        return _ModelView(self, instance)

    def select(self, instances):
        selected = copy.copy(self)
        selected.count = len(instances)
        selected._varying = None
        selected.columns = {name: values[instances].copy() for name, values in self.columns.items()}
        selected.options = {name: values[instances].copy() for name, values in self.options.items()}
        selected.statistics = {name: values[instances].copy() for name, values in self.statistics.items()}
        selected.fixed_positions = self.fixed_positions[instances].copy()
        selected.fixed_rotations = self.fixed_rotations[instances].copy()
        return selected

    def clone(self):
        return self.select(np.arange(self.count))

    def _arrays(self, model):
        """The model's column arrays, which keep their memory for the life of the model."""
        entry = self._bound.get(id(model))
        if entry is None:
            entry = self._bound[id(model)] = (model, {name: getattr(model, name) for name in self.columns})
        return entry[1]

    def varying(self):
        """The names of the columns, options and statistics whose rows are not all bit-equal."""
        if self._varying is None:

            def differ(table):
                names = []
                for name, values in table.items():
                    bits = np.ascontiguousarray(values).view(np.uint8).reshape(len(values), -1)
                    if not (bits == bits[:1]).all():
                        names.append(name)
                return tuple(names)

            self._varying = (differ(self.columns), differ(self.options), differ(self.statistics))
        return self._varying

    def _tables(self, names):
        if names is None:
            return self.columns.items(), self.options.items(), self.statistics.items()
        return tuple(
            [(name, table[name]) for name in subset]
            for table, subset in zip((self.columns, self.options, self.statistics), names, strict=True)
        )

    def capture(self, instance, physics, names=None):
        """Read the model of ``physics`` into row ``instance``: all of it, or the ``(columns, options, statistics)`` in ``names``."""
        model, data = physics.model.ptr, physics.data.ptr
        arrays = self._arrays(model)
        columns, options, statistics = self._tables(names)
        for name, values in columns:
            values[instance] = arrays[name]
        for name, values in options:
            values[instance] = getattr(model.opt, name)
        for name, values in statistics:
            values[instance] = getattr(model.stat, name)
        self.fixed_positions[instance] = data.geom_xpos[self.fixed_indices]
        self.fixed_rotations[instance] = data.geom_xmat[self.fixed_indices].reshape(-1, 3, 3)

    def restore(self, instance, physics, names=None):
        """Select an environment without recomputing its previously derived constants; ``names`` as in ``capture``."""
        model = physics.model.ptr
        arrays = self._arrays(model)
        columns, options, statistics = self._tables(names)
        for name, values in columns:
            arrays[name][:] = values[instance]
        for name, values in options:
            if values.ndim == 1:
                setattr(model.opt, name, values[instance].item())
            else:
                getattr(model.opt, name)[:] = values[instance]
        for name, values in statistics:
            setattr(model.stat, name, values[instance].item())


@dataclass
class _Prepared:
    state: object
    models: WarpModelBatch
    physics_values: np.ndarray | None


class WarpResetPool:
    """Native DMC resets with independent per-environment saved state.

    Domains whose reset is mostly MuJoCo work use one owner per batch thread, each
    resetting a shard of the environments. Other domains use one owner. Transient native
    work never changes the installed environment records. Model preparation can fail
    without committing another environment's partial updates.
    """

    def __init__(self, key, num_envs, seed, model_fields, option_fields):
        self.key, self.num_envs = key, int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self._closed = False
        self._owners, self._pool, self._batch = [], None, None
        try:
            count = batch_threads(self.num_envs) if key in THREADED_RESETS else 1
            self._owners = [make_local(key, 1, "cpu", seed) for _ in range(count)]
            self._pool = ThreadPoolExecutor(count) if count > 1 else None
            self._owner = self._owners[0]
            self.envs = self._owner.envs
            self.fields, self.action_dim = self._owner.fields, self._owner.action_dim
            self.schema = self._owner.schema
            reference = copy.copy(self.envs[0].physics.model.ptr)
            self.models = WarpModelBatch(reference, self.num_envs, model_fields, option_fields)
            episode = (MODEL_FIELDS[key][0] for key in self.fields if key in MODEL_FIELDS)
            self._episode_models = tuple(name for name in dict.fromkeys(episode) if name in self.models.columns)
            # This holder uses the native physics layout/column validation, but
            # only the owners' PhysicsInstances ever apply native changes.
            self.physics = None if self._owner.physics is None else copy.copy(self._owner.physics)
            if self.physics is not None:
                self.physics.models = self.models
                self.physics._data = None
                self.physics.values = np.tile(self.physics.spec.nominal, (self.num_envs, 1))
            self._state = self._columns = None
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

    def _select_owner(self, owner, columns, instance, values, restore=None):
        """Install environment ``instance`` in ``owner``: its model, its physics ``values`` and the RNG of State ``columns``.

        ``restore`` limits the model rows written to the ``(columns, options, statistics)`` it names.
        """
        env = owner.envs[0]
        if self._state is not None:
            self.models.restore(instance, env.physics, restore)
            if self.physics is not None:
                # Compare proposed values with this environment, not the previously
                # selected owner. Reapplying unchanged variation can recompute
                # constants from a later episode's target or rounded geometry.
                owner.physics.values = self.physics.values[instance : instance + 1].copy()
        if values is not None:
            owner.configure_physics(values[instance : instance + 1])
        if columns is not None:
            for key in self.fields:
                if key in MODEL_FIELDS:
                    name, index = MODEL_FIELDS[key]
                    getattr(env.physics.named.model, name)[index] = columns[key][instance]
            env.task.random.set_state(
                (
                    "MT19937",
                    columns["rng", "keys"][instance].astype(np.uint32),
                    columns["rng", "position"][instance].item(),
                    int(columns["rng", "has_gauss"][instance].item()),
                    columns["rng", "cached_gaussian"][instance].item(),
                )
            )
        return env

    def _batch_task(self):
        """The CPU batch task of this domain on a mujoco.batch of the nominal model, if its reset runs batched."""
        if self._batch is None:
            task = BATCH_TASKS[self.key]
            if not hasattr(task, "begin_reset"):
                self._batch = False
            else:
                from mujoco.batch import Batch

                reference = self.models.reference
                b = Batch(reference, self.num_envs, nthread=batch_threads(self.num_envs), persistent=True)
                self._batch = task(b, [reference] * self.num_envs)
        return self._batch or None

    def _reset_batched(self, task, ids, seed, models, columns, cumulative, model_fields, restore, capture):
        """Reset environments ``ids`` with the batch task's reset hooks, which run the MuJoCo work of all environments together.

        Each environment runs ``begin_reset`` on the owner, then the batch runs ``batch_reset`` for all environments, then each environment
        runs ``end_reset`` on the owner again with the RNG state its ``begin_reset`` left.
        """
        # mujoco.batch takes environments in increasing order.
        order = np.argsort(ids, kind="stable")
        instances, owner, begun, rng = ids[order], self._owners[0], [], []
        for instance in instances:
            base = self._select_owner(owner, self._columns, instance, None, restore)._env
            if seed is not None:
                base.task._random = np.random.RandomState(int(seed) + int(instance))
            base._reset_next_step, base._step_count = False, 0
            begun.append(task.begin_reset(base, instance))
            rng.append(base.task.random.get_state())
        for k, (instance, value) in enumerate(zip(instances, task.batch_reset(instances, begun), strict=True)):
            env = self._select_owner(owner, self._columns, instance, None, restore)
            env._env.task.random.set_state(rng[k])
            task.end_reset(env._env, instance, value)
            models.capture(instance, env.physics, capture)
            write_row(columns, order[k], env, cumulative[order[k]], model_fields)

    def prepare_reset(self, seed=None, instances=None, *, initial=False):
        self._check_open()
        ids = np.arange(self.num_envs) if instances is None else np.asarray(instances, dtype=np.int64)
        values = None if self.physics is None else self.physics.values.copy()
        models = self.models if initial else self.models.clone()
        cumulative = (
            np.zeros(len(ids)) if instances is None else self._columns["task", "cumulative_reward"][ids, 0].copy()
        )

        columns = Columns(self.schema, len(ids), self.fields)
        model_fields = model_columns(columns)
        # A reset writes only the State's episode-drawn model entries, and the installed physics values are those
        # of each environment. Model values that all environments share stay installed in an owner between environments.
        restore = capture = None
        if not initial and len(ids):
            restore = self.models.varying()
            capture = (tuple(dict.fromkeys((*restore[0], *self._episode_models))), *restore[1:])
            for owner in self._owners:
                self.models.restore(ids[0], owner.envs[0].physics)

        def reset_shard(owner, positions):
            for position in positions:
                instance = ids[position]
                env = self._select_owner(owner, self._columns, instance, None, restore)
                # make_local performed the first environment's registered initial
                # reset already. Reuse it without consuming a second RNG draw.
                if not (initial and instance == 0):
                    base = env._env
                    if seed is not None:
                        base.task._random = np.random.RandomState(int(seed) + int(instance))
                    base._reset_next_step, base._step_count = False, 0
                    with env.physics.reset_context():
                        base.task.initialize_episode(env.physics)
                models.capture(instance, env.physics, capture)
                write_row(columns, position, env, cumulative[position], model_fields)

        nominal = self.physics is None or np.array_equal(
            values[ids], np.broadcast_to(self.physics.spec.nominal, values[ids].shape)
        )
        task = None if initial or not nominal else self._batch_task()
        shards = [shard for shard in np.array_split(np.arange(len(ids)), len(self._owners)) if len(shard)]
        if task is not None:
            self._reset_batched(task, ids, seed, models, columns, cumulative, model_fields, restore, capture)
        elif self._pool is None or len(shards) < 2:
            reset_shard(self._owners[0], range(len(ids)))
        else:
            list(self._pool.map(lambda job: reset_shard(*job), zip(self._owners, shards, strict=False)))
        extra = None if self.physics is None else {key: value[ids] for key, value in self.physics.export().items()}
        reset = columns_to_state(columns, "cpu", extra)
        state = reset if instances is None else self._state.clone()
        if instances is not None:
            state[ids] = reset
        return _Prepared(state, models, values)

    def validate(self, state):
        self._check_open()
        validate_state(self.schema, state, self.num_envs, self.key)
        return None if self.physics is None else self.physics.rows_from_state(unfold(state))

    def prepare_state(self, state):
        values = self.validate(state)
        host = state.cpu().clone()
        columns = field_arrays(self.schema, host, self.fields)
        changed = np.zeros(self.num_envs, dtype=bool)
        if values is not None:
            changed |= np.any(values != self.physics.values, axis=1)
        for key in self.fields:
            if key in MODEL_FIELDS:
                changed |= np.any(columns[key] != self._columns[key], axis=1)
        models = self.models.clone() if changed.any() else self.models
        for instance in np.flatnonzero(changed):
            env = self._select_owner(self._owner, columns, instance, values)
            env.physics.forward()
            models.capture(instance, env.physics)
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
            self.schema.set(state, field_name(key), value)
        return self.prepare_state(state)

    def commit(self, prepared):
        self._state, self.models = prepared.state, prepared.models
        self._columns = field_arrays(self.schema, self._state, self.fields)
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
