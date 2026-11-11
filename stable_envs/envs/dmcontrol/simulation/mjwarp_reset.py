"""One native reset owner with compact per-lane model and RNG checkpoints."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from ...._execution.state_columns import require_columns
from .factory import make_local
from .state import MODEL_FIELDS, export_state


class _ModelView:
    """Read-only model attributes; large static assets belong to one prototype."""

    def __init__(self, batch, lane, option=False):
        self._batch, self._lane, self._option = batch, lane, option

    def __getattr__(self, name):
        batch = self._batch
        if name == "opt" and not self._option:
            return _ModelView(batch, self._lane, option=True)
        columns = batch.options if self._option else batch.columns
        if name in columns:
            return columns[name][self._lane]
        owner = batch.reference.opt if self._option else batch.reference
        return getattr(owner, name)


class WarpModelBatch:
    """Native model columns needed by Warp, without duplicating native assets."""

    def __init__(self, reference, count, model_fields, option_fields):
        self.reference = reference
        self.count = count
        names = dict.fromkeys((*model_fields, "site_size", "geom_gap", "pair_gap"))
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
        selected.fixed_positions = self.fixed_positions[lanes].copy()
        selected.fixed_rotations = self.fixed_rotations[lanes].copy()
        return selected

    def clone(self):
        return self.select(np.arange(self.count))

    def capture(self, lane, physics):
        model, data = physics.model.ptr, physics.data.ptr
        for name, values in self.columns.items():
            values[lane] = getattr(model, name)
        for name, values in self.options.items():
            values[lane] = getattr(model.opt, name)
        self.fixed_positions[lane] = data.geom_xpos[self.fixed_indices]
        self.fixed_rotations[lane] = data.geom_xmat[self.fixed_indices].reshape(-1, 3, 3)


@dataclass
class _Prepared:
    state: object
    models: WarpModelBatch
    physics_values: np.ndarray | None


class _ResetOwners:
    """Export each reused owner immediately, before moving to the next lane."""

    def __init__(self, count, generate):
        self.count, self.generate = count, generate

    def __len__(self):
        return self.count

    def __iter__(self):
        return self.generate()


class WarpResetPool:
    """Native DMC resets with one owner and independent per-lane saved state.

    Transient native work never changes the installed lane records. Model
    preparation can fail without committing another lane's partial updates.
    """

    def __init__(self, key, num_envs, seed, model_fields, option_fields):
        self.key, self.num_envs = key, int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self._owner = make_local(key, 1, "cpu", seed)
        self.envs = self._owner.envs
        self.fields, self.action_dim = self._owner.fields, self._owner.action_dim
        self._closed = False
        try:
            reference = copy.copy(self.envs[0].physics.model.ptr)
            self.models = WarpModelBatch(reference, self.num_envs, model_fields, option_fields)
            # This holder uses the native physics layout/column validation, but
            # only the single owner's PhysicsLanes ever applies native changes.
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

    def _select_owner(self, state, lane, values):
        owner, env = self._owner, self.envs[0]
        if values is not None:
            try:
                owner.configure_physics(values[lane : lane + 1])
            except Exception:
                # Native apply may fail after writing an earlier model field.
                # Force the next selection to reinstall every physics knob.
                owner.physics.values.fill(np.nan)
                raise
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

        def generate():
            for lane in ids:
                env = self._select_owner(self._state, lane, values)
                # make_local performed the first lane's registered initial
                # reset already. Reuse it without consuming a second RNG draw.
                if not (initial and lane == 0):
                    if seed is not None:
                        env.task._random = np.random.RandomState(int(seed) + int(lane))
                    env.reset()
                models.capture(lane, env.physics)
                yield env

        extra = None if self.physics is None else {key: value[ids] for key, value in self.physics.export().items()}
        reset = export_state(_ResetOwners(len(ids), generate), cumulative, self.fields, "cpu", extra)
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
            env = self._select_owner(host, lane, values)
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
            self._owner.close()
            self._closed = True
