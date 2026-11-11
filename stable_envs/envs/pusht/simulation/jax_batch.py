"""Device-resident PushT simulation with source reset and rendering metadata."""

import os

import numpy as np
import torch
from tensordict import TensorDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.dlpack

from ...._core.tensordicts import build_tensordict
from .jax_physics import advance
from .numeric import MAX_SHAPES, geometry_columns, install_latest
from .replay import HISTORY_KEYS, validate_seed
from .replay_batch import PushTReplayAdapter, lane_options

enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64


def from_torch(value):
    value = value.detach().contiguous()
    if value.data_ptr() % (16 if value.is_cuda else 64):
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


class PushTJaxAdapter(PushTReplayAdapter):
    """Step PushT lanes with batched JAX physics; CPU owners supply resets, restores and Variation.

    With ``processes``, owners in worker processes reset the lanes (see ``PushTResetPool``). After a
    worker reset, the workers own the lanes' CPU engines and the batch columns hold their static fields;
    ``envs`` rebuilds the parent owners from the columns when it is read.
    """

    def __init__(self, batch, device="cpu", seed=0, *, discrete=False, config=None, processes=0, cpus=None):
        device = torch.device(device)
        if device.type == "cuda":
            try:
                devices = jax.devices("gpu")
                devices[device.index if device.index is not None else torch.cuda.current_device()]
            except (RuntimeError, IndexError) as exc:
                raise RuntimeError(
                    f"PushT JAX cannot use {device}; install CUDA support with "
                    "pip install 'stable-envs[pusht-gpu]' and ensure JAX can see the selected GPU"
                ) from exc
        self._pool, self._pool_owns = None, False
        super().__init__(batch, device, seed, discrete=discrete, config=config)
        try:
            self._initialize_device()
            if processes:
                from .reset_pool import PushTResetPool

                groups = {
                    "static": self._static_columns(),
                    "current": self._current_columns(),
                    "geometry": geometry_columns(self.envs, MAX_SHAPES),
                }
                self._pool = PushTResetPool(
                    discrete, self.template.config, batch, seed, groups, processes=processes, cpus=cpus
                )
        except BaseException:
            self.close()
            raise

    @property
    def envs(self):
        if self._stale:
            self._materialize(sorted(self._stale))
        return self._envs

    @envs.setter
    def envs(self, owners):
        self._envs, self._stale = owners, set()

    @property
    def template(self):
        if 0 in self._stale:
            self._materialize([0])
        return self._envs[0]

    def _materialize(self, lanes):
        """Rebuild the parent owners of ``lanes`` from the batch columns of a worker reset."""
        for lane in lanes:
            row = {key: np.asarray(value[lane]) for key, value in self._columns.items()}
            owner = self._envs[lane].reconstruct_anchor(row)
            owner._install_metadata({k.removeprefix("owned/"): v for k, v in row.items() if k.startswith("owned/")})
            self._envs[lane].close()
            self._envs[lane] = owner
            self._stale.discard(lane)

    def _adopt(self, columns):
        """Install the static columns that the workers wrote; the workers now own the lanes."""
        self._columns = columns
        self._stale = set(range(self.num_envs))
        self._pool_owns = True

    def _static_columns(self):
        return self._columns if self._pool_owns else super()._static_columns()

    def _initialize_device(self, fields=None, geometry=None):
        if fields is None:
            fields = super().render_fields()
        self._fixed = {key: torch.as_tensor(value, device=self.device).clone() for key, value in fields.items()}
        self._fixed["recipe/version"].fill_(3)
        self._latest_tags = torch.tensor([[4, 0], [4, 1]], dtype=torch.uint8, device=self.device)
        self._symmetries = torch.tensor(
            [2 * np.pi, 2 * np.pi, 2 * np.pi, np.pi, np.pi / 2, np.pi, 2 * np.pi, np.pi / 2],
            dtype=torch.float64,
            device=self.device,
        )
        if geometry is None:
            geometry = geometry_columns(self.envs)
        shapes = int(geometry["shapes"].max())
        self._circle = bool(geometry["circle"].all())
        self._body = torch.tensor(geometry["body"], device=self.device)
        self._geometry = tuple(
            torch.tensor(
                geometry[key][:, :, :shapes] if key in ("vertices", "enabled") else geometry[key], device=self.device
            )
            for key in ("vertices", "centers", "inverse", "radius", "enabled")
        )
        count = shapes + shapes * 8 if self._circle else 2 * shapes**2 + 16 * shapes
        self._bias = torch.zeros(self.num_envs, 2, 3, dtype=torch.float64, device=self.device)
        self._impulse = torch.zeros(self.num_envs, count, dtype=torch.float64, device=self.device)
        self._identifiers = torch.full((self.num_envs, count), -1, dtype=torch.int64, device=self.device)
        self._lengths = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._actions = torch.zeros(self.num_envs, 0, 2, dtype=torch.float64, device=self.device)
        self._codes = torch.zeros(self.num_envs, 0, dtype=torch.uint8, device=self.device)
        self._contacts = self._fixed["current/contacts"]
        self._refresh_current()

    def _refresh_current(self):
        for index, name in enumerate(("agent", "block")):
            for field, value in (
                ("position", self._body[:, index, :2]),
                ("velocity", self._body[:, index, 2:4]),
                ("angle", self._body[:, index, 4]),
                ("angular_velocity", self._body[:, index, 5]),
            ):
                self._fixed[f"current/body/{name}/{field}"] = value
        self._fixed["current/observation"] = torch.cat(
            (self._body[:, 0, :2], self._body[:, 1, :2], self._body[:, 1, 4:5] % (2 * np.pi), self._body[:, 0, 2:4]), 1
        )
        self._fixed["current/contacts"] = self._contacts

    def _solver_state(self):
        return {"bias": self._bias, "impulse": self._impulse, "identifiers": self._identifiers}

    def get_state(self):
        width = int(self._lengths.max())
        fields = dict(self._fixed)
        fields.update({"current/jax/" + key: value for key, value in self._solver_state().items()})
        fields.update(
            {
                "history/actions": self._actions[:, :width],
                "history/dtypes": self._codes[:, :width],
                "history/lengths": self._lengths,
                "history/valid": torch.arange(width, device=self.device) < self._lengths[:, None],
            }
        )
        groups, owned = {}, {}
        for key, value in fields.items():
            groups.setdefault(value.dtype, []).append(key)
        for keys in groups.values():
            flat = torch.cat([fields[key].reshape(-1) for key in keys])
            offset = 0
            for key in keys:
                value = fields[key]
                owned[key] = flat[offset : offset + value.numel()].view(value.shape)
                offset += value.numel()
        tree = {}
        for key, value in owned.items():
            node = tree
            *parents, leaf = key.split("/")
            for parent in parents:
                node = node.setdefault(parent, {})
            node[leaf] = value
        return build_tensordict(tree, [self.num_envs], self.device)

    def _swap(self, staged):
        super()._swap(staged)
        self._initialize_device()

    def reset(self, seed=None, options=None):
        latest = tuple(
            self._fixed["current/" + key].cpu().numpy() for key in ("latest/value", "latest/type", "latest_present")
        )
        if self._pool is None:
            install_latest(self.envs, *latest)
            return super().reset(seed, options)
        validate_seed(seed)
        space = self.template.env.variation_space
        lanes = (
            options
            if isinstance(options, list)
            else [lane_options(space, options, self.num_envs, lane) for lane in range(self.num_envs)]
        )
        # Workers that do not own the lanes first take the owners' metadata from the columns.
        sync = None if self._pool_owns else self._static_columns()
        static, current, geometry = self._pool.reset(seed, lanes, latest, sync)
        self._adopt(static)
        self._initialize_device({**static, **current}, geometry)
        return self.get_state()

    def install_variation(self, lanes, immediate=frozenset()):
        if not self._pool_owns:
            return super().install_variation(lanes, immediate)
        self._adopt(self._pool.install_variation(lanes, immediate))
        self._install_columns(self._columns)

    def variation_rows(self, attribute, keys):
        if self._pool_owns:
            return self._pool.variation_rows(attribute, keys)
        return super().variation_rows(attribute, keys)

    def variation_changed(self):
        super().variation_changed()
        self._install_columns(self._static_columns())

    def _install_columns(self, columns):
        for key, value in columns.items():
            if key != "recipe/version":
                self._fixed[key] = torch.as_tensor(value, device=self.device).clone()

    def render_fields(self):
        return dict(self._fixed)

    def step(self, actions, *, active=None, capture_state=True):
        actions = torch.as_tensor(actions, device=self.device)
        discrete = self.template.discrete
        allowed = (torch.int32, torch.int64) if discrete else (torch.float32, torch.float64)
        if actions.shape != (self.num_envs, 2) or actions.dtype not in allowed:
            raise ValueError("PushT actions require shape [num_envs, 2] and a supported dtype")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("PushT action must be finite")
        if discrete and not bool(((actions >= 0) & (actions < 16)).all()):
            raise ValueError("PushT discrete indices must be in [0, 16)")
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        else:
            active = torch.as_tensor(active, device=self.device)
            if active.shape != (self.num_envs,) or active.dtype != torch.bool:
                raise ValueError("PushT active mask requires bool [num_envs]")
        if discrete:
            polar = (actions.to(torch.float64) + 0.5) / 16
            theta = polar[:, 1] * (2 * np.pi)
            relative = torch.stack((polar[:, 0] * theta.cos(), polar[:, 0] * theta.sin()), 1).to(torch.float32)
        else:
            relative = actions
        target = (self._body[:, 0, :2].to(relative.dtype) + relative * 100).to(torch.float64)
        with enable_x64():
            outputs = advance(
                from_torch(self._body),
                from_torch(self._bias),
                from_torch(self._impulse),
                from_torch(self._identifiers),
                tuple(from_torch(v) for v in self._geometry),
                from_torch(target),
                self._circle,
            )
            body, bias, impulse, identifiers, contacts, segments, segment_count = (
                torch.from_dlpack(v) for v in outputs
            )
        self._body = torch.where(active[:, None, None], body, self._body)
        self._bias = torch.where(active[:, None, None], bias, self._bias)
        self._impulse = torch.where(active[:, None], impulse, self._impulse)
        self._identifiers = torch.where(active[:, None], identifiers, self._identifiers)
        self._contacts = torch.where(active, contacts.to(torch.int64), self._contacts)
        width = int(self._lengths.max()) + 1
        if width > self._actions.shape[1]:
            capacity = max(16, 2 * self._actions.shape[1])
            self._actions = torch.cat(
                (self._actions, self._actions.new_zeros(self.num_envs, capacity - self._actions.shape[1], 2)), 1
            )
            self._codes = torch.cat(
                (self._codes, self._codes.new_zeros(self.num_envs, capacity - self._codes.shape[1])), 1
            )
        lane = torch.arange(self.num_envs, device=self.device)
        index = self._lengths
        self._actions[lane, index] = torch.where(active[:, None], actions.to(torch.float64), 0.0)
        code = (
            (torch.int32, torch.int64).index(actions.dtype) + 2
            if discrete
            else (torch.float32, torch.float64).index(actions.dtype)
        )
        self._codes[lane, index] = torch.where(active, code, 0).to(torch.uint8)
        self._lengths = self._lengths + active
        self._fixed["current/latest/value"] = torch.where(
            active[:, None], relative.to(torch.float64), self._fixed["current/latest/value"]
        )
        tags = self._fixed["current/latest/type"]
        new_tags = self._latest_tags[0 if relative.dtype == torch.float32 else 1]
        self._fixed["current/latest/type"] = torch.where(active[:, None], new_tags, tags)
        self._fixed["current/latest_present"] = self._fixed["current/latest_present"] | active
        self._fixed["current/render_segments"] = torch.where(
            active[:, None, None], segments, self._fixed["current/render_segments"]
        )
        self._fixed["current/render_segment_count"] = torch.where(
            active, segment_count.to(torch.int64), self._fixed["current/render_segment_count"]
        )
        self._refresh_current()
        observation = self._fixed["current/observation"]
        goal = self._fixed["current/goal/value"]
        distance = (goal - observation).norm(dim=1)
        shape = self._fixed["owned/variation/block/shape/current/value"].to(torch.int64)
        symmetry = self._symmetries[shape]
        angle = (goal[:, 4] - observation[:, 4]).abs() % symmetry
        success = ((goal[:, :4] - observation[:, :4]).norm(dim=1) < 20) & (
            torch.minimum(angle, symmetry - angle) < np.pi / 9
        )
        return TensorDict(
            dict(
                **({"state": self.get_state()} if capture_state else {}),
                reward=-distance,
                success=success,
                terminated=success.clone(),
                truncated=torch.zeros_like(success),
            ),
            [self.num_envs],
            device=self.device,
        )

    def _reconstruct_state(self, state):
        from .replay_batch import state_columns

        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise ValueError("PushT full State requires batch [num_envs]")
        base = self.template.get_state()
        keys = [tuple(key.split("/")) for key in base]
        extras = {("current", "jax", key) for key in self._solver_state()}
        actual_keys = set(state.keys(True, True))
        if not set(keys) <= actual_keys:
            raise ValueError("PushT full State keys differ")
        version = state["recipe", "version"]
        new = bool((version == 3).all())
        if actual_keys != set(keys) | (extras if new else set()):
            raise ValueError("PushT full State keys differ")
        fields = state_columns(state.select(*keys), base, self.num_envs)
        rows = [{k: np.asarray(v[i]).copy() for k, v in fields.items()} for i in range(self.num_envs)]
        # Checks and replays depend only on the configuration, which every lane shares.
        template = self.template
        for row in rows:
            if row["recipe/version"] not in (2, 3):
                raise ValueError("PushT State version mismatch")
            checked = dict(row, **{"recipe/version": np.array(2, np.int64)})
            template.validate(checked)
        staged = object.__new__(type(self))
        staged.__dict__ = self.__dict__.copy()
        staged.envs, staged._sources, staged._columns, staged._prepared = [], [None] * self.num_envs, {}, None
        staged._pool, staged._pool_owns = None, False
        try:
            for row in rows:
                staged.envs.append(
                    template.reconstruct(row) if row["recipe/version"] == 2 else template.reconstruct_anchor(row)
                )
            staged._initialize_device()
            if new:
                staged._body = (
                    torch.stack(
                        [
                            torch.cat(
                                [
                                    state["current", "body", name, "position"],
                                    state["current", "body", name, "velocity"],
                                    state["current", "body", name, "angle"][:, None],
                                    state["current", "body", name, "angular_velocity"][:, None],
                                ],
                                1,
                            )
                            for name in ("agent", "block")
                        ],
                        1,
                    )
                    .to(self.device)
                    .clone()
                )
                for key, sample in staged._solver_state().items():
                    value = state["current", "jax", key]
                    if value.shape != sample.shape or value.dtype != sample.dtype or not bool(value.isfinite().all()):
                        raise ValueError(f"Invalid PushT JAX solver cache: {key}")
                    setattr(staged, "_" + key, value.to(self.device).clone())
                if bool((staged._impulse < 0).any() | (staged._identifiers < -1).any()):
                    raise ValueError("Invalid PushT JAX contact cache")
                staged._contacts = state["current", "contacts"].to(self.device).clone()
                for key in (
                    "latest/value",
                    "latest/type",
                    "latest_present",
                    "render_segments",
                    "render_segment_count",
                ):
                    staged._fixed["current/" + key] = (
                        state[tuple(("current/" + key).split("/"))].to(self.device).clone()
                    )
                staged._refresh_current()
                staged._actions = torch.tensor(fields["history/actions"], device=self.device)
                staged._codes = torch.tensor(fields["history/dtypes"], device=self.device)
                staged._lengths = torch.tensor(fields["history/lengths"], device=self.device)
                for owner, row in zip(staged.envs, rows, strict=True):
                    owner._install_metadata(
                        {k.removeprefix("owned/"): v for k, v in row.items() if k.startswith("owned/")}
                    )
                staged.variation_changed()
                actual = staged.get_state()
                for key, value in actual.items(True, True):
                    expected = state[key].to(self.device)
                    if "/".join(key) in HISTORY_KEYS:
                        expected = expected[:, : value.shape[1]]
                    if not torch.equal(value, expected):
                        raise ValueError(f"PushT replay disagrees with supplied State: {'/'.join(key)}")
            elif any(row["recipe/version"] != 2 for row in rows):
                raise ValueError("PushT State cannot mix physics versions")
        except BaseException:
            staged.close()
            raise
        return staged

    def validate(self, state):
        staged = self._reconstruct_state(state)
        staged.close()

    def prepare_restore(self, state):
        if self._prepared is not None:
            raise RuntimeError("A PushT restore is already prepared")
        self._prepared = self._reconstruct_state(state)

    def commit_restore(self):
        if self._prepared is None:
            raise RuntimeError("No PushT restore is prepared")
        staged, self._prepared = self._prepared, None
        for owner in self._envs:
            owner.close()
        pool = self._pool
        self.__dict__.update(staged.__dict__)
        self._pool = pool

    def discard_restore(self):
        staged, self._prepared = self._prepared, None
        if staged is not None:
            staged.close()

    def close(self):
        self.discard_restore()
        try:
            if self._pool is not None:
                self._pool.close()
        finally:
            self._pool = None
            for owner in self._envs:
                owner.close()
            self.envs = []
