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
from .replay import HISTORY_KEYS
from .replay_batch import PushTReplayAdapter

enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64


def from_torch(value):
    value = value.detach().contiguous()
    if value.data_ptr() % (16 if value.is_cuda else 64):
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


class PushTJaxAdapter(PushTReplayAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self._initialize_device()
        except BaseException:
            self.close()
            raise

    def _initialize_device(self):
        fields = super().render_fields()
        self._fixed = {key: torch.as_tensor(value, device=self.device).clone() for key, value in fields.items()}
        self._fixed["recipe/version"].fill_(3)
        bodies = []
        geometry = []
        shapes = max(len(getattr(owner.env, name).shapes) for owner in self.envs for name in ("agent", "block"))
        self._circle = all(int(owner.env.variation_space["agent"]["shape"].value) == 0 for owner in self.envs)
        for owner in self.envs:
            row, vertices, centers, inverse, enabled = [], [], [], [], []
            radius = 0.0
            for name in ("agent", "block"):
                body = getattr(owner.env, name)
                row.append([*body.position, *body.velocity, body.angle, body.angular_velocity])
                polys = np.zeros((shapes, 4, 2), np.float64)
                valid = np.zeros(shapes, np.bool_)
                for index, shape in enumerate(
                    sorted(
                        body.shapes,
                        key=lambda shape: (
                            tuple(np.array(shape.get_vertices()).ravel()) if hasattr(shape, "get_vertices") else ()
                        ),
                    )
                ):
                    if hasattr(shape, "get_vertices"):
                        polys[index] = shape.get_vertices()
                    else:
                        radius = shape.radius
                    valid[index] = True
                # Nonzero padded edges avoid NaNs in inactive polygon contacts.
                polys[~valid] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                if name == "agent" and not np.isfinite(body.mass):
                    polys[:] = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                vertices.append(polys)
                centers.append(body.center_of_gravity)
                inverse.append([1 / body.mass, 1 / body.moment])
                enabled.append(valid)
            bodies.append(row)
            geometry.append(
                (np.array(vertices), np.array(centers), np.array(inverse), np.array(radius), np.array(enabled))
            )
        self._body = torch.tensor(np.array(bodies), device=self.device)
        self._geometry = tuple(torch.tensor(np.stack([v[i] for v in geometry]), device=self.device) for i in range(5))
        count = shapes + shapes * 8 if self._circle else 2 * shapes**2 + 16 * shapes
        self._bias = torch.zeros(self.num_envs, 2, 3, dtype=torch.float64, device=self.device)
        self._impulse = torch.zeros(self.num_envs, count, dtype=torch.float64, device=self.device)
        self._identifiers = torch.full((self.num_envs, count), -1, dtype=torch.int64, device=self.device)
        self._lengths = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._actions = torch.zeros(self.num_envs, 0, 2, dtype=torch.float64, device=self.device)
        self._codes = torch.zeros(self.num_envs, 0, dtype=torch.uint8, device=self.device)
        self._contacts = self._fixed["current/contacts"]
        self._refresh_current()
        self._anchor = {
            key: value.clone()
            for key, value in {
                "body": self._body,
                "latest/value": self._fixed["current/latest/value"],
                "latest/type": self._fixed["current/latest/type"],
                "latest_present": self._fixed["current/latest_present"],
                "contacts": self._contacts,
                "render_segments": self._fixed["current/render_segments"],
                "render_segment_count": self._fixed["current/render_segment_count"],
            }.items()
        }

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

    def get_state(self):
        width = int(self._lengths.max())
        fields = dict(self._fixed)
        fields.update({"anchor/jax/" + key: value for key, value in self._anchor.items()})
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
        latest = self._fixed["current/latest/value"].cpu().numpy()
        present = self._fixed["current/latest_present"].cpu().numpy()
        tags = self._fixed["current/latest/type"].cpu().numpy()
        for lane, owner in enumerate(self.envs):
            owner.env.latest_action = (
                latest[lane].astype(np.float32 if tags[lane, 1] == 0 else np.float64) if present[lane] else None
            )
        return super().reset(seed, options)

    def variation_changed(self):
        super().variation_changed()
        for key, value in self._static_columns().items():
            if key != "recipe/version":
                self._fixed[key] = torch.as_tensor(value, device=self.device).clone()

    def render_fields(self):
        return dict(self._fixed)

    def step(self, actions, *, active=None, capture_state=True, codes=None):
        actions = torch.as_tensor(actions, device=self.device)
        discrete = self.template.discrete
        allowed = (torch.int32, torch.int64) if discrete else (torch.float32, torch.float64)
        if actions.shape != (self.num_envs, 2) or actions.dtype not in allowed:
            raise ValueError("PushT actions require shape [num_envs, 2] and a supported dtype")
        if not bool(
            torch.isfinite(actions).all()
            & (actions >= (-0 if discrete else -1)).all()
            & (actions < 16 if discrete else actions <= 1).all()
        ):
            raise ValueError("PushT actions must be finite and within the action space")
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
        if codes is not None and not discrete:
            single = self._body[:, 0, :2].to(torch.float32) + relative.to(torch.float32) * 100
            target = torch.where((codes == 0)[:, None], single.to(torch.float64), target)
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
        self._codes[lane, index] = torch.where(active, code if codes is None else codes, 0).to(torch.uint8)
        self._lengths = self._lengths + active
        self._fixed["current/latest/value"] = torch.where(
            active[:, None], relative.to(torch.float64), self._fixed["current/latest/value"]
        )
        tags = self._fixed["current/latest/type"]
        new_tags = torch.tensor(
            [4, 0 if relative.dtype == torch.float32 else 1], dtype=torch.uint8, device=self.device
        )
        if codes is not None and not discrete:
            new_tags = torch.stack((torch.full_like(codes, 4), codes), 1)
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
        symmetry = torch.tensor(
            [2 * np.pi, 2 * np.pi, 2 * np.pi, np.pi, np.pi / 2, np.pi, 2 * np.pi, np.pi / 2],
            device=self.device,
            dtype=torch.float64,
        )[shape]
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
        extras = {tuple(("anchor/jax/" + key).split("/")) for key in self._anchor}
        actual_keys = set(state.keys(True, True))
        if not set(keys) <= actual_keys:
            raise ValueError("PushT full State keys differ")
        version = state["recipe", "version"]
        new = bool((version == 3).all())
        if actual_keys != set(keys) | (extras if new else set()):
            raise ValueError("PushT full State keys differ")
        for key, sample in self._anchor.items():
            if new:
                value = state[tuple(("anchor/jax/" + key).split("/"))]
                if value.shape != sample.shape or value.dtype != sample.dtype or not bool(torch.isfinite(value).all()):
                    raise ValueError("Invalid PushT JAX anchor")
        fields = state_columns(state.select(*keys), base, self.num_envs)
        rows = [{k: np.asarray(v[i]).copy() for k, v in fields.items()} for i in range(self.num_envs)]
        for owner, row in zip(self.envs, rows, strict=True):
            if row["recipe/version"] not in (2, 3):
                raise ValueError("PushT State version mismatch")
            checked = dict(row, **{"recipe/version": np.array(2, np.int64)})
            owner.validate(checked)
        staged = object.__new__(type(self))
        staged.__dict__ = self.__dict__.copy()
        staged.envs, staged._sources, staged._columns, staged._prepared = [], [None] * self.num_envs, {}, None
        try:
            for owner, row in zip(self.envs, rows, strict=True):
                staged.envs.append(
                    owner.reconstruct(row) if row["recipe/version"] == 2 else owner.reconstruct_anchor(row)
                )
            staged._initialize_device()
            if new:
                staged._anchor = {
                    key: state[tuple(("anchor/jax/" + key).split("/"))].to(self.device).clone() for key in self._anchor
                }
                staged._body = staged._anchor["body"].clone()
                staged._contacts = staged._anchor["contacts"].clone()
                for key in (
                    "latest/value",
                    "latest/type",
                    "latest_present",
                    "render_segments",
                    "render_segment_count",
                ):
                    staged._fixed["current/" + key] = staged._anchor[key].clone()
                staged._refresh_current()
                length = int(fields["history/lengths"].max())
                for t in range(length):
                    codes = torch.tensor(fields["history/dtypes"][:, t], device=self.device)
                    actions = torch.tensor(fields["history/actions"][:, t], device=self.device)
                    if self.template.discrete:
                        actions = actions.to(torch.int64)
                    else:
                        # Float32 multiplication rounds before addition to the double position.
                        actions = torch.where(
                            (codes == 0)[:, None], actions.to(torch.float32).to(torch.float64), actions
                        )
                    staged.step(
                        actions,
                        active=torch.tensor(t < fields["history/lengths"], device=self.device),
                        capture_state=False,
                        codes=codes,
                    )
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
        for owner in self.envs:
            owner.close()
        self.__dict__.update(staged.__dict__)

    def discard_restore(self):
        staged, self._prepared = self._prepared, None
        if staged is not None:
            staged.close()

    def set_state(self, state):
        self.prepare_restore(state)
        self.commit_restore()
