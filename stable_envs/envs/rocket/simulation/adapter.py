from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.process import BROADCAST, PER_LANE, fresh_copy, leaf_tensordict
from ...._execution.protocol import SimulatorAdapter
from .actions import validate_actions
from .replay import RocketReplay, generator_state, install_generator, is_zero_view


def check_values(template, values, num_envs):
    """Check one ``{path: value}`` per lane against the variation space of engine ``template``."""
    if len(values) != num_envs:
        raise ValueError("Expected one Rocket Variation row per lane")
    for row in values:
        for key, value in row.items():
            node = template.env.variation_space._walk(key)
            if hasattr(node, "spaces") or not node.contains(value):
                raise ValueError(f"Invalid Rocket Variation: {key}")


def state_columns(state, fields, template, batch):
    """Check a ``[batch]`` state against ``fields`` and its rows with engine ``template``; return the host columns.

    The lanes of one batch share one configuration, so one engine checks every row.
    """
    if not isinstance(state, TensorDict) or state.batch_size != torch.Size([batch]):
        raise ValueError("Rocket full State requires batch [num_envs]")
    leaves = dict(state.items(True, True))
    if leaves.keys() != fields.keys():
        raise ValueError("Rocket full State keys differ")
    for key, (shape, dtype) in fields.items():
        if leaves[key].shape != (batch, *shape) or leaves[key].dtype != dtype:
            raise ValueError(f"Invalid Rocket State leaf: {key}")
    # One transfer per leaf; per-row device reads cost one copy per row and leaf.
    columns = {"/".join(k): v.detach().cpu().numpy() for k, v in leaves.items()}
    floats = [v.reshape(batch, -1) for v in columns.values() if v.dtype.kind not in "biu"]
    if floats and not np.isfinite(np.concatenate(floats, 1)).all():
        for key, value in columns.items():
            if value.dtype.kind not in "biu" and not np.isfinite(value).all():
                raise ValueError(f"Invalid Rocket State leaf: {tuple(key.split('/'))}")
    template.validate_batch(columns, checked=True)
    return columns


def split_rows(columns):
    """Host columns -> one ``{key: value}`` row per lane."""
    batch = len(columns["history/length"])
    return [{k: v[i, ...].copy() for k, v in columns.items()} for i in range(batch)]


class RocketAdapter(SimulatorAdapter):
    """Batch CPU Rocket owners with CPU/CUDA state storage.

    Validate every action before stepping any lane. Reset and restoration replace
    the engines one lane at a time, so the batch holds at most one extra engine.
    Reset checks the arguments of every lane before it replaces an engine.
    Restoration keeps the State row of each replaced lane until it commits, and a
    discarded restoration replays these rows.
    """

    process_methods = {"configure": PER_LANE, "rgba": BROADCAST, "variation_rows": BROADCAST}
    process_attributes = ("fields",)
    action_dim = 7

    def __init__(self, batch, device, seed, config=None, frame=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = batch, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Rocket supports CPU dynamics and CPU/CUDA State storage")
        self.envs, self._prepared, self._static = [], None, None
        # ``(size, supersample)`` of the TinyRenderer frames ``render_live`` draws; None draws none.
        self.frame, self._scene = frame, None
        try:
            for lane in range(batch):
                self.envs.append(RocketReplay(config, None if seed is None else seed + lane))
        except BaseException:
            self.close()
            raise
        self.fields = {
            tuple(k.split("/")): (v.shape, torch.from_numpy(v).dtype) for k, v in self.envs[0].get_state().items()
        }

    @property
    def template(self):
        """A lane whose configuration and variation space describe the batch."""
        return self.envs[0]

    @property
    def process_static_leaves(self):
        return [tuple(key.split("/")) for key in self.envs[0].static_keys()]

    def process_step_state(self):
        """Return ``{key tuple: host array}`` for the State leaves that a step can change."""
        rows = [env.step_leaves() for env in self.envs]
        return {tuple(key.split("/")): np.stack([row[key] for row in rows]) for key in rows[0]}

    def _static_leaves(self):
        """``{key tuple: tensor}`` on ``device`` of the other State leaves, stacked once per reset or restore."""
        if self._static is None:
            rows = [env.full_leaves() for env in self.envs]
            self._static, zero = {}, set()
            for key in self.process_static_leaves:
                column = [row["/".join(key)] for row in rows]
                if all(is_zero_view(value) for value in column):
                    # A large zero leaf of every lane, such as an unused image, is not stacked.
                    value = torch.zeros((), dtype=self.fields[key][1], device=self.device)
                    self._static[key] = value.expand(self.num_envs, *column[0].shape)
                    zero.add(key)
                else:
                    self._static[key] = torch.from_numpy(np.stack(column)).to(self.device)
                    if not self._static[key].any():
                        zero.add(key)
            # Zero leaves are not copied.
            self._zero = frozenset(zero)
        return self._static

    @property
    def zero_state_leaves(self):
        """Keys of the State leaves that are zero and that a step does not change."""
        self._static_leaves()
        return self._zero

    def get_state(self):
        step, static = self.process_step_state(), self._static_leaves()
        leaves = {
            key: torch.as_tensor(step[key], device=self.device)
            if key in step
            else fresh_copy(static[key], key in self._zero)
            for key in self.fields
        }
        return leaf_tensordict(leaves, self.num_envs, self.device)

    def rgba(self):
        return torch.as_tensor(np.stack([e.rgba() for e in self.envs]), device=self.device)

    def validate(self, state):
        return state_columns(state, self.fields, self.template, self.num_envs)

    def prepare_restore(self, state):
        """Replace each lane's engine with one replayed from ``state``; on failure restore the replaced lanes."""
        if self._prepared is not None:
            raise RuntimeError("A Rocket restore is already prepared")
        rows = split_rows(self.validate(state))
        previous, self._static = [], None
        try:
            for lane, env in enumerate(self.envs):
                row = env.get_state()
                staged = env.reconstruct(rows[lane])
                rows[lane] = None
                self.envs[lane] = staged
                previous.append(row)
                env.close()
        except BaseException:
            self._replay(previous)
            raise
        self._prepared = previous

    def _replay(self, rows):
        """Replace the engines of the first ``len(rows)`` lanes with engines replayed from ``rows``."""
        self._static = None
        for lane, row in enumerate(rows):
            env = self.envs[lane]
            self.envs[lane] = env.reconstruct(row)
            env.close()

    def commit_restore(self):
        if self._prepared is None:
            raise RuntimeError("No Rocket restore is prepared")
        self._prepared = None

    def discard_restore(self):
        previous, self._prepared = self._prepared, None
        if previous:
            self._replay(previous)

    def set_state(self, state):
        self.prepare_restore(state)
        self.commit_restore()

    def reset(self, seed=None, options=None, values=None):
        # Reset keeps only the persistent configuration and owned RNG of a lane. It needs no
        # physics replay because an authoritative reset discards the old episode.
        if values is not None and len(values) != self.num_envs:
            raise ValueError("Expected one Rocket Variation row per lane")
        lanes = [
            (None if seed is None else seed + lane, None if values is None else values[lane])
            for lane in range(self.num_envs)
        ]
        for env, (lane_seed, row) in zip(self.envs, lanes, strict=True):
            env.check_reset(lane_seed, options, row)
        self._static = None
        for lane, (env, (lane_seed, row)) in enumerate(zip(self.envs, lanes, strict=True)):
            target = RocketReplay(env.config, initialize=False)
            try:
                target.env.variation_space = copy.deepcopy(env.env.variation_space)
                install_generator(target.env, generator_state(env.env))
                install_generator(target.env.action_space, generator_state(env.env.action_space))
                target.reset(lane_seed, options, row)
            except BaseException:
                target.close()
                raise
            self.envs[lane] = target
            env.close()
        return self.get_state()

    def check_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 7):
            raise ValueError("Rocket actions require shape [num_envs, 7]")
        validate_actions(actions)
        if active is not None:
            active = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active, dtype=bool)
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 7):
            raise ValueError("Rocket actions require shape [num_envs, 7]")
        validate_actions(actions)
        results = [
            env.step(action) if active is None or active[i] else self._last_outputs[i]
            for i, (env, action) in enumerate(zip(self.envs, actions, strict=True))
        ]
        self._last_outputs = results
        return TensorDict(
            dict(
                **({"state": self.get_state()} if capture_state else {}),
                reward=torch.tensor([r[1] for r in results], dtype=torch.float64, device=self.device),
                success=torch.tensor([r[4]["env_complete"] for r in results], dtype=torch.bool, device=self.device),
                terminated=torch.tensor([r[2] for r in results], dtype=torch.bool, device=self.device),
                truncated=torch.tensor([r[3] for r in results], dtype=torch.bool, device=self.device),
            ),
            [self.num_envs],
            device=self.device,
        )

    @property
    def frame_shape(self):
        return None if self.frame is None else (self.frame[0], self.frame[0], 3)

    def draw(self, body, links, colors, size, ss):
        """Draw explicit body poses, camera link poses and colors with TinyRenderer."""
        from ..render import draw_tiny, tiny_scene

        if self._scene is None:
            self._scene = tiny_scene()
        return draw_tiny(self._scene, body, links, colors, size, ss)

    def render_live(self):
        """Draw every lane's running scene with TinyRenderer, as ``render`` draws its state."""
        body, links, colors = (np.stack(column) for column in zip(*(env.render_fields() for env in self.envs)))
        return self.draw(body, links, colors, *self.frame)

    def frame_fields(self):
        """Every lane's body poses, camera link pose, colours and float32 camera view matrix, for a GPU frame."""
        import pybullet

        from ..render import frame_views

        body, links, colors = (np.stack(column) for column in zip(*(env.render_fields() for env in self.envs)))
        return {"body": body, "link": links, "rgba": colors, "view": frame_views(links, pybullet)}

    def configure(self, values):
        check_values(self.envs[0], values, self.num_envs)
        self._static = None
        for env, row in zip(self.envs, values, strict=True):
            env.configure(row)

    def variation_rows(self, attribute, keys):
        """Return each lane's variation leaves ``keys`` as one flat row of ``attribute`` values."""
        return np.stack(
            [
                np.concatenate(
                    [np.asarray(getattr(env.env.variation_space._walk(k), attribute)).reshape(-1) for k in keys]
                )
                for env in self.envs
            ]
        )

    def close(self):
        self._prepared = None
        for env in self.envs:
            env.close()
        self.envs = []
        if self._scene is not None:
            import pybullet

            pybullet.disconnect(self._scene[0])
            self._scene = None
