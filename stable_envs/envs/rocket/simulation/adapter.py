from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.process import BROADCAST, PER_LANE
from ...._execution.protocol import SimulatorAdapter
from .actions import validate_action
from .replay import RocketReplay, generator_state, install_generator


def check_values(template, values, num_envs):
    """Check one ``{path: value}`` per lane against the Variation space of engine ``template``."""
    if len(values) != num_envs:
        raise ValueError("Expected one Rocket Variation row per lane")
    for row in values:
        for key, value in row.items():
            node = template.env.variation_space._walk(key)
            if hasattr(node, "spaces") or not node.contains(value):
                raise ValueError(f"Invalid Rocket Variation: {key}")


def state_rows(state, fields, engines):
    """Check State against ``fields`` and each row with an engine of ``engines``; return the host rows."""
    batch = len(engines)
    if not isinstance(state, TensorDict) or state.batch_size != torch.Size([batch]):
        raise ValueError("Rocket full State requires batch [num_envs]")
    if set(state.keys(True, True)) != set(fields):
        raise ValueError("Rocket full State keys differ")
    for key, (shape, dtype) in fields.items():
        if state[key].shape != (batch, *shape) or state[key].dtype != dtype:
            raise ValueError(f"Invalid Rocket State leaf: {key}")
    # One transfer per leaf; per-row device reads cost one copy per row and leaf.
    columns = {"/".join(k): v.detach().cpu().numpy() for k, v in state.items(True, True)}
    for key, value in columns.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Invalid Rocket State leaf: {tuple(key.split('/'))}")
    rows = [{k: v[i, ...].copy() for k, v in columns.items()} for i in range(batch)]
    for env, row in zip(engines, rows, strict=True):
        env.validate(row)
    return rows


class RocketAdapter(SimulatorAdapter):
    """Batch CPU Rocket owners with CPU/CUDA State storage.

    Validate every action before stepping any lane. Stage reset and restoration
    for the whole batch before replacing its current engines.
    """

    process_methods = {"configure": PER_LANE, "rgba": BROADCAST, "variation_rows": BROADCAST}
    process_attributes = ("fields",)
    action_dim = 7

    def __init__(self, batch, device, seed, config=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = batch, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Rocket supports CPU dynamics and CPU/CUDA State storage")
        self.envs, self._prepared = [], None
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
        """A lane whose configuration and Variation space describe the batch."""
        return self.envs[0]

    def get_state(self):
        rows = [e.get_state() for e in self.envs]
        return TensorDict(
            {
                tuple(k.split("/")): torch.as_tensor(np.stack([r[k] for r in rows]), device=self.device)
                for k in rows[0]
            },
            [self.num_envs],
            device=self.device,
        )

    def rgba(self):
        return torch.as_tensor(np.stack([e.rgba() for e in self.envs]), device=self.device)

    def validate(self, state):
        return state_rows(state, self.fields, self.envs)

    def prepare_restore(self, state):
        """Replay every lane of ``state`` into new engines; the live lanes do not change."""
        if self._prepared is not None:
            raise RuntimeError("A Rocket restore is already prepared")
        rows = self.validate(state)
        staged = []
        try:
            for env, row in zip(self.envs, rows, strict=True):
                staged.append(env.reconstruct(row))
        except BaseException:
            for env in staged:
                env.close()
            raise
        self._prepared = staged

    def commit_restore(self):
        if self._prepared is None:
            raise RuntimeError("No Rocket restore is prepared")
        previous, self.envs, self._prepared = self.envs, self._prepared, None
        for env in previous:
            env.close()

    def discard_restore(self):
        staged, self._prepared = self._prepared, None
        for env in staged or ():
            env.close()

    def set_state(self, state):
        self.prepare_restore(state)
        self.commit_restore()

    def reset(self, seed=None, options=None, values=None):
        # Stage reset with persistent configuration and owned RNG only; no physics
        # replay is needed because authoritative reset discards the old episode.
        staged = []
        try:
            for lane, env in enumerate(self.envs):
                target = RocketReplay(env.config, initialize=False)
                staged.append(target)
                target.env.variation_space = copy.deepcopy(env.env.variation_space)
                install_generator(target.env, generator_state(env.env))
                install_generator(target.env.action_space, generator_state(env.env.action_space))
                target.reset(None if seed is None else seed + lane, options, None if values is None else values[lane])
        except BaseException:
            for env in staged:
                env.close()
            raise
        previous, self.envs = self.envs, staged
        for env in previous:
            env.close()
        return self.get_state()

    def check_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 7):
            raise ValueError("Rocket actions require shape [num_envs, 7]")
        for action in actions:
            validate_action(action)
        if active is not None:
            active = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active, dtype=bool)
        return actions, active

    def step(self, actions, *, active=None, capture_state=True):
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 7):
            raise ValueError("Rocket actions require shape [num_envs, 7]")
        for action in actions:
            validate_action(action)
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

    def configure(self, values):
        check_values(self.envs[0], values, self.num_envs)
        for env, row in zip(self.envs, values, strict=True):
            env.configure(row)

    def variation_rows(self, attribute, keys):
        """Return each lane's Variation leaves ``keys`` as one flat row of ``attribute`` values."""
        return np.stack(
            [
                np.concatenate(
                    [np.asarray(getattr(env.env.variation_space._walk(k), attribute)).reshape(-1) for k in keys]
                )
                for env in self.envs
            ]
        )

    def close(self):
        self.discard_restore()
        for env in self.envs:
            env.close()
        self.envs = []
