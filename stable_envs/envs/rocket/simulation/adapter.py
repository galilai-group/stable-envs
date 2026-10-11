from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from .actions import validate_action
from .replay import RocketReplay, generator_state, install_generator


class RocketAdapter(SimulatorAdapter):
    """Batch CPU Rocket owners with CPU/CUDA State storage.

    Validate every action before stepping any lane. Stage reset and restoration
    for the whole batch before replacing its current engines."""

    def __init__(self, batch, device, seed, config=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = batch, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Rocket supports CPU dynamics and CPU/CUDA State storage")
        self.envs = []
        try:
            for lane in range(batch):
                self.envs.append(RocketReplay(config, None if seed is None else seed + lane))
        except BaseException:
            self.close()
            raise
        self.fields = {
            tuple(k.split("/")): (v.shape, torch.from_numpy(v).dtype) for k, v in self.envs[0].get_state().items()
        }

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

    def validate(self, state):
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise ValueError("Rocket full State requires batch [num_envs]")
        if set(state.keys(True, True)) != set(self.fields):
            raise ValueError("Rocket full State keys differ")
        for key, (shape, dtype) in self.fields.items():
            if (
                state[key].shape != (self.num_envs, *shape)
                or state[key].dtype != dtype
                or not torch.isfinite(state[key]).all()
            ):
                raise ValueError(f"Invalid Rocket State leaf: {key}")
        rows = [{"/".join(k): v.detach().cpu().numpy().copy() for k, v in row.items(True, True)} for row in state]
        for env, row in zip(self.envs, rows, strict=True):
            env.validate(row)
        return rows

    def set_state(self, state):
        rows = self.validate(state)
        staged = []
        try:
            for env, row in zip(self.envs, rows, strict=True):
                staged.append(env.reconstruct(row))
        except BaseException:
            for env in staged:
                env.close()
            raise
        previous, self.envs = self.envs, staged
        for env in previous:
            env.close()

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

    def step(self, actions, *, active=None):
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
                state=self.get_state(),
                reward=torch.tensor([r[1] for r in results], dtype=torch.float64, device=self.device),
                success=torch.tensor([r[4]["env_complete"] for r in results], dtype=torch.bool, device=self.device),
                terminated=torch.tensor([r[2] for r in results], dtype=torch.bool, device=self.device),
                truncated=torch.tensor([r[3] for r in results], dtype=torch.bool, device=self.device),
            ),
            [self.num_envs],
            device=self.device,
        )

    def configure(self, values):
        for env, row in zip(self.envs, values, strict=True):
            for key, value in row.items():
                node = env.env.variation_space._walk(key)
                if hasattr(node, "spaces") or not node.contains(value):
                    raise ValueError(f"Invalid Rocket Variation: {key}")
        for env, row in zip(self.envs, values, strict=True):
            env.configure(row)

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []
