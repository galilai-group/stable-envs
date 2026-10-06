from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from .pusht_replay import HISTORY_KEYS, PushTReplay, validate_seed


class PushTReplayAdapter:
    """Batch CPU PushT owners with CPU/CUDA State storage.

    Stage every lane before installing a reset or restored State. Export pads
    histories to the longest lane with zero, invalid entries; padding adds no
    episode time limit."""

    def __init__(self, batch, device="cpu", seed=0, *, discrete=False, config=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("PushT batch must be a positive integer")
        validate_seed(seed)
        self.num_envs, self.device = batch, torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("PushT supports CPU physics with CPU/CUDA State storage")
        self.envs = []
        try:
            for lane in range(batch):
                self.envs.append(
                    PushTReplay(discrete=discrete, config=config, seed=None if seed is None else int(seed) + lane)
                )
        except BaseException:
            self.close()
            raise

    def get_state(self):
        rows = [env.get_state() for env in self.envs]
        capacity = max(int(row["history/lengths"]) for row in rows)
        result = {}
        for key in rows[0]:
            if key in HISTORY_KEYS:
                sample = rows[0][key]
                values = np.zeros((self.num_envs, capacity, *sample.shape[1:]), sample.dtype)
                for lane, row in enumerate(rows):
                    values[lane, : len(row[key])] = row[key]
            else:
                values = np.stack([row[key] for row in rows])
            result[tuple(key.split("/"))] = torch.as_tensor(values, device=self.device)
        return TensorDict(result, [self.num_envs], device=self.device)

    def validate(self, state):
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise ValueError("PushT full State requires batch [num_envs]")
        expected = self.envs[0].get_state()
        if set(state.keys(True, True)) != {tuple(key.split("/")) for key in expected}:
            raise ValueError("PushT full State keys differ")
        # Transfer each whole field once, then take owned lane copies on the host.
        fields = {}
        for key, sample in expected.items():
            value = state[tuple(key.split("/"))]
            if value.dtype != torch.from_numpy(sample).dtype or value.shape[0] != self.num_envs:
                raise ValueError(f"Invalid PushT State dtype/batch: {key}")
            fields[key] = value.detach().cpu().numpy()
        rows = [
            {key: np.asarray(value[lane]).copy() for key, value in fields.items()} for lane in range(self.num_envs)
        ]
        for env, row in zip(self.envs, rows, strict=True):
            env.validate(row)
        return rows

    def _swap(self, staged):
        previous, self.envs = self.envs, staged
        for env in previous:
            env.close()

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
        self._swap(staged)

    def reset(self, seed=None, options=None):
        validate_seed(seed)
        staged = []
        try:
            for lane, env in enumerate(self.envs):
                staged.append(env.reset_candidate(None if seed is None else int(seed) + lane, options))
        except BaseException:
            for env in staged:
                env.close()
            raise
        self._swap(staged)
        return self.get_state()

    def step(self, actions, *, active=None):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        if not isinstance(actions, np.ndarray) or actions.shape != (self.num_envs, 2):
            raise ValueError("PushT actions require shape [num_envs, 2]")
        for env, action in zip(self.envs, actions, strict=True):
            env.validate_action(action)
        if active is None:
            active = np.ones(self.num_envs, np.bool_)
        elif torch.is_tensor(active):
            active = active.detach().cpu().numpy()
        if not isinstance(active, np.ndarray) or active.dtype != np.bool_ or active.shape != (self.num_envs,):
            raise ValueError("PushT active mask requires bool [num_envs]")
        outputs = []
        for env, action, enabled in zip(self.envs, actions, active, strict=True):
            if enabled:
                out = env.step(action)
                outputs.append((out[1], out[2], out[3]))
            else:
                success, distance = env.env.eval_state(env.env.goal_state, env.env._get_obs())
                outputs.append((-distance, success, False))
        return TensorDict(
            dict(
                state=self.get_state(),
                reward=torch.tensor([v[0] for v in outputs], dtype=torch.float64, device=self.device),
                success=torch.tensor([v[1] for v in outputs], dtype=torch.bool, device=self.device),
                terminated=torch.tensor([v[1] for v in outputs], dtype=torch.bool, device=self.device),
                truncated=torch.tensor([v[2] for v in outputs], dtype=torch.bool, device=self.device),
            ),
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []
