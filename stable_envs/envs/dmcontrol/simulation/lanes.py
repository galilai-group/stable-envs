from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from ...._execution.state_columns import require_columns
from . import physics as _physics
from .state import export_state, restore_lane


class DMCLanes(SimulatorAdapter):
    """Independent CPU dm_control lanes; each action advances the source twice.

    ``device`` controls returned tensor placement. State holds physics, lifecycle,
    task RNG, episode-drawn model entries and the lane's physics Variation.
    """

    def __init__(self, key, num_envs, device, seed, make, fields):
        self.key = key
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [make(int(seed) + i) for i in range(self.num_envs)]
        self.physics = _physics.make_lanes(key, [env.physics.model.ptr for env in self.envs])
        self.fields = {**fields, **_physics.state_fields(key)}
        self._base_fields = fields
        self.action_dim = int(self.envs[0].action_spec().shape[0])
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    @property
    def physics_layout(self):
        return None if self.physics is None else self.physics.layout

    @property
    def physics_dim(self):
        return 0 if self.physics is None else self.physics.spec.dim

    def configure_physics(self, values):
        self._check_open()
        if self.physics is None:
            raise ValueError(f"{self.key} declares no physics Variation")
        self.physics.configure(values)
        # dm_control's step begins with mj_step2, which reuses the position and velocity stages in data.
        for env in self.envs:
            env.physics.forward()

    def reset(self, seed=None):
        self._check_open()
        for i, env in enumerate(self.envs):
            if seed is not None:
                env.task._random = np.random.RandomState(int(seed) + i)
            env.reset()
        self._cumulative.fill(0)
        return self.get_state()

    def get_state(self):
        self._check_open()
        extra = None if self.physics is None else self.physics.export()
        return export_state(self.envs, self._cumulative, self.fields, self.device, extra)

    def validate(self, state):
        """Check every lane of ``state`` and return its physics rows, or None without physics Variation."""
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError(f"{self.key} requires full TensorDict State with batch [num_envs]")
        rows = None if self.physics is None else self.physics.rows_from_state(state)
        require_columns(state, {key: ((self.num_envs, width), dtype) for key, (width, dtype) in self.fields.items()})
        return rows

    def set_state(self, state):
        rows = self.validate(state)
        host = state.cpu()
        if rows is not None:
            self.physics.configure(rows)
        for i, env in enumerate(self.envs):
            restore_lane(env, host[i], self._base_fields, self._cumulative, i)

    def check_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"DMC actions must be finite with shape {(self.num_envs, self.action_dim)}")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or not active.all():
                raise ValueError(
                    "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
                )
        return actions, None

    def _info(self, env, lane):
        """Return ``(success, info)`` for task outputs beyond reward, or None."""
        return None

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        if active is not None and not bool(torch.as_tensor(active).all()):
            raise ValueError(
                "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
            )
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"{self.key} action must have finite shape (num_envs, {self.action_dim})")
        rewards, observations, infos = [], [], []
        for i, env in enumerate(self.envs):
            reward = 0.0
            for _ in range(2):
                result = env.step(actions[i].astype(env.action_spec().dtype))
                reward += result.reward or 0.0
            self._cumulative[i] += reward
            rewards.append(reward)
            observations.append(np.concatenate([v.flatten() for v in result.observation.values()], dtype=np.float32))
            infos.append(self._info(env, i))
        out = {
            **({"state": self.get_state()} if capture_state else {}),
            "reward": torch.tensor(rewards, dtype=torch.float64, device=self.device),
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "observation": torch.as_tensor(np.stack(observations), device=self.device),
        }
        score = torch.tensor(self._cumulative / 1000.0, dtype=torch.float64, device=self.device)
        if infos[0] is None:
            out["score"] = score
        else:
            out["success"] = torch.tensor([s for s, _ in infos], dtype=torch.float64, device=self.device)
            out["score"] = score
            out["info"] = TensorDict(
                {
                    key: torch.as_tensor(np.stack([np.asarray(info[key]) for _, info in infos]), device=self.device)
                    for key in infos[0][1]
                },
                [self.num_envs],
                device=self.device,
            )
        return TensorDict(out, [self.num_envs], device=self.device)

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
