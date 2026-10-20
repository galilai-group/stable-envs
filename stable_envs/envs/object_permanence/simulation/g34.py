"""Owned MuJoCo lanes for the source-authored G34 animation.

All motion is prescribed by the source frame. The only action is zero (advance
one frame); this is an autonomous video environment, not a force-control task.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from ..scenes.g34 import FPS, FRAME_COUNT, SUBSTEPS, authored_pose, make_model, set_pose

STATE_MASK = int(mujoco.mjtState.mjSTATE_INTEGRATION)
INTEGRATION = ("physics", "integration")
ELAPSED = ("lifecycle", "elapsed_steps")


class OcclusionBallAdapter(SimulatorAdapter):
    """Advance independent MuJoCo lanes, retaining every engine integration input.

    No RNG is consumed: every seed selects the same authored nominal sequence.
    Model/configuration are fixed. The State mask, model and field layout must
    match when transferring a snapshot between instances.
    """

    def __init__(self, num_envs, device="cpu"):
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.num_envs, self.device = num_envs, torch.device(device)
        # The immutable model is shared; every lane owns its writable MjData.
        self.model = make_model()
        self.lanes = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.state_width = mujoco.mj_stateSize(self.model, STATE_MASK)
        before_pose = STATE_MASK & (int(mujoco.mjtState.mjSTATE_MOCAP_POS) - 1)
        self.pose_offset = mujoco.mj_stateSize(self.model, before_pose)
        self.fields = {INTEGRATION: (self.state_width, torch.float64), ELAPSED: (1, torch.int64)}
        self.elapsed = np.zeros(num_envs, np.int64)
        self._closed = False
        self.reset()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OcclusionBallAdapter is closed")

    def reset(self, seed=None):
        self._check_open()
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        position, quaternion = authored_pose(0)
        for data in self.lanes:
            mujoco.mj_resetData(self.model, data)
            set_pose(self.model, data, position, quaternion)
        self.elapsed.fill(0)
        return self.get_state()

    def get_state(self):
        self._check_open()
        rows = np.empty((self.num_envs, self.state_width), np.float64)
        for i, data in enumerate(self.lanes):
            mujoco.mj_getState(self.model, data, rows[i], STATE_MASK)
        return TensorDict(
            {
                INTEGRATION: torch.from_numpy(rows).to(self.device),
                ELAPSED: torch.from_numpy(self.elapsed.copy()[:, None]).to(self.device),
            },
            [self.num_envs],
            device=self.device,
        )

    def validate_state(self, state, *, batch=None):
        """Validate all lanes, including semantic constraints, before any mutation."""
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full TensorDict State, not compact render rows")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError("State must have batch [num_envs]")
        keys = {(key,) if isinstance(key, str) else key for key in state.keys(True, True)}
        if keys != set(self.fields):
            raise ValueError("State must contain exactly physics.integration and lifecycle.elapsed_steps")
        count = state.batch_size[0]
        for key, (width, dtype) in self.fields.items():
            value = state[key]
            if value.shape != (count, width) or value.dtype != dtype or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid State field {key}: shape, dtype or finite values")
        steps = state[ELAPSED].detach().cpu().numpy()[:, 0]
        rows = state[INTEGRATION].detach().cpu().numpy()
        if np.any((steps < 0) | (steps >= FRAME_COUNT)):
            raise ValueError("elapsed_steps must be in [0, 119]")
        if not np.allclose(rows[:, 0], steps / FPS, atol=1e-12, rtol=0):
            raise ValueError("MuJoCo time and source frame disagree")
        quaternions = rows[:, self.pose_offset + 3 : self.pose_offset + 7]
        if not np.allclose(np.linalg.norm(quaternions, axis=-1), 1, atol=1e-12, rtol=0):
            raise ValueError("Mocap quaternion must have unit norm")

    def set_state(self, state):
        self._check_open()
        self.validate_state(state, batch=self.num_envs)
        host = state.detach().cpu()
        for i, data in enumerate(self.lanes):
            mujoco.mj_setState(self.model, data, host[INTEGRATION][i].numpy(), STATE_MASK)
            mujoco.mj_forward(self.model, data)
        self.elapsed[:] = host[ELAPSED][:, 0].numpy()

    def step(self, actions, *, active=None):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if (
            actions.shape != (self.num_envs, 1)
            or actions.dtype.kind != "f"
            or not np.isfinite(actions).all()
            or np.any(actions != 0)
        ):
            raise ValueError("G34 accepts only finite floating-point zero actions with shape [B, 1]")
        if active is None:
            mask = np.ones(self.num_envs, bool)
        else:
            mask = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active)
            if mask.shape != (self.num_envs,) or mask.dtype != np.bool_:
                raise ValueError("active must be a boolean mask with shape [B]")
        if np.any(mask & (self.elapsed >= FRAME_COUNT - 1)):
            raise RuntimeError("Episode finished; reset before stepping again")
        for i in np.flatnonzero(mask):
            data = self.lanes[i]
            mujoco.mj_step(self.model, data, nstep=SUBSTEPS)
            self.elapsed[i] += 1
            set_pose(self.model, data, *authored_pose(self.elapsed[i]))
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.zeros(self.num_envs, dtype=torch.float64, device=self.device),
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.as_tensor(self.elapsed >= FRAME_COUNT - 1, device=self.device),
            },
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            self.lanes.clear()
            self._closed = True
