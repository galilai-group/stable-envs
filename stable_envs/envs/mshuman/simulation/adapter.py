"""MS-Human-700 physics on the shared batched MuJoCo backend: ``mujoco.batch`` on CPU, MuJoCo Warp on CUDA."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.mjbatch import NativeBatch, WarpBatch
from ...._execution.protocol import SimulatorAdapter

#: State keys and the MjData fields they hold.
_FIELDS = {"qpos": "qpos", "qvel": "qvel", "act": "act", "time": "time", "warmstart": "qacc_warmstart"}


class _MSHumanAdapter(SimulatorAdapter):
    batch: NativeBatch | WarpBatch

    def __init__(self, num_envs: int, device, model):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.action_dim = int(model.nu)
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def get_state(self) -> TensorDict:
        self._check_open()
        physics = self.batch.read(_FIELDS.values())
        physics["time"] = physics["time"][:, None]
        return TensorDict(
            {key: physics[name].to(torch.float32) for key, name in _FIELDS.items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def _physics(self, state) -> dict:
        return {name: state[key] for key, name in _FIELDS.items()}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.batch.close()


class MSHumanBatchAdapter(_MSHumanAdapter):
    """CPU MuJoCo simulations stepped together in one ``mujoco.batch`` on a native thread pool."""

    def __init__(self, num_envs: int, device, model):
        super().__init__(num_envs, device, model)
        self.batch = NativeBatch(model, self.num_envs, self.device, keyframe=0)
        limited = model.actuator_ctrllimited.astype(bool)
        self._ctrl_low = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self._ctrl_high = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)
        self.reset()

    @property
    def b(self):
        """The ``mujoco.batch.Batch``, or None once closed."""
        return None if self._closed else self.batch.b

    def _quantize_state(self) -> None:
        """Keep the complete public checkpoint lossless at its float32 dtype."""
        for name in _FIELDS.values():
            value = self.batch.numpy(name)
            value[...] = value.astype(np.float32)

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        self.batch.reset()
        self._quantize_state()
        return self.get_state()

    def set_state(self, state) -> None:
        self._check_open()
        self.batch.set_physics({k: v.detach().cpu().to(torch.float32) for k, v in self._physics(state).items()})

    def step(self, actions) -> TensorDict:
        self._check_open()
        if torch.is_tensor(actions):
            # NumPy has no bfloat16.
            actions = (actions.float() if actions.dtype == torch.bfloat16 else actions).detach().cpu().numpy()
        ctrl = np.asarray(actions, np.float32).reshape(self.num_envs, self.action_dim)
        # Clamp as MuJoCo Warp does, NaN to the lower bound: native MuJoCo would instead zero the
        # lane's controls on a NaN, inf or huge value and warn, which raises below.
        self.batch.step(np.fmin(np.fmax(ctrl, self._ctrl_low), self._ctrl_high))
        self._quantize_state()
        if (bad := self.batch.unstable_envs()).size:
            raise RuntimeError(f"Physics state is invalid in lanes {bad.tolist()}")
        return TensorDict({"state": self.get_state()}, batch_size=[self.num_envs], device=self.device)


class MSHumanWarpAdapter(_MSHumanAdapter):
    """One MuJoCo Warp world per lane, simulated on CUDA ``device``."""

    def __init__(self, num_envs: int, device, model):
        super().__init__(num_envs, device, model)
        self.batch = WarpBatch(model, self.num_envs, self.device, keyframe=0)

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        self.batch.reset()
        return self.get_state()

    def set_state(self, state) -> None:
        self._check_open()
        self.batch.set_physics(self._physics(state))

    def step(self, actions) -> TensorDict:
        self._check_open()
        self.batch.step(torch.as_tensor(actions, device=self.device).reshape(self.num_envs, self.action_dim))
        return TensorDict({"state": self.get_state()}, batch_size=[self.num_envs], device=self.device)
