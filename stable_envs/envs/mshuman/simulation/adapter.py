"""MS-Human-700 physics on the shared batched MuJoCo backend: ``mujoco.batch``, or MuJoCo Warp on CUDA by request."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.state_schema import Field, StateSchema
from ...._core.tensordicts import build_tensordict
from ...._execution.mjbatch import NativeBatch, WarpBatch
from ...._execution.protocol import SimulatorAdapter

#: Model dimensions: positions, velocities, muscle activations and controls.
NQ, NV, NA, NU = 36, 36, 100, 100
#: The State group: every field is float32.
GROUP = "mshuman/float32"
STATE = StateSchema(
    "MSHuman",
    [
        Field("qpos", (NQ,), torch.float32, GROUP),
        Field("qvel", (NV,), torch.float32, GROUP),
        Field("act", (NA,), torch.float32, GROUP),
        Field("time", (1,), torch.float32, GROUP),
        Field("warmstart", (NV,), torch.float32, GROUP),
    ],
)
#: State fields and the MjData fields they hold.
_FIELDS = {"qpos": "qpos", "qvel": "qvel", "act": "act", "time": "time", "warmstart": "qacc_warmstart"}


def state_of(group: torch.Tensor) -> TensorDict:
    """The State that holds ``group``, a ``(batch, size)`` tensor."""
    return build_tensordict({"mshuman": {"float32": group}}, [len(group)], group.device)


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

    def _physics(self, state) -> dict:
        views = STATE.views(state)
        return {name: views[key] for key, name in _FIELDS.items()}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.batch.close()


class MSHumanBatchAdapter(_MSHumanAdapter):
    """CPU MuJoCo simulations stepped together in one ``mujoco.batch`` on a native thread pool."""

    def __init__(self, num_envs: int, device, model):
        super().__init__(num_envs, device, model)
        self.batch = NativeBatch(model, self.num_envs, "cpu", keyframe=0)
        # Float32 copy of the batch's State, in one group; the field views are cut once.
        self._live = np.zeros((self.num_envs, STATE.groups[GROUP].size), np.float32)
        self._fields = STATE.views({GROUP: self._live}, numpy=True)
        limited = model.actuator_ctrllimited.astype(bool)
        self._ctrl_low = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self._ctrl_high = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)
        self.reset()

    @property
    def b(self):
        """The ``mujoco.batch.Batch``, or None once closed."""
        return None if self._closed else self.batch.b

    def _sync(self, write_back: bool = False) -> None:
        """Copy the batch's State into the float32 group; with ``write_back``, keep the batch lossless at float32."""
        for key, name in _FIELDS.items():
            value = self.batch.numpy(name)
            field = self._fields[key]
            np.copyto(field, value.reshape(field.shape), casting="same_kind")
            if write_back:
                value[...] = field.reshape(value.shape)

    def _snapshot(self) -> TensorDict:
        """The State of the float32 group, on the env device, in new memory."""
        group = torch.from_numpy(self._live.copy() if self.device.type == "cpu" else self._live)
        return state_of(group.to(self.device))

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        self.batch.reset()
        self._sync(write_back=True)
        return self._snapshot()

    def get_state(self) -> TensorDict:
        self._check_open()
        self._sync()
        return self._snapshot()

    def set_state(self, state) -> None:
        self._check_open()
        self.batch.set_physics({k: v.detach().cpu().to(torch.float32) for k, v in self._physics(state).items()})

    def step(self, actions, *, active=None) -> TensorDict:
        self._check_open()
        if torch.is_tensor(active):
            active = active.detach().cpu().numpy()
        if torch.is_tensor(actions):
            # NumPy has no bfloat16.
            actions = (actions.float() if actions.dtype == torch.bfloat16 else actions).detach().cpu().numpy()
        ctrl = np.asarray(actions, np.float32).reshape(self.num_envs, self.action_dim)
        # Clamp as MuJoCo Warp does, NaN to the lower bound: native MuJoCo would instead zero the
        # environment's controls on a NaN, inf or huge value and warn, which raises below.
        self.batch.step(np.fmin(np.fmax(ctrl, self._ctrl_low), self._ctrl_high), active=active)
        self._sync(write_back=True)
        if (bad := self.batch.unstable_envs()).size:
            raise RuntimeError(f"Physics state is invalid in environments {bad.tolist()}")
        return TensorDict({"state": self._snapshot()}, batch_size=[self.num_envs], device=self.device)


class MSHumanWarpAdapter(_MSHumanAdapter):
    """One MuJoCo Warp world per environment, simulated on CUDA ``device``."""

    def __init__(self, num_envs: int, device, model):
        super().__init__(num_envs, device, model)
        self.batch = WarpBatch(model, self.num_envs, self.device, keyframe=0)

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        self.batch.reset()
        return self.get_state()

    def get_state(self) -> TensorDict:
        self._check_open()
        data = self.batch.data
        values = {key: data(name).to(torch.float32) for key, name in _FIELDS.items()}
        values["time"] = values["time"][:, None]
        return state_of(STATE.pack(values)[GROUP])

    def set_state(self, state) -> None:
        self._check_open()
        self.batch.set_physics(self._physics(state))

    def step(self, actions, *, active=None) -> TensorDict:
        self._check_open()
        ctrl = torch.as_tensor(actions, device=self.device).reshape(self.num_envs, self.action_dim)
        self.batch.step(ctrl, active=active)
        return TensorDict({"state": self.get_state()}, batch_size=[self.num_envs], device=self.device)
