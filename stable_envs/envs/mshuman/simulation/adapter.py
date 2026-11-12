"""MS-Human-700 physics: native ``mujoco.batch`` on CPU, MuJoCo Warp on CUDA."""

from __future__ import annotations

import os

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from mujoco.batch import Batch
from tensordict import TensorDict

from ...._core.config import log
from ...._core.warp_cuda import require_warp_cuda
from ...._execution.policy import cpu_affinity
from ...._execution.protocol import SimulatorAdapter

#: Most mujoco.batch threads, as the ``dmc_threads`` default; also at most one per CPU in the affinity and per lane.
_MAX_THREADS = 8


def _log_warning(message: str) -> None:
    log("WARNING", f"MuJoCo: {message}")


class MSHumanBatchAdapter(SimulatorAdapter):
    """CPU MuJoCo simulations stepped together in one ``mujoco.batch`` on a native thread pool."""

    def __init__(self, num_envs: int, device, model):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.action_dim = int(model.nu)
        self._closed = False
        # Without a handler MuJoCo prints warnings and appends them to MUJOCO_LOG.TXT in the working directory.
        if mujoco.get_mju_user_warning() is None:
            mujoco.set_mju_user_warning(_log_warning)
        cpus = len(cpu_affinity() or ()) or os.cpu_count() or 1
        self.b = Batch(model, self.num_envs, nthread=min(_MAX_THREADS, cpus, self.num_envs))
        limited = model.actuator_ctrllimited.astype(bool)
        self._ctrl_low = np.where(limited, model.actuator_ctrlrange[:, 0], -np.inf)
        self._ctrl_high = np.where(limited, model.actuator_ctrlrange[:, 1], np.inf)
        self.reset()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _quantize_state(self) -> None:
        """Keep the complete public checkpoint lossless at its float32 dtype."""
        for value in (self.b.qpos, self.b.qvel, self.b.act, self.b.time, self.b.qacc_warmstart):
            value[...] = value.astype(np.float32)

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        self.b.reset(keyframe=0)
        self._quantize_state()
        self._warnings = self.b.warning[:, :, 1].copy()
        return self.get_state()

    def get_state(self) -> TensorDict:
        self._check_open()
        fields = {
            "qpos": self.b.qpos,
            "qvel": self.b.qvel,
            "act": self.b.act,
            "time": self.b.time[:, None],
            "warmstart": self.b.qacc_warmstart,
        }
        return TensorDict(
            {key: torch.from_numpy(value.astype(np.float32)) for key, value in fields.items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def set_state(self, state) -> None:
        self._check_open()
        for field, target in (
            ("qpos", self.b.qpos),
            ("qvel", self.b.qvel),
            ("act", self.b.act),
            ("time", self.b.time),
            ("warmstart", self.b.qacc_warmstart),
        ):
            target[:] = state[field].detach().cpu().to(torch.float32).numpy().reshape(target.shape)
        self._warnings = self.b.warning[:, :, 1].copy()
        self.b.forward()

    def step(self, actions) -> TensorDict:
        self._check_open()
        if torch.is_tensor(actions):
            # NumPy has no bfloat16.
            actions = (actions.float() if actions.dtype == torch.bfloat16 else actions).detach().cpu().numpy()
        ctrl = np.asarray(actions, np.float32).reshape(self.num_envs, self.action_dim)
        # Clamp as MuJoCo Warp does, NaN to the lower bound: native MuJoCo would instead zero the
        # lane's controls on a NaN, inf or huge value and warn, which raises below.
        self.b.ctrl[:] = np.fmin(np.fmax(ctrl, self._ctrl_low), self._ctrl_high)
        self.b.step()
        self._quantize_state()
        if (bad := np.flatnonzero((self.b.warning[:, :, 1] > self._warnings).any(axis=1))).size:
            raise RuntimeError(f"Physics state is invalid in lanes {bad.tolist()}")
        return TensorDict({"state": self.get_state()}, batch_size=[self.num_envs], device=self.device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.b = None


class MSHumanWarpAdapter(SimulatorAdapter):
    """One MuJoCo Warp world per lane, simulated on CUDA ``device``."""

    def __init__(self, num_envs: int, device, model):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.action_dim = int(model.nu)
        require_warp_cuda("MS-Human-700 GPU physics")
        self._wp = f"cuda:{self.device.index or 0}"
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        self._init = {k: getattr(data, k).copy() for k in ("qpos", "qvel", "act", "qacc_warmstart")}
        with wp.ScopedDevice(self._wp):
            self._m = mjw.put_model(model)
            self._d = mjw.put_data(
                model,
                data,
                nworld=self.num_envs,
                nconmax=model.nconmax,
                njmax=model.njmax,
            )
            mjw.step(self._m, self._d)
            self._restore_data()
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                mjw.step(self._m, self._d)
            self._step_graph = capture.graph
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _buf(self, name: str) -> torch.Tensor:
        return wp.to_torch(getattr(self._d, name))

    def _torch_stream(self):
        return wp.stream_from_torch(self._wp)

    def _restore_data(self) -> None:
        for name, value in self._init.items():
            self._buf(name).copy_(torch.as_tensor(value, device=self._wp))
        self._buf("ctrl").zero_()
        self._buf("time").zero_()
        mjw.forward(self._m, self._d)

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        with wp.ScopedDevice(self._wp), wp.ScopedStream(self._torch_stream()):
            self._restore_data()
            wp.synchronize_stream(self._wp)
        return self.get_state()

    def get_state(self) -> TensorDict:
        self._check_open()
        fields = {
            "qpos": self._buf("qpos"),
            "qvel": self._buf("qvel"),
            "act": self._buf("act"),
            "time": self._buf("time").reshape(self.num_envs, 1),
            "warmstart": self._buf("qacc_warmstart"),
        }
        return TensorDict(
            {k: v.clone().to(self.device, torch.float32) for k, v in fields.items()},
            batch_size=[self.num_envs],
            device=self.device,
        )

    def set_state(self, state) -> None:
        self._check_open()
        with wp.ScopedDevice(self._wp), wp.ScopedStream(self._torch_stream()):
            for field, name in (
                ("qpos", "qpos"),
                ("qvel", "qvel"),
                ("act", "act"),
                ("warmstart", "qacc_warmstart"),
                ("time", "time"),
            ):
                buf = self._buf(name)
                buf.copy_(state[field].reshape(buf.shape).to(self._wp, buf.dtype))
            mjw.forward(self._m, self._d)
            wp.synchronize_stream(self._wp)

    def step(self, actions) -> TensorDict:
        self._check_open()
        action = torch.as_tensor(actions, device=self._wp).reshape(self.num_envs, self.action_dim)
        with wp.ScopedDevice(self._wp), wp.ScopedStream(self._torch_stream()):
            self._buf("ctrl").copy_(action.to(self._buf("ctrl").dtype))
            wp.capture_launch(self._step_graph)
            wp.synchronize_stream(self._wp)
        return TensorDict({"state": self.get_state()}, batch_size=[self.num_envs], device=self.device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._m = self._d = self._step_graph = None
