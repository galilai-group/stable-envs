"""MuJoCo Warp physics for MS-Human-700."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import torch
import warp as wp
from tensordict import TensorDict

from ...._core.warp_cuda import require_warp_cuda
from ...._execution.protocol import SimulatorAdapter


class WarpAdapter(SimulatorAdapter):
    """One MuJoCo Warp world per lane, simulated on ``device``."""

    def __init__(self, num_envs: int, device, model):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.action_dim = int(model.nu)
        if self.device.type == "cuda":
            require_warp_cuda("MS-Human-700 GPU physics")
        self._wp = "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
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
                nconmax=self.num_envs * model.nconmax,
                njmax=self.num_envs * model.njmax,
            )
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _buf(self, name: str) -> torch.Tensor:
        return wp.to_torch(getattr(self._d, name))

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        with wp.ScopedDevice(self._wp):
            for name, value in self._init.items():
                self._buf(name).copy_(torch.as_tensor(value, device=self._wp))
            self._buf("ctrl").zero_()
            self._buf("time").zero_()
            mjw.forward(self._m, self._d)
            wp.synchronize()
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
        with wp.ScopedDevice(self._wp):
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
            wp.synchronize()

    def step(self, actions) -> TensorDict:
        self._check_open()
        action = torch.as_tensor(actions, device=self._wp).reshape(self.num_envs, self.action_dim)
        with wp.ScopedDevice(self._wp):
            self._buf("ctrl").copy_(action.to(self._buf("ctrl").dtype))
            mjw.step(self._m, self._d)
            wp.synchronize()
        return TensorDict({"state": self.get_state()}, batch_size=[self.num_envs], device=self.device)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._m = self._d = None
