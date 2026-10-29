"""MS-Human-700 musculoskeletal locomotion on MuJoCo Warp."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
from tensordict import TensorDict

from ..._core.mujoco_gl import mujoco_renderer
from ..._core.warp_cuda import require_warp_cuda
from ...env import Env, render_states_arg, require_contiguous_out
from ...registry import register
from ...spaces import Dict
from .assets import _COMMIT, _XML
from .assets import model as _model

_NQ, _NV, _NA, _NU = 36, 36, 100, 100


@register
class MSHumanLocomotion(Env):
    """MS-Human-700 whole-body locomotion: 100 muscles actuate 36 DoF, controls in ``[0, 1]``."""

    family = "mshuman"
    reference_id = f"LNSGroup/MS-Human-700/{_XML}@{_COMMIT}"
    default_size = 128
    action_dim = _NU
    action_low, action_high = 0.0, 1.0
    state_dim = _NQ
    state_layout = (
        ("qpos", _NQ, 0),
        ("qvel", _NV, None),
        ("act", _NA, None),
        ("time", 1, None),
        ("warmstart", _NV, None),
    )

    def __init__(self, num_envs: int = 1, **kwargs):
        super().__init__(num_envs, **kwargs)
        self._renderers: dict[int, tuple] = {}
        if self.device.type == "cuda":
            require_warp_cuda("MS-Human-700 GPU physics")
        self._wp = "cpu" if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"
        self._model = _model()
        dims = (self._model.nq, self._model.nv, self._model.na, self._model.nu)
        if dims != (_NQ, _NV, _NA, _NU):
            raise ValueError(f"MS-Human-700 model dimensions changed: {dims}")
        data = mujoco.MjData(self._model)
        mujoco.mj_resetDataKeyframe(self._model, data, 0)
        mujoco.mj_forward(self._model, data)
        self._init = {k: getattr(data, k).copy() for k in ("qpos", "qvel", "act", "qacc_warmstart")}
        with wp.ScopedDevice(self._wp):
            self._m = mjw.put_model(self._model)
            self._d = mjw.put_data(
                self._model, data, nworld=self.B,
                nconmax=self.B * self._model.nconmax, njmax=self.B * self._model.njmax,
            )

    def make_variation_space(self) -> Dict:
        return Dict({})

    def _buf(self, name: str) -> torch.Tensor:
        return wp.to_torch(getattr(self._d, name))

    def reset(self, seed: int | None = None):
        self._check_open("reset")
        with wp.ScopedDevice(self._wp):
            for name, value in self._init.items():
                self._buf(name).copy_(torch.as_tensor(value, device=self._wp))
            self._buf("ctrl").zero_()
            self._buf("time").zero_()
            mjw.forward(self._m, self._d)
            wp.synchronize()
        return self.get_state()

    def get_state(self):
        self._check_open("get_state")
        fields = {
            "qpos": self._buf("qpos"),
            "qvel": self._buf("qvel"),
            "act": self._buf("act"),
            "time": self._buf("time").reshape(self.B, 1),
            "warmstart": self._buf("qacc_warmstart"),
        }
        return TensorDict(
            {k: v.clone().to(self.device, torch.float32) for k, v in fields.items()},
            batch_size=[self.B], device=self.device,
        )

    def set_state(self, state):
        self._check_open("set_state")
        self._validate_state(state, batch=self.B)
        with wp.ScopedDevice(self._wp):
            for field, name in (("qpos", "qpos"), ("qvel", "qvel"), ("act", "act"),
                                ("warmstart", "qacc_warmstart"), ("time", "time")):
                buf = self._buf(name)
                buf.copy_(state[field].reshape(buf.shape).to(self._wp, buf.dtype))
            mjw.forward(self._m, self._d)
            wp.synchronize()

    def _step(self, actions):
        self._check_open("step")
        action = torch.as_tensor(actions, device=self._wp).reshape(self.B, self.action_dim)
        with wp.ScopedDevice(self._wp):
            self._buf("ctrl").copy_(action.to(self._buf("ctrl").dtype))
            mjw.step(self._m, self._d)
            wp.synchronize()
        return self._step_result(self.get_state())

    def _renderer(self, size: int):
        hit = self._renderers.get(size)
        if hit is None:
            hit = (mujoco_renderer(self._model, height=size, width=size), mujoco.MjData(self._model))
            self._renderers[size] = hit
        return hit

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        require_contiguous_out(f"{type(self).__name__}.render", out)
        states = render_states_arg(self, states)
        size = int(size or self.size)
        renderer, data = self._renderer(size)
        rows = states.reshape(-1, self.state_dim).detach().cpu().numpy()
        frames = np.empty((len(rows), size, size, 3), np.uint8)
        for i, qpos in enumerate(rows):
            data.qpos[:] = qpos
            mujoco.mj_forward(self._model, data)
            renderer.update_scene(data)
            frames[i] = renderer.render()
        result = torch.as_tensor(frames, device=self.device).reshape(*states.shape[:-1], size, size, 3)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def _close(self):
        for renderer, _data in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        self._m = self._d = None
