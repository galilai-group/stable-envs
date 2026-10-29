"""MS-Human-700 musculoskeletal locomotion on MuJoCo Warp."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from ..._core.mujoco_gl import mujoco_renderer
from ...env import Env, render_states_arg, require_contiguous_out
from ...registry import register
from ...spaces import Dict
from .assets import _COMMIT, _XML
from .assets import model as _model
from .simulation.adapter import WarpAdapter

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
        self._model = _model()
        dims = (self._model.nq, self._model.nv, self._model.na, self._model.nu)
        if dims != (_NQ, _NV, _NA, _NU):
            raise ValueError(f"MS-Human-700 model dimensions changed: {dims}")
        self._simulator = WarpAdapter(self.B, self.device, self._model)

    def make_variation_space(self) -> Dict:
        return Dict({})

    def reset(self, seed: int | None = None):
        self._check_open("reset")
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._validate_state(state, batch=self.B)
        self._simulator.set_state(state)

    def _step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

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
        if hasattr(self, "_simulator"):
            self._simulator.close()
