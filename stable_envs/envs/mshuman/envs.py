"""MS-Human-700 musculoskeletal locomotion: MuJoCo Warp physics on CUDA, native ``mujoco.batch`` on CPU."""

from __future__ import annotations

import threading

import mujoco
import numpy as np
import torch

from ..._core.mujoco_gl import mujoco_renderer
from ..._core.stepping import _stack_states, _threads
from ..._execution.mjbatch import resolve_backend
from ...env import Env, render_states_arg, require_contiguous_out
from ...registry import register
from ...render._resources import retain_process_lifetime
from ...spaces import Dict
from .assets import _COMMIT, _XML
from .assets import model as _model
from .simulation.adapter import MSHumanBatchAdapter, MSHumanWarpAdapter

_NQ, _NV, _NA, _NU = 36, 36, 100, 100
#: With ``simulation_backend="auto"``, the smallest CUDA batch that steps on MuJoCo Warp; smaller CUDA batches
#: step faster on mujoco.batch.
MJWARP_MIN_ENVS = 256
_LOCK = threading.RLock()
_RENDERERS: dict[int, tuple] = {}


def _renderer(model, size: int):
    if size not in _RENDERERS:
        entry = (mujoco_renderer(model, height=size, width=size), mujoco.MjData(model))
        retain_process_lifetime(entry, "MS-Human OpenGL context must remain alive with sibling contexts")
        _RENDERERS[size] = entry
    return _RENDERERS[size]


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

    def __init__(self, num_envs: int = 1, *, simulation_backend="default", mjwarp_min_envs=MJWARP_MIN_ENVS, **kwargs):
        """Select physics at construction.

        ``simulation_backend="default"`` steps CUDA batches on MuJoCo Warp and CPU batches on native
        ``mujoco.batch``. ``"auto"`` steps CUDA batches of ``mjwarp_min_envs`` or more on MuJoCo Warp and every
        other batch on native ``mujoco.batch``. ``"native"`` and ``"mjwarp"`` select explicitly.
        """
        resolve_backend(1, "cpu", "native", mjwarp_min_envs)
        if simulation_backend not in (None, "default", "auto", "native", "mjwarp"):
            raise ValueError("simulation_backend must be 'default', 'auto', 'native' or 'mjwarp'")
        super().__init__(num_envs, **kwargs)
        self.simulation_backend = resolve_backend(self.B, self.device, simulation_backend, mjwarp_min_envs)
        self._model = _model()
        dims = (self._model.nq, self._model.nv, self._model.na, self._model.nu)
        if dims != (_NQ, _NV, _NA, _NU):
            raise ValueError(f"MS-Human-700 model dimensions changed: {dims}")
        adapter = MSHumanWarpAdapter if self.simulation_backend == "mjwarp" else MSHumanBatchAdapter
        self._simulator = adapter(self.B, self.device, self._model)

    def make_variation_space(self) -> Dict:
        return Dict({})

    def reset(self, seed: int | None = None, options: dict | None = None):
        self._check_open("reset")
        self.draw_reset_values(seed, options)
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

    def _stack_states(self, values):
        # Torch copies tensors over 32768 elements on OpenMP threads, which then spin against the batch's threads.
        with _threads(1):
            return _stack_states(values)

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        require_contiguous_out(f"{type(self).__name__}.render", out)
        states = render_states_arg(self, states)
        size = int(size or self.size)
        rows = states.reshape(-1, self.state_dim).detach().cpu().numpy()
        frames = np.empty((len(rows), size, size, 3), np.uint8)
        with _LOCK:
            renderer, data = _renderer(self._model, size)
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
        if hasattr(self, "_simulator"):
            self._simulator.close()
