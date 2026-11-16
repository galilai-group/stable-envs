"""MS-Human-700 musculoskeletal locomotion: MuJoCo Warp physics on CUDA, native ``mujoco.batch`` on CPU."""

from __future__ import annotations

import threading

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ..._core.mujoco_gl import mujoco_renderer
from ..._core.state_schema import get_field
from ..._core.stepping import _stack_states, _threads
from ..._execution.backends import measure_env
from ..._execution.mjbatch import mujoco_backend
from ...env import Env, require_contiguous_out
from ...registry import register
from ...render._resources import retain_process_lifetime
from ...spaces import Dict
from .assets import _COMMIT, _XML
from .assets import model as _model
from .simulation.adapter import NA as _NA
from .simulation.adapter import NQ as _NQ
from .simulation.adapter import NU as _NU
from .simulation.adapter import NV as _NV
from .simulation.adapter import STATE, MSHumanBatchAdapter, MSHumanWarpAdapter

#: Shipped CUDA switch points: the smallest batch that steps on MuJoCo Warp; smaller batches step on ``mujoco.batch``.
#: The setting and the tuning cache (``stable-envs tune``) take precedence.
#: docs/performance-tuning.md (Physics backends) says how the shipped values were measured.
#: The simulation path, which keys the switch point in the tuning cache.
SIMULATION_PATH = "mshuman-sim/locomotion"
CUDA_MIN_ENVS: int | None = 16  # RTX 3090 8, RTX A5000 16
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
    simulation_backends = ("native", "mjwarp")
    state_solver_caches = ("warmstart",)
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

    def __init__(self, num_envs: int = 1, *, simulation_backend="auto", **kwargs):
        """Select physics at construction.

        ``simulation_backend="auto"`` steps CPU batches on native ``mujoco.batch``. On CUDA it steps small batches
        on native ``mujoco.batch`` and large ones on MuJoCo Warp, from the measured ``large_backend_min_envs``
        switch point. ``"native"`` and ``"mjwarp"`` select explicitly.
        """
        if simulation_backend not in (None, "auto", "native", "mjwarp"):
            raise ValueError("simulation_backend must be 'auto', 'native' or 'mjwarp'")
        if simulation_backend == "mjwarp" and torch.device(Env.resolve_device(kwargs.get("device"))).type != "cuda":
            raise ValueError("simulation_backend='mjwarp' requires device='cuda'")
        super().__init__(num_envs, **kwargs)
        self._model = _model()
        dims = (self._model.nq, self._model.nv, self._model.na, self._model.nu)
        if dims != (_NQ, _NV, _NA, _NU):
            raise ValueError(f"MS-Human-700 model dimensions changed: {dims}")

        device = self.device
        cls = type(self)
        measure = measure_env(lambda backend, n: cls(num_envs=n, device=device, simulation_backend=backend))
        path, requested = SIMULATION_PATH, simulation_backend
        self.simulation_backend_path = path
        self.simulation_backend = mujoco_backend(
            path, self.B, self.device, requested, cuda_min_envs=CUDA_MIN_ENVS, measure=measure
        )
        # nq == nv: every joint is a hinge or a slide with one qpos.
        self.state_names = tuple(
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            for j in range(self._model.njnt)
        )
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

    def _validate_state(self, state, *, batch=None):
        if not isinstance(state, TensorDict):
            raise TypeError("Expected full State (TensorDict); compact RenderState rows cannot restore dynamics")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
        STATE.validate(state, finite=True)

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            return super()._pack_render_state(state)
        n = int(state.batch_size[0]) if state.batch_size else 1
        qpos = get_field(state, "qpos").reshape(n, _NQ)
        return qpos.to(device=self.device, dtype=torch.float32, copy=True)

    @property
    def state_spec(self):
        return STATE.spec(self.device)

    def _stack_states(self, values):
        # Torch copies tensors over 32768 elements on OpenMP threads, which then spin against the batch's threads.
        with _threads(1):
            return _stack_states(values)

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        require_contiguous_out(f"{type(self).__name__}.render", out)
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
