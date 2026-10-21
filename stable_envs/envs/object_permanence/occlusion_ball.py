"""Public state-driven environment for the authored G34 nominal trajectory."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from tensordict import TensorDict

from ...env import Env
from ...registry import register
from ...spaces import Dict
from .rendering.g34_native import render_rows
from .scenes.g34 import FRAME_COUNT, authored_pose
from .simulation.g34 import INTEGRATION, OcclusionBallAdapter


@register
class MuJoCoOcclusionBall(Env):
    """G34 nominal, source-authored ball occlusion on CPU MuJoCo.

    The only action is zero: advance one of 119 intervals at 24 FPS. Reset is
    source frame 1; step 119 reaches frame 120 and truncates without autoreset.
    This is prescribed animation, not contact dynamics or a controllable RL task.

    State: physics.integration is MuJoCo's float64 mjSTATE_INTEGRATION vector;
    lifecycle.elapsed_steps is int64 [1]. Model and configuration must match on
    restore. No random draws occur. RenderState is float32 xyz + wxyz [7].

    RGB is native MuJoCo and deliberately differs from Blender. Source fidelity
    is certified by geometry depth/masks and poses under the alignment contract;
    backend="raycast" uses CPU or CUDA faceted ray hits over a cached static plate;
    its ball shading differs from native RGB. Default auto selects native OpenGL.
    """

    family, domain = "object_permanence", "occlusion_ball"
    name = "mujoco-occlusion-ball"
    reference_id = "mujoco/G34/nominal"
    reference_versions = {"mujoco": "3.12.0"}
    # Legacy parity metadata refers only to this native MuJoCo renderer. The
    # original Blender source has an explicitly separate geometry-only contract.
    geometry_reference = "object-permanence/G34/nominal@824baddcdad087810d35638bf8f273b884f74975"
    state_dim = 7
    state_names = ("ball_x", "ball_y", "ball_z", "ball_qw", "ball_qx", "ball_qy", "ball_qz")
    default_size = 90
    action_dim, action_dtype = 1, torch.float64
    action_low = action_high = 0.0
    execution_mode = "DELEGATED_CPU"
    seed_independent_trajectory = True  # the source nominal animation has no random reset or control

    def __init__(self, num_envs=1, size=None, device="cpu", backend="auto", seed=0, supersample=1):
        device = self.resolve_device(device)
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("MuJoCoOcclusionBall supports CPU and CUDA output")
        if torch.device(device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA output requires a CUDA-enabled PyTorch and an available GPU")
        if backend not in ("auto", "opengl", "raycast") or supersample != 1:
            raise ValueError("Supported backends are opengl and raycast with supersample=1")
        height = self.default_size if size is None else size
        if not isinstance(height, int) or isinstance(height, bool) or height < 9 or height % 9:
            raise ValueError("size is image height and must be a positive multiple of 9")
        super().__init__(num_envs=num_envs, size=height, device=device, backend=backend, seed=seed)
        self.frame_shape = (height, height * 16 // 9)
        self._simulator = OcclusionBallAdapter(num_envs, self.device)
        self.reset(seed=seed)

    def make_variation_space(self):
        return Dict({})

    def _layout(self):
        return tuple((key, width, None) for key, (width, _) in self._simulator.fields.items())

    def _layout_dtypes(self):
        return {key: dtype for key, (_, dtype) in self._simulator.fields.items()}

    def _validate_state(self, state, *, batch=None):
        self._check_open("state")
        self._simulator.validate_state(state, batch=batch)

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        if options:
            raise ValueError("G34 nominal has no reset overrides")
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step(self, actions, *, active=None):
        return self._simulator.step(actions, active=active)

    def _pack_render_state(self, state):
        offset = self._simulator.pose_offset
        # Convert on the source device first: CPU replay uploads 28 bytes per
        # lane, rather than a strided float64 integration vector or RGB image.
        return state[INTEGRATION][:, offset : offset + 7].to(dtype=torch.float32).to(self.device)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, 7)
        return TensorDict({"ball_position": rows[:, :3].clone(), "ball_quaternion": rows[:, 3:].clone()}, [len(rows)])

    def sample_states(self, batch, rng=None):
        rng = self.rng if rng is None else rng
        return np.concatenate(authored_pose(rng.integers(0, FRAME_COUNT, size=batch)), axis=-1).astype(np.float32)

    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        if supersample not in (None, 1):
            raise ValueError("Only supersample=1 is supported")
        for value in (variation, style):
            if value is None:
                continue
            if isinstance(value, Mapping):
                nonempty = bool(value)
            else:
                nonempty = value.numel() != 0 if torch.is_tensor(value) else np.asarray(value).size != 0
            if nonempty:
                raise ValueError("G34 nominal has no appearance overrides")
        height = self.size if size is None else size
        if not isinstance(height, int) or height < 9 or height % 9:
            raise ValueError("size must be a positive multiple of 9")
        lead = tuple(states.shape[:-1])
        width = height * 16 // 9
        if out is not None:
            if not torch.is_tensor(out) or tuple(out.shape) != (*lead, height, width, 3):
                raise ValueError("out must match the requested frame shape")
            flat_out = out.reshape(-1, height, width, 3)
        else:
            flat_out = None
        images = render_rows(
            states, height=height, width=width, device=self.device, out=flat_out, backend=self.backend
        )
        return out if out is not None else images.reshape(*lead, height, width, 3)

    def geometry(self, state=None):
        """Diagnostic camera-axis depth and visible object IDs from the RGB scene."""
        self._check_open("geometry")
        state = self.get_state() if state is None else state
        self._validate_state(state)
        rows = self._pack_render_state(state)
        options = dict(height=self.frame_shape[0], width=self.frame_shape[1], device=self.device, backend=self.backend)
        depth = render_rows(rows, mode="depth", **options)
        ids = render_rows(rows, mode="ids", **options)
        depth.masked_fill_(ids == 0, 0)
        return dict(depth=depth, object_ids=ids)

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
