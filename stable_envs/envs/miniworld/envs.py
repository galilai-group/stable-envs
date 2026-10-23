"""Miniworld's MovingBlockWorld, the 3D Dynamic Blockworld of Flow Equivariant World Models.

Each lane runs the hlillemark/Miniworld fork's ``MovingBlockWorld``: an agent in one room of coloured boxes
and balls that each move by a fixed velocity every step, bouncing off walls and each other. The default
``preset="flowm"`` is the FloWM textured dataset configuration (16 x 16 grid room, 90 degree turns, unit
forward steps, randomised wall, floor and block textures, 500 steps); ``preset="upstream"`` is the env as
the fork registers it. Keyword arguments override single options of either preset.

The action is one of the fork's nine ids: turn left 0, turn right 1, forward 2, back 3, do nothing 4,
pick up 5, drop 6, toggle 7, done 8. Reward is always 0 and episodes end only by the step limit. Frames are
the agent view Miniworld draws with OpenGL through pyglet (``env.render()`` in ``rgb_array`` mode), at
``size`` x ``size``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

if importlib.util.find_spec("miniworld") is None or importlib.util.find_spec("pyglet") is None:
    raise ImportError("Miniworld needs the miniworld extra: pip install 'stable-envs[miniworld]'")

from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env
from ...registry import register
from ...spaces import Dict
from .simulation.adapter import (
    ACTION_HIGH,
    FORK_COMMIT,
    FORK_URL,
    MovingBlockWorldAdapter,
    render_rows,
    resolve_config,
)


def _require_fork() -> None:
    # Checked on disk: importing miniworld imports pyglet.gl, which must wait for configure_pyglet().
    locations = importlib.util.find_spec("miniworld").submodule_search_locations or ()
    if not any((Path(p) / "envs" / "moving_blockworld.py").is_file() for p in locations):
        raise ImportError(
            "MovingBlockWorld comes from the hlillemark/Miniworld fork, not the PyPI miniworld package: "
            f"pip install 'miniworld @ git+{FORK_URL}@{FORK_COMMIT}'"
        )


@register
class MiniWorldMovingBlockWorld(Env):
    """MovingBlockWorld lanes with complete restorable State and Miniworld's own frames.

    State: the agent pose and carried block, the room textures, each block's kind, colour, texture, size,
    pose and velocity, the env's PCG64 generator and the step count. ``set_state`` rebuilds a lane's room
    and blocks when its world differs from the one it holds.
    """

    family = "miniworld"
    domain = "moving_blockworld"
    name = "miniworld-moving-blockworld"
    reference_id = "MiniWorld-MovingBlockWorld-v0"
    execution_mode = "DELEGATED_CPU"
    action_dim = 1
    action_dtype = torch.int64
    action_low, action_high = 0, ACTION_HIGH
    default_size = 64

    def __init__(
        self, num_envs=1, size=None, supersample=1, device=None, backend="auto", seed=0, preset="flowm", **options
    ):
        """Build ``num_envs`` lanes of ``preset`` (``"flowm"`` or ``"upstream"``) updated with ``options``.

        ``options`` take ``MovingBlockWorld`` keyword arguments (``room_size`` for its ``size``), and
        ``turn_step``, ``forward_step``, ``cam_fov_y``, ``heading_zero``, ``wrap_heading`` and
        ``max_episode_steps`` as the dataset script sets them.
        """
        _require_fork()
        if supersample != 1:
            raise ValueError("Miniworld draws its frames without supersampling")
        if backend not in ("auto", "opengl"):
            raise ValueError("Miniworld draws with OpenGL; backend must be 'auto' or 'opengl'")
        self.config = resolve_config(preset, options)
        device = Env.resolve_device(device)
        self._simulator = MovingBlockWorldAdapter(self.config, int(num_envs), int(size or self.default_size), device)
        try:
            super().__init__(num_envs, size, supersample, device, backend, seed)
            self.reset(seed=seed)
        except BaseException:
            self._simulator.close()
            raise

    def make_variation_space(self):
        return Dict({})

    def _layout(self):
        return tuple((key, width, None) for key, (width, _) in self._simulator.fields.items())

    def _layout_dtypes(self):
        return {key: dtype for key, (_, dtype) in self._simulator.fields.items()}

    def _validate_state(self, state, *, batch=None):
        self._simulator.validate(state, batch=batch)

    def reset(self, seed=None):
        """Reset every lane; lane ``i`` of ``reset(seed)`` is ``MovingBlockWorld.reset(seed=seed + i)``."""
        self._check_open("reset")
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        """Restore every lane from a State with ``num_envs`` lanes; the State is copied."""
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active)

    def render(self, state=None):
        """Draw ``(N, size, size, 3)`` uint8 frames from a State of any batch, or from the live lanes."""
        self._check_open("render")
        if state is None:
            return self._simulator.frames()
        self._simulator.validate(state)
        return render_rows(self.config, self.size, self._simulator, state).to(self.device)

    def _render_live(self):
        return self._simulator.frames()

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
