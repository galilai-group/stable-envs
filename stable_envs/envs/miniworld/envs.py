"""Miniworld's MovingBlockWorld, the 3D Dynamic Blockworld of Flow Equivariant World Models.

Each environment is the hlillemark/Miniworld fork's ``MovingBlockWorld``: an agent in one room of colored boxes
and balls that each move by a fixed velocity every step and bounce off walls. All environments step together
in a Numba kernel (CPU) or a Triton kernel (CUDA) that follow the fork's step rules; resets sample the fork's reset
distribution from a counter-based generator. The default ``preset="flowm"`` is the FloWM dynamic
dataset: plain colored boxes, each moving one cell per step in a cardinal direction through the agent and
each other, in a 16 x 16 grid room with 90 degree turns, unit forward steps and 500 steps.
Each environment draws 6 to 10 blocks per episode, as the dataset script does.
``preset="flowm-textured"`` is the paper's textured split (random wall, floor and block textures, and balls);
``preset="upstream"`` is the env as the fork registers it. Keyword arguments override single options of either preset.

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

from ..._core.stepping import _stack_states
from ..._execution import render_pool
from ...env import Env
from ...registry import register
from ...spaces import Dict
from .simulation.adapter import (
    ACTION_HIGH,
    FORK_COMMIT,
    FORK_URL,
    RenderRows,
    pack_rows,
    render_here,
    resolve_config,
)
from .simulation.batched import TRITON_MIN_ENVS, BatchedMovingBlockWorldAdapter, execution_mode, miniworld_backend


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
    """MovingBlockWorld environments stepped in one batch, with complete restorable State and Miniworld's own frames.

    State: the agent pose and carried block, the room textures, each block's kind, color, texture, size,
    pose and velocity, the reset generator's seed and episode, and the step count.
    """

    family = "miniworld"
    domain = "moving_blockworld"
    name = "miniworld-moving-blockworld"
    reference_id = "MiniWorld-MovingBlockWorld-v0"
    #: Numba on CPU and Triton on CUDA; ``__init__`` sets the mode of the selected backend.
    execution_mode = "AUTO_NUMBA_TRITON"
    #: Every step returns new output tensors.
    _step_outputs_owned = True
    action_dim = 1
    action_dtype = torch.int64
    action_low, action_high = 0, ACTION_HIGH
    default_size = 64

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=1,
        device=None,
        backend="auto",
        seed=0,
        preset="flowm",
        *,
        simulation_backend=None,
        triton_min_envs=TRITON_MIN_ENVS,
        **options,
    ):
        """Build ``num_envs`` environments of ``preset`` (``"flowm"``, ``"flowm-textured"`` or ``"upstream"``) updated with ``options``.

        ``options`` take ``MovingBlockWorld`` keyword arguments (``room_size`` for its ``size``), and
        ``turn_step``, ``forward_step``, ``cam_fov_y``, ``heading_zero``, ``wrap_heading`` and
        ``max_episode_steps`` as the dataset script sets them.

        ``simulation_backend`` selects the step: ``"numba"`` (a CPU kernel, one environment per thread),
        ``"triton"`` (a CUDA kernel, one environment per thread) or ``"torch"`` (vectorized Torch on the env's
        device). The default is Triton on CUDA and Numba on CPU; ``"auto"`` also steps CUDA batches of fewer than
        ``triton_min_envs`` environments with Numba. Every backend returns State and outputs on ``device``.
        """
        _require_fork()
        if supersample != 1:
            raise ValueError("Miniworld draws its frames without supersampling")
        if backend not in ("auto", "opengl"):
            raise ValueError("Miniworld draws with OpenGL; backend must be 'auto' or 'opengl'")
        self.config = resolve_config(preset, options)
        device = Env.resolve_device(device)
        self.simulation_backend = miniworld_backend(device, int(num_envs), simulation_backend, triton_min_envs)
        self.execution_mode = execution_mode(self.simulation_backend)
        self._simulator = BatchedMovingBlockWorldAdapter(
            self.config, int(num_envs), int(size or self.default_size), device, self.simulation_backend
        )
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
        """Reset every environment; environment ``i`` of ``reset(seed)`` draws from seed ``seed + i``.

        ``reset()`` draws each environment's next episode from its own seed.
        """
        self._check_open("reset")
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        """Restore every environment from a State with ``num_envs`` rows; the State is copied."""
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _stack_states(self, values):
        """One step's State: copy its two flat buffers instead of stacking each leaf."""
        stacked = self._simulator.stack_one(values[0]) if len(values) == 1 else None
        return _stack_states(values) if stacked is None else stacked

    def _step(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active)

    def render(self, state=None, *, size=None, supersample=None, variation=None, style=None, out=None):
        """Draw ``(N, size, size, 3)`` uint8 frames from a State of any batch, or from the live environments.

        ``size`` defaults to the env's; ``out`` takes a contiguous uint8 buffer of the frames' shape. Miniworld
        has no supersampling and no Variation, so ``supersample`` other than 1, ``variation`` and ``style``
        raise. A State batch of at least ``render_worker_min_rows`` rows draws in render workers
        (``miniworld-render`` tuning path); the frames are the same as drawn in this process.
        """
        self._check_open("render")
        if supersample not in (None, 1):
            raise ValueError("Miniworld draws its frames without supersampling")
        if variation or style is not None:
            raise ValueError("Miniworld has no Variation; render takes State only")
        size = self.size if size is None else int(size)
        if size < 1:
            raise ValueError(f"size must be a positive int, got {size}")
        state = self._simulator.get_state() if state is None else state
        self._simulator.validate(state)
        rows = pack_rows(state, self._simulator.fields)
        frames = render_pool.render_rows(
            self,
            (type(self).__name__, repr(sorted(self.config.items())), size),
            rows,
            None,
            (size, size, 3),
            "miniworld-render",
            factory=RenderRows(self.config, size),
            software_gl=True,
        )
        if frames is None:
            frames = torch.from_numpy(render_here(self.config, size, rows))
        if out is None:
            return frames.to(self.device)
        if tuple(out.shape) != tuple(frames.shape) or out.dtype != torch.uint8:
            raise ValueError(
                f"out must be a uint8 buffer of shape {tuple(frames.shape)}, got {tuple(out.shape)} {out.dtype}"
            )
        return out.copy_(frames)

    def _render_live(self):
        return self.render()

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()
