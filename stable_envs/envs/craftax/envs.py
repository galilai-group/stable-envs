"""Craftax and Craftax-Classic: the whole batch in one stacked JAX pytree, advanced by one jitted vmap.

Batch row ``i`` follows stable-worldmodel's ``CraftaxWrapper`` built with ``seed + i``. State holds each
leaf of the upstream ``EnvState`` under ``engine``, the lane's ``EnvParams`` variation values under
``physics`` and the wrapper's PRNG key under ``key``, one row per environment; uint32 PRNG words are stored bit for bit as int32. Frames are Craftax's pixel observation
scaled to uint8 as stable-worldmodel renders it. The symbolic tasks render the same frames from their
state with the pixel renderer of their game, and return the symbolic vector as ``observation``.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.config import get_config, log
from ..._core.stepping import _stack_states
from ..._core.style import StyleTensor
from ..._core.tensordicts import build_tensordict
from ...env import Env, require_contiguous_out
from ...registry import register
from ...spaces import Box, Dict, Discrete
from .params import CLASSIC_PARAMS, CRAFTAX_PARAMS, INT_RANGES

if importlib.util.find_spec("craftax") is None or importlib.util.find_spec("jax") is None:
    raise ImportError("Craftax needs the craftax extra: pip install 'stable-envs[craftax]'")

# JAX reserves most of the GPU when its GPU backend starts and leaves torch little of it. Importing
# stable_envs sets this before the caller's own JAX code can start the backend.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


class CraftaxEnv(Env):
    """Batched Craftax lanes over the upstream JAX simulator."""

    family = "craftax"
    action_dim = 1
    action_dtype = torch.int64
    action_low = 0
    # XLA steps on threads of its own, so the bookkeeping around a CPU step runs in one torch thread.
    native_step_threads = True
    _step_outputs_owned = True
    _frames_owned = True
    visual_fidelity = {
        "mse_mean": 0.0,
        "mse_p95": 0.0,
        "mse_max": 0.0,
        "measured_at": (
            "native size, per-frame MSE over pixels and channels on the 0-255 scale, over 2048 frames: 8 "
            "environments stepped 256 times with default_rng(20260811) actions from reset(seed=20260811), against "
            "upstream Craftax 1.6.1 single-environment jit frames on the same device; NVIDIA RTX A5000, "
            "jax 0.11.2, torch 2.14.1+cu126. The CPU frames give the same values"
        ),
        "reference_versions": {"craftax": "1.6.1"},
        "measured_with": {"jax": "0.11.2", "torch": "2.14.1+cu126"},
    }
    visual_note = (
        "The frames are upstream Craftax's pixel observation, drawn by a port of its renderer for the whole "
        "batch in one vmap and truncated to uint8 as stable-worldmodel does. They equal upstream frames bit "
        "for bit on CUDA and on the CPU."
    )

    def __init__(self, num_envs=1, size=None, supersample=1, device=None, backend="auto", seed=0):
        from .kernel import kernel

        if size not in (None, self.default_size) or supersample != 1:
            raise ValueError(
                f"{type(self).__name__} draws Craftax's own {self.frame_shape} frame; size and supersample are fixed"
            )
        if backend not in ("auto", "jax"):
            raise ValueError(f"{type(self).__name__} runs on the JAX backend, not {backend!r}")
        self._k = kernel(self.reference_id)
        super().__init__(num_envs, size, supersample, device, backend, seed)
        self._device = self._k.device(self.device)
        self._keys = self._k.seed_keys(int(seed), self.B, self._device)
        self._state = None
        self._varied = False
        #: The device flag of the last step with device actions, read by the next step, reset or set_state.
        self._pending = None

    @staticmethod
    def resolve_device(device: str | None) -> str:
        """Resolve the device as ``Env`` does, but let ``"auto"`` pick the CPU when JAX has no GPU backend.

        Args:
            device: A device string, ``"auto"``, or None for the global ``default_device``.

        Returns:
            The device string to use.
        """
        requested = device if device is not None else (get_config()["default_device"] or "auto")
        resolved = Env.resolve_device(device)
        if str(requested) == "auto" and resolved.startswith("cuda"):
            from .kernel import has_gpu

            if not has_gpu():
                log(
                    "INFO",
                    "JAX has no GPU backend, so device='auto' chose the CPU for Craftax. "
                    "Install the craftax-gpu extra to step Craftax on CUDA.",
                )
                return "cpu"
        return resolved

    def make_variation_space(self):
        """Upstream ``EnvParams`` fields as physics leaves; each lane takes its values at reset."""
        leaves: dict[str, dict] = {}
        for path, _, dtype, default in self._k.params:
            group, name = path.split(".")
            if dtype == np.bool_:
                space = Discrete(2, init_value=int(default))
            elif dtype == np.int32:
                low, high = INT_RANGES[path]
                space = Discrete(high - low + 1, start=low, init_value=int(default))
            else:
                space = Box(low=0.0, high=1.0, shape=(1,), init_value=[default])
            leaves.setdefault(group, {})[name] = space
        return Dict({group: Dict(spaces) for group, spaces in leaves.items()})

    def sample_variation(self, batch=None, paths=None, rng=None):
        """Sample variation rows as ``Env`` does, with integer leaves inside their ``[start, start + n)`` range."""
        out = super().sample_variation(batch, paths, rng)
        for path, offset, _ in self.variation_space.flat_layout():
            node = self.variation_space._walk(path)
            if hasattr(node, "n") and (paths is None or path in paths):
                out[:, offset] += int(node.start)
        return out

    def _lane_values(self) -> dict[str, np.ndarray]:
        """The installed variation as per-lane parameter values."""
        rows = self.get_variation().detach().cpu().numpy()
        columns = {path: offset for path, offset, _ in self.variation_space.flat_layout()}
        out = {}
        for path, _, dtype, _ in self._k.params:
            column = rows[: self.B, columns[path]]
            out[path] = np.rint(column).astype(dtype) if dtype != np.float32 else column.astype(np.float32)
        return out

    def _adopt_values(self, values: Mapping[str, np.ndarray]) -> None:
        """Install the parameter values of a restored state as the variation."""
        rows = self.get_variation().clone()
        columns = {path: offset for path, offset, _ in self.variation_space.flat_layout()}
        for path, value in values.items():
            rows[:, columns[path]] = torch.as_tensor(value.astype(np.float32), device=rows.device)
        self._style = StyleTensor(rows, self.variation_space.flat_layout())
        self._variation_is_default = False

    def _layout(self):
        return tuple((f.key, f.width, None) for f in self._k.fields)

    def _layout_dtypes(self):
        return {f.key: f.dtype for f in self._k.fields}

    def _live(self, what: str):
        self._check_open(what)
        if self._state is None:
            raise RuntimeError(f"{type(self).__name__}.{what}: call reset() first")
        return self._state

    def _snapshot(self, views, batch_size=None) -> TensorDict:
        tree: dict = {}
        for key, value in views.items():
            node = tree
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = value
        return build_tensordict(tree, [self.B] if batch_size is None else batch_size, self.device)

    def _stack_states(self, values):
        """Stack the packed blocks behind step snapshots instead of each leaf."""
        rows = [self._k.bases(value) for value in values]
        if any(row is None for row in rows):
            return _stack_states(values)
        blocks = [torch.stack(column, dim=1) for column in zip(*rows, strict=True)]
        return self._snapshot(self._k.split(blocks), [self.B, len(values)])

    def _checked(self, state, what: str, batch: int | None = None):
        self._check_open(what)
        if not isinstance(state, TensorDict) or len(state.batch_size) != 1:
            raise TypeError(f"{type(self).__name__}.{what}: expected a State TensorDict with one batch dimension")
        if batch is not None and state.batch_size[0] != batch:
            raise ValueError(
                f"{type(self).__name__}.{what}: expected State batch [{batch}], got {list(state.batch_size)}"
            )
        return state

    def get_state(self):
        """An independent snapshot: each leaf a ``(num_envs, width)`` view of blocks only it uses."""
        return self._snapshot(self._k.export(self._live("get_state"), self._keys))

    def set_state(self, state):
        """Restore every lane from a state with ``num_envs`` lanes; the state is copied."""
        self._check_pending()
        state = self._checked(state, "set_state", self.B)
        values = self._k.values(state)
        for path, value in values.items():
            space = self.variation_space._walk(path)
            low, high = (space.start, space.start + space.n - 1) if hasattr(space, "n") else (space.low, space.high)
            if not ((value >= low) & (value <= high)).all():
                raise ValueError(f"State field physics.{path} is outside [{low}, {high}]")
        self._state, self._keys = self._k.restore(state, self.device)
        self._varied = self._k.varied(values)
        self._adopt_values(values)

    def reset(self, seed=None, options=None):
        """Reset every lane; lane ``i`` of ``reset(seed)`` is stable-worldmodel's reset with ``seed + i``."""
        self._check_open("reset")
        self._check_pending()
        self.draw_reset_values(seed, options)
        if seed is not None:
            self._keys = self._k.seed_keys(int(seed), self.B, self._device)
        values = self._lane_values()
        self._varied = self._k.varied(values)
        self._state, self._keys, views = self._k.reset(self._keys, values, self._device)
        return self._snapshot(views)

    def _actions(self, actions):
        """Device actions, and whether only the step itself can check their range.

        Host actions are checked here. Device actions outside the range are clamped to one step past
        it, so int32 cannot wrap them into the range, and the step rejects them on the device.
        """
        if torch.is_tensor(actions) and actions.device.type != "cpu" and self.device.type != "cpu":
            a = actions.to(self.device).reshape(self.B)
            if a.is_floating_point():
                raise self._invalid_actions()
            return a.to(torch.int64).clamp(-1, self.action_high + 1), True
        host = actions.cpu() if torch.is_tensor(actions) else torch.as_tensor(np.asarray(actions))
        host = host.reshape(self.B)
        if host.is_floating_point() or bool(((host < 0) | (host > self.action_high)).any()):
            raise self._invalid_actions()
        if self.device.type == "cpu":
            return host, False
        # Pageable host memory is staged before the call returns, so the copy need not wait for the stream.
        return host.to(self.device, non_blocking=not host.is_pinned()), False

    def _invalid_actions(self):
        return ValueError(f"{type(self).__name__}: actions must be integers in [0, {self.action_high}]")

    def _check_pending(self):
        """Raise for a step whose device actions were out of range; that step left every lane unchanged."""
        pending, self._pending = self._pending, None
        if pending is not None and self._k.read_flag(pending):
            raise ValueError(
                f"{type(self).__name__}: actions must be integers in [0, {self.action_high}]; "
                "the last step got others and left every environment unchanged"
            )

    def _advance(self, actions, active, snapshot: bool, frame: bool = False):
        self._check_open("step")
        self._check_pending()
        state = self._live("step")
        a, deferred = self._actions(actions)
        out = self._k.step(self._keys, state, a, active, self._device, snapshot, frame, self._varied)
        self._state, self._keys, reward, done, observation, views, frames, invalid = out
        if deferred:
            # A JAX read waits for this step only; a torch read would wait for all queued GPU work.
            invalid.copy_to_host_async()
            self._pending = invalid
        result = {
            "reward": reward,
            "terminated": done,
            "truncated": torch.zeros(self.B, dtype=torch.bool, device=self.device),
        }
        if views is not None:
            result["state"] = self._snapshot(views)
        if observation is not None:
            result["observation"] = observation
        if frames is not None:
            result["frame"] = frames
        return build_tensordict(result, [self.B], self.device)

    def _step(self, actions, *, active=None):
        return self._advance(actions, active, snapshot=True)

    def _advance_without_snapshot(self, actions, *, active=None):
        return self._advance(actions, active, snapshot=False)

    def _step_pipeline(self):
        """Steps that draw their frame in the same jitted call when the caller selects that frame."""
        pipeline = self.__dict__.get("_fused_steps")
        if pipeline is None:
            pipeline = self.__dict__["_fused_steps"] = _FusedSteps(self)
        return pipeline

    def _render_state_input(self, state, *, size=None, supersample=None, variation=None, style=None, out=None):
        """Draw ``(N, H, W, 3)`` uint8 frames from a state of any batch, or from the live lanes.

        Every leaf is physics, so ``variation`` is checked and does not change the frames.
        """
        name = type(self).__name__
        if size not in (None, self.default_size) or supersample not in (None, 1):
            raise ValueError(
                f"{name}.render draws Craftax's own {self.frame_shape} frame; size and supersample are fixed"
            )
        require_contiguous_out(f"{name}.render", out)
        if state is None:
            frames = self._k.render(self._live("render"))
        else:
            frames = self._k.render_state(self._checked(state, "render"), self.device)
        if variation is not None or style is not None:
            self.resolve_style(style, variation, batch=len(frames))
        if out is None:
            return frames
        target = torch.as_tensor(out)
        if tuple(target.shape) != tuple(frames.shape):
            raise ValueError(f"{name}.render: out has shape {tuple(target.shape)}, the frames {tuple(frames.shape)}")
        target.copy_(frames)
        return out

    def _render_live(self):
        return self._k.render(self._live("render"))

    def _close(self):
        self._state = None
        self._pending = None


class _FusedSteps:
    """The step pipeline of a Craftax env: each submitted step runs when it is received."""

    live_frames = True

    def __init__(self, env: CraftaxEnv):
        self._env = env
        self._pending: tuple | None = None

    def submit_step(self, actions, *, active=None, capture_state=True, capture_frame=False) -> None:
        self._pending = (actions, active, capture_state, capture_frame)

    def receive_step(self) -> TensorDict:
        if self._pending is None:
            raise RuntimeError("receive_step() without a submitted step")
        actions, active, snapshot, frame = self._pending
        self._pending = None
        return self._env._advance(actions, active, snapshot=snapshot, frame=frame)

    def abort_step(self) -> None:
        self._pending = None


_CLASSIC_LEAVES = frozenset(path for path, _ in CLASSIC_PARAMS)
_CRAFTAX_LEAVES = frozenset(path for path, _ in CRAFTAX_PARAMS)


@register
class CraftaxClassicPixels(CraftaxEnv):
    """Craftax-Classic; the frame is the observation."""

    reference_id = "Craftax-Classic-Pixels-v1"
    frame_shape = (63, 63)
    default_size = native_size = 63
    action_high = 16
    dynamics_variation_leaves = reset_value_paths = reset_sample_paths = _CLASSIC_LEAVES


@register
class CraftaxClassicSymbolic(CraftaxEnv):
    """Craftax-Classic with its 1345-wide symbolic observation; frames come from the pixel renderer."""

    reference_id = "Craftax-Classic-Symbolic-v1"
    frame_shape = (63, 63)
    default_size = native_size = 63
    action_high = 16
    dynamics_variation_leaves = reset_value_paths = reset_sample_paths = _CLASSIC_LEAVES


@register
class CraftaxPixels(CraftaxEnv):
    """Craftax; the frame is the observation."""

    reference_id = "Craftax-Pixels-v1"
    frame_shape = (130, 110)
    default_size = native_size = 130
    action_high = 42
    dynamics_variation_leaves = reset_value_paths = reset_sample_paths = _CRAFTAX_LEAVES


@register
class CraftaxSymbolic(CraftaxEnv):
    """Craftax with its 8268-wide symbolic observation; frames come from the pixel renderer."""

    reference_id = "Craftax-Symbolic-v1"
    frame_shape = (130, 110)
    default_size = native_size = 130
    action_high = 42
    dynamics_variation_leaves = reset_value_paths = reset_sample_paths = _CRAFTAX_LEAVES
