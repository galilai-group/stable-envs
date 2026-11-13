"""Craftax and Craftax-Classic: every lane in one stacked JAX pytree, advanced by one jitted vmap.

Lane ``i`` follows stable-worldmodel's ``CraftaxWrapper`` built with ``seed + i``. State holds each
leaf of the upstream ``EnvState`` under ``engine`` and the wrapper's PRNG key under ``key``, one row
per lane; uint32 PRNG words are stored bit for bit as int32. Frames are Craftax's pixel observation
scaled to uint8 as stable-worldmodel renders it. The symbolic tasks render the same frames from their
State with the pixel renderer of their game, and return the symbolic vector as ``observation``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.stepping import _stack_states
from ..._core.tensordicts import build_tensordict
from ...env import Env
from ...registry import register
from ...spaces import Dict

if importlib.util.find_spec("craftax") is None or importlib.util.find_spec("jax") is None:
    raise ImportError("Craftax needs the craftax extra: pip install 'stable-envs[craftax]'")


class CraftaxEnv(Env):
    """Batched Craftax lanes over the upstream JAX simulator."""

    family = "craftax"
    action_dim = 1
    action_dtype = torch.int64
    action_low = 0

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

    def make_variation_space(self):
        return Dict({})

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
        """Restore every lane from a State with ``num_envs`` lanes; the State is copied."""
        self._state, self._keys = self._k.restore(self._checked(state, "set_state", self.B), self.device)

    def reset(self, seed=None):
        """Reset every lane; lane ``i`` of ``reset(seed)`` is stable-worldmodel's reset with ``seed + i``."""
        self._check_open("reset")
        if seed is not None:
            self._keys = self._k.seed_keys(int(seed), self.B, self._device)
        self._state, self._keys, views = self._k.reset(self._keys)
        return self._snapshot(views)

    def _actions(self, actions):
        """Device actions, and for device inputs a pending flag that is true when any is out of range."""
        if torch.is_tensor(actions) and actions.device.type != "cpu" and self.device.type != "cpu":
            a = actions.to(self.device).reshape(self.B)
            if a.is_floating_point():
                raise self._invalid_actions()
            # Read after the step is queued, so the check does not stall the host before dispatch.
            return a, ((a < 0) | (a > self.action_high)).any()
        host = actions.cpu() if torch.is_tensor(actions) else torch.as_tensor(np.asarray(actions))
        host = host.reshape(self.B)
        if host.is_floating_point() or bool(((host < 0) | (host > self.action_high)).any()):
            raise self._invalid_actions()
        if self.device.type == "cpu":
            return host, None
        # Pageable host memory is staged before the call returns, so the copy need not wait for the stream.
        return host.to(self.device, non_blocking=not host.is_pinned()), None

    def _invalid_actions(self):
        return ValueError(f"{type(self).__name__}: actions must be integers in [0, {self.action_high}]")

    def _step(self, actions, *, active=None):
        state = self._live("step")
        a, invalid = self._actions(actions)
        out = self._k.step(self._keys, state, a, active)
        # A rejected step leaves the live lanes as they were: JAX arrays are immutable.
        if invalid is not None and bool(invalid):
            raise self._invalid_actions()
        self._state, self._keys, reward, done, observation, views = out
        result = {
            "state": self._snapshot(views),
            "reward": reward,
            "terminated": done,
            "truncated": torch.zeros(self.B, dtype=torch.bool, device=self.device),
        }
        if observation is not None:
            result["observation"] = observation
        return build_tensordict(result, [self.B], self.device)

    def render(self, state=None):
        """Draw ``(N, H, W, 3)`` uint8 frames from a State of any batch, or from the live lanes."""
        if state is None:
            return self._k.render(self._live("render"))
        return self._k.render_state(self._checked(state, "render"), self.device)

    def _render_live(self):
        return self._k.render(self._live("render"))

    def _close(self):
        self._state = None


@register
class CraftaxClassicPixels(CraftaxEnv):
    """Craftax-Classic; the frame is the observation."""

    reference_id = "Craftax-Classic-Pixels-v1"
    frame_shape = (63, 63)
    default_size = native_size = 63
    action_high = 16


@register
class CraftaxClassicSymbolic(CraftaxEnv):
    """Craftax-Classic with its 1345-wide symbolic observation; frames come from the pixel renderer."""

    reference_id = "Craftax-Classic-Symbolic-v1"
    frame_shape = (63, 63)
    default_size = native_size = 63
    action_high = 16


@register
class CraftaxPixels(CraftaxEnv):
    """Craftax; the frame is the observation."""

    reference_id = "Craftax-Pixels-v1"
    frame_shape = (130, 110)
    default_size = native_size = 130
    action_high = 42


@register
class CraftaxSymbolic(CraftaxEnv):
    """Craftax with its 8268-wide symbolic observation; frames come from the pixel renderer."""

    reference_id = "Craftax-Symbolic-v1"
    frame_shape = (130, 110)
    default_size = native_size = 130
    action_high = 42
