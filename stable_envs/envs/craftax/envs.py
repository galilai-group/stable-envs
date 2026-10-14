"""Craftax and Craftax-Classic: every lane in one stacked JAX pytree, advanced by one jitted vmap.

Lane ``i`` follows stable-worldmodel's ``CraftaxWrapper`` built with ``seed + i``. State holds each
leaf of the upstream ``EnvState`` under ``engine`` and the wrapper's PRNG key under ``key``, one row
per lane; uint32 PRNG words are stored bit for bit as int32. Frames are Craftax's pixel observation
scaled to uint8 as stable-worldmodel renders it. The symbolic tasks render the same frames from their
State with the pixel renderer of their game, and return the symbolic vector as ``observation``.
"""

from __future__ import annotations

import importlib.util

import torch
from tensordict import TensorDict

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

    def _hidden_state(self):
        return self._k.export(self._live("get_state"), self._keys)

    def _restore_state(self, state):
        self._state, self._keys = self._k.restore(state, self.device, copy=True)

    def reset(self, seed=None):
        """Reset every lane; lane ``i`` of ``reset(seed)`` is stable-worldmodel's reset with ``seed + i``."""
        self._check_open("reset")
        if seed is not None:
            self._keys = self._k.seed_keys(int(seed), self.B, self._device)
        self._state, self._keys = self._k.reset(self._keys)
        return self.get_state()

    def _step(self, actions, *, active=None):
        state = self._live("step")
        a = torch.as_tensor(actions, device=self.device).reshape(self.B)
        if a.is_floating_point() or bool(((a < 0) | (a > self.action_high)).any()):
            raise ValueError(f"{type(self).__name__}: actions must be integers in [0, {self.action_high}]")
        self._state, self._keys, reward, done, observation = self._k.step(self._keys, state, a, active)
        result = {
            "state": self.get_state(),
            "reward": torch.from_dlpack(reward),
            "terminated": torch.from_dlpack(done),
            "truncated": torch.zeros(self.B, dtype=torch.bool, device=self.device),
        }
        if observation is not None:
            result["observation"] = torch.from_dlpack(observation)
        return TensorDict(result, batch_size=[self.B], device=self.device)

    def render(self, state=None):
        """Draw ``(N, H, W, 3)`` uint8 frames from a State of any batch, or from the live lanes."""
        if state is None:
            return self._k.render(self._live("render"))
        self._check_open("render")
        self._validate_state(state)
        return self._k.render(self._k.restore(state, self.device, copy=False)[0])

    def _close(self):
        self._state = None


@register
class CraftaxClassicPixels(CraftaxEnv):
    """Craftax-Classic; the frame is the observation."""

    reference_id = "Craftax-Classic-Pixels-v1"
    frame_shape = (63, 63)
    default_size = 63
    action_high = 16


@register
class CraftaxClassicSymbolic(CraftaxEnv):
    """Craftax-Classic with its 1345-wide symbolic observation; frames come from the pixel renderer."""

    reference_id = "Craftax-Classic-Symbolic-v1"
    frame_shape = (63, 63)
    default_size = 63
    action_high = 16


@register
class CraftaxPixels(CraftaxEnv):
    """Craftax; the frame is the observation."""

    reference_id = "Craftax-Pixels-v1"
    frame_shape = (130, 110)
    default_size = 130
    action_high = 42


@register
class CraftaxSymbolic(CraftaxEnv):
    """Craftax with its 8268-wide symbolic observation; frames come from the pixel renderer."""

    reference_id = "Craftax-Symbolic-v1"
    frame_shape = (130, 110)
    default_size = 130
    action_high = 42
