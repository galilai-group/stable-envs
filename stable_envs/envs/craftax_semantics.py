"""Four bounded source configurations over the real Classic and full Craftax engines."""

from __future__ import annotations

import torch

from ..env import Env
from ..registry import register
from ..spaces import Dict
from ..specs import CompositeSpec, Spec


class CraftaxSemantics(Env):
    family = "craftax"
    domain = "craftax"
    execution_mode = "DELEGATED_CPU"
    state_dim = 0
    state_names = ()
    action_dim = 1
    action_dtype = torch.int64
    action_low = 0
    fidelity_tier = "parity"
    CLASSIC = False
    PIXELS = False
    default_size = 10
    default_supersample = 1

    def __init__(self, num_envs=1, size=None, supersample=1, device="cpu", backend="auto", seed=0):
        from .._adapters.craftax import CraftaxAdapter

        if size is not None and (type(size) is not int or size != self.default_size) or supersample != 1:
            raise ValueError("Craftax recovery supports only the registered tile size and supersample=1")
        if backend not in {"auto", "jax"}:
            raise ValueError("Craftax semantics delegates to the upstream JAX backend")
        self._simulator = CraftaxAdapter(self.CLASSIC, self.PIXELS, num_envs, self.resolve_device(device), seed)
        super().__init__(num_envs, size, supersample, device, backend, seed)
        self.execution_mode = "DELEGATED_GPU" if self.device.type == "cuda" else "DELEGATED_CPU"
        self.action_high = self._simulator.upstream.num_actions - 1
        self.frame_shape = (63, 63) if self.CLASSIC else (130, 110)

    def make_variation_space(self):
        return Dict({})

    def set_variation(self, variation=None, **overrides):
        self._check_open("set_variation")
        if overrides or variation is not None:
            raise ValueError("Registered Craftax configurations expose no Variation controls")
        return self._style

    @property
    def state_spec(self):
        tree = {}
        for key, (shape, dtype) in self._simulator.fields.items():
            node = tree
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = Spec(shape, dtype=dtype, device=self.device)
        return CompositeSpec(tree, device=self.device)

    def _layout(self):
        import math

        return tuple((k, math.prod(s), None) for k, (s, _) in self._simulator.fields.items())

    def _layout_dtypes(self):
        return {k: dtype for k, (_, dtype) in self._simulator.fields.items()}

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        return self._simulator.reset(seed, options)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def step(self, actions):
        self._check_open("step")
        state = self._simulator.step(actions)
        return self._step_result(
            state,
            reward=self._simulator.reward,
            terminated=self._simulator.terminated,
            truncated=self._simulator.truncated,
            done=self._simulator.terminated,
        )

    def transition(self, state, action):
        self._check_open("transition")
        saved = self.get_state()
        try:
            self.set_state(state)
            return self.step(action)["state"]
        finally:
            self.set_state(saved)

    def render(self, state=None, **kwargs):
        raise NotImplementedError("Craftax rendering integration is pending; symbolic variants do not produce frames")

    def sample_states(self, batch, rng=None):
        raise NotImplementedError("Craftax snapshot sampling requires an explicit upstream trajectory")

    def _close(self):
        self._simulator.close()


@register
class CraftaxClassicPixels(CraftaxSemantics):
    """Classic registered pixel task with exhaustive delegated JAX simulator State."""

    CLASSIC = True
    PIXELS = True
    default_size = 7
    reference_id = "Craftax-Classic-Pixels-v1"


@register
class CraftaxClassicSymbolic(CraftaxSemantics):
    """Classic symbolic task; learner output selection remains outside stable-envs."""

    CLASSIC = True
    default_size = 7
    reference_id = "Craftax-Classic-Symbolic-v1"


@register
class CraftaxPixels(CraftaxSemantics):
    """Full Craftax registered pixel task with exhaustive delegated JAX State."""

    PIXELS = True
    reference_id = "Craftax-Pixels-v1"


@register
class CraftaxSymbolic(CraftaxSemantics):
    """Full Craftax symbolic task; symbolic vectors cannot restore dynamics."""

    reference_id = "Craftax-Symbolic-v1"
