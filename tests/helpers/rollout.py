from __future__ import annotations

from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs._core.dynamics import has_dynamics
from stable_envs.testing._sampling import has_render_rows, render_any


class BasePolicy:
    """Test-policy interface: get_action receives batched render rows and returns batched actions."""

    def __init__(self, **kwargs: Any) -> None:
        self.env = None

    def set_env(self, env) -> None:
        self.env = env

    def reset(self, batch: int) -> None:
        """Clear policy state at the start of a rollout; the base implementation does nothing."""

    def get_action(self, obs: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Draw seeded uniform actions within configured or environment bounds.

    ``discrete`` chooses the action type: ``False`` draws floats uniform in ``[low, high)``
    (bounds default to the env's ``action_low``/``action_high``); ``True`` draws integers in
    the inclusive ``[low, high]`` range (bounds default to the env's ``action_spec``) with
    the spec's dtype. ``None`` decides once, in ``set_env``, from the action spec's dtype.
    """

    def __init__(self, low=None, high=None, seed: int = 0, discrete: bool | None = None, **kw):
        super().__init__(**kw)
        self.low, self.high, self.seed = low, high, int(seed)
        self.discrete = discrete
        self._given = (low, high)  # set_env replaces low/high with the bounds it adopts
        self._range = None
        self._gen = None

    def set_env(self, env) -> None:
        self.env = env
        spec = getattr(env, "action_spec", None)
        dtype = getattr(spec, "dtype", torch.float32)
        discrete = (not dtype.is_floating_point) if self.discrete is None else self.discrete
        given_low, given_high = self._given
        if discrete:
            low = int(spec.low if given_low is None else given_low)
            high = int(spec.high if given_high is None else given_high)
            self._range = (low, high, dtype if not dtype.is_floating_point else torch.int64)
        else:
            low = getattr(env, "action_low", -1.0) if given_low is None else given_low
            high = getattr(env, "action_high", 1.0) if given_high is None else given_high
            self._range = (low, high, torch.float32)
        self.low, self.high = low, high
        self._discrete = discrete
        self._gen = torch.Generator(device=env.device).manual_seed(self.seed)

    def reset(self, batch: int) -> None:
        if self.env is not None:
            self._gen = torch.Generator(device=self.env.device).manual_seed(self.seed)

    def get_action(self, obs, **kw) -> torch.Tensor:
        env = self.env
        low, high, dtype = self._range
        shape = (env.B, env.action_dim)
        if self._discrete:
            return torch.randint(low, high + 1, shape, generator=self._gen, device=env.device, dtype=dtype)
        lo = torch.as_tensor(low, device=env.device, dtype=dtype)
        hi = torch.as_tensor(high, device=env.device, dtype=dtype)
        u = torch.rand(*shape, generator=self._gen, device=env.device)
        return lo + u * (hi - lo)


class ScriptedPolicy(BasePolicy):
    """Adapt a callable fn(render_rows, t) to the test-policy interface."""

    def __init__(self, fn, **kw):
        super().__init__(**kw)
        self.fn = fn

    def get_action(self, obs, t: int = 0, **kw) -> torch.Tensor:
        return self.fn(obs, t)


@torch.no_grad()
def sample(
    env,
    steps: int = 100,
    batch: int | None = None,
    policy: BasePolicy | None = None,
    style=None,
    render: bool = False,
    seed: int | None = None,
    init_states=None,
) -> dict:
    """Collect a seeded test rollout as render projections and actions.

    Return states [B, T+1, D], actions [B, T, A] (None when T=0), and style.
    With render=True, also return uint8 frames [B, T+1, H, W, 3]. These stored
    projections cannot resume simulation. Delegated CPU environments require
    full State for init_states, even when outputs live on CUDA. For an env
    without render rows, states is a TensorDict with batch [B, T+1] that holds
    the State fields whose shape is the same at every step.
    """
    if not has_dynamics(env):
        raise NotImplementedError(
            f"{type(env).__name__} has a renderer but no batched simulator, so it cannot generate "
            f"trajectories. Supply states you already have, or roll out the official environment and "
            f"render the states it produces."
        )

    B = int(batch or env.B)
    if B != env.B:
        raise ValueError(f"batch {B} != env batch {env.B}; construct the env with num_envs={B}")
    if seed is not None:
        env.rng = np.random.default_rng(seed)

    policy = policy or RandomPolicy(seed=seed or 0)
    policy.set_env(env)
    policy.reset(B)

    if getattr(env, "execution_mode", None) == "DELEGATED_CPU":
        if style is not None:
            env.set_variation(style)
        env.reset(seed=seed)
        if init_states is not None:
            env.set_state(init_states)

        def render_rows():
            return env._pack_render_state(env.get_state()) if has_render_rows(env) else env.get_state()
    else:
        env._set_render_state(init_states if init_states is not None else env.sample_states(B), style=style)
        render_rows = env._get_render_state
    st = [render_rows().clone()]
    acts = []
    for t in range(int(steps)):
        a = policy.get_action(render_rows(), t=t)
        a = torch.as_tensor(a, device=env.device)
        a = (a if not a.is_floating_point() else a.to(torch.float32)).reshape(B, -1)
        env.step(a, frames=None)
        acts.append(a)
        st.append(render_rows().clone())

    out = {
        "states": _stack_steps(st),
        "actions": torch.stack(acts, dim=1) if acts else None,
        "style": env.get_style().clone(),
    }
    if render:
        # frames[:, t] is strided; assign the result rather than passing it as out=.
        draw = env._render if has_render_rows(env) else lambda states, style: render_any(env, states)
        first = draw(out["states"][:, 0], style=out["style"])
        frames = torch.empty((B, int(steps) + 1, *first.shape[1:]), dtype=torch.uint8, device=env.device)
        frames[:, 0] = first
        for t in range(1, int(steps) + 1):
            frames[:, t] = draw(out["states"][:, t], style=out["style"])
        out["frames"] = frames
    return out


def _stack_steps(states: list):
    """Stack per-step rows, or the per-step State fields whose shape does not change over time."""
    if not isinstance(states[0], TensorDict):
        return torch.stack(states, dim=1)
    keys = [k for k, v in states[0].items(True, True) if all(s[k].shape == v.shape for s in states[1:])]
    return torch.stack([s.select(*keys) for s in states], dim=1)
