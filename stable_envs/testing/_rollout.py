from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .._dynamics import has_dynamics


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
    """Draw seeded uniform actions within configured or environment bounds."""

    def __init__(self, low=None, high=None, seed: int = 0, **kw):
        super().__init__(**kw)
        self.low, self.high, self.seed = low, high, int(seed)
        self._gen = None

    def set_env(self, env) -> None:
        self.env = env
        if self.low is None:
            self.low = getattr(env, "action_low", -1.0)
        if self.high is None:
            self.high = getattr(env, "action_high", 1.0)
        self._gen = torch.Generator(device=env.device).manual_seed(self.seed)

    def reset(self, batch: int) -> None:
        if self.env is not None:
            self._gen = torch.Generator(device=self.env.device).manual_seed(self.seed)

    def get_action(self, obs, **kw) -> torch.Tensor:
        env = self.env
        lo = torch.as_tensor(self.low, device=env.device, dtype=torch.float32)
        hi = torch.as_tensor(self.high, device=env.device, dtype=torch.float32)
        u = torch.rand(env.B, env.action_dim, generator=self._gen, device=env.device)
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
    full State for init_states, even when outputs live on CUDA.
    """
    if not has_dynamics(env):
        raise NotImplementedError(
            f"{type(env).__name__} ships a renderer but no batched GPU simulator, so it cannot generate "
            f"trajectories. That is a deliberate split: rendering and simulation are separable, and an "
            f"env earns a step() only where its dynamics can be ported AND checked. Either supply "
            f"states you already have, or roll out the official environment and render the states it "
            f"produces."
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
            env.set_state(init_states)  # Requires full State, never a lossy corpus row.

        def render_rows():
            return env._pack_render_state(env.get_state())
    else:
        env._set_render_state(init_states if init_states is not None else env.sample_states(B), style=style)
        render_rows = env._get_render_state
    st = [render_rows().clone()]
    acts = []
    for t in range(int(steps)):
        a = policy.get_action(render_rows(), t=t)
        a = torch.as_tensor(a, device=env.device, dtype=torch.float32).reshape(B, -1)
        env.step(a, frames=None)
        acts.append(a)
        st.append(render_rows().clone())

    out = {
        "states": torch.stack(st, dim=1),
        "actions": torch.stack(acts, dim=1) if acts else None,
        "style": env.get_style().clone(),
    }
    if render:
        # frames[:, t] is strided; assign the result rather than passing it as out=.
        first = env._render(out["states"][:, 0], style=out["style"])
        frames = torch.empty((B, int(steps) + 1, *first.shape[1:]), dtype=torch.uint8, device=env.device)
        frames[:, 0] = first
        for t in range(1, int(steps) + 1):
            frames[:, t] = env._render(out["states"][:, t], style=out["style"])
        out["frames"] = frames
    return out
