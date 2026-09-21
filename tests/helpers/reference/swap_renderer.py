"""Validation helper: official dynamics, stable-envs pixels.

This is a certification/experiment tool, not a public sibling-library API.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from stable_envs.registry import make


def swap_renderer(
    gym_env,
    se_name: str,
    read_state: Callable[[Any], np.ndarray],
    style=None,
    size: int | None = None,
    device: str = "cuda",
):
    """Keep an official env's dynamics and replace only its ``render()``."""
    env = make(se_name, num_envs=1, size=size, device=device)
    if style is not None:
        env.set_style(np.asarray(style, np.float32).reshape(1, -1))
    u = getattr(gym_env, "unwrapped", gym_env)
    original = u.render

    def render(*a, **kw):
        return env.render(np.asarray(read_state(u), np.float32).reshape(1, -1))[0].cpu().numpy()

    u.render = render
    u._stable_envs_original_render = original
    u._stable_envs_renderer = env
    return gym_env


def pusht_state(u) -> np.ndarray:
    """``swm/PushT-v1`` / ``gym_pusht/PushT-v0`` -> the 5-float PushT render state."""
    return np.array([*u.agent.position, *u.block.position, u.block.angle], np.float64)
