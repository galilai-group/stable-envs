"""The one call through which the paper harness draws stable-envs frames from stored inputs."""

from __future__ import annotations

import torch


def draw(env, batch) -> torch.Tensor:
    """Draw ``batch`` with ``env.render``: a full State, or the rows that ``sample_states`` returns."""
    return env.render(batch)
