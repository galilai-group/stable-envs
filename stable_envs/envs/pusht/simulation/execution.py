"""Construct the shared CPU/CUDA JAX simulator."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from .jax_batch import PushTJaxAdapter


def make_pusht_adapter(
    num_envs: int, device: str | torch.device, seed: int | None, *, discrete: bool, config: dict
) -> PushTJaxAdapter:
    from .jax_batch import PushTJaxAdapter

    return PushTJaxAdapter(num_envs, device, seed, discrete=discrete, config=config)
