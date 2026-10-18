"""Build TensorDicts from values already on the right device and batch, without per-leaf validation."""

from __future__ import annotations

import torch
from tensordict import TensorDict


def _device(device) -> torch.device:
    # The device TensorDict(..., device=device) records.
    device = torch.device(device)
    if device.index is None:
        if device.type == "cuda":
            return torch.device("cuda", torch.cuda.current_device())
        if device.type not in ("cpu", "meta"):
            return torch.device(device.type, 0)
    return device


def _fits(source: dict, batch: torch.Size, device: torch.device) -> bool:
    n = len(batch)
    for key, value in source.items():
        if type(key) is not str:
            return False
        cls = type(value)
        if cls is dict:
            if not _fits(value, batch, device):
                return False
        elif cls is torch.Tensor:
            if value.device != device or value.shape[:n] != batch:
                return False
        elif cls is TensorDict:
            if value.device != device or value.batch_size[:n] != batch or value._has_names():
                return False
        else:
            return False
    return True


def build_tensordict(source: dict, batch_size, device) -> TensorDict:
    """``TensorDict(source, batch_size, device=device)``; skips the per-leaf checks when every leaf already fits."""
    batch = torch.Size(batch_size)
    target = _device(device)
    if _fits(source, batch, target):
        return TensorDict._new_unsafe(source, batch_size=batch, device=target)
    return TensorDict(source, batch_size=batch, device=device)
