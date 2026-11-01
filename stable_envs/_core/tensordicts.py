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


# Byte alignment of each leaf in the transfer buffer; a multiple of every element size.
_ALIGN = 64


def to_device_batched(td: TensorDict, device) -> TensorDict:
    """Return ``td`` on ``device`` with one host-to-device copy for all of its CPU leaves.

    The moved leaves are views into one new device buffer, so they keep their values, dtypes and
    shapes and share no memory with ``td``. Leaves already on ``device`` stay as they are.
    """
    device = _device(device)
    leaves: dict[int, torch.Tensor] = {}
    for value in td.values(True, True):
        if value.device.type == "cpu" and id(value) not in leaves:
            leaves[id(value)] = value
    offsets, total = {}, 0
    for key, value in leaves.items():
        offsets[key] = total
        total += -(-value.numel() * value.element_size() // _ALIGN) * _ALIGN
    host = torch.empty(total, dtype=torch.uint8)
    for key, value in leaves.items():
        size = value.numel() * value.element_size()
        if size:
            host[offsets[key] : offsets[key] + size] = value.contiguous().reshape(-1).view(torch.uint8)
    moved = host.to(device)

    def place(value):
        key = id(value)
        if key not in leaves:
            return value
        size = value.numel() * value.element_size()
        return moved[offsets[key] : offsets[key] + size].view(value.dtype).view(value.shape)

    return td.apply(place, device=device)
