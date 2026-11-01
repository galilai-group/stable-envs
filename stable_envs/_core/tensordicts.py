"""Build TensorDicts from values already on the right device and batch, without per-leaf validation."""

from __future__ import annotations

import threading

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
_STAGING = threading.local()


def _staging(size: int, *, pinned: bool) -> torch.Tensor:
    """Return ``size`` bytes of this thread's host staging memory; a new buffer would fault its pages each call."""
    key = "pinned" if pinned else "pageable"
    buffer = getattr(_STAGING, key, None)
    if buffer is None or buffer.numel() < size:
        buffer = torch.empty(max(size, 1 << 20), dtype=torch.uint8, pin_memory=pinned)
        setattr(_STAGING, key, buffer)
    return buffer[:size]


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
    host = _staging(total, pinned=device.type == "cuda")
    flat = host.numpy()
    for key, value in leaves.items():
        size = value.numel() * value.element_size()
        if size:
            # NumPy copies in this thread; a torch copy of a large leaf would wake the intra-op pool.
            flat[offsets[key] : offsets[key] + size] = value.contiguous().reshape(-1).view(torch.uint8).numpy()
    # The copy is synchronous, so the staging memory is free again when it returns.
    moved = host.to(device) if device.type != "cpu" else host.clone()

    def place(node):
        data = {}
        for key, value in node.items():
            if isinstance(value, TensorDict):
                data[key] = place(value)
            elif id(value) in leaves:
                start, size = offsets[id(value)], value.numel() * value.element_size()
                data[key] = moved[start : start + size].view(value.dtype).view(value.shape)
            else:
                data[key] = value
        return TensorDict._new_unsafe(data, batch_size=node.batch_size, device=device)

    return place(td)
