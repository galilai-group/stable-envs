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


def _build(source: dict, batch: torch.Size, device: torch.device) -> TensorDict | None:
    """Wrap ``source`` in one pass when every leaf already fits; None when a leaf needs validation."""
    n = len(batch)
    # A tensor fits when its leading dims equal the batch; one batch dim is the common case.
    rows = batch[0] if n == 1 else None
    out = {}
    for key, value in source.items():
        if type(key) is not str:
            return None
        cls = type(value)
        if cls is torch.Tensor:
            if value.device != device:
                return None
            if rows is not None:
                if not value.ndim or value.shape[0] != rows:
                    return None
            elif value.shape[:n] != batch:
                return None
        elif cls is dict:
            value = _build(value, batch, device)
            if value is None:
                return None
        elif cls is TensorDict:
            if value.device != device or value.batch_size[:n] != batch or value._has_names():
                return None
        else:
            return None
        out[key] = value
    return TensorDict._new_unsafe(out, batch_size=batch, device=device)


def build_tensordict(source: dict, batch_size, device) -> TensorDict:
    """``TensorDict(source, batch_size, device=device)``; skips the per-leaf checks when every leaf already fits."""
    batch = torch.Size(batch_size)
    built = _build(source, batch, _device(device))
    return TensorDict(source, batch_size=batch, device=device) if built is None else built


class _Irregular(Exception):
    """A node the leaf walks below do not handle; the caller falls back to the TensorDict method."""


def _dense(td) -> bool:
    return type(td) is TensorDict and not td._has_names()


def _view_node(node, skip=()):
    out = {}
    for key, value in node._tensordict.items():
        if key in skip:
            continue
        if _dense(value):
            value = _view_node(value)
        elif type(value) is not torch.Tensor:
            raise _Irregular
        out[key] = value
    return TensorDict._new_unsafe(out, batch_size=node.batch_size, device=node.device)


def _leaf_slots(node, slots):
    for key, value in node._tensordict.items():
        if type(value) is torch.Tensor:
            slots.append((node._tensordict, key, value))
        else:
            _leaf_slots(value, slots)
    return slots


def exclude_leaves(td: TensorDict, exclude=(), *, clone: bool = False) -> TensorDict:
    """``td.exclude(*exclude)``, cloned when ``clone``, without per-leaf validation; keys are top-level strings."""
    if not _dense(td):
        out = td.exclude(*exclude)
        return out.clone() if clone else out
    try:
        out = _view_node(td, frozenset(exclude))
    except _Irregular:
        out = td.exclude(*exclude)
        return out.clone() if clone else out
    if clone:
        # The copies TensorDict.clone makes: one foreach add for the leaves it can add, clone for the rest.
        slots = _leaf_slots(out, [])
        added = [slot for slot in slots if slot[2].dtype is not torch.bool and not slot[2].requires_grad]
        for (parent, key, _), copy in zip(added, torch._foreach_add([slot[2] for slot in added], 0) if added else ()):
            parent[key] = copy
        for parent, key, value in slots:
            if value.dtype is torch.bool or value.requires_grad:
                parent[key] = value.clone()
    return out


def _stack_node(nodes, batch, device):
    out = {}
    single = len(nodes) == 1
    for key, value in nodes[0]._tensordict.items():
        if single and type(value) is torch.Tensor:
            out[key] = torch.stack((value,), 1)
            continue
        items = [node._tensordict[key] for node in nodes]
        if type(value) is torch.Tensor:
            out[key] = torch.stack(items, 1)
        elif _dense(value):
            out[key] = _stack_node(items, batch, device)
        else:
            raise _Irregular
    return TensorDict._new_unsafe(out, batch_size=batch, device=device)


def stack_leaves(values) -> TensorDict:
    """Stack TensorDicts of one structure along a new dim 1, leaf by leaf."""
    first = values[0]
    batch = torch.Size([*first.batch_size, len(values)])
    if all(_dense(value) and value.device == first.device for value in values):
        try:
            return _stack_node(values, batch, first.device)
        except (_Irregular, KeyError):
            pass
    return first.apply(lambda *leaves: torch.stack(leaves, dim=1), *values[1:], batch_size=batch)
