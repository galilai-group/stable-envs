"""Build TensorDicts from values already on the right device and batch, without per-leaf validation."""

from __future__ import annotations

import math

import numpy as np
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


def _direct_node(source: dict, batch: torch.Size, device) -> TensorDict:
    node = TensorDict.__new__(TensorDict)
    node._device = device
    node._tensordict = source
    node._batch_size = batch
    node._td_dim_names = None
    return node


def _direct_nodes_match() -> bool:
    """Whether ``_direct_node`` sets exactly the attributes ``TensorDict._new_unsafe`` sets, in this tensordict."""
    try:
        batch, value = torch.Size([1]), torch.zeros(1)
        made = TensorDict._new_unsafe({"x": value}, batch_size=batch, device=torch.device("cpu"))
        direct = _direct_node({"x": value}, batch, torch.device("cpu"))
        return (
            type(made._tensordict) is dict
            and vars(made).keys() == vars(direct).keys()
            and all(
                getattr(made, k) is getattr(direct, k) or getattr(made, k) == getattr(direct, k)
                for k in vars(made)
                if k != "_tensordict"
            )
        )
    except Exception:  # noqa: BLE001 - any difference keeps the library constructor
        return False


_DIRECT_NODES = _direct_nodes_match()


def new_node(source: dict, batch: torch.Size, device) -> TensorDict:
    """``TensorDict._new_unsafe(source, batch_size=batch, device=device)`` for a dict of finished children.

    ``source`` becomes the node's storage. When this tensordict's constructor sets only the attributes checked at
    import, the node is built directly, which costs about half as much.
    """
    if _DIRECT_NODES:
        return _direct_node(source, batch, device)
    return TensorDict._new_unsafe(source, batch_size=batch, device=device)


def nested_builder(paths):
    """``build(leaves, node)``: the nested TensorDict from leaves in ``paths`` order; ``node(dict)`` makes one node."""
    tree: dict = {}
    for index, key in enumerate(paths):
        node = tree
        for part in key[:-1]:
            node = node.setdefault(part, {})
        if not isinstance(node, dict) or key[-1] in node:
            raise ValueError(f"State key {key} clashes with another key")
        node[key[-1]] = index

    def source(node):
        items = (f"{name!r}: {source(c) if type(c) is dict else f'L[{c}]'}" for name, c in node.items())
        return "N({" + ", ".join(items) + "})"

    return eval("lambda L, N: " + source(tree))  # noqa: S307 - the source holds only key literals


def _build(source: dict, batch: torch.Size, device: torch.device, index: int | None) -> TensorDict | None:
    """Wrap ``source`` in one pass when every leaf already fits; None when a leaf needs validation.

    ``index`` is ``Tensor.get_device()`` of a tensor on ``device`` (-1 on CPU), or None to compare devices.
    """
    n = len(batch)
    # A tensor fits when its leading dims equal the batch; one batch dim is the common case.
    rows = batch[0] if n == 1 else None
    out = {}
    for key, value in source.items():
        if type(key) is not str:
            return None
        cls = type(value)
        if cls is torch.Tensor:
            # get_device() returns an int, without building a torch.device for each leaf.
            if (value.get_device() != index) if index is not None else (value.device != device):
                return None
            if rows is not None:
                if not value.ndim or value.size(0) != rows:
                    return None
            elif value.shape[:n] != batch:
                return None
        elif cls is dict:
            value = _build(value, batch, device, index)
            if value is None:
                return None
        elif cls is TensorDict:
            if value.device != device or value.batch_size[:n] != batch or value._has_names():
                return None
        else:
            return None
        out[key] = value
    return new_node(out, batch, device)


def build_tensordict(source: dict, batch_size, device) -> TensorDict:
    """``TensorDict(source, batch_size, device=device)``; skips the per-leaf checks when every leaf already fits."""
    batch = torch.Size(batch_size)
    device = _device(device)
    index = -1 if device.type == "cpu" else device.index if device.type == "cuda" else None
    built = _build(source, batch, device, index)
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


def _stack_node(nodes, batch, device, view):
    out = {}
    single = len(nodes) == 1
    for key, value in nodes[0]._tensordict.items():
        if single and type(value) is torch.Tensor:
            out[key] = value.unsqueeze(1) if view else torch.stack((value,), 1)
            continue
        items = [node._tensordict[key] for node in nodes]
        if type(value) is torch.Tensor:
            out[key] = torch.stack(items, 1)
        elif _dense(value):
            out[key] = _stack_node(items, batch, device, view)
        else:
            raise _Irregular
    return TensorDict._new_unsafe(out, batch_size=batch, device=device)


def stack_leaves(values, *, view: bool = False) -> TensorDict:
    """Stack TensorDicts of one structure along a new dim 1, leaf by leaf.

    With ``view`` and one dense TensorDict, each leaf is a view of the input leaf, without a copy. Use it only for
    a TensorDict that the caller owns.
    """
    first = values[0]
    batch = torch.Size([*first.batch_size, len(values)])
    if all(_dense(value) and value.device == first.device for value in values):
        try:
            return _stack_node(values, batch, first.device, view)
        except (_Irregular, KeyError):
            pass
    return first.apply(lambda *leaves: torch.stack(leaves, dim=1), *values[1:], batch_size=batch)


_NUMPY = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
}


def _leaves(node, prefix, out, batch, kept, device):
    """Collect the CPU leaves of ``node`` in ``out`` and the leaves already on ``device`` in ``kept``."""
    if node.batch_size != batch:
        raise _Irregular
    for key, value in node._tensordict.items():
        if type(value) is torch.Tensor:
            if value.device == device:
                kept.append(((*prefix, key), value))
            elif value.dtype not in _NUMPY or value.device.type != "cpu" or value.requires_grad:
                raise _Irregular
            else:
                out.append(((*prefix, key), value))
        elif _dense(value) and value._tensordict:
            _leaves(value, (*prefix, key), out, batch, kept, device)
        else:
            raise _Irregular
    return out


#: Byte alignment of each dtype group in the pinned transfer buffer; a multiple of every dtype's size.
_ALIGN = 16
#: ``_Transfer`` per leaf structure that ``to_device`` moved; cleared when it holds this many.
_TRANSFERS_HELD = 64
_transfers: dict = {}


class _Transfer:
    """Where the leaves of one TensorDict structure go in the pinned buffer, and how the device copy splits.

    Each dtype is one block; inside it, leaves of one shape sit next to each other, so one ``unbind`` splits them.
    """

    def __init__(self, signature, kept_keys=()):
        groups: dict = {}
        for index, (_, shape, dtype) in enumerate(signature):
            groups.setdefault(dtype, {}).setdefault(tuple(shape), []).append(index)
        self.groups, total = [], 0
        for dtype, runs in groups.items():
            order = [i for indices in runs.values() for i in indices]
            sizes = [len(indices) * math.prod(shape) for shape, indices in runs.items()]
            nbytes = sum(sizes) * torch.empty((), dtype=dtype).element_size()
            self.groups.append((dtype, order, total, total + nbytes, sizes, list(runs.items())))
            total += -(-nbytes // _ALIGN) * _ALIGN
        self.total = max(total, _ALIGN)
        self.build = nested_builder([key for key, _, _ in signature] + list(kept_keys))


def to_device(td: TensorDict, device) -> TensorDict:
    """``td.to(device)`` for a CPU TensorDict, with one host-to-device copy for all its leaves.

    The leaves are packed by dtype into one pinned buffer, which moves in one asynchronous copy; each returned leaf
    is a view of the device copy. Leaves already on ``device`` stay as they are. A TensorDict that isn't dense, or
    that has a leaf NumPy can't hold, moves leaf by leaf.
    """
    device = _device(device)
    if device.type != "cuda" or not _dense(td) or td.device != torch.device("cpu"):
        return td.to(device)
    kept: list = []
    try:
        leaves = _leaves(td, (), [], td.batch_size, kept, device)
    except _Irregular:
        return td.to(device)
    # Leaves already on the device stay as they are, as with ``td.to``; only their keys enter the plan.
    signature = (tuple((key, value.shape, value.dtype) for key, value in leaves), tuple(key for key, _ in kept))
    plan = _transfers.get(signature)
    if plan is None:
        if len(_transfers) >= _TRANSFERS_HELD:
            _transfers.clear()
        plan = _transfers[signature] = _Transfer(*signature)
    parts: list = [None] * len(leaves)
    batch = td.batch_size
    if not leaves:
        return plan.build([value for _, value in kept], lambda node: new_node(node, batch, device))
    host = torch.empty(plan.total, dtype=torch.uint8, pin_memory=True)
    packed = host.numpy()
    for _, order, start, stop, _, _ in plan.groups:
        packed[start:stop] = torch._C._nn.flatten_dense_tensors([leaves[i][1] for i in order]).numpy().view(np.uint8)
    # The pinned block isn't reused before this copy ends: the host allocator records the copy's stream.
    moved = host.to(device, non_blocking=True)
    for dtype, _, start, stop, sizes, runs in plan.groups:
        block = moved[start:stop].view(dtype)
        for (shape, indices), piece in zip(runs, block.split_with_sizes(sizes) if len(runs) > 1 else (block,)):
            if len(indices) == 1:
                parts[indices[0]] = piece.view(shape)
            else:
                for i, leaf in zip(indices, piece.view(len(indices), *shape).unbind(0), strict=True):
                    parts[i] = leaf
    parts.extend(value for _, value in kept)
    return plan.build(parts, lambda node: new_node(node, batch, device))
