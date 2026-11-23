"""Build TensorDicts from values already on the right device and batch, without per-leaf validation."""

from __future__ import annotations

import math
import operator
import sys
import threading

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


def node_factory():
    """``new_node`` itself, or the function it calls when that is the direct constructor: one call less per node."""
    return _direct_node if _DIRECT_NODES else new_node


def nested_builder(paths, *, with_paths: bool = False, indices=None):
    """``build(leaves, node)``: the nested TensorDict from leaves in ``paths`` order; ``node(dict)`` makes one node.

    With ``with_paths``, ``node(dict, path)`` also gets the key path of the node, ``()`` for the root. With
    ``indices``, the leaf of ``paths[k]`` is ``leaves[indices[k]]``.
    """
    tree: dict = {}
    for index, key in zip(range(len(paths)) if indices is None else indices, paths, strict=True):
        node = tree
        for part in key[:-1]:
            node = node.setdefault(part, {})
        if not isinstance(node, dict) or key[-1] in node:
            raise ValueError(f"State key {key} clashes with another key")
        node[key[-1]] = index

    def source(node, path):
        items = (
            f"{name!r}: {source(c, (*path, name) if with_paths else ()) if type(c) is dict else f'L[{c}]'}"
            for name, c in node.items()
        )
        return "N({" + ", ".join(items) + "}" + (f", {path!r})" if with_paths else ")")

    return eval("lambda L, N: " + source(tree, ()))  # noqa: S307 - the source holds only key literals


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


def _one_storage(td) -> bool:
    """Whether every leaf of a dense ``td`` uses one storage; stops at the first leaf that doesn't."""
    pointer = None
    stack = [td._tensordict]
    while stack:
        for value in stack.pop().values():
            if type(value) is torch.Tensor:
                current = value.untyped_storage().data_ptr()
                if pointer is None:
                    pointer = current
                elif current != pointer:
                    return False
            elif _dense(value):
                stack.append(value._tensordict)
            else:
                return False
    return pointer is not None


def _clone_packed(slots) -> bool:
    """Replace each ``(parent, key, leaf)`` slot by a copy, in one copy; False when the leaves aren't packed.

    Packed leaves lie inside one small span of one storage, as an export makes them.
    """
    first = slots[0][2]
    storage = first.untyped_storage()
    spans = [
        (value.storage_offset() * value.element_size(), value)
        for _, _, value in slots
        if value.is_contiguous() and not value.requires_grad
    ]
    shared = all(value.untyped_storage().data_ptr() == storage.data_ptr() for _, value in spans)
    if len(spans) == len(slots) and shared:
        lo = min(start for start, _ in spans)
        hi = max(start + value.nbytes for start, value in spans)
        aligned = all((start - lo) % value.element_size() == 0 for start, value in spans)
        if aligned and hi - lo <= 2 * sum(value.nbytes for _, value in spans):
            whole = torch.empty(0, dtype=torch.uint8, device=first.device).set_(storage)
            copy = whole[lo:hi].clone().untyped_storage()
            # Leaves of one dtype, shape and stride at evenly spaced offsets come from one view and one unbind.
            i = 0
            while i < len(slots):
                start, value = spans[i]
                j, step = i + 1, None
                while j < len(slots):
                    after, other = spans[j]
                    gap = after - spans[j - 1][0]
                    like = (
                        other.dtype == value.dtype and other.shape == value.shape and other.stride() == value.stride()
                    )
                    if not like or gap <= 0 or gap % value.element_size() or (step is not None and gap != step):
                        break
                    step, j = gap, j + 1
                size = value.element_size()
                base = torch.empty(0, dtype=value.dtype, device=value.device)
                if j - i == 1:
                    slots[i][0][slots[i][1]] = base.set_(copy, (start - lo) // size, value.shape, value.stride())
                else:
                    run = base.set_(copy, (start - lo) // size, (j - i, *value.shape), (step // size, *value.stride()))
                    for (parent, key, _), leaf in zip(slots[i:j], run.unbind(0), strict=True):
                        parent[key] = leaf
                i = j
            return True
    return False


def stack_leaves(values, *, view: bool = False) -> TensorDict:
    """Stack TensorDicts of one structure along a new dim 1, leaf by leaf.

    With ``view`` and one dense TensorDict, each leaf is a view of the input leaf, without a copy. Use it only for
    a TensorDict that the caller owns.
    """
    first = values[0]
    batch = torch.Size([*first.batch_size, len(values)])
    if all(_dense(value) and value.device == first.device for value in values):
        try:
            if len(values) == 1 and not view and _one_storage(first):
                # One copy of packed leaves instead of a copy per leaf.
                out = _stack_node(values, batch, first.device, True)
                slots = _leaf_slots(out, [])
                # A single leaf costs less to stack than to copy through its storage.
                if len(slots) > 1 and _clone_packed(slots):
                    return out
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


def _leaves(node, prefix, out, batch, kept, device, batches):
    """Collect the CPU leaves of ``node`` in ``out`` and the leaves already on ``device`` in ``kept``.

    ``batches`` gets ``(path, batch size)`` of each node whose batch size extends the root's, such as a time axis.
    """
    if node.batch_size != batch:
        if node.batch_size[: len(batch)] != batch:
            raise _Irregular
        batch = node.batch_size
        batches.append((prefix, batch))
    for key, value in node._tensordict.items():
        if type(value) is torch.Tensor:
            if value.device == device:
                kept.append(((*prefix, key), value))
            elif value.dtype not in _NUMPY or value.device.type != "cpu" or value.requires_grad:
                raise _Irregular
            else:
                out.append(((*prefix, key), value))
        elif _dense(value) and value._tensordict:
            _leaves(value, (*prefix, key), out, batch, kept, device, batches)
        else:
            raise _Irregular
    return out


#: Byte alignment of each dtype group in the transfer buffer; a multiple of every dtype's size.
_ALIGN = 16
#: ``_Transfer`` per leaf structure that ``to_device`` or ``pack_to_device`` moved; cleared when it holds this many.
_TRANSFERS_HELD = 64
_transfers: dict = {}
_TORCH = {np.dtype(kind): dtype for dtype, kind in _NUMPY.items()}
_SHAPE = operator.attrgetter("shape")
_DTYPE = operator.attrgetter("dtype")


class _Transfer:
    """Where the leaves of one TensorDict structure go in the transfer buffer, and how the device copy splits.

    Each dtype is one block; inside it, leaves of one shape sit next to each other, so one ``unbind`` splits them.
    ``signature`` holds ``(key, shape, torch dtype)`` per moved leaf; ``kept_keys`` the keys of leaves already on the
    device, which follow the moved ones; ``batches`` ``(node path, batch size)`` of each node whose batch size
    extends its parent's.
    """

    def __init__(self, signature, kept_keys=(), batches=(), copies=None):
        groups: dict = {}
        for index, (_, shape, dtype) in enumerate(signature):
            groups.setdefault(dtype, {}).setdefault(tuple(shape), []).append(index)
        self.groups, total = [], 0
        #: ``(dtype, first byte, shape)`` of each moved leaf, by signature index.
        self.places: dict = {}
        for dtype, runs in groups.items():
            order = [i for indices in runs.values() for i in indices]
            sizes = [len(indices) * math.prod(shape) for shape, indices in runs.items()]
            nbytes = sum(sizes) * dtype.itemsize
            self.groups.append((dtype, _NUMPY[dtype], order, total, total + nbytes, sizes, list(runs.items())))
            offset = total
            for shape, indices in runs.items():
                for i in indices:
                    self.places[i] = (dtype, offset, shape)
                    offset += math.prod(shape) * dtype.itemsize
            total += -(-nbytes // _ALIGN) * _ALIGN
        #: ``(key, source index, shape)`` of each mirror: a copy of a moved leaf below node ``copies[0]``, under
        #: ``copies[1]`` with a time axis of length one, cut from a second copy of the packed leaves that follows
        #: them in the buffer.
        self.mirrors = ()
        if copies is not None:
            source, target, before = copies
            depth = len(source)
            self.mirrors = tuple(
                ((*target, *key[depth:]), i, (shape[0], 1, *shape[1:]))
                for i, (key, shape, _) in enumerate(signature)
                if key[:depth] == source
            )
        self.main = max(total, _ALIGN)
        self.total = 2 * self.main if self.mirrors else self.main
        self.count = len(signature) + len(self.mirrors)
        self.buffers: dict[int, np.ndarray] = {}
        #: Per thread: device blocks of earlier calls with their leaves and the counts that show them unused.
        self.held: dict[int, list] = {}
        keys = [key for key, _, _ in signature] + [key for key, _, _ in self.mirrors] + list(kept_keys)
        indices = list(range(len(keys)))
        if self.mirrors:
            # The copies sit before key ``before`` in the result, as they would if they were moved leaves.
            first = len(signature)
            at = next((k for k, key in enumerate(keys[:first]) if key == before), first)
            copied = indices[first : first + len(self.mirrors)]
            indices = indices[:at] + copied + indices[at:first] + indices[first + len(self.mirrors) :]
            keys = [keys[i] for i in indices]
        self.build = nested_builder(keys, with_paths=True, indices=indices)
        #: Batch size of every node below the root whose batch size differs from the root's.
        self.batches: dict = {}
        explicit = dict(batches)
        for key in keys:
            batch = None
            for depth in range(1, len(key)):
                path = key[:depth]
                batch = explicit.get(path, batch)
                if batch is not None:
                    self.batches[path] = batch

    def staging(self) -> np.ndarray:
        """This thread's host buffer for the packed leaves.

        The buffer is pageable: a copy from pageable memory returns once the driver has staged the bytes, so the
        buffer can be packed again right after, with no pinned block or event per copy.
        """
        thread = threading.get_ident()
        buffer = self.buffers.get(thread)
        if buffer is None:
            buffer = self.buffers[thread] = np.empty(self.total, np.uint8)
        return buffer

    def send(self, packed: np.ndarray, device: torch.device) -> list:
        """The leaves of a device copy of ``packed``, in signature order.

        The copy goes into the block of an earlier call when no tensor uses that block or its leaves any more, as
        after the caller dropped that result: its leaves come back as they are, with no allocation and no new
        tensor. Work on the device that read the block is queued before the copy.
        """
        source = torch.from_numpy(packed)
        pool = self.held.setdefault(threading.get_ident(), [])
        for block, parts, uses, refs in pool:
            if block.device == device and _uses(block) == uses and _refs(parts) == refs:
                block.copy_(source, non_blocking=True)
                return parts
        block = source.to(device, non_blocking=True)
        parts = [None] * self.count
        self.split(block, parts)
        if _COUNTS_USES:
            pool.append((block, parts, _uses(block), _refs(parts)))
            if len(pool) > 2:
                pool.pop(0)
        return parts

    def host_parts(self, packed: np.ndarray) -> list:
        """The leaves of ``packed`` as CPU tensors, in signature order; they share its memory."""
        parts: list = [None] * self.count
        for _, kind, _, start, stop, sizes, runs in self.groups:
            block, offset = packed[start:stop].view(kind), 0
            for (shape, indices), size in zip(runs, sizes, strict=True):
                piece = block[offset : offset + size].reshape(len(indices), *shape)
                offset += size
                for i, row in zip(indices, piece, strict=True):
                    parts[i] = torch.from_numpy(row)
        first = self.count - len(self.mirrors)
        for j, (_, source, shape) in enumerate(self.mirrors):
            dtype, start, _ = self.places[source]
            start += self.main
            stop = start + math.prod(shape) * dtype.itemsize
            parts[first + j] = torch.from_numpy(packed[start:stop].view(_NUMPY[dtype]).reshape(shape))
        return parts

    def node(self, batch, device):
        """``node(dict, path)`` for the builder: a node with the batch size of its path."""
        batches = self.batches
        if not batches:
            return lambda source, path: new_node(source, batch, device)
        return lambda source, path: new_node(source, batches.get(path, batch), device)

    def split(self, moved, parts):
        """Fill ``parts`` with views of the device copy ``moved``, one per moved leaf."""
        for dtype, _, _, start, stop, sizes, runs in self.groups:
            block = moved[start:stop].view(dtype)
            for (shape, indices), piece in zip(runs, block.split_with_sizes(sizes) if len(runs) > 1 else (block,)):
                if len(indices) == 1:
                    parts[indices[0]] = piece.view(shape)
                else:
                    for i, leaf in zip(indices, piece.view(len(indices), *shape).unbind(0), strict=True):
                        parts[i] = leaf
        first = self.count - len(self.mirrors)
        for j, (_, source, shape) in enumerate(self.mirrors):
            dtype, start, _ = self.places[source]
            start += self.main
            parts[first + j] = moved[start : start + math.prod(shape) * dtype.itemsize].view(dtype).view(shape)


def _uses(tensor) -> int:
    return torch._C._storage_Use_Count(tensor.untyped_storage()._cdata)


def _refs(parts) -> int:
    return sum(sys.getrefcount(part) for part in parts)


#: Whether this torch counts the users of a storage; without it, no device block is reused.
_COUNTS_USES = hasattr(torch._C, "_storage_Use_Count")


def _plan(signature) -> _Transfer:
    plan = _transfers.get(signature)
    if plan is None:
        if len(_transfers) >= _TRANSFERS_HELD:
            _transfers.clear()
        plan = _transfers[signature] = _Transfer(*signature)
    return plan


def to_device(td: TensorDict, device) -> TensorDict:
    """``td.to(device)`` for a CPU TensorDict, with one host-to-device copy for all its leaves.

    The leaves are packed by dtype into one host buffer, which moves in one copy; each returned leaf is a view of
    the device copy, which can be the block of an earlier result that nothing uses any more. Leaves already on ``device`` stay as they are. A TensorDict that isn't dense, or
    that has a leaf NumPy can't hold, moves leaf by leaf.
    """
    device = _device(device)
    if td.device == device:
        # Every leaf of a TensorDict that records a device is on it: there is nothing to move.
        return td
    if device.type != "cuda" or not _dense(td) or td.device != torch.device("cpu"):
        return td.to(device)
    kept: list = []
    batches: list = []
    try:
        leaves = _leaves(td, (), [], td.batch_size, kept, device, batches)
    except _Irregular:
        return td.to(device)
    # Leaves already on the device stay as they are, as with ``td.to``; only their keys enter the plan.
    signature = (
        tuple((key, value.shape, value.dtype) for key, value in leaves),
        tuple(key for key, _ in kept),
        tuple(batches),
    )
    plan = _plan(signature)
    parts: list = []
    if leaves:
        packed = plan.staging()
        for _, kind, order, start, stop, _, _ in plan.groups:
            torch.cat([leaves[i][1].reshape(-1) for i in order], out=torch.from_numpy(packed[start:stop].view(kind)))
        parts = plan.send(packed, device)
    if kept:
        parts = [*parts, *(value for _, value in kept)]
    return plan.build(parts, plan.node(td.batch_size, device))


def leaf_tree(leaves) -> dict:
    """The nested dict of CPU tensors of ``{key path: array}``, for ``build_tensordict``."""
    tree: dict = {}
    for key, value in leaves.items():
        node = tree
        for part in key[:-1]:
            node = node.setdefault(part, {})
        node[key[-1]] = torch.from_numpy(value)
    return tree


def pack_to_device(leaves, batch, device, batches=(), copies=None) -> TensorDict:
    """The TensorDict of host NumPy ``leaves``, ``[(key path, array)]``, on ``device`` in one copy.

    The arrays are packed by dtype straight into one host buffer, with no intermediate tensor per leaf. On an
    accelerator each returned leaf is a view of the device copy, which can be the block of an earlier result that
    nothing uses any more; on the CPU each leaf is a view of a new buffer that the result owns. ``batches`` holds
    ``(node path, batch size)`` of each node whose batch size extends its parent's; the batch size of a node is its
    parent's unless listed. ``copies``, ``(source node path, target node path, key)``, adds a copy of every leaf
    below the source node under the target node, before ``key`` in the result, with a time axis of length one after
    the batch axis. The buffer then holds the packed leaves twice, written with one memory copy, and the copies
    come from the second half, so they share no memory with their sources.
    """
    device = _device(device)
    batch = torch.Size(batch)
    keys, arrays = zip(*leaves)
    # The structure of the leaves, read without a Python step per leaf.
    signature = (keys, tuple(map(_SHAPE, arrays)), tuple(map(_DTYPE, arrays)), tuple(batches), copies)
    plan = _transfers.get(signature)
    if plan is None:
        if len(_transfers) >= _TRANSFERS_HELD:
            _transfers.clear()
        layout = tuple(zip(keys, signature[1], (_TORCH[kind] for kind in signature[2])))
        plan = _transfers[signature] = _Transfer(layout, (), signature[3], signature[4])
    cpu = device.type == "cpu"
    # On the CPU the leaves are views of a new buffer, which the result owns.
    packed = np.empty(plan.total, np.uint8) if cpu else plan.staging()
    for _, kind, order, start, stop, _, _ in plan.groups:
        np.concatenate([arrays[i].reshape(-1) for i in order], out=packed[start:stop].view(kind))
    if plan.mirrors:
        packed[plan.main :] = packed[: plan.main]
    parts = plan.host_parts(packed) if cpu else plan.send(packed, device)
    return plan.build(parts, plan.node(batch, device))


def splice_rows(rows: torch.Tensor, fresh: TensorDict, kept: TensorDict, history=()) -> TensorDict:
    """A copy of ``kept`` with the batch ``rows`` (a bool mask) taken from ``fresh``.

    ``history`` names the ``/``-joined leaves whose second axis grows with the episode, such as an action history;
    they are zero-padded to the wider of the two. Any other leaf must have the same shape in both.
    """
    out = kept.clone()
    for key in fresh.keys(True, True):
        new, old = fresh.get(key), kept.get(key)
        if new.shape != old.shape:
            name = "/".join((key,) if isinstance(key, str) else key)
            if name not in history or new.ndim < 2 or new.shape[2:] != old.shape[2:]:
                raise ValueError(f"State field {name} has shape {tuple(new.shape)}, expected {tuple(old.shape)}")
            width = max(new.shape[1], old.shape[1])
            new, old = (
                torch.cat((v, v.new_zeros(v.shape[0], width - v.shape[1], *v.shape[2:])), 1) for v in (new, old)
            )
        mask = rows.to(new.device).view(-1, *[1] * (new.ndim - 1))
        out.set(key, torch.where(mask, new, old.to(new.device)))
    return out
