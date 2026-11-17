"""The result of one action of every environment as views of one copy of a packed device buffer.

A family whose step is a fixed sequence of device kernels captures it once as a CUDA graph (``GraphStep``). The graph
also copies the State and the outputs into one byte buffer, so each step costs an action copy, a graph replay and one
buffer copy, whatever the batch size. The result has the leaves and nodes that ``stepping`` builds for a single
action: the State, the outputs (with a time axis for a sequence), ``valid``, and the ``states`` and ``state_valid``
of the action. The State and its ``states`` copy are separate regions of the buffer, so no leaf shares memory with
another.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from tensordict import TensorDict

from .config import log
from .tensordicts import _device, node_factory

#: ``(key path, dtype, row shape, row strides, row stride, element offset)`` of one leaf in a packed buffer.
Leaf = tuple[tuple[str, ...], torch.dtype, tuple[int, ...], tuple[int, ...], int, int]


def result_builder(
    num_envs: int,
    device: torch.device,
    state: Sequence[Leaf],
    outputs: Sequence[Leaf],
    states: Sequence[Leaf],
    ones: tuple[int, int],
    *,
    sequence: bool,
    with_states: int | None,
    final_state: bool,
    shared: Sequence[tuple[str, ...]] = (),
) -> Callable[..., TensorDict]:
    """``build(buffer, extra=None)``: the step result whose leaves are views of the byte tensor ``buffer``.

    ``state`` and ``states`` lay out the two copies of the State, ``outputs`` the task outputs, and ``ones`` gives
    the element offsets of two all-true bool columns of ``num_envs`` rows: ``valid`` and ``state_valid``.
    ``with_states`` is the number of ``states`` of the result, 1 or 0, or None for none. The ``shared`` outputs
    come from ``extra`` (``{key path: (num_envs, ...) tensor}``) as they are, after the others.
    The builder is generated once per layout, so a call runs no loop over the leaves.
    """
    n = num_envs
    plan = []
    if final_state:
        for path, dtype, shape, strides, row, offset in state:
            plan.append((("state", *path), dtype, (n, *shape), (row, *strides), offset))
    time = (1,) if sequence else ()
    for path, dtype, shape, strides, row, offset in outputs:
        plan.append((path, dtype, (n, *time, *shape), (row, *(row,) * len(time), *strides), offset))
    plan += [(path, None, None, None, None) for path in shared]
    plan.append((("valid",), torch.bool, (n, 1), (1, 1), ones[0]))
    k = with_states
    if k is not None:
        for path, dtype, shape, strides, row, offset in states:
            plan.append((("states", *path), dtype, (n, k, *shape), (row, row, *strides), offset))
        plan.append((("state_valid",), torch.bool, (n, k), (1, 1), ones[1]))
    batches = {(): torch.Size([n]), ("states",): torch.Size([n, k or 0])}
    if sequence:
        batches.update({path[:1]: torch.Size([n, 1]) for path, *_ in outputs if len(path) > 1})
        batches.update({path[:1]: torch.Size([n, 1]) for path in shared if len(path) > 1})
    names: dict = {"N": node_factory(), "D": _device(device)}
    dtypes: dict = {}
    tree: dict = {}
    for path, dtype, shape, strides, offset in plan:
        parent = tree
        for part in path[:-1]:
            parent = parent.setdefault(part, {})
        if dtype is None:
            names[f"P{len(names)}"] = path
            parent[path[-1]] = f"E[P{len(names) - 1}]{'.unsqueeze(1)' if sequence else ''}"
            continue
        flat = dtypes.setdefault(dtype, f"V{len(dtypes)}")
        names[f"S{len(names)}"] = (shape, strides)
        parent[path[-1]] = f"{flat}.as_strided(*S{len(names) - 1}, {offset})"

    def source(node, path):
        size = f"B{len(names)}"
        names[size] = batches.get(path[:1], batches[()])
        items = ", ".join(f"{k!r}: {source(c, (*path, k)) if type(c) is dict else c}" for k, c in node.items())
        return f"N({{{items}}}, {size}, D)"

    body = source(tree, ())
    views = "".join(f"{flat} = X.view(T{i}); " for i, flat in enumerate(dtypes.values()))
    for i, dtype in enumerate(dtypes):
        names[f"T{i}"] = dtype
    code = f"def build(X, E=None):\n    {views}\n    return {body}\n"
    exec(code, names)  # noqa: S102 - the source holds only key literals and names of this function
    return names["build"]


def _leaves(td: TensorDict, prefix: tuple = ()) -> list[tuple[tuple[str, ...], torch.Tensor]]:
    out = []
    for key, value in td.items():
        if isinstance(value, TensorDict):
            out += _leaves(value, (*prefix, key))
        else:
            out.append(((*prefix, key), value))
    return out


class GraphStep:
    """One action of every environment, replayed as a CUDA graph whose result is one copy of a packed buffer.

    ``compute(actions, *sources)`` advances every environment by the ``(num_envs, ...)`` actions and returns a
    TensorDict with the ``state`` and the task outputs. It must only launch device work: no host reads and no shapes
    that depend on values. ``live()`` returns the tensors it writes. The first call runs ``compute`` once to learn
    the result layout and puts them back. ``key()`` returns a value that changes when ``compute`` would read or write
    other tensors; a new key captures again.

    ``sources()``, when given, returns tensors that ``compute`` reads through copies the graph owns, such as the
    Variation rows, which a new Variation replaces. A replay first copies a source that is a new tensor or was
    written since the last copy, so a new Variation of the same shape needs no new capture. ``shared()``, when
    given, returns ``{key path: tensor}`` outputs that ``compute`` leaves out and that the result holds as they
    are, such as a large image that steps never write.
    """

    def __init__(self, num_envs: int, device: torch.device, compute, live, key, *, sources=None, shared=None):
        self.num_envs, self.device = int(num_envs), _device(device)
        self._compute, self._live, self._key = compute, live, key
        self._sources, self._shared = sources, shared
        self._graphs: dict = {}
        self.failed = False

    def result(self, actions: torch.Tensor, sequence: bool, state_indices, final_state: bool) -> TensorDict | None:
        """The step result of ``actions``, or None when the step can't be captured."""
        if self.failed:
            return None
        sources = () if self._sources is None else tuple(self._sources())
        key = (self._key(), actions.dtype, tuple(actions.shape), tuple(tuple(s.shape) for s in sources))
        entry = self._graphs.get(key)
        if entry is None:
            if len(self._graphs) >= 8:
                self._graphs.clear()
            entry = self._graphs[key] = self._capture(actions, sources)
            if entry is None:
                return None
        graph, static, copies, buffer, layout, builders = entry
        extra = None if self._shared is None else self._shared()
        shape = (sequence, None if state_indices is None else len(state_indices), final_state)
        build = builders.get(shape)
        if build is None:
            build = builders[shape] = result_builder(
                self.num_envs,
                self.device,
                *layout,
                sequence=shape[0],
                with_states=shape[1],
                final_state=shape[2],
                shared=() if extra is None else tuple(extra),
            )
        for i, source in enumerate(sources):
            seen = copies[i]
            if seen[1] is not source or seen[2] != source._version:
                seen[0].copy_(source)
                copies[i] = (seen[0], source, source._version)
        static.copy_(actions)
        if torch.cuda.current_device() == self.device.index:
            graph.replay()
        else:
            with torch.cuda.device(self.device):
                graph.replay()
        # In-place writes of a replay don't bump version counters; caches keyed on them must see the step.
        torch.autograd.graph.increment_version(self._live())
        return build(buffer.clone(), extra)

    def _capture(self, actions: torch.Tensor, sources):
        with torch.cuda.device(self.device):
            return self._capture_here(actions, sources)

    def _capture_here(self, actions: torch.Tensor, sources):
        static = torch.empty(actions.shape, dtype=actions.dtype, device=self.device)
        static.copy_(actions)
        copies = [(source.clone(), source, source._version) for source in sources]
        inputs = [copy for copy, _, _ in copies]
        live = self._live()
        held = [tensor.clone() for tensor in live]
        current = torch.cuda.current_stream(self.device)
        try:
            # One eager run on a side stream compiles lazily built kernels and gives the result layout.
            stream = torch.cuda.Stream(self.device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                probe = self._compute(static, *inputs)
            current.wait_stream(stream)
            leaves = _leaves(probe)
            regions, offset = [], 0
            for path, value in leaves:
                regions.append((path, value, offset))
                offset += -(-value.numel() * value.element_size() // 8) * 8
            state = [(path, value, start) for path, value, start in regions if path[0] == "state"]
            states = []
            for path, value, _ in state:
                states.append((path, value, offset))
                offset += -(-value.numel() * value.element_size() // 8) * 8
            ones = offset
            offset += -(-2 * self.num_envs // 8) * 8
            buffer = torch.zeros(offset, dtype=torch.uint8, device=self.device)
            buffer[ones : ones + 2 * self.num_envs].fill_(1)

            def slot(value, start):
                return buffer[start : start + value.numel() * value.element_size()].view(value.dtype).view(value.shape)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = dict(_leaves(self._compute(static, *inputs)))
                for path, value, start in (*regions, *states):
                    slot(value, start).copy_(captured[path])
        except Exception as error:  # noqa: BLE001 - step without the graph, and say so
            self.failed = True
            log(
                "WARNING",
                f"CUDA graph capture of the step failed ({type(error).__name__}: {error}); steps are slower.",
            )
            return None
        finally:
            for tensor, value in zip(live, held, strict=True):
                tensor.copy_(value)

        def spec(path, value, start, drop=0) -> Leaf:
            size = value.element_size()
            tail = tuple(value.shape[1:])
            strides = []
            stride = 1
            for dim in reversed(tail):
                strides.insert(0, stride)
                stride *= dim
            return (path[drop:], value.dtype, tail, tuple(strides), stride, start // size)

        layout = (
            [spec(path, value, start, 1) for path, value, start in state],
            [spec(path, value, start) for path, value, start in regions if path[0] != "state"],
            [spec(path, value, start, 1) for path, value, start in states],
            (ones, ones + self.num_envs),
        )
        return graph, static, copies, buffer, layout, {}
