"""Run the lanes of a local simulator adapter in persistent spawned worker processes."""

from __future__ import annotations

import builtins
import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import torch
from tensordict import TensorDict

from .._core.tensordicts import build_tensordict
from .protocol import SimulatorAdapter
from .requests import ProcessRequests, lane_slices
from .shared_memory import allocate_buffers, buffer_views, key_path, owned_group
from .state_columns import require_columns
from .workers import pin_worker, send_failure, send_ready

#: Step frame requests: workers draw the frame, or write the State leaves a step changes for the parent to draw.
FRAME_DRAW, FRAME_FIELDS = 1, 2
#: ``process_methods`` kinds: split the first argument by lane, or send every worker the same arguments.
PER_LANE, BROADCAST = "per_lane", "broadcast"


def lane_rows(value, rows):
    """Return the ``rows`` lanes of a per-lane argument; mappings split each value."""
    if isinstance(value, dict):
        return {key: lane_rows(item, rows) for key, item in value.items()}
    return value[rows]


def gather(replies):
    """Join the per-worker return values of one request in lane order."""
    first = replies[0]
    if first is None:
        return None
    if isinstance(first, np.ndarray):
        return np.concatenate(replies)
    if isinstance(first, (TensorDict, torch.Tensor)):
        return torch.cat(replies)
    if isinstance(first, list):
        return [item for reply in replies for item in reply]
    return replies


class _Views(dict):
    """Shared views keyed by flat paths; a nested key such as ``("state", ("rng", "seed"))`` also resolves."""

    def __missing__(self, key):
        flat = tuple(part for item in key for part in key_path(item))
        if flat == key:
            raise KeyError(key)
        return self[flat]


def leaf_tensordict(leaves, batch, device):
    """Build a ``[batch]`` TensorDict from ``{key tuple: tensor}`` leaves on ``device``, in their order."""
    tree = {}
    for key, value in leaves.items():
        branch = tree
        for part in key[:-1]:
            branch = branch.setdefault(part, {})
        branch[key[-1]] = value
    return build_tensordict(tree, [batch], device)


def _clone_all(tensors):
    return {key: value.clone() for key, value in tensors.items()}


def _shared_state(views, rows, count, leaves):
    values = {path[1:]: torch.from_numpy(view[rows].copy()) for path, view in views.items() if path[0] == "state"}
    values.update((key, torch.from_numpy(value)) for key, value in leaves.items())
    return TensorDict(values, [count])


def process_worker(conn, factory, start, count, batch, seed, cpus, buffers, specs):
    """Own lanes ``start:start + count`` of ``factory(count, seed + start)`` and serve parent requests."""
    adapter = None
    try:
        pin_worker(cpus)
        adapter = factory(count, None if seed is None else seed + start)
        views = buffer_views(buffers, specs, batch)
        rows = slice(start, start + count)
        groups = {group: [path for path in views if path[0] == group] for group in ("state", "task", "fields")}
        methods = getattr(adapter, "process_methods", {})

        def write(group, values):
            for path in groups[group]:
                views[path][rows] = values[path[1:]].numpy()

        # Leaves that only resets, restores and process methods change stay in the buffers between steps.
        static = bool(getattr(adapter, "process_static_leaves", ()))

        write("state", adapter.get_state())
        send_ready(conn)
        while True:
            command, payload = conn.recv()
            reply = None
            if command == "close":
                conn.send((True, None))
                break
            if command == "step":
                actions, active, capture, frame = payload
                result = adapter.step(
                    actions,
                    active=None if active is None else torch.from_numpy(active),
                    capture_state=capture and not static,
                )
                write("task", result)
                if static and (capture or frame == FRAME_FIELDS):
                    for key, value in adapter.process_step_state().items():
                        views[("state", *key)][rows] = value
                elif capture:
                    write("state", result["state"])
                if frame == FRAME_DRAW:
                    views[("frame",)][rows] = adapter.render_live()
                elif frame == FRAME_FIELDS and groups["fields"]:
                    fields = adapter.frame_fields()
                    for path in groups["fields"]:
                        views[path][rows] = fields[path[1]]
            elif command == "get_state":
                write("state", adapter.get_state())
            elif command == "reset":
                lane_seed, args = payload
                write("state", adapter.reset(lane_seed, *args))
            elif command == "prepare_restore":
                try:
                    adapter.prepare_restore(_shared_state(views, rows, count, payload))
                except Exception as exc:
                    adapter.discard_restore()
                    reply = (type(exc).__name__, str(exc))
            elif command == "commit_restore":
                adapter.commit_restore()
            elif command == "discard_restore":
                adapter.discard_restore()
            elif command == "call":
                name, args = payload
                if name not in methods:
                    raise ValueError(f"{type(adapter).__name__} does not declare process method {name!r}")
                value = getattr(adapter, name)
                reply = value(*args) if callable(value) else value
            else:
                raise ValueError(f"Unknown worker command {command!r}")
            conn.send((True, reply))
    except BaseException:
        send_failure(conn)
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            conn.close()


class ProcessAdapter(ProcessRequests, SimulatorAdapter):
    """Split a local adapter's lanes across spawned workers with shared State and task buffers.

    ``factory(count, seed)`` builds the local adapter that one worker owns; it
    must be picklable. Lane ``i`` gets ``seed + i``, as in one local batch.
    The local adapter declares ``process_methods``, which run on every worker
    through ``call``, and ``process_attributes``, which the parent copies from
    a one-lane prototype. The parent runs the local ``check_step`` before a
    step reaches any worker. Restore prepares every worker before any worker
    commits. Received outputs are copies that the next request cannot change.
    """

    #: Family name in error messages.
    name = "Simulation"
    #: State leaves that have no fixed shape; restore sends them through the pipe.
    _pipe_leaves: frozenset[tuple[str, ...]] = frozenset()

    def __init__(
        self,
        factory: Callable[[int, int | None], SimulatorAdapter],
        num_envs: int,
        device: str | torch.device = "cpu",
        seed: int | None = 0,
        *,
        processes: int = 8,
        cpus: list[int] | None = None,
    ):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide at least one distinct core per process")
        self.num_envs, self.device = num_envs, torch.device(device)
        self._host_outputs = False
        self._closed, self._pending, self._capture, self._capture_frame = False, None, False, False
        self._pending_frame = None
        self._connections, self._processes = [], []
        self._metadata = None
        self._static, self._static_cache, self._spare, self._copier = frozenset(), None, None, None
        self.process_methods: dict[str, str] = {}
        try:
            # The one-lane prototype stays open for parent-side queries and checks.
            self._metadata = prototype = factory(1, seed)
            self._local = type(prototype)
            self.process_methods = dict(prototype.process_methods)
            self._static = frozenset(key_path(key) for key in getattr(prototype, "process_static_leaves", ()))
            for attribute in prototype.process_attributes:
                setattr(self, attribute, getattr(prototype, attribute))
            state = prototype.get_state()
            task = prototype.step(prototype.probe_actions(), capture_state=False)
            self._specs = {
                ("task", *key_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                for key, value in task.exclude("state").items(True, True)
            }
            self._specs.update(
                {
                    ("state", *key_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                    for key, value in state.items(True, True)
                    if key_path(key) not in self._pipe_leaves
                }
            )
            # Local adapters that draw their running lanes draw step frames in the workers.
            frame_shape = getattr(prototype, "frame_shape", None) if hasattr(prototype, "render_live") else None
            if frame_shape is not None:
                self._specs[("frame",)] = (tuple(frame_shape), np.dtype(np.uint8))
            if self.frame_mode == FRAME_FIELDS and hasattr(prototype, "frame_fields"):
                self._specs.update(
                    {
                        ("fields", key): (tuple(value.shape[1:]), value.dtype)
                        for key, value in prototype.frame_fields().items()
                    }
                )
            self.state_fields = {
                key_path(key): (tuple(value.shape[1:]), value.dtype)
                for key, value in state.items(True, True)
                if key_path(key) not in self._pipe_leaves
            }
            self._slices = lane_slices(num_envs, processes)
            context = mp.get_context("spawn")
            self._buffers = allocate_buffers(context, self._specs, num_envs)
            self._views = _Views(buffer_views(self._buffers, self._specs, num_envs))
            self._spawn_workers(
                context,
                self._worker_target(),
                cpus,
                lambda rows, group: (
                    factory,
                    rows.start,
                    rows.stop - rows.start,
                    num_envs,
                    seed,
                    group,
                    self._buffers,
                    self._specs,
                ),
            )
        except BaseException:
            self._abort()
            raise

    @property
    def _worker_family(self):
        return self.name

    def _worker_target(self):
        return process_worker

    def _abort(self):
        try:
            super()._abort()
        finally:
            if self._copier is not None:
                self._copier.shutdown(wait=False)
                self._copier, self._spare = None, None
            if self._metadata is not None:
                self._metadata.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _owned(self, group):
        return owned_group(self._views, group, self.num_envs, self._output_device)

    def _state_leaves(self):
        """Copy the shared State leaves to ``{key: tensor}``; leaves that only resets, restores and calls
        change come from a cache."""
        if not self._static:
            return {
                key[1:]: torch.from_numpy(view.copy()).to(self._output_device)
                for key, view in self._views.items()
                if key[0] == "state"
            }
        if self._static_cache is None:
            self._static_cache = {
                key[1:]: torch.from_numpy(view.copy()) for key, view in self._views.items() if key[1:] in self._static
            }
            self._spare = None
        cache = self._static_cache
        # A thread copies the cache for the next snapshot while the workers step.
        spare = self._spare.result() if self._spare is not None else _clone_all(cache)
        if self._copier is None:
            self._copier = ThreadPoolExecutor(1)
        self._spare = self._copier.submit(_clone_all, cache)
        values = {}
        for key, view in self._views.items():
            if key[0] == "state":
                value = spare[key[1:]] if key[1:] in spare else torch.from_numpy(view.copy())
                values[key[1:]] = value.to(self._output_device)
        return values

    def _owned_state(self):
        return leaf_tensordict(self._state_leaves(), self.num_envs, self._output_device)

    def _state(self):
        return self._owned_state()

    def __getattr__(self, name):
        if name in self.__dict__.get("process_methods", {}):
            return partial(self.call, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def call(self, name: str, *args):
        """Run the declared local method ``name`` on every worker and join the results in lane order.

        A ``per_lane`` method receives the rows of its first argument that its worker owns
        and every other argument unchanged.
        """
        kind = self.process_methods.get(name)
        if kind not in (PER_LANE, BROADCAST):
            raise ValueError(f"{name!r} is not a declared process method")
        self._static_cache = None
        payloads = [
            (name, (lane_rows(args[0], rows), *args[1:]) if kind == PER_LANE else args) for rows in self._slices
        ]
        return gather(self._request("call", payloads))

    def get_state(self) -> TensorDict:
        self._request("get_state")
        return self._state()

    def _reset(self, seed, args=None):
        """Reset every worker with ``seed + first lane`` and its entry of ``args``."""
        payloads = [
            (None if seed is None else int(seed) + rows.start, () if args is None else tuple(args[index]))
            for index, rows in enumerate(self._slices)
        ]
        self._static_cache = None
        self._request("reset", payloads)
        return self._state()

    def reset(self, seed: int | None = None) -> TensorDict:
        return self._reset(seed)

    def validate(self, state: TensorDict) -> object:
        """Reject State that does not fit the shared buffers; families add their own checks."""
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError(f"{self.name} requires full TensorDict State with batch [num_envs]")
        require_columns(
            state, {key: ((self.num_envs, *shape), dtype) for key, (shape, dtype) in self.state_fields.items()}
        )

    def set_state(self, state: TensorDict) -> None:
        """Restore every lane, or no lane when any worker rejects its rows."""
        self._check_idle()
        self.validate(state)
        self._static_cache = None
        host = state.cpu()
        leaves = {}
        for key in (key_path(key) for key in host.keys(True, True)):
            value, view = host[key].numpy(), self._views.get(("state", *key))
            if view is not None and view.shape == value.shape and view.dtype == value.dtype:
                view[:] = value
            else:
                # Workers check leaves that do not fit the shared buffers.
                leaves[key] = value
        replies = self._request(
            "prepare_restore", [{key: value[rows] for key, value in leaves.items()} for rows in self._slices]
        )
        failures = [reply for reply in replies if reply is not None]
        if failures:
            self._request("discard_restore")
            self._request("get_state")
            kind, message = failures[0]
            error = getattr(builtins, kind, None)
            raise (error if isinstance(error, type) and issubclass(error, Exception) else RuntimeError)(message)
        self._request("commit_restore")

    #: ``receive_step`` accepts ``host=True``.
    host_outputs = True
    #: How ``capture_frame`` obtains a frame: ``FRAME_DRAW`` or ``FRAME_FIELDS`` (see ``_fields_frame``).
    frame_mode = FRAME_DRAW

    @property
    def live_frames(self) -> bool:
        """Whether ``submit_step(capture_frame=True)`` returns the frame of the running lanes."""
        return ("frame",) in self._specs and (self.frame_mode == FRAME_DRAW or bool(self._static))

    def _frame_fields(self) -> object:
        """Copy the fields a ``FRAME_FIELDS`` frame needs: the ``frame_fields`` buffers by default."""
        fields = {key[1]: view.copy() for key, view in self._views.items() if key[0] == "fields"}
        if not fields:
            raise NotImplementedError
        return fields

    def _fields_frame(self, fields) -> torch.Tensor:
        """Draw a step frame from ``_frame_fields`` output; families with ``FRAME_FIELDS`` define it."""
        raise NotImplementedError

    def take_frame(self) -> torch.Tensor | None:
        """Draw the frame of the last received ``FRAME_FIELDS`` step, or return None.

        The leaves were copied on receipt, so the frame can be drawn while the next step runs.
        """
        fields, self._pending_frame = self._pending_frame, None
        return None if fields is None else self._fields_frame(fields)

    def submit_step(self, actions, *, active=None, capture_state: bool = True, capture_frame: bool = False) -> None:
        """Submit one transition; receive its result before submitting another."""
        self._check_idle()
        if capture_frame and not self.live_frames:
            raise ValueError(f"{self.name} workers do not draw frames")
        actions, active = self._local.check_step(self, actions, active)
        capture_state, capture_frame = bool(capture_state), bool(capture_frame)
        self._submit(
            "step",
            [
                (
                    actions[rows],
                    None if active is None else active[rows],
                    capture_state,
                    capture_frame and self.frame_mode,
                )
                for rows in self._slices
            ],
        )
        self._capture, self._capture_frame = capture_state, capture_frame
        self._submitted(actions, active)

    def _submitted(self, actions, active):
        """Record a dispatched step; ``actions`` and ``active`` are the checked host arrays."""

    def receive_step(self, host: bool = False) -> TensorDict:
        """Wait for the transition and copy its outputs before workers can reuse the buffers.

        With ``host``, the outputs stay on the CPU, so the caller can move them to ``device`` together.
        """
        self._check_open()
        if self._pending != "step":
            raise RuntimeError(f"No {self.name} step in flight")
        self._receive()
        self._host_outputs = bool(host)
        try:
            result = self._owned("task")
            if self._capture:
                result["state"] = self._state()
            if self._capture_frame and self.frame_mode == FRAME_FIELDS:
                self._pending_frame = self._frame_fields()
            elif self._capture_frame:
                result["frame"] = torch.from_numpy(self._views[("frame",)]).clone().to(self._output_device)
        finally:
            self._host_outputs = False
        return result

    @property
    def _output_device(self) -> torch.device:
        """The device of step outputs: ``device``, or the CPU while ``receive_step(host=True)`` copies them."""
        return torch.device("cpu") if self._host_outputs else self.device

    def step(self, actions, *, active=None, capture_state: bool = True) -> TensorDict:
        self.submit_step(actions, active=active, capture_state=capture_state)
        return self.receive_step()

    def abort_step(self) -> None:
        """Terminate workers and close the adapter after a failed consumer."""
        self._abort()
