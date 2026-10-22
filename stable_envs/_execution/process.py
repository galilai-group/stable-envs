"""Run the lanes of a local simulator adapter in persistent spawned worker processes."""

from __future__ import annotations

import builtins
import multiprocessing as mp
from collections.abc import Callable
from functools import partial

import numpy as np
import torch
from tensordict import TensorDict

from .protocol import SimulatorAdapter
from .requests import ProcessRequests, lane_slices
from .shared_memory import allocate_buffers, buffer_views, key_path, owned_group
from .state_columns import require_columns
from .workers import pin_worker, send_failure, send_ready

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
        groups = {group: [path for path in views if path[0] == group] for group in ("state", "task")}
        methods = getattr(adapter, "process_methods", {})

        def write(group, values):
            for path in groups[group]:
                views[path][rows] = values[path[1:]].numpy()

        write("state", adapter.get_state())
        send_ready(conn)
        while True:
            command, payload = conn.recv()
            reply = None
            if command == "close":
                conn.send((True, None))
                break
            if command == "step":
                actions, active, capture = payload
                result = adapter.step(
                    actions, active=None if active is None else torch.from_numpy(active), capture_state=capture
                )
                write("task", result)
                if capture:
                    write("state", result["state"])
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
    Subclasses validate inputs in the parent, so that a rejected request
    reaches no worker. Restore prepares every worker before any worker
    commits. Received outputs are copies that the next request cannot change.
    Methods named in the local adapter's ``process_methods`` run on every
    worker through ``call``.
    """

    _worker_family = "Simulation"
    _step_family = "Simulation"
    #: Name in the default State and action errors.
    _state_family = "Simulation"
    #: Keep the specification prototype as ``_metadata`` for parent-side queries.
    _keep_metadata = False
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
        self._closed, self._pending, self._capture = False, None, False
        self._connections, self._processes = [], []
        self._metadata = None
        self.process_methods: dict[str, str] = {}
        try:
            prototype = factory(1, seed)
            try:
                self._inspect(prototype)
                self.process_methods = dict(getattr(prototype, "process_methods", {}))
                state = prototype.get_state()
                task = self._task_template(prototype)
                if self._keep_metadata:
                    self._metadata = prototype
            finally:
                if self._metadata is not prototype:
                    prototype.close()
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

    def _worker_target(self):
        return process_worker

    def _inspect(self, prototype):
        """Read family metadata from a one-lane prototype before workers start."""

    def _task_template(self, prototype):
        """Return one prototype step whose task outputs define the shared task buffers."""
        return prototype.step(self._probe_actions(prototype), capture_state=False)

    def _probe_actions(self, prototype):
        return np.zeros((1, self.action_dim))

    def _abort(self):
        try:
            super()._abort()
        finally:
            if self._metadata is not None:
                self._metadata.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _owned(self, group):
        return owned_group(self._views, group, self.num_envs, self.device)

    def _state(self):
        return self._owned("state")

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
        self._request("reset", payloads)
        return self._state()

    def reset(self, seed: int | None = None) -> TensorDict:
        return self._reset(seed)

    def validate(self, state: TensorDict) -> object:
        """Reject State that does not fit the shared buffers; families add their own checks."""
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError(f"{self._state_family} requires full TensorDict State with batch [num_envs]")
        require_columns(
            state, {key: ((self.num_envs, *shape), dtype) for key, (shape, dtype) in self.state_fields.items()}
        )

    def set_state(self, state: TensorDict) -> None:
        """Restore every lane, or no lane when any worker rejects its rows."""
        self._check_idle()
        self.validate(state)
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

    def _host_step(self, actions, active):
        """Return host ``(actions, active)`` arrays or raise before any worker steps."""
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"{self._state_family} action must have finite shape (num_envs, {self.action_dim})")
        return actions, self._host_active(active)

    def _host_active(self, active):
        if active is None:
            return None
        active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
        if active.shape != (self.num_envs,) or active.dtype != np.bool_:
            raise ValueError("active must be a bool vector with shape [num_envs]")
        return active

    def submit_step(self, actions, *, active=None, capture_state: bool = True) -> None:
        """Submit one transition; receive its result before submitting another."""
        self._check_idle()
        actions, active = self._host_step(actions, active)
        capture_state = bool(capture_state)
        self._submit(
            "step", [(actions[rows], None if active is None else active[rows], capture_state) for rows in self._slices]
        )
        self._capture = capture_state

    def receive_step(self) -> TensorDict:
        """Wait for the transition and copy its outputs before workers can reuse the buffers."""
        self._check_open()
        if self._pending != "step":
            raise RuntimeError(f"No {self._step_family} step in flight")
        self._receive()
        result = self._owned("task")
        if self._capture:
            result["state"] = self._state()
        return result

    def step(self, actions, *, active=None, capture_state: bool = True) -> TensorDict:
        self.submit_step(actions, active=active, capture_state=capture_state)
        return self.receive_step()

    def abort_step(self) -> None:
        """Terminate workers and close the adapter after a failed consumer."""
        self._abort()
