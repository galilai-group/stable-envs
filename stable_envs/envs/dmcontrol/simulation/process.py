from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from ...._execution.requests import ProcessRequests
from ...._execution.shared_memory import allocate_buffers, buffer_views, owned_group
from ...._execution.state_columns import require_columns
from .factory import make_local


def _path(key):
    return key if isinstance(key, tuple) else (key,)


def _worker(conn, domain, start, count, batch, seed, cpus, buffers, specs):
    adapter = None
    try:
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
        adapter = make_local(domain, count, seed=seed + start)
        views = buffer_views(buffers, specs, batch)
        rows = slice(start, start + count)

        def write_state(state):
            for key, value in state.items(True, True):
                views[("state", *_path(key))][rows] = value.numpy()

        def write_task(result):
            for key, value in result.exclude("state").items(True, True):
                views[("task", *_path(key))][rows] = value.numpy()

        write_state(adapter.get_state())
        conn.send(
            (
                True,
                dict(
                    pid=os.getpid(),
                    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                ),
            )
        )
        while True:
            command, payload = conn.recv()
            if command == "close":
                conn.send((True, None))
                break
            if command in ("step", "advance"):
                output = adapter.step(payload, capture_state=command == "step")
                write_task(output)
                if command == "step":
                    write_state(output["state"])
            elif command == "get_state":
                write_state(adapter.get_state())
            elif command == "set_state":
                state = TensorDict(
                    {key[1:]: torch.from_numpy(value[rows]) for key, value in views.items() if key[0] == "state"},
                    [count],
                )
                adapter.set_state(state)
            elif command == "reset":
                write_state(adapter.reset(payload))
            elif command == "configure":
                adapter.configure_physics(payload)
            else:
                raise ValueError(f"Unknown DMC worker command {command}")
            conn.send((True, None))
    except BaseException:
        try:
            conn.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            conn.close()


class DMCProcessAdapter(ProcessRequests, SimulatorAdapter):
    """Run DMC adapters in persistent spawned processes with typed shared outputs.

    Copy State and task tensors into parent-owned storage after each completed
    request. Physics stays on CPU even when outputs are delivered on CUDA.
    """

    _worker_family = "DMC"

    def __init__(self, domain, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide at least one distinct core per process")
        self.domain, self.num_envs, self.device = domain, num_envs, torch.device(device)
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        # Discover exact nested task shapes/dtypes from one independent owner;
        # no parent-side live batch or renderer contexts are constructed here.
        prototype = make_local(domain, 1, seed=int(seed))
        try:
            self.action_dim = prototype.envs[0].action_spec().shape[0]
            initial = prototype.get_state()
            self.fields = {key: (value.shape[1], value.dtype) for key, value in initial.items(True, True)}
            result = prototype.step(np.zeros((1, self.action_dim)))
            self._specs = {
                ("task", *_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                for key, value in result.exclude("state").items(True, True)
            }
            self._specs.update(
                {
                    ("state", *_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                    for key, value in initial.items(True, True)
                }
            )
            physics = prototype.physics
            self.physics_layout = None if physics is None else physics.layout
            self._physics_dim = 0 if physics is None else physics.spec.dim
        finally:
            prototype.close()
        self._slices = [
            slice(int(lanes[0]), int(lanes[-1]) + 1) for lanes in np.array_split(np.arange(num_envs), processes)
        ]
        context = mp.get_context("spawn")
        self._buffers = allocate_buffers(context, self._specs, num_envs)
        self._views = buffer_views(self._buffers, self._specs, num_envs)
        groups = [None] * processes if cpus is None else [group.tolist() for group in np.array_split(cpus, processes)]
        try:
            for rows, group in zip(self._slices, groups, strict=True):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(
                        child,
                        domain,
                        rows.start,
                        rows.stop - rows.start,
                        num_envs,
                        int(seed),
                        group,
                        self._buffers,
                        self._specs,
                    ),
                )
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    raise
                finally:
                    child.close()
                self._connections.append(parent)
                self._processes.append(process)
            self.worker_info = self._responses(timeout=300)
        except BaseException:
            self._abort()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("DMCProcessAdapter is closed")

    def _owned(self, group):
        return owned_group(self._views, group, self.num_envs, self.device)

    def get_state(self):
        self._request("get_state")
        return self._owned("state")

    def set_state(self, state):
        self._check_idle()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("DMC requires full TensorDict State with batch [num_envs]")
        require_columns(state, {key: ((self.num_envs, width), dtype) for key, (width, dtype) in self.fields.items()})
        host = state.cpu()
        for key in self.fields:
            self._views[("state", *_path(key))][:] = host[key].numpy()
        self._request("set_state")

    def reset(self, seed=None):
        self._request("reset", [None if seed is None else int(seed) + rows.start for rows in self._slices])
        return self._owned("state")

    def submit_step(self, actions, *, active=None, capture_state=True):
        self._check_idle()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"DMC actions must be finite with shape {(self.num_envs, self.action_dim)}")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or not active.all():
                raise ValueError(
                    "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
                )
        self._submit("step" if capture_state else "advance", [actions[rows] for rows in self._slices])

    def receive_step(self):
        self._check_open()
        if self._pending not in ("step", "advance"):
            raise RuntimeError("No DMC step in flight")
        capture = self._pending == "step"
        self._receive()
        result = self._owned("task")
        if capture:
            result["state"] = self._owned("state")
        return result

    def step(self, actions, *, capture_state=True):
        self.submit_step(actions, capture_state=capture_state)
        return self.receive_step()

    def abort_step(self):
        self._abort()

    def configure_physics(self, values):
        self._check_idle()
        if self.physics_layout is None:
            raise ValueError(f"{self.domain} declares no physics Variation")
        values = np.asarray(values, np.float64).reshape(self.num_envs, self._physics_dim).copy()
        if not np.isfinite(values).all():
            raise ValueError("Configuration values must be finite")
        self._request("configure", [values[rows] for rows in self._slices])
