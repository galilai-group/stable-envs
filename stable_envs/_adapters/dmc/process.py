from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np
import torch
from tensordict import TensorDict

from ..process_requests import ProcessRequests
from ..shared_memory import allocate_buffers, buffer_views, owned_group
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
                name, values, force = payload
                method = getattr(adapter, name)
                method(values, force=True) if force else method(values)
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


class DMCProcessAdapter(ProcessRequests):
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
        self._config = {}
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
            self._configuration_methods = {
                name
                for name in (
                    "configure_gravity",
                    "configure_density",
                    "configure_proximal_density",
                    "configure_upper_arm_density",
                )
                if hasattr(prototype, name)
            }
            for name in ("proximal_density", "upper_arm_density"):
                if hasattr(prototype, "_" + name):
                    self._config[name] = np.repeat(getattr(prototype, "_" + name), num_envs)
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
        for key, (width, dtype) in self.fields.items():
            if key not in state.keys(True, True):
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if value.shape != (self.num_envs, width) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid State field {key}")
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

    def _configure(self, name, values, width):
        self._check_idle()
        method = "configure_" + name
        if method not in self._configuration_methods:
            raise ValueError(f"{self.domain} does not support {method}")
        shape = (self.num_envs,) if width == 1 else (self.num_envs, width)
        values = np.asarray(values, np.float64).reshape(shape).copy()
        if not np.isfinite(values).all():
            raise ValueError("Configuration values must be finite")
        force = name in self._config
        if force and np.array_equal(values, self._config[name]):
            return
        # Density changes reset the source batch; force every shard to reset even if its values are unchanged.
        self._request("configure", [(method, values[rows], force) for rows in self._slices])
        if force:
            self._config[name] = values

    def configure_gravity(self, values):
        self._configure("gravity", values, 3)

    def configure_density(self, values):
        self._configure("density", values, 2)

    def configure_proximal_density(self, values):
        self._configure("proximal_density", values, 1)

    def configure_upper_arm_density(self, values):
        self._configure("upper_arm_density", values, 1)
