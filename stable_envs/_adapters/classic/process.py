from __future__ import annotations

import multiprocessing as mp
import os
import traceback

import numpy as np
import torch
from tensordict import TensorDict

from .. import SimulatorAdapter
from ..process_requests import ProcessRequests
from ..shared_memory import allocate_buffers, buffer_views, owned_group
from .local import ClassicControlAdapter, _from_limbs, validate_classic_state


def _path(key):
    return key if isinstance(key, tuple) else (key,)


def _worker(conn, start, count, batch, seed, cpus, buffers, specs):
    adapter: ClassicControlAdapter | None = None
    try:
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
        adapter = ClassicControlAdapter("acrobot", count)
        views = buffer_views(buffers, specs, batch)
        rows = slice(start, start + count)

        def write(group, value):
            for key, tensor in value.items(True, True):
                views[(group, *_path(key))][rows] = tensor.numpy()

        write("state", adapter.reset(seed + start))
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
                actions, active = payload
                result = adapter.step(actions, active=None if active is None else torch.from_numpy(active))
                write("task", result.exclude("state"))
                if command == "step":
                    write("state", result["state"])
            elif command == "get_state":
                write("state", adapter.get_state())
            elif command == "set_state":
                # Parent validates every lane's schema and PCG64 before dispatch.
                adapter.set_state(
                    TensorDict(
                        {
                            key[1:]: torch.from_numpy(value[rows].copy())
                            for key, value in views.items()
                            if key[0] == "state"
                        },
                        [count],
                    )
                )
            elif command == "reset":
                seed, options = payload
                write("state", adapter.reset(seed, options))
            elif command == "configure":
                adapter.configure(payload)
            else:
                raise ValueError(f"Unknown Acrobot worker command {command}")
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


class AcrobotProcessAdapter(ProcessRequests, SimulatorAdapter):
    """Run Acrobot CPU simulators in spawned processes with shared State/task columns.

    The parent validates all lanes before dispatch and returns owned output copies.
    ProcessRequests manages the single pending request and worker cleanup.
    """

    _worker_family = "Acrobot"

    def __init__(self, domain, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        if domain != "acrobot":
            raise ValueError("The process candidate is bounded to Acrobot")
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide one distinct core per process")
        self.domain, self.num_envs, self.device = domain, num_envs, torch.device(device)
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        seed = int(seed)
        np.random.SeedSequence(seed)
        prototype = ClassicControlAdapter(domain, 1)
        try:
            state = prototype.reset(seed)
            self.fields = prototype.fields
            result = prototype.step(np.ones((1, 1), np.int64))
            self._specs = {
                ("task", *_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                for key, value in result.exclude("state").items(True, True)
            }
            self._specs.update(
                {
                    ("state", *_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                    for key, value in state.items(True, True)
                }
            )
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
                        rows.start,
                        rows.stop - rows.start,
                        num_envs,
                        seed,
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
            raise RuntimeError("AcrobotProcessAdapter is closed")

    def _owned(self, group):
        return owned_group(self._views, group, self.num_envs, self.device)

    def get_state(self):
        self._request("get_state")
        return self._owned("state")

    def set_state(self, state):
        self._check_idle()
        validate_classic_state(self.domain, state, batch=self.num_envs)
        host = state.cpu()
        # Validate every lane's RNG assignment before dispatch so malformed State cannot partially restore owners.
        probe = np.random.PCG64(0)
        for row in host:
            probe.state = dict(
                bit_generator="PCG64",
                state=dict(
                    state=_from_limbs(row["rng", "state"].numpy()), inc=_from_limbs(row["rng", "increment"].numpy())
                ),
                has_uint32=row["rng", "has_uint32"].item(),
                uinteger=row["rng", "uinteger"].item(),
            )
        for key in self.fields:
            self._views[("state", *_path(key))][:] = host[key].numpy()
        self._request("set_state")

    def reset(self, seed=None, options=None):
        self._check_idle()
        if seed is not None:
            seed = int(seed)
            np.random.SeedSequence(seed)
        self._request("reset", [(None if seed is None else seed + rows.start, options) for rows in self._slices])
        return self._owned("state")

    def configure(self, values):
        self._check_idle()
        columns = {key: np.asarray(value) for key, value in values.items()}
        if any(value.shape != (self.num_envs,) for value in columns.values()):
            raise ValueError("Physical parameters must have one scalar per lane")
        self._request("configure", [{key: value[rows] for key, value in columns.items()} for rows in self._slices])

    def submit_step(self, actions, *, active=None, capture_state=True):
        self._check_idle()
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        # Preserve source scalar types and Python-list action indexing, including valid negative indices.
        converted = ClassicControlAdapter._actions(self, actions)
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        for i, action in enumerate(converted):
            if active is None or active[i]:
                [-1.0, 0.0, 1.0][action]
        self._submit(
            "step" if capture_state else "advance",
            [(actions[rows], None if active is None else active[rows]) for rows in self._slices],
        )

    def receive_step(self):
        self._check_open()
        if self._pending not in ("step", "advance"):
            raise RuntimeError("No Acrobot step in flight")
        capture = self._pending == "step"
        self._receive()
        result = self._owned("task")
        if capture:
            result["state"] = self._owned("state")
        return result

    def step(self, actions, *, active=None, capture_state=True):
        self.submit_step(actions, active=active, capture_state=capture_state)
        return self.receive_step()

    def abort_step(self):
        self._abort()
