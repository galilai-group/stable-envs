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
from .maze import OGBMazeAdapter


def _path(key):
    return key if isinstance(key, tuple) else (key,)


def _worker(conn, start, count, batch, seed, cpus, buffers, specs):
    adapter, prepared = None, None
    try:
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
        adapter = OGBMazeAdapter(count, seed=seed + start)
        views = buffer_views(buffers, specs, batch)
        rows = slice(start, start + count)

        def write(group, value):
            for key, tensor in value.items(True, True):
                views[(group, *_path(key))][rows] = tensor.numpy()

        write("state", adapter.get_state())
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
            reply = None
            if command == "close":
                conn.send((True, None))
                break
            if command in ("step", "advance"):
                actions, active = payload
                active = None if active is None else torch.from_numpy(active)
                result = adapter.step(actions, active=active, capture_state=command == "step")
                write("task", result.exclude("state"))
                if command == "step":
                    write("state", result["state"])
            elif command == "get_state":
                write("state", adapter.get_state())
            elif command == "prepare_restore":
                prepared = None
                state = TensorDict(
                    {
                        key[1:]: torch.from_numpy(value[rows].copy())
                        for key, value in views.items()
                        if key[0] == "state"
                    },
                    [count],
                )
                try:
                    adapter.validate(state)
                except (ValueError, TypeError) as exc:
                    reply = dict(prepared=False, kind=type(exc).__name__, error=str(exc))
                else:
                    prepared = state
                    reply = dict(prepared=True)
            elif command == "discard_restore":
                prepared = None
            elif command == "commit_restore":
                if prepared is None:
                    raise RuntimeError("Maze restore was not prepared")
                adapter.set_state(prepared)
                prepared = None
            elif command == "reset":
                write("state", adapter.reset(payload))
            else:
                raise ValueError(f"Unknown Maze worker command {command}")
            conn.send((True, reply))
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


class OGBMazeProcessAdapter(ProcessRequests, SimulatorAdapter):
    """Run OGBench Maze simulators and isolated RNGs in spawned processes.

    State and task outputs use typed shared buffers and are copied on receipt.
    Validate every shard before committing a checkpoint; validation failure leaves
    live simulators available, while worker failure closes the whole adapter.
    """

    _worker_family = "OGBench Maze"

    def __init__(self, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide one distinct core per process")
        self.num_envs, self.device = num_envs, torch.device(device)
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        prototype = OGBMazeAdapter(1, seed=int(seed))
        try:
            state = prototype.get_state()
            self.fields = {key: (tuple(value.shape[1:]), value.dtype) for key, value in state.items(True, True)}
            result = prototype.step(np.zeros((1, 2)))
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
            raise RuntimeError("OGBMazeProcessAdapter is closed")

    def _owned(self, group):
        return owned_group(self._views, group, self.num_envs, self.device)

    def get_state(self):
        self._request("get_state")
        return self._owned("state")

    def set_state(self, state):
        self._check_idle()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("OGBMaze requires full TensorDict State with batch [num_envs]")
        for key, (shape, dtype) in self.fields.items():
            if key not in state.keys(True, True):
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if value.shape != (self.num_envs, *shape) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid State field {key}")
        host = state.cpu()
        for key in self.fields:
            self._views[("state", *_path(key))][:] = host[key].numpy()
        replies = self._request("prepare_restore")
        errors = [reply for reply in replies if not reply["prepared"]]
        if errors:
            self._request("discard_restore")
            error = errors[0]
            raise {"TypeError": TypeError, "ValueError": ValueError}[error["kind"]](error["error"])
        # Commit only after all shards validate; a failure during commit aborts every owner.
        self._request("commit_restore")

    def reset(self, seed=None):
        self._check_idle()
        if seed is not None:
            seed = int(seed)
            # Check the complete lane range before any worker resets.
            np.random.RandomState(seed)
            np.random.RandomState(seed + self.num_envs - 1)
        self._request("reset", [None if seed is None else int(seed) + rows.start for rows in self._slices])
        return self._owned("state")

    def submit_step(self, actions, *, active=None, capture_state=True):
        self._check_idle()
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if actions.shape != (self.num_envs, 2) or not np.isfinite(actions).all():
            raise ValueError("OGBMaze action must have finite shape (num_envs, 2)")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        payloads = [(actions[rows], None if active is None else active[rows]) for rows in self._slices]
        self._submit("step" if capture_state else "advance", payloads)

    def receive_step(self):
        self._check_open()
        if self._pending not in ("step", "advance"):
            raise RuntimeError("No Maze step in flight")
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
