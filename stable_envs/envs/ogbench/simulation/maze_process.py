from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from ...._execution.requests import SharedOutputProcess, lane_slices
from ...._execution.shared_memory import allocate_buffers, buffer_views, key_path
from ...._execution.state_columns import require_columns
from ...._execution.workers import pin_worker, send_failure, send_ready
from .maze import OGBMazeAdapter


def _worker(conn, start, count, batch, seed, cpus, buffers, specs, body):
    adapter, prepared = None, None
    try:
        pin_worker(cpus)
        adapter = OGBMazeAdapter(count, seed=seed + start, **body)
        views = buffer_views(buffers, specs, batch)
        rows = slice(start, start + count)

        def write(group, value):
            for key, tensor in value.items(True, True):
                views[(group, *key_path(key))][rows] = tensor.numpy()

        write("state", adapter.get_state())
        send_ready(conn)
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
        send_failure(conn)
    finally:
        try:
            if adapter is not None:
                adapter.close()
        finally:
            conn.close()


class OGBMazeProcessAdapter(SharedOutputProcess, SimulatorAdapter):
    """Run OGBench Maze simulators and isolated RNGs in spawned processes.

    State and task outputs use typed shared buffers and are copied on receipt.
    Validate every shard before committing a restore; validation failure leaves
    live simulators available, while worker failure closes the whole adapter.
    """

    _worker_family = "OGBench Maze"
    _step_family = "Maze"

    def __init__(self, num_envs, device="cpu", seed=0, *, processes=8, cpus=None, loco="point", maze="large"):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide one distinct core per process")
        self.num_envs, self.device = num_envs, torch.device(device)
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        body = dict(loco=loco, maze=maze)
        prototype = OGBMazeAdapter(1, seed=int(seed), **body)
        try:
            self.action_dim = prototype.action_dim
            state = prototype.get_state()
            self.fields = {key: (tuple(value.shape[1:]), value.dtype) for key, value in state.items(True, True)}
            result = prototype.step(np.zeros((1, self.action_dim)))
            self._specs = {
                ("task", *key_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                for key, value in result.exclude("state").items(True, True)
            }
            self._specs.update(
                {
                    ("state", *key_path(key)): (tuple(value.shape[1:]), value.numpy().dtype)
                    for key, value in state.items(True, True)
                }
            )
        finally:
            prototype.close()
        self._slices = lane_slices(num_envs, processes)
        context = mp.get_context("spawn")
        self._buffers = allocate_buffers(context, self._specs, num_envs)
        self._views = buffer_views(self._buffers, self._specs, num_envs)
        try:
            self._spawn_workers(
                context,
                _worker,
                cpus,
                lambda rows, group: (
                    rows.start,
                    rows.stop - rows.start,
                    num_envs,
                    int(seed),
                    group,
                    self._buffers,
                    self._specs,
                    body,
                ),
            )
        except BaseException:
            self._abort()
            raise

    def set_state(self, state):
        self._check_idle()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("OGBMaze requires full TensorDict State with batch [num_envs]")
        require_columns(state, {key: ((self.num_envs, *shape), dtype) for key, (shape, dtype) in self.fields.items()})
        host = state.cpu()
        for key in self.fields:
            self._views[("state", *key_path(key))][:] = host[key].numpy()
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
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"OGBMaze action must have finite shape (num_envs, {self.action_dim})")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        payloads = [(actions[rows], None if active is None else active[rows]) for rows in self._slices]
        self._submit("step" if capture_state else "advance", payloads)

    def step(self, actions, *, active=None, capture_state=True):
        self.submit_step(actions, active=active, capture_state=capture_state)
        return self.receive_step()
