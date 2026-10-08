from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import traceback

import numpy as np
import torch
from tensordict import TensorDict

from ..process_requests import ProcessRequests
from .adapter import FetchAdapter


def _arrays(buffers, specs, batch):
    return {
        key: np.frombuffer(buffers[key], dtype=dtype).reshape(batch, width) for key, (width, dtype) in specs.items()
    }


def _worker(conn, start, count, batch, task, reward_type, seed, threads, cpus, buffers, specs):
    adapter: FetchAdapter | None = None
    try:
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
        adapter = FetchAdapter(count, "cpu", None if seed is None else seed + start, task, reward_type)
        adapter._set_workers(min(threads, count))
        views = _arrays(buffers, specs, batch)
        rows = slice(start, start + count)

        def write_state(state):
            for key in adapter.fields:
                views[("state", key)][rows] = state[key].numpy()

        def write_task(result):
            for key in ("reward", "terminated", "truncated", "success"):
                views[("task", key)][rows, 0] = result[key].numpy()

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
            result = None
            if command == "close":
                conn.send((True, None))
                break
            if command in ("step", "advance"):
                actions, active = payload
                output = getattr(adapter, command)(actions, active=active)
                write_task(output)
                if command == "step":
                    write_state(output["state"])
            elif command == "get_state":
                write_state(adapter.get_state())
            elif command == "set_state":
                state = TensorDict(
                    {key: torch.from_numpy(views[("state", key)][rows]) for key in adapter.fields}, [count]
                )
                adapter.set_state(state)
            elif command == "reset":
                seed, variations = payload
                write_state(adapter.reset(seed, variations))
                result = adapter.object_mass
            elif command == "mass":
                adapter.configure_object_mass(payload)
            elif command == "visual":
                result = adapter.reset_visual_colors(payload)
            elif command == "limits":
                adapter._set_episode_limits(payload)
            else:
                raise ValueError(f"Unknown Fetch worker command: {command}")
            conn.send((True, result))
    except BaseException:
        try:
            conn.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if adapter is not None:
            adapter.close()
        conn.close()


class FetchProcessAdapter(ProcessRequests, FetchAdapter):
    """Fetch simulators distributed across persistent spawned CPU processes.

    Typed shared buffers carry State and task outputs; parent-side copies own their
    storage before workers advance again. Optional threads run within each worker;
    rendering and output-device transfer remain in the parent.
    """

    _worker_family = "Fetch"

    def __init__(
        self,
        num_envs,
        device="cpu",
        seed=0,
        task="PickAndPlace",
        reward_type="sparse",
        *,
        processes=8,
        threads_per_process=1,
        cpus=None,
    ):
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if type(threads_per_process) is not int or threads_per_process < 1:
            raise ValueError("threads_per_process must be positive")
        self.num_envs, self.device = int(num_envs), torch.device(device)
        self._closed = False
        self._pending = None
        self._connections, self._processes = [], []
        # Keep one model for specs and variation; child processes own the live simulator batch.
        prototype = FetchAdapter(1, "cpu", 0, task, reward_type)
        try:
            self.fields, self.has_object = prototype.fields, prototype.has_object
            self._model = prototype.reference_model()
            self._mass = np.repeat(prototype.object_mass, self.num_envs)
        finally:
            prototype.close()
        self._workers = processes * threads_per_process
        self._threads_per_process = threads_per_process
        self._slices = []
        start = 0
        for lanes in np.array_split(np.arange(self.num_envs), processes):
            self._slices.append(slice(start, start + len(lanes)))
            start += len(lanes)
        self._specs = {
            ("state", key): (width, np.float64 if dtype == torch.float64 else np.int64)
            for key, (width, dtype) in self.fields.items()
        }
        self._specs.update(
            {
                ("task", "reward"): (1, np.float64 if reward_type == "dense" else np.float32),
                ("task", "terminated"): (1, np.bool_),
                ("task", "truncated"): (1, np.bool_),
                ("task", "success"): (1, np.float32),
            }
        )
        context = mp.get_context("spawn")
        ctypes_by_dtype = {
            np.float64: ctypes.c_double,
            np.float32: ctypes.c_float,
            np.int64: ctypes.c_int64,
            np.bool_: ctypes.c_bool,
        }
        self._buffers = {
            key: context.RawArray(ctypes_by_dtype[dtype], self.num_envs * width)
            for key, (width, dtype) in self._specs.items()
        }
        self._views = _arrays(self._buffers, self._specs, self.num_envs)
        groups = [None] * processes if cpus is None else np.array_split(cpus, processes)
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide at least one distinct core per process")
        try:
            for rows, group in zip(self._slices, groups):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(
                        child,
                        rows.start,
                        rows.stop - rows.start,
                        self.num_envs,
                        task,
                        reward_type,
                        seed,
                        threads_per_process,
                        None if group is None else group.tolist(),
                        self._buffers,
                        self._specs,
                    ),
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            self.worker_info = self._responses(timeout=300)
        except BaseException:
            self._abort()
            raise

    def _state(self):
        return TensorDict(
            {
                key: torch.as_tensor(self._views[("state", key)].copy(), dtype=dtype, device=self.device)
                for key, (_, dtype) in self.fields.items()
            },
            [self.num_envs],
            device=self.device,
        )

    def _task(self):
        return TensorDict(
            {
                key: torch.as_tensor(self._views[("task", key)][:, 0].copy(), device=self.device)
                for key in ("reward", "terminated", "truncated", "success")
            },
            [self.num_envs],
            device=self.device,
        )

    def reference_model(self):
        self._check_open()
        return self._model

    def configure_object_mass(self, values):
        self._check_open()
        values = np.asarray(values, np.float64).reshape(self.num_envs).copy()
        self._request("mass", [values[rows] for rows in self._slices])
        self._mass = values

    def reset_visual_colors(self, values):
        self._check_open()
        if len(values) != self.num_envs:
            raise ValueError("Expected one variation dictionary per lane")
        return [value for group in self._request("visual", [values[rows] for rows in self._slices]) for value in group]

    def _set_episode_limits(self, limits):
        self._check_open()
        limits = np.asarray(limits)
        if limits.shape != (self.num_envs,) or limits.dtype.kind not in "iu" or np.any(limits < 1):
            raise ValueError("Expected positive integer episode limits, one per lane")
        self._request("limits", [limits[rows] for rows in self._slices])

    def reset(self, seed: int | None = None, variations=None) -> TensorDict:
        self._check_open()
        if variations is not None and len(variations) != self.num_envs:
            raise ValueError("Expected one variation dictionary per lane")
        masses = self._request(
            "reset",
            [
                (None if seed is None else int(seed) + rows.start, None if variations is None else variations[rows])
                for rows in self._slices
            ],
        )
        self._mass = np.concatenate(masses)
        return self._state()

    def get_state(self) -> TensorDict:
        self._request("get_state")
        return self._state()

    def set_state(self, state: TensorDict) -> None:
        # Reject before writing shared storage owned by an in-flight step.
        self._check_idle()
        self._validate_state(state)
        host = state.cpu()
        for key in self.fields:
            self._views[("state", key)][:] = host[key].numpy()
        self._request("set_state")

    def _actions(self, actions, active):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 4):
            raise ValueError("Fetch action must have shape (num_envs,4)")
        if torch.is_tensor(active):
            active = active.detach().cpu().numpy()
        active = None if active is None else np.asarray(active, dtype=bool)
        if active is not None and active.shape != (self.num_envs,):
            raise ValueError("Fetch active mask must have shape (num_envs,)")
        return [(actions[rows], None if active is None else active[rows]) for rows in self._slices]

    def advance(self, actions: torch.Tensor | np.ndarray, *, active=None) -> TensorDict:
        self._request("advance", self._actions(actions, active))
        return self._task()

    def step(self, actions: torch.Tensor | np.ndarray, *, active=None) -> TensorDict:
        self._request("step", self._actions(actions, active))
        result = self._task()
        result["state"] = self._state()
        return result

    def submit_step(self, actions: torch.Tensor | np.ndarray, *, active=None, capture_state: bool = True) -> None:
        """Submit one transition; receive its result before submitting another."""
        self._submit("step" if capture_state else "advance", self._actions(actions, active))

    def receive_step(self) -> TensorDict:
        """Wait for completion and copy outputs before workers can reuse shared storage."""
        self._check_open()
        if self._pending not in ("step", "advance"):
            raise RuntimeError("No Fetch step in flight")
        capture_state = self._pending == "step"
        self._receive()
        result = self._task()
        if capture_state:
            result["state"] = self._state()
        return result

    def abort_step(self) -> None:
        """Terminate workers and close the adapter after a failed consumer."""
        self._abort()
