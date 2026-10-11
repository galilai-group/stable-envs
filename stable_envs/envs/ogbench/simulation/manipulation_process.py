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
from .manipulation import OGBManipulationAdapter


def _worker(conn, family, start, count, batch, seed, reward_task_id, cpus, buffers, specs):
    adapter: OGBManipulationAdapter | None = None
    try:
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
        adapter = OGBManipulationAdapter(
            family, count, seed=None if seed is None else seed + start, reward_task_id=reward_task_id
        )
        views, rows = buffer_views(buffers, specs, batch), slice(start, start + count)

        def write(group, value):
            for key, tensor in value.items(True, True):
                key = (key,) if isinstance(key, str) else key
                views[(group, *key)][rows] = tensor.numpy()

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
                result = adapter.step(
                    actions,
                    active=None if active is None else torch.from_numpy(active),
                    capture_state=command == "step",
                )
                write("task", result.exclude("state"))
                if command == "step":
                    write("state", result["state"])
            elif command == "get_state":
                write("state", adapter.get_state())
            elif command == "prepare_restore":
                state = TensorDict(
                    {
                        key[1:]: torch.from_numpy(value[rows].copy())
                        for key, value in views.items()
                        if key[0] == "state"
                    },
                    [count],
                )
                try:
                    adapter.prepare_restore(state)
                except Exception as exc:
                    # Preparation failure leaves live worlds untouched; discard staged replacements on every owner.
                    adapter.discard_restore()
                    reply = dict(prepared=False, kind=type(exc).__name__, error=str(exc))
                else:
                    reply = dict(prepared=True)
            elif command == "discard_restore":
                adapter.discard_restore()
            elif command == "commit_restore":
                adapter.commit_restore()
            elif command == "reset":
                write("state", adapter.reset(*payload))
            elif command == "configure":
                adapter.configure(payload)
                write("state", adapter.get_state())
            else:
                raise ValueError(f"Unknown Cube/Scene worker command {command}")
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


class OGBManipulationProcessAdapter(ProcessRequests, SimulatorAdapter):
    """Run Cube/Scene worlds in spawned processes with staged checkpoint restore.

    The parent retains one metadata instance for validation and configuration.
    Workers prepare all replacement worlds before committing any restore; expected
    preparation failures discard staged worlds without changing live simulation.
    Received State and task outputs are copied from typed shared buffers.
    """

    _worker_family = "OGBench manipulation"
    execution_mode = "DELEGATED_CPU"

    def __init__(self, family, num_envs=1, device="cpu", seed=0, reward_task_id=None, *, processes=8, cpus=None):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        if type(processes) is not int or not 1 <= processes <= num_envs:
            raise ValueError("processes must be between one and num_envs")
        if cpus is not None and (len(set(cpus)) != len(cpus) or len(cpus) < processes):
            raise ValueError("CPU allocation must provide one distinct core per process")
        self.family, self.num_envs, self.device = family, num_envs, torch.device(device)
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        self._metadata = None
        try:
            self._metadata = OGBManipulationAdapter(family, 1, seed=seed, reward_task_id=reward_task_id)
            self.fields, self.nq = self._metadata.fields, self._metadata.nq
            state = self._metadata.get_state()
            self._specs = {
                ("state", *key): (tuple(value.shape[1:]), value.numpy().dtype)
                for key, value in state.items(True, True)
            }
            self._specs.update(
                {
                    ("task", name): ((), np.dtype(dtype))
                    for name, dtype in (
                        ("reward", np.float64),
                        ("success", bool),
                        ("terminated", bool),
                        ("truncated", bool),
                    )
                }
            )
            self._slices = [
                slice(int(lanes[0]), int(lanes[-1]) + 1) for lanes in np.array_split(np.arange(num_envs), processes)
            ]
            context = mp.get_context("spawn")
            self._buffers = allocate_buffers(context, self._specs, num_envs)
            self._views = buffer_views(self._buffers, self._specs, num_envs)
            groups = [None] * processes if cpus is None else [g.tolist() for g in np.array_split(cpus, processes)]
            for rows, group in zip(self._slices, groups, strict=True):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(
                        child,
                        family,
                        rows.start,
                        rows.stop - rows.start,
                        num_envs,
                        seed,
                        reward_task_id,
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
            raise RuntimeError("OGBManipulationProcessAdapter is closed")

    def _abort(self):
        try:
            super()._abort()
        finally:
            if self._metadata is not None:
                self._metadata.close()

    def _owned(self, group):
        return owned_group(self._views, group, self.num_envs, self.device)

    def get_state(self):
        self._request("get_state")
        return self._owned("state")

    def validate(self, state):
        self._check_open()
        # Metadata validation is independent of workers advancing the next transition.
        return self._metadata.validate(state, _batch_size=self.num_envs)

    def set_state(self, state):
        self._check_idle()
        rows = self.validate(state)
        for key in self.fields:
            for lane, row in enumerate(rows):
                self._views[("state", *key)][lane] = row[key]
        replies = self._request("prepare_restore")
        errors = [reply for reply in replies if not reply["prepared"]]
        if errors:
            self._request("discard_restore")
            self._request("get_state")  # Discard requested bytes as well as staged worlds.
            error = errors[0]
            kind = {"ValueError": ValueError, "TypeError": TypeError}.get(error["kind"], RuntimeError)
            raise kind(error["error"])
        self._request("commit_restore")

    def reset(self, seed=None, options=None):
        self._check_idle()
        if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        lanes = self._metadata.envs[0].lane_options(options, self.num_envs)
        self._request(
            "reset", [(None if seed is None else int(seed) + rows.start, lanes[rows]) for rows in self._slices]
        )
        return self._owned("state")

    def configure(self, values):
        self._check_idle()
        if len(values) != self.num_envs:
            raise ValueError("Variation batch mismatch")
        for value in values:
            self._metadata.envs[0].validate_options({"variation": [], "variation_values": value})
        self._request("configure", [values[rows] for rows in self._slices])

    def make_variation_space(self):
        self._check_open()
        return self._metadata.make_variation_space()

    def variation_values(self, layout, *, anchor=False):
        self._check_idle()
        namespace = "anchor" if anchor else "current"
        values = np.concatenate(
            [self._views["state", "variation", namespace, name].reshape(self.num_envs, -1) for name, _, _ in layout],
            axis=1,
        )
        return torch.from_numpy(values).to(self.device)

    def submit_step(self, actions, *, active=None, capture_state=True):
        self._check_idle()
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if (
            actions.shape != (self.num_envs, 5)
            or actions.dtype not in (np.float32, np.float64)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        self._submit(
            "step" if capture_state else "advance",
            [(actions[rows], None if active is None else active[rows]) for rows in self._slices],
        )

    def receive_step(self):
        self._check_open()
        if self._pending not in ("step", "advance"):
            raise RuntimeError("No Cube/Scene step in flight")
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
