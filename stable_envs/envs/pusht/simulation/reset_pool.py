"""Reset the PushT lanes of the CUDA physics in worker processes."""

from __future__ import annotations

import builtins
import multiprocessing as mp

import numpy as np

from ...._execution.requests import ProcessRequests, lane_slices
from ...._execution.shared_memory import allocate_buffers, buffer_views
from ...._execution.workers import pin_worker, send_failure, send_ready
from .numeric import MAX_SHAPES, geometry_columns, install_latest
from .replay_batch import PushTReplayAdapter


class _Lanes(PushTReplayAdapter):
    """Worker owners whose reset keeps the previous owners until ``settle``."""

    _previous = None

    def _swap(self, staged):
        self._previous, self.envs = self.envs, staged

    def settle(self, commit):
        """Close the previous owners, or with ``commit=False`` the reset owners, and keep the others."""
        previous, self._previous = self._previous, None
        if previous is None:
            return
        if not commit:
            previous, self.envs = self.envs, previous
        for env in previous:
            env.close()

    def close(self):
        self.settle(True)
        super().close()


def reset_worker(conn, discrete, config, start, count, batch, seed, cpus, buffers, specs):
    """Own lanes ``start:start + count`` and write the columns of each reset into the shared buffers."""
    lanes = None
    try:
        pin_worker(cpus)
        lanes = _Lanes(count, "cpu", None if seed is None else seed + start, discrete=discrete, config=config)
        views = buffer_views(buffers, specs, batch)
        rows = slice(start, start + count)
        owned = [key for group, key in specs if group == "static" and key.startswith("owned/")]

        def write(group, columns):
            for key, value in columns.items():
                views[(group, key)][rows] = value

        send_ready(conn)
        while True:
            command, payload = conn.recv()
            reply = None
            if command == "close":
                conn.send((True, None))
                break
            if command == "reset":
                sync, lane_seed, options, latest = payload
                if sync:
                    for lane, env in enumerate(lanes.envs):
                        env._install_metadata(
                            {key.removeprefix("owned/"): views[("static", key)][start + lane].copy() for key in owned}
                        )
                install_latest(lanes.envs, *latest)
                try:
                    lanes.reset(lane_seed, options)
                except Exception as exc:  # noqa: BLE001 - the parent discards the reset and raises it
                    reply = (type(exc).__name__, str(exc))
                else:
                    write("static", lanes._static_columns())
                    write("current", lanes._current_columns())
                    write("geometry", geometry_columns(lanes.envs, MAX_SHAPES))
            elif command == "settle":
                lanes.settle(payload)
            elif command == "install_variation":
                lanes.install_variation(*payload)
                write("static", lanes._static_columns())
            elif command == "variation_rows":
                reply = lanes.variation_rows(*payload)
            else:
                raise ValueError(f"Unknown worker command {command!r}")
            conn.send((True, reply))
    except BaseException:
        send_failure(conn)
    finally:
        try:
            if lanes is not None:
                lanes.close()
        finally:
            conn.close()


class PushTResetPool(ProcessRequests):
    """Reset contiguous lane slices in spawned workers that own the lanes' CPU engines.

    ``groups`` maps ``static``, ``current`` and ``geometry`` to one-lane-per-row sample columns,
    which fix the shared buffers. Lane ``i`` gets ``seed + i``, as in one local batch. A reset that
    fails in any worker changes no worker.
    """

    _worker_family = "PushT reset"

    def __init__(self, discrete, config, num_envs, seed, groups, *, processes, cpus=None):
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        self._slices = lane_slices(num_envs, processes)
        specs = {
            (group, key): (value.shape[1:], value.dtype)
            for group, columns in groups.items()
            for key, value in columns.items()
        }
        context = mp.get_context("spawn")
        self._buffers = allocate_buffers(context, specs, num_envs)
        self._views = buffer_views(self._buffers, specs, num_envs)
        try:
            self._spawn_workers(
                context,
                reset_worker,
                cpus,
                lambda rows, group: (
                    discrete,
                    config,
                    rows.start,
                    rows.stop - rows.start,
                    num_envs,
                    seed,
                    group,
                    self._buffers,
                    specs,
                ),
                daemon=True,
            )
        except BaseException:
            self._abort()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError("PushTResetPool is closed")

    def _columns(self, group):
        return {key: view.copy() for (name, key), view in self._views.items() if name == group}

    def reset(self, seed, options, latest, sync=None):
        """Reset every lane with ``options[i]`` and the ``latest`` action columns; return owned columns.

        ``sync`` holds static columns whose ``owned`` fields the workers install before the reset.
        Returns the ``static``, ``current`` and ``geometry`` columns.
        """
        if sync is not None:
            for key, value in sync.items():
                if key.startswith("owned/"):
                    self._views[("static", key)][:] = value
        payloads = [
            (
                sync is not None,
                None if seed is None else int(seed) + rows.start,
                options[rows],
                tuple(column[rows] for column in latest),
            )
            for rows in self._slices
        ]
        failures = [reply for reply in self._request("reset", payloads) if reply is not None]
        if failures:
            self._request("settle", [False] * len(self._slices))
            kind, message = failures[0]
            error = getattr(builtins, kind, None)
            raise (error if isinstance(error, type) and issubclass(error, Exception) else RuntimeError)(message)
        columns = tuple(self._columns(group) for group in ("static", "current", "geometry"))
        self._request("settle", [True] * len(self._slices))
        return columns

    def install_variation(self, lanes, immediate):
        """Apply ``PushTReplayAdapter.install_variation`` in the workers; return the static columns."""
        self._request("install_variation", [(lanes[rows], immediate) for rows in self._slices])
        return self._columns("static")

    def variation_rows(self, attribute, keys):
        return np.concatenate(self._request("variation_rows", [(attribute, keys)] * len(self._slices)))
