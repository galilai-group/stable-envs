"""Draw PushT state rows in worker processes."""

from __future__ import annotations

import builtins
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

from ..._execution.requests import ProcessRequests, lane_slices
from ..._execution.workers import pin_worker, send_failure, send_ready


def render_worker(conn, cpus):
    """Draw the rows of each request into the shared frame block that the request names."""
    import cv2

    from .projection import PushTProjection

    projection, block = None, None
    try:
        pin_worker(cpus)
        cv2.setNumThreads(1)
        projection = PushTProjection()
        send_ready(conn)
        while True:
            command, payload = conn.recv()
            if command == "close":
                conn.send((True, None))
                break
            reply = None
            if payload is not None:
                name, start, fields, size = payload
                if block is None or block.name != name:
                    if block is not None:
                        block.close()
                    # Spawned workers share the parent's resource tracker; the parent unlinks the block.
                    block = shared_memory.SharedMemory(name=name)
                count = len(fields["recipe/config"])
                rows = [{key: value[lane] for key, value in fields.items()} for lane in range(count)]
                frames = np.ndarray((start + count, size, size, 3), np.uint8, buffer=block.buf)[start:]
                try:
                    projection.render(rows, size, out=frames)
                except Exception as exc:
                    reply = (type(exc).__name__, str(exc))
                del frames
            conn.send((True, reply))
    except BaseException:
        send_failure(conn)
    finally:
        if projection is not None:
            projection.close()
        if block is not None:
            block.close()
        conn.close()


class PushTRenderPool(ProcessRequests):
    """Draw state rows in spawned workers, each with its own projection.

    Each worker draws a contiguous slice of the rows into one shared frame block.
    A frame depends only on its row, so the split does not change the frames.
    """

    _worker_family = "PushT render"

    def __init__(self, processes, cpus=None):
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        self._slices = lane_slices(processes, processes)
        self._block = None
        try:
            self._spawn_workers(mp.get_context("spawn"), render_worker, cpus, lambda rows, group: (group,))
        except BaseException:
            self._abort()
            raise

    @property
    def processes(self):
        return len(self._processes)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("PushTRenderPool is closed")

    def render(self, fields, size):
        """Return owned ``[N, size, size, 3]`` uint8 frames of the host column ``fields``."""
        count = len(fields["recipe/config"])
        nbytes = count * size * size * 3
        if self._block is None or self._block.size < nbytes:
            self._release()
            self._block = shared_memory.SharedMemory(create=True, size=max(nbytes, 1))
        parts = lane_slices(count, min(count, self.processes))
        payloads = [(self._block.name, rows.start, {k: v[rows] for k, v in fields.items()}, size) for rows in parts]
        replies = self._request("render", payloads + [None] * (self.processes - len(parts)))
        failures = [reply for reply in replies if reply is not None]
        if failures:
            kind, message = failures[0]
            error = getattr(builtins, kind, None)
            raise (error if isinstance(error, type) and issubclass(error, Exception) else RuntimeError)(message)
        return np.ndarray((count, size, size, 3), np.uint8, buffer=self._block.buf).copy()

    def _release(self):
        if self._block is not None:
            self._block.close()
            self._block.unlink()
            self._block = None

    def close(self):
        try:
            super().close()
        finally:
            self._release()
