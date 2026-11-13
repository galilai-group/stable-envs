"""Persistent worker processes that draw Cube/Scene states through OpenGL into one shared frame buffer."""

from __future__ import annotations

import atexit
import mmap
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from multiprocessing import reduction

import numpy as np
import torch

from ... import tuning
from ..._core.config import log
from ..._execution.workers import send_failure, send_ready
from . import manipulation_opengl

_LOCK = threading.Lock()
_SIZE = ("renderer", "size")


class _WorkerError(RuntimeError):
    """A worker raised while drawing and remains ready for the next request."""


_POOLS: dict[tuple[str, str, int], _Pool | None] = {}


def _worker(conn, family, device_type, threads):
    try:
        torch.set_num_threads(1)
        if threads is not None:
            os.environ["LP_NUM_THREADS"] = str(threads)
        frames = None
        send_ready(conn)
        while True:
            message = conn.recv()
            if message is None:
                break
            if message[0] == "buffer":
                fd = reduction.recv_handle(conn)
                if frames is not None:
                    frames.close()
                frames = mmap.mmap(fd, message[1])
                os.close(fd)
                continue
            try:
                _, offset, columns = message
                size = int(columns[_SIZE][0, 0])
                images = np.ndarray((len(columns[_SIZE]), size, size, 3), np.uint8, frames, offset)
                manipulation_opengl.draw(family, columns, device_type, images)
            except Exception:  # noqa: BLE001 - the parent raises it; this worker stays usable
                send_failure(conn)
            else:
                conn.send((True, None))
    except BaseException:
        send_failure(conn)


class _Pool:
    """Workers with their own OpenGL contexts; each request splits rows into contiguous slices."""

    def __init__(self, family, device_type, workers):
        self.family, self.device_type = family, device_type
        self.processes, self.connections = [], []
        self.capacity, self.frames, self.fd = 0, None, -1
        context = mp.get_context("spawn")
        threads = _llvmpipe_threads(device_type, workers)
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker, args=(child, family, device_type, threads), daemon=True, name="ogb-opengl"
                )
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            self._replies(self.connections, timeout=300)
        except BaseException:
            self.close()
            raise

    def _replies(self, connections, timeout=600):
        errors = []
        deadline = time.monotonic() + timeout
        for conn, process in zip(connections, self.processes):
            while not conn.poll(0.01):
                if not process.is_alive() or time.monotonic() >= deadline:
                    raise RuntimeError(f"Cube/Scene OpenGL worker {process.pid} exited or timed out")
            ok, value = conn.recv()
            if not ok:
                errors.append(value)
        if errors:
            raise _WorkerError("Cube/Scene OpenGL worker failed:\n" + "\n".join(errors))

    def _reserve(self, nbytes):
        if nbytes <= self.capacity:
            return
        capacity = max(nbytes, 2 * self.capacity)
        fd = _anonymous_file()
        try:
            os.ftruncate(fd, capacity)
            frames = mmap.mmap(fd, capacity)
            for conn, process in zip(self.connections, self.processes):
                conn.send(("buffer", capacity))
                reduction.send_handle(conn, fd, process.pid)
        except BaseException:
            os.close(fd)
            raise
        if self.frames is not None:
            self.frames.close()
            os.close(self.fd)
        self.capacity, self.frames, self.fd = capacity, frames, fd

    def draw(self, columns, size):
        """Return (B, size, size, 3) uint8 frames of validated NumPy state columns; the array views the buffer."""
        n = len(columns[_SIZE])
        frame = size * size * 3
        self._reserve(n * frame)
        bounds = np.linspace(0, n, min(n, len(self.processes)) + 1).astype(int)
        active = []
        for conn, lo, hi in zip(self.connections, bounds[:-1].tolist(), bounds[1:].tolist()):
            conn.send(("draw", lo * frame, {key: value[lo:hi] for key, value in columns.items()}))
            active.append(conn)
        self._replies(active)
        return np.ndarray((n, size, size, 3), np.uint8, self.frames)

    def close(self):
        for conn in self.connections:
            try:
                conn.send(None)
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for conn in self.connections:
            conn.close()
        self.processes, self.connections = [], []
        if self.frames is not None:
            self.frames.close()
            os.close(self.fd)
            self.frames = None


def _anonymous_file():
    """Return a descriptor of an unlinked file in shared memory, which no process leaves behind."""
    fd, path = tempfile.mkstemp(prefix="stable-envs-ogb-", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
    os.unlink(path)
    return fd


def _llvmpipe_threads(device_type, workers):
    if device_type != "cpu" or "LP_NUM_THREADS" in os.environ:
        return None
    return max(1, len(os.sched_getaffinity(0)) // workers)


def workers(batch, device_type="cpu"):
    """Number of OpenGL worker processes a batch of ``batch`` states uses; 0 draws in this process."""
    if batch < 2 or not sys.platform.startswith("linux") or mp.current_process().daemon:
        return 0
    count = min(tuning.get("ogb_gl_workers"), len(os.sched_getaffinity(0)))
    if device_type == "cpu":
        # Single-threaded software workers beat one multi-threaded context once each worker has a state.
        return count if count >= 2 and batch >= count else 0
    # NVIDIA frames are fast: below about eight states per worker, inter-process traffic outweighs the parallel draws.
    count = min(count, batch // 8)
    return count if count >= 2 else 0


def _pool(family, device_type):
    count = min(tuning.get("ogb_gl_workers"), len(os.sched_getaffinity(0)))
    key = (family, device_type, count)
    if key not in _POOLS:
        for other in [k for k in _POOLS if k[:2] == key[:2]]:
            if (stale := _POOLS.pop(other)) is not None:
                stale.close()
        try:
            _POOLS[key] = _Pool(family, device_type, count)
        except Exception as exc:  # noqa: BLE001 - drawing in this process stays correct
            log("WARNING", f"Cube/Scene OpenGL workers did not start; drawing in this process: {exc}")
            _POOLS[key] = None
    return _POOLS[key], key


def draw(family, columns, device_type, size):
    """Frames of validated NumPy state columns drawn by worker processes, or None when they draw in this process.

    The returned array is a view of the shared buffer, valid until the next call.
    """
    if not workers(len(columns[_SIZE]), device_type):
        return None
    with _LOCK:
        pool, key = _pool(family, device_type)
        if pool is None:
            return None
        try:
            return pool.draw(columns, size)
        except _WorkerError:
            raise
        except BaseException:
            pool.close()
            del _POOLS[key]
            raise


@atexit.register
def _close_all():
    for pool in _POOLS.values():
        if pool is not None:
            pool.close()
    _POOLS.clear()
