"""Draw CPU frames in persistent worker processes that write into one shared frame block."""

from __future__ import annotations

import atexit
import builtins
import ctypes
import multiprocessing as mp
import os
import sys
import threading
import weakref
from collections.abc import Callable, Hashable
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from .. import tuning
from .._core.config import log
from .requests import ProcessRequests, lane_slices
from .workers import pin_worker, send_failure, send_ready

#: True inside a render worker, which draws its rows in-process.
IN_WORKER = False

_LOCK = threading.Lock()
_POOLS: dict[Hashable, RenderPool | int] = {}
# Attempts to start one pool; after that its batches draw in-process.
_STARTS = 3
_SCHED_GETCPU = getattr(ctypes.CDLL(None), "sched_getcpu", None) if sys.platform == "linux" else None


def _current_cpu() -> int | None:
    """The CPU this thread runs on, or None where the platform does not report it."""
    cpu = _SCHED_GETCPU() if _SCHED_GETCPU is not None else -1
    return cpu if cpu >= 0 else None


def _worker(conn, factory, cpus, software_gl):
    global IN_WORKER
    IN_WORKER = True
    block = None
    try:
        if software_gl and cpus and "LP_NUM_THREADS" not in os.environ:
            # Mesa reads this when it creates the context: one rasterizer thread per CPU of this worker.
            os.environ["LP_NUM_THREADS"] = str(len(cpus))
        pin_worker(cpus)
        draw = factory()
        send_ready(conn)
        import torch.multiprocessing  # noqa: F401 - registers the reductions that receive shared tensors

        while True:
            command, payload = conn.recv()
            if command == "close":
                conn.send((True, None))
                break
            reply = None
            if payload is not None:
                target, start, count, shape, args = payload
                if isinstance(target, str):
                    if block is None or block.name != target:
                        if block is not None:
                            block.close()
                        # Spawned workers share the parent's resource tracker; the parent unlinks the block.
                        block = shared_memory.SharedMemory(name=target)
                    frames = np.ndarray((start + count, *shape), np.uint8, buffer=block.buf)[start:]
                else:
                    frames = target[start : start + count].numpy()
                try:
                    draw(args, frames)
                except Exception as exc:  # noqa: BLE001 - the parent raises it; this worker stays usable
                    reply = (type(exc).__name__, str(exc))
                del frames
            conn.send((True, reply))
    except BaseException:
        send_failure(conn)
    finally:
        if block is not None:
            block.close()
        conn.close()


def _drop_block(block):
    try:
        block.close()
    except BufferError:
        pass
    block.unlink()


class RenderPool(ProcessRequests):
    """Spawned workers that each build one drawer with ``factory()`` and draw contiguous row slices.

    A drawer is called as ``draw(args, frames)`` and writes ``frames`` in place. A frame must depend
    only on its row, so the split does not change the frames.
    """

    def __init__(
        self, factory: Callable[[], Callable], processes: int, cpus=None, *, software_gl=False, name="Render"
    ):
        self._worker_family = name
        self._closed, self._pending = False, None
        self._connections, self._processes = [], []
        self._slices = lane_slices(processes, processes)
        self._block = None
        self.lock = threading.Lock()
        context = mp.get_context("spawn")
        groups = [None] * processes if cpus is None else [g.tolist() for g in np.array_split(cpus, processes)]
        self._groups = groups
        try:
            for group in groups:
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, factory, group, software_gl), daemon=True)
                try:
                    process.start()
                finally:
                    child.close()
                self._connections.append(parent)
                self._processes.append(process)
            self.worker_info = self._responses(timeout=900)
        except BaseException:
            self._abort()
            raise

    @property
    def processes(self) -> int:
        return len(self._processes)

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{self._worker_family} pool is closed")

    def _responses(self, timeout=None):
        # A large batch can take minutes; the wait still ends when a worker exits.
        return super()._responses(timeout=24 * 3600 if timeout is None else timeout)

    def render(
        self,
        count: int,
        shape: tuple[int, ...],
        args_of: Callable[[slice], Any],
        out=None,
        during: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Return ``(count, *shape)`` uint8 frames; worker ``k`` draws the rows ``args_of(slice_k)`` describes.

        Without ``out``, the workers write into a new shared block that the result views, so the frames
        are not copied again. With ``out``, they write into a block kept across calls and the frames are
        copied into ``out``. ``during()`` runs in this process while the workers draw; an error it raises
        is raised after they finish, and no frames return.
        """
        nbytes = max(count * int(np.prod(shape)), 1)
        if out is None:
            # A fresh block per call: the workers fault its pages in parallel and the result views it.
            block = shared_memory.SharedMemory(create=True, size=nbytes)
        else:
            # One block serves every call that copies into ``out``.
            if self._block is None or self._block.size < nbytes:
                self._release()
                self._block = shared_memory.SharedMemory(create=True, size=nbytes)
            block = self._block
        target = block.name
        parts = lane_slices(count, min(count, self.processes))
        payloads = [(target, rows.start, rows.stop - rows.start, shape, args_of(rows)) for rows in parts]
        # A woken worker pinned to the CPU of this process preempts it, so that worker is woken last.
        cpu = _current_cpu()
        order = sorted(range(self.processes), key=lambda i: cpu in (self._groups[i] or ()))
        self._submit("render", payloads + [None] * (self.processes - len(parts)), order)
        if during is not None:
            try:
                during()
            except BaseException:
                try:
                    self._receive()
                finally:
                    if out is None:
                        _drop_block(block)
                raise
        replies = self._receive()
        failures = [reply for reply in replies if reply is not None]
        if failures:
            if out is None:
                _drop_block(block)
            kind, message = failures[0]
            error = getattr(builtins, kind, None)
            raise (error if isinstance(error, type) and issubclass(error, Exception) else RuntimeError)(message)
        frames = np.ndarray((count, *shape), np.uint8, buffer=block.buf)
        if out is None:
            weakref.finalize(frames, _drop_block, block)
            return frames
        np.copyto(out, frames)
        return out

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


class EnvRows:
    """Picklable factory of a drawer that renders compact rows and appearance rows with one in-process env."""

    def __init__(self, name: str, **kwargs):
        self.name, self.kwargs = name, kwargs

    def __call__(self):
        import torch

        from .. import registry

        env = registry.REGISTRY[self.name](num_envs=1, device="cpu", **self.kwargs)

        def draw(args, frames):
            rows, style, options = args
            env._render(torch.from_numpy(rows), style=torch.from_numpy(style), out=torch.from_numpy(frames), **options)

        return draw


class ReferenceRows:
    """Picklable factory of a drawer that renders compact rows with the env class's reference renderer."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self):
        from .. import registry

        cls = registry.REGISTRY[self.name]
        env = cls.reference_env()

        def draw(rows, frames):
            for row, frame in zip(rows, frames, strict=True):
                cls.reference_set_state(env, row)
                frame[...] = env.render()

        return draw


def render_rows(
    env, key: Hashable, rows, style, shape, path: str, out=None, *, factory=None, software_gl=False, **options
):
    """Draw host ``rows`` and appearance ``style`` of a CPU env in its render workers.

    Return the frames as a tensor, or None when this batch draws in-process. ``options`` are passed to
    the workers' ``_render``. With ``factory``, workers draw with ``factory()`` and receive only the rows.
    """
    import torch

    workers = pool(
        key,
        factory
        or EnvRows(type(env).__name__, size=env.size, supersample=env.ss, **getattr(env, "render_worker_kwargs", {})),
        len(rows),
        path,
        int(shape[1]),
        software_gl=software_gl,
        name=f"{type(env).__name__} render",
    )
    if workers is None:
        return None
    rows = np.ascontiguousarray(rows)
    style = None if factory else np.ascontiguousarray(style)
    with workers.lock:
        frames = workers.render(
            len(rows),
            tuple(shape),
            (lambda part: rows[part]) if factory else (lambda part: (rows[part], style[part], options)),
            None if out is None else out.numpy(),
        )
    return out if out is not None else torch.from_numpy(frames)


def worker_count(count: int, path: str, size: int, min_rows: int | None = None) -> int:
    """Return the number of render workers for ``count`` rows on render path ``path``, or 0 to draw in-process.

    ``min_rows`` replaces the fallback of the ``render_worker_min_rows`` knob for this call site.
    """
    if IN_WORKER or sys.platform != "linux" or mp.current_process().daemon or not hasattr(os, "sched_getaffinity"):
        return 0
    if min_rows is None:
        threshold = tuning.get("render_worker_min_rows", path=path, size=size, batch=count)
    else:
        threshold = tuning.get("render_worker_min_rows", path=path, size=size, batch=count, default=min_rows)
    if count < int(threshold or 0):
        return 0
    workers = min(
        int(tuning.get("render_workers", path=path, size=size, batch=count) or 0), len(os.sched_getaffinity(0))
    )
    return workers if workers >= 2 else 0


def pool(
    key: Hashable,
    factory: Callable[[], Callable],
    count: int,
    path: str,
    size: int,
    *,
    min_rows: int | None = None,
    **kwargs,
) -> RenderPool | None:
    """Return the shared pool for ``key`` when ``count`` rows use workers, or None to draw in-process.

    ``min_rows`` is as in :func:`worker_count`. A pool that fails to start is logged; after ``_STARTS``
    failures its key draws in-process.
    """
    workers = worker_count(count, path, size, min_rows)
    if not workers:
        return None
    with _LOCK:
        full = (key, workers)
        item = _POOLS.get(full)
        if isinstance(item, int) and item >= _STARTS:
            return None
        if not isinstance(item, RenderPool):
            for other in [k for k in _POOLS if k[0] == key and k != full]:
                stale = _POOLS.pop(other)
                if isinstance(stale, RenderPool):
                    stale.close()
            try:
                _POOLS[full] = RenderPool(factory, workers, sorted(os.sched_getaffinity(0)), **kwargs)
            except Exception as exc:  # noqa: BLE001 - drawing in this process gives the same frames
                failures = (item or 0) + 1
                log(
                    "WARNING",
                    f"{kwargs.get('name', 'Render')} workers did not start ({failures} of {_STARTS} tries); drawing in this process: {exc}",
                )
                _POOLS[full] = failures
                return None
        return _POOLS[full]


def close_all() -> None:
    """Close every render pool."""
    with _LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for item in pools:
        if isinstance(item, RenderPool):
            try:
                item.close()
            except Exception:  # noqa: BLE001 - exiting
                pass


atexit.register(close_all)
