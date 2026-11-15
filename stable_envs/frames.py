"""Offline frames: draw stored states ahead in large ``env.render`` calls and hand them out a batch at a time.

The chunk size comes from memory alone, never from timing: two chunks of frames and renderer working memory
per row must fit in a quarter of the free memory of the device, and one chunk of frames stays under
``MAX_CHUNK_BYTES`` so that every renderer can index it with 32-bit offsets. A chunk holds at most the family's
``Env.render_ahead_rows`` and never fewer rows than a batch.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping

import numpy as np
import torch
from tensordict import TensorDictBase

from .env import Env, State

#: Most bytes of frames in one chunk; below 2**31, so that 32-bit offsets reach every pixel.
MAX_CHUNK_BYTES = 1 << 30
#: Least working memory per row; it covers the per-world buffers of the MuJoCo Warp renderers (DMCReacher at 84x84
#: holds about 0.9 MiB per row).
MIN_ROW_WORK = 4 << 20
#: Share of the free device memory that the frames and working buffers of render-ahead may use.
MEMORY_SHARE = 4


def _host_available() -> int:
    """Bytes the process can still allocate on the host: MemAvailable, and the cgroup limit where one is set."""
    out = []
    try:
        with open("/proc/meminfo") as f:
            out += [int(line.split()[1]) * 1024 for line in f if line.startswith("MemAvailable:")]
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/memory.max") as f, open("/sys/fs/cgroup/memory.current") as g:
            limit = f.read().strip()
            if limit != "max":
                out.append(int(limit) - int(g.read()))
    except (OSError, ValueError):
        pass
    if not out:
        try:
            out.append(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
        except (ValueError, OSError, AttributeError):
            out.append(MAX_CHUNK_BYTES)
    return max(0, min(out))


def _free_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    return _host_available()


def chunk_rows(env: Env, batch_size: int, total: int, size: int | None = None, supersample: int | None = None) -> int:
    """Rows of one render-ahead chunk: ``total`` when it fits, else a multiple of ``batch_size`` and at least one batch.

    A chunk holds at most ``env.render_ahead_rows`` rows. A row needs two frames (the chunk that the caller holds and the next one) and renderer working memory: a
    float32 RGBA image at the supersampled size, and at least ``MIN_ROW_WORK``.
    """
    b = max(1, int(batch_size))
    if total <= b:
        return max(1, int(total))
    h, w = (size, size) if size is not None else tuple(env.frame_spec.shape[:2])
    ss = int(supersample if supersample is not None else getattr(env, "ss", 1) or 1)
    frame = int(h) * int(w) * 3
    per_row = 2 * frame + max(16 * int(h) * int(w) * ss * ss, MIN_ROW_WORK)
    rows = min(_free_bytes(env.device) // MEMORY_SHARE // per_row, MAX_CHUNK_BYTES // frame)
    rows = min(rows, int(env.render_ahead_rows))
    return int(total) if rows >= total else max(b, rows // b * b)


class FrameLoader:
    """Iterate over the frames of stored states, ``batch_size`` frames at a time, in order.

    The loader draws a chunk of many batches with one ``env.render`` call and yields its slices, so a small
    batch costs the same per frame as a large one. It picks the chunk size from the free memory of
    ``env.device``. Each batch has the bytes of ``env.render`` of its states, except where a renderer's bytes
    depend on the batch size: OGBCube and OGBScene on CUDA draw 16 or more states in OpenGL worker processes,
    whose frames can differ from one process by 1 level in a few pixels.

    Args:
        env: The environment that draws. Its live state doesn't change.
        states: A full State with batch ``(N,)``, or compact rows ``(N, state_dim)`` as ``sample_states``
            returns them (NumPy or Torch).
        batch_size: Frames per yielded batch. The last batch is shorter when ``N`` is not a multiple.
        indices: Optional order of the rows to draw, for example a permutation drawn ahead. A sequence or
            1-D array of row indices; rows may repeat. None draws every row once in order.
        chunk_size: Rows per ``env.render`` call. None (the default) picks it from device memory.
        **render_kwargs: Passed to every ``env.render`` call, for example ``size`` or ``variation``. A
            ``variation`` of ``(N, d)`` rows, one per stored state, is sliced with the states (and ``indices``).
            A Mapping or one shared row applies to every batch.

    Yields:
        ``(batch_size, H, W, 3)`` uint8 frames on ``env.device``.

    Raises:
        TypeError: If ``states`` is neither a State nor an array of rows, or ``out`` is passed.
        ValueError: If ``batch_size`` or ``chunk_size`` is not positive, ``indices`` is not 1-D, a
            ``variation`` has more than one row but not one per stored state, or compact rows draw under a
            per-environment variation with a ``batch_size`` other than ``num_envs``.
    """

    def __init__(
        self,
        env: Env,
        states: State | torch.Tensor | np.ndarray,
        batch_size: int,
        *,
        indices=None,
        chunk_size: int | None = None,
        **render_kwargs,
    ) -> None:
        if int(batch_size) < 1:
            raise ValueError(f"FrameLoader: batch_size must be positive, got {batch_size}")
        if chunk_size is not None and int(chunk_size) < 1:
            raise ValueError(f"FrameLoader: chunk_size must be positive, got {chunk_size}")
        if "out" in render_kwargs:
            raise TypeError("FrameLoader: out is not accepted; the loader allocates the frames of each chunk")
        if isinstance(states, TensorDictBase):
            if len(states.batch_size) != 1:
                raise ValueError(f"FrameLoader: expected a State with batch (N,), got {list(states.batch_size)}")
            count = int(states.batch_size[0])
        else:
            states = torch.as_tensor(states)
            if states.ndim != 2:
                raise ValueError(f"FrameLoader: expected rows (N, state_dim), got shape {tuple(states.shape)}")
            count = int(states.shape[0])
        stored = count
        variation = render_kwargs.get("variation")
        self._variation = None
        if variation is not None and not isinstance(variation, Mapping):
            rows = torch.as_tensor(variation)
            if rows.ndim == 2 and rows.shape[0] > 1:
                if rows.shape[0] != stored:
                    raise ValueError(
                        f"FrameLoader: variation has {rows.shape[0]} rows; pass one row per stored state "
                        f"({stored}) or one shared row"
                    )
                self._variation = rows
                render_kwargs = {k: v for k, v in render_kwargs.items() if k != "variation"}
        if indices is not None:
            indices = torch.as_tensor(indices, dtype=torch.long)
            if indices.ndim != 1:
                raise ValueError(f"FrameLoader: indices must be 1-D, got shape {tuple(indices.shape)}")
            count = int(indices.shape[0])
        self.env = env
        self.states = states
        self.indices = indices
        self.batch_size = int(batch_size)
        self.render_kwargs = render_kwargs
        self._count = count
        self._chunk = int(chunk_size) if chunk_size is not None else None
        if self._per_environment() and self.batch_size != env.B:
            raise ValueError(
                f"FrameLoader: compact rows draw under the environment's per-environment variation of {env.B} "
                f"rows, so batch_size must be {env.B}, got {self.batch_size}"
            )
        if self._per_environment() and isinstance(variation, Mapping) and count % self.batch_size:
            raise ValueError(
                "FrameLoader: a variation Mapping over a per-environment variation needs a row count that is a "
                f"multiple of batch_size ({self.batch_size})"
            )

    def __len__(self) -> int:
        return -(-self._count // self.batch_size)

    @property
    def chunk_size(self) -> int:
        """Rows that one ``env.render`` call draws."""
        if self._chunk is not None:
            return self._chunk
        if not self._chunks_allowed():
            return self.batch_size
        kw = self.render_kwargs
        return chunk_rows(self.env, self.batch_size, self._count, kw.get("size"), kw.get("supersample"))

    def _per_environment(self) -> bool:
        """Whether compact rows draw under a per-environment variation: row ``i`` of a batch with lane ``i``."""
        variation = self.render_kwargs.get("variation")
        if isinstance(self.states, TensorDictBase) or self._variation is not None:
            return False
        if variation is not None and not isinstance(variation, Mapping):
            return False
        style = getattr(self.env, "_style", None)
        return style is not None and not bool(getattr(style, "shared", True))

    def _chunks_allowed(self) -> bool:
        """Whether a chunk may hold more rows than a batch and keep the same frames. Rows that draw under a
        per-environment variation keep one batch per call."""
        return not self._per_environment()

    def _take(self, values, start: int, stop: int):
        if self.indices is None and start == 0 and stop == self._count:
            return values.to(self.env.device)
        if self.indices is None:
            return values[start:stop].to(self.env.device)
        return values[self.indices[start:stop].to(values.device)].to(self.env.device)

    def _kwargs(self, start: int, stop: int) -> dict:
        if self._variation is not None:
            return {**self.render_kwargs, "variation": self._take(self._variation, start, stop)}
        if stop - start < self.batch_size and self._per_environment():
            # A short last batch draws its rows with the first lanes' variation, as a full batch does.
            return {**self.render_kwargs, "variation": self.env._style.values[: stop - start]}
        return self.render_kwargs

    def __iter__(self) -> Iterator[torch.Tensor]:
        b, n = self.batch_size, self._count
        chunk = self.chunk_size
        if chunk % b and chunk < n:
            chunk = max(b, chunk // b * b)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            frames = self.env.render(self._take(self.states, start, stop), **self._kwargs(start, stop))
            for i in range(0, stop - start, b):
                yield frames[i : i + b]
