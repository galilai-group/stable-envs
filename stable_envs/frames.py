"""Offline frames: draw stored states ahead in large ``env.render`` calls and hand them out a batch at a time.

The chunk size comes from memory alone, never from timing: two chunks of frames and a renderer working buffer
per row must fit in a quarter of the free memory of the device, and one chunk of frames stays under
``MAX_CHUNK_BYTES`` so that every renderer can index it with 32-bit offsets.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import numpy as np
import torch
from tensordict import TensorDictBase

from .env import Env, State

#: Most bytes of frames in one chunk; below 2**31, so that 32-bit offsets reach every pixel.
MAX_CHUNK_BYTES = 1 << 30
#: Share of the free device memory that the frames and working buffers of render-ahead may use.
MEMORY_SHARE = 4


def _free_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
    try:
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return MAX_CHUNK_BYTES


def chunk_rows(env: Env, batch_size: int, total: int, size: int | None = None, supersample: int | None = None) -> int:
    """Rows of one render-ahead chunk: a multiple of ``batch_size`` that fits in memory, at most ``total``.

    A row needs two frames (the chunk that the caller holds and the next one) and a float32 RGBA image at the
    supersampled size as renderer working memory.
    """
    b = max(1, int(batch_size))
    if total <= b:
        return max(1, int(total))
    h, w = (size, size) if size is not None else tuple(env.frame_spec.shape[:2])
    ss = int(supersample if supersample is not None else getattr(env, "ss", 1) or 1)
    frame = int(h) * int(w) * 3
    per_row = 2 * frame + 16 * int(h) * int(w) * ss * ss
    rows = min(_free_bytes(env.device) // MEMORY_SHARE // per_row, MAX_CHUNK_BYTES // frame, total)
    return max(b, rows // b * b)


class FrameLoader:
    """Iterate over the frames of stored states, ``batch_size`` frames at a time, in order.

    The loader draws a chunk of many batches with one ``env.render`` call and yields its slices, so a small
    batch costs the same per frame as a large one. It picks the chunk size from the free memory of
    ``env.device``. Each batch has the same bytes as ``env.render`` of its states.

    Args:
        env: The environment that draws. Its live state doesn't change.
        states: A full State with batch ``(N,)``, or compact rows ``(N, state_dim)`` as ``sample_states``
            returns them (NumPy or Torch).
        batch_size: Frames per yielded batch. The last batch is shorter when ``N`` is not a multiple.
        indices: Optional order of the rows to draw, for example a permutation drawn ahead. A sequence or
            1-D array of row indices; rows may repeat. None draws every row once in order.
        chunk_size: Rows per ``env.render`` call. None (the default) picks it from device memory.
        **render_kwargs: Passed to every ``env.render`` call, for example ``size`` or ``variation``.

    Yields:
        ``(batch_size, H, W, 3)`` uint8 frames on ``env.device``.

    Raises:
        TypeError: If ``states`` is neither a State nor an array of rows, or ``out`` is passed.
        ValueError: If ``batch_size`` or ``chunk_size`` is not positive, or ``indices`` is not 1-D.
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

    def _chunks_allowed(self) -> bool:
        """Whether a chunk may hold more rows than a batch and keep the same frames.

        A per-row ``variation=`` fixes the batch to its rows. Rows without a full variation row draw
        under the environment's variation, which must then be one shared row.
        """
        variation = self.render_kwargs.get("variation")
        shape = () if variation is None or isinstance(variation, dict) else tuple(np.shape(variation))
        if len(shape) > 1 and shape[0] > 1:
            return False
        if isinstance(self.states, TensorDictBase) or shape:
            return True
        style = getattr(self.env, "_style", None)
        return style is None or bool(getattr(style, "shared", True))

    def _take(self, start: int, stop: int):
        if self.indices is None:
            return self.states[start:stop].to(self.env.device)
        return self.states[self.indices[start:stop].to(self.states.device)].to(self.env.device)

    def __iter__(self) -> Iterator[torch.Tensor]:
        b, n = self.batch_size, self._count
        chunk = self.chunk_size
        if chunk % b and chunk < n:
            chunk = max(b, chunk // b * b)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            frames = self.env.render(self._take(start, stop), **self.render_kwargs)
            for i in range(0, stop - start, b):
                yield frames[i : i + b]
