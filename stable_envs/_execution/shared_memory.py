"""Typed views and owned snapshots for process outputs; families define their schemas."""

import ctypes
from collections.abc import Mapping
from multiprocessing.context import BaseContext
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

BufferSpecs = Mapping[tuple[str, ...], tuple[tuple[int, ...], np.dtype]]


def allocate_buffers(context: BaseContext, specs: BufferSpecs, batch: int) -> dict[tuple[str, ...], Any]:
    return {
        key: context.RawArray(ctypes.c_ubyte, int(batch * np.prod(shape) * dtype.itemsize))
        for key, (shape, dtype) in specs.items()
    }


def buffer_views(buffers: Mapping, specs: BufferSpecs, batch: int) -> dict[tuple[str, ...], np.ndarray]:
    return {
        key: np.frombuffer(buffers[key], dtype=dtype).reshape(batch, *shape) for key, (shape, dtype) in specs.items()
    }


def owned_group(
    views: Mapping[tuple[str, ...], np.ndarray], group: str, batch: int, device: torch.device
) -> TensorDict:
    # Copy before the next request: workers will overwrite these buffers.
    return TensorDict(
        {key[1:]: torch.from_numpy(value.copy()).to(device) for key, value in views.items() if key[0] == group},
        [batch],
        device=device,
    )
