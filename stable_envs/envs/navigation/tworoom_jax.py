"""Batched TwoRoom physics with Torch/JAX buffer exchange.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.

Torch owns the live state. Steps exchange device buffers through DLPack and return
new position arrays. Borrowed inputs are not retained between steps or donated to
JAX, so public state snapshots keep independent storage. Packing position, action
and style into one device buffer amortizes the DLPack stream handoff and handles
strided or unaligned inputs. The first call compiles the physics kernel;
masking, observations and task outputs remain in the shared navigation code.
"""

from __future__ import annotations

import functools
import os

import torch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.dlpack  # noqa: E402
import jax.numpy as jnp  # noqa: E402

try:
    from jax import enable_x64
except ImportError:  # Older JAX keeps the context manager under experimental.
    from jax.experimental import enable_x64


def apply_collisions(position, proposed, *, radius, axis, thickness, doors, sizes, number):
    """Resolve border, wall and doorway collisions."""
    lower = (14.0 + radius.astype(jnp.float64)).astype(jnp.float32)
    upper = (210.0 - radius.astype(jnp.float64)).astype(jnp.float32)
    bounded = jnp.clip(proposed, lower, upper)
    half = (thickness.astype(jnp.int32) // 2).astype(jnp.float64)
    low = (112.0 - half - radius.astype(jnp.float64)).astype(jnp.float32)
    high = (112.0 + half + radius.astype(jnp.float64)).astype(jnp.float32)
    across = jnp.where(axis > 0.5, 0, 1)
    along = 1 - across
    started_low = position[across] < 112.0
    # Truncate door count, all three slots retain static shape.
    active = jnp.arange(3) < jnp.trunc(number)
    inside = (bounded[along] >= doors - sizes - 1.75) & (bounded[along] <= doors + sizes + 1.75)
    doorway = jnp.any(active & inside)
    blocked = jnp.where(started_low, bounded[across] > low, bounded[across] < high) & ~doorway
    pushed = jnp.where(started_low, low - 0.5, high + 0.5)
    return bounded.at[across].set(jnp.where(blocked, pushed, bounded[across]))


@functools.lru_cache(maxsize=8)
def _compiled(layout):
    fields = {path: slice(offset, offset + width) for path, offset, width in layout}

    def advance(position, action, style):
        leaf = lambda path: style[fields[path]]
        # Preserve Torch's separate float32 multiply/add rounding for arbitrary
        # continuous actions; a barrier alone does not prevent LLVM contracting FMA.
        clipped = jnp.clip(action, -1.0, 1.0).astype(jnp.float64)
        rounded = lambda x: jax.lax.reduce_precision(x, exponent_bits=8, mantissa_bits=23).astype(jnp.float32)
        delta = rounded(clipped * leaf("agent.speed")[0].astype(jnp.float64))
        proposed = rounded(position.astype(jnp.float64) + delta.astype(jnp.float64))
        return apply_collisions(
            position,
            proposed,
            radius=leaf("agent.radius")[0],
            axis=leaf("wall.axis")[0],
            thickness=leaf("wall.thickness")[0],
            doors=leaf("door.position"),
            sizes=leaf("door.size"),
            number=leaf("door.number")[0],
        )

    def advance_packed(row):
        return advance(row[:2], row[2:4], row[4:])

    return jax.jit(jax.vmap(advance_packed))


def from_torch(value):
    """Import a Torch tensor into JAX without changing its device."""
    value = value.detach().contiguous()
    # A contiguous time-slice view can still be unaligned for XLA's CPU/CUDA kernels.
    alignment = 64 if value.device.type == "cpu" else 16
    if value.data_ptr() % alignment:
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


class Kernel:
    """JIT-compiled TwoRoom dynamics with DLPack exchange on CPU or CUDA."""

    def __init__(self, device, layout):
        if device.type not in ("cpu", "cuda"):
            raise ValueError("TwoRoom supports cpu or cuda devices")
        try:
            devices = jax.devices("gpu" if device.type == "cuda" else "cpu")
            self.device = devices[
                device.index
                if device.index is not None
                else (torch.cuda.current_device() if device.type == "cuda" else 0)
            ]
        except (RuntimeError, IndexError) as exc:
            raise RuntimeError("TwoRoom JAX device unavailable; CUDA needs stable-envs[navigation-gpu]") from exc
        self._advance = _compiled(tuple((path, offset, width) for path, (offset, width) in layout.items()))

    def advance(self, position, action, style):
        # Scope x64 to this kernel instead of changing the caller's global JAX configuration.
        # No donation: Torch still owns every input buffer, and public snapshots must survive.
        with enable_x64(), jax.default_device(self.device):
            # One stream handoff costs less than importing three small buffers.
            inputs = torch.cat((position, action, style), dim=1)
            return torch.from_dlpack(self._advance(from_torch(inputs)))
