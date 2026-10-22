"""JAX TwoRoom dynamics adapted from fracapuano/envX at d26c817c1124229c819be574f137cd68bda4380a.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.

Torch owns the live State. Steps exchange device buffers through DLPack and return
new arrays. Borrowed inputs are not retained between steps.
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
except ImportError:  # Older JAX kept the context manager under experimental.
    from jax.experimental import enable_x64


def apply_collisions(position, proposed, *, radius, axis, thickness, doors, sizes, number):
    """Resolve walls and doorways with stable-envs boundary rounding."""
    lower = (14.0 + radius.astype(jnp.float64)).astype(jnp.float32)
    upper = (210.0 - radius.astype(jnp.float64)).astype(jnp.float32)
    bounded = jnp.clip(proposed, lower, upper)
    half = (thickness.astype(jnp.int32) // 2).astype(jnp.float64)
    low = (112.0 - half - radius.astype(jnp.float64)).astype(jnp.float32)
    high = (112.0 + half + radius.astype(jnp.float64)).astype(jnp.float32)
    across = jnp.where(axis > 0.5, 0, 1)
    along = 1 - across
    started_low = position[across] < 112.0
    # Truncate door count, as main does; all three slots retain static shape.
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

    def step(position, target, action, style, active):
        agent = jax.vmap(advance)(position, action, style)
        if active is not None:
            agent = jnp.where(active[:, None], agent, position)
        leaf = lambda path: style[:, fields[path]]
        vertical = leaf("wall.axis") > 0.5
        coordinates = leaf("door.position")
        centers = jnp.full_like(coordinates, 112.0)
        doors = jnp.stack((jnp.where(vertical, centers, coordinates), jnp.where(vertical, coordinates, centers)), -1)
        doors = jnp.where((jnp.arange(3) < jnp.trunc(leaf("door.number")))[:, :, None], doors, 0.0)
        observation = jnp.concatenate((agent, target, doors.reshape(-1, 6)), axis=1)
        # Main's two-element CPU torch.norm accumulates x*x, then fma(y, y, sum),
        # and rounds sqrt to float32. A reassociated reduction can flip success
        # at exactly 16 pixels. Float64 intermediates reproduce that order on GPU.
        delta = (agent - target).astype(jnp.float64)
        first = jax.lax.optimization_barrier((delta[:, 0] * delta[:, 0]).astype(jnp.float32))
        squared = jax.lax.optimization_barrier(
            (delta[:, 1] * delta[:, 1] + first.astype(jnp.float64)).astype(jnp.float32)
        )
        distance = jax.lax.optimization_barrier(jnp.sqrt(squared.astype(jnp.float64)).astype(jnp.float32))
        return agent, observation, distance.astype(jnp.float64), distance < 16.0

    return jax.jit(step)


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
        self._step = _compiled(tuple((path, offset, width) for path, (offset, width) in layout.items()))

    def step(self, position, target, action, style, active):
        # Scope x64 to this kernel instead of changing the caller's global JAX configuration.
        # No donation: Torch still owns every input buffer, and public snapshots must survive.
        with enable_x64(), jax.default_device(self.device):
            inputs = tuple(from_torch(value) for value in (position, target, action, style))
            result = self._step(*inputs, None if active is None else from_torch(active))
            return tuple(torch.from_dlpack(value) for value in result)
