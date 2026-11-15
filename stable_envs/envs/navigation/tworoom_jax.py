"""Batched TwoRoom physics with Torch/JAX buffer exchange.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.

Torch owns the live state; steps exchange device buffers through DLPack. ``Kernel.step`` runs the physics, the
masking, the observation and the task outputs in one jitted call. It imports the actions, and the step mask when
there is one. The rows, the target and the Variation stay on the JAX side between steps, as copies that are taken
again when their Torch tensors change. The call returns one new buffer, and the step result is a TensorDict of
views of it. Torch's stream waits for that buffer, so a later Torch write to the borrowed actions cannot race the
kernel. No buffer is donated, so public state snapshots keep independent storage. The first call compiles.
"""

from __future__ import annotations

import functools
import os

import torch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.dlpack  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ..._core.state_layout import StateLayout  # noqa: E402

try:
    from jax._src.lib import _jax as _xla
except ImportError:
    _xla = None
# The protocol layers of jax.dlpack and torch.from_dlpack cost more than the step at small batches;
# the jaxlib calls under them do the same exchange.
if not all(hasattr(_xla, name) for name in ("dlpack_managed_tensor_to_buffer", "buffer_to_dlpack_managed_tensor")):
    _xla = None

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


#: The leaves of a TwoRoom step result, in TensorDict order, as ``(key, per-row shape, dtype)``.
OUTPUT_LEAVES = (
    ("state/agent/position", (2,), torch.float32),
    ("state/target/position", (2,), torch.float32),
    ("reward", (), torch.float64),
    ("terminated", (), torch.bool),
    ("truncated", (), torch.bool),
    ("observation", (10,), torch.float32),
    ("info/proprio", (2,), torch.float32),
    ("info/state", (2,), torch.float32),
    ("info/goal_state", (2,), torch.float32),
    ("info/distance_to_target", (), torch.float64),
)


def _physics(layout):
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

    return fields, jax.vmap(advance)


@functools.lru_cache(maxsize=8)
def _compiled(layout):
    _, advance = _physics(layout)
    return jax.jit(lambda packed: advance(packed[:, :2], packed[:, 2:4], packed[:, 4:]))


#: Float32 words per row of the packed step output: the float32 leaves, then the float64 leaves.
_FLOAT_WORDS = 20
_WORDS = _FLOAT_WORDS + 4


@functools.lru_cache(maxsize=8)
def _compiled_step(layout, width):
    fields, advance = _physics(layout)

    def step(source, inputs, fixed):
        n = inputs.shape[0]
        # The previous packed output starts with the rows; after a change outside the step, the rows themselves.
        rows = source[: 2 * n].reshape(n, 2) if source.ndim == 1 else source
        style, target = fixed[:, :width], fixed[:, width:]
        column = lambda path: style[:, fields[path]]
        moved = advance(rows, inputs[:, :2], style)
        # A third input column is the step mask.
        agent = moved if inputs.shape[1] == 2 else jnp.where(inputs[:, 2:3] > 0.5, moved, rows)
        axis = column("wall.axis") > 0.5
        positions = column("door.position")
        center = jnp.full_like(positions, 112.0)
        doors = jnp.stack((jnp.where(axis, center, positions), jnp.where(axis, positions, center)), axis=-1)
        live = jnp.arange(positions.shape[1]) < jnp.trunc(column("door.number"))
        doors = jnp.where(live[..., None], doors, 0.0).reshape(n, -1)
        observation = jnp.concatenate((agent, target, doors), axis=1)
        # Torch's two-element CPU norm: round x*x to float32, add the exact y*y in float64, round, then sqrt.
        delta = (agent - target).astype(jnp.float64)
        first = jnp.square(delta[:, 0]).astype(jnp.float32).astype(jnp.float64)
        squared = (first + delta[:, 1] * delta[:, 1]).astype(jnp.float32).astype(jnp.float64)
        distance = jnp.sqrt(squared).astype(jnp.float32).astype(jnp.float64)
        doubles = jnp.concatenate((jnp.zeros(n, jnp.float64), distance))
        flags = jnp.concatenate((distance < 16.0, jnp.zeros(-(-2 * n // 4) * 4 - n, jnp.bool_))).astype(jnp.uint8)
        words = (
            # StateLayout's order: each dtype's leaves grouped by shape, in State order within a shape.
            *(v.reshape(-1) for v in (agent, target, agent, agent, target, observation)),
            jax.lax.bitcast_convert_type(doubles, jnp.float32).reshape(-1),
            jax.lax.bitcast_convert_type(flags.reshape(-1, 4), jnp.float32),
        )
        return jnp.concatenate(words)

    return jax.jit(step)


@functools.cache
def _jax_device(kind, index):
    return jax.devices("gpu" if kind == "cuda" else "cpu")[index or 0]


@functools.cache
def _external_stream(device):
    """Torch's handle on the stream that JAX makes external buffers wait for on ``device``."""
    return torch.cuda.ExternalStream(device.get_stream_for_external_ready_events())


def from_torch(value, device=None):
    """Import a Torch tensor into JAX without changing its device."""
    value = value.detach().contiguous()
    # A contiguous time-slice view can still be unaligned for XLA's CPU/CUDA kernels.
    alignment = 64 if value.device.type == "cpu" else 16
    if value.data_ptr() % alignment:
        value = value.clone()
    if _xla is None:
        return jax.dlpack.from_dlpack(value, copy=False)
    device = device or _jax_device(value.device.type, value.device.index)
    if not value.is_cuda:
        return _xla.dlpack_managed_tensor_to_buffer(torch.utils.dlpack.to_dlpack(value), device, None, False, 1)
    # Make JAX's stream for external buffers wait for the work that Torch queued on the value.
    stream = _external_stream(device)
    ready = torch.cuda.Event()
    ready.record(torch.cuda.current_stream(value.device))
    stream.wait_event(ready)
    capsule = torch.utils.dlpack.to_dlpack(value)
    return _xla.dlpack_managed_tensor_to_buffer(capsule, device, stream.cuda_stream, False, 2)


def to_torch(array, device):
    """Export a JAX array on ``device`` to Torch, ready on Torch's current stream."""
    if _xla is None:
        return torch.from_dlpack(array)
    stream = None
    if device.type == "cuda":
        # DLPack names the legacy default stream 1.
        stream = torch.cuda.current_stream(device).cuda_stream or 1
    capsule = _xla.buffer_to_dlpack_managed_tensor(array.addressable_data(0), stream=stream)
    return torch.utils.dlpack.from_dlpack(capsule)


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
        key = tuple((path, offset, width) for path, (offset, width) in layout.items())
        self._advance = _compiled(key)
        self._step = _compiled_step(key, max(offset + width for _, offset, width in key))
        self._fixed: tuple = ()
        self._packed: tuple = ()
        self._outputs: StateLayout | None = None

    def advance(self, position, action, style):
        # Scope x64 to this kernel instead of changing the caller's global JAX configuration.
        # No donation: Torch still owns every input buffer, and public snapshots must survive.
        with enable_x64(), jax.default_device(self.device):
            # One stream handoff costs less than importing three small buffers.
            inputs = torch.cat((position, action, style), dim=1)
            return to_torch(self._advance(from_torch(inputs, self.device)), position.device)

    def step(self, rows, action, active, style, target):
        """Advance ``rows`` in place by ``action`` where ``active`` (all lanes when None) and return the step result.

        The result holds the state and the task outputs, as TwoRoom's shared Torch task code computes them.
        """
        n = rows.shape[0]
        if self._outputs is None or self._outputs.num_envs != n:
            self._outputs = StateLayout(OUTPUT_LEAVES, n)
        # Torch's stream waits for the output, so later Torch writes to a borrowed input wait for the kernel too.
        if active is None:
            inputs = from_torch(action, self.device)
        else:
            inputs = from_torch(torch.cat((action, active.reshape(n, 1).to(action.dtype)), dim=1), self.device)
        # The rows, the Variation and the target stay on the JAX side until their Torch tensors change.
        key = (id(style), style._version, id(target), target._version)
        if self._fixed[:1] != (key,):
            self._fixed = (key, style, target, from_torch(torch.cat((style, target), 1), self.device))
        fixed = self._fixed[3]
        packed = self._packed
        current = packed and packed[0] is rows and packed[1] == rows._version
        source = packed[2] if current else from_torch(rows.clone(), self.device)
        # Every input is committed to the device, so the call needs no default device.
        with enable_x64():
            out = self._step(source, inputs, fixed)
        words = to_torch(out, rows.device)
        sizes = (_FLOAT_WORDS * n, (_WORDS - _FLOAT_WORDS) * n, words.numel() - _WORDS * n)
        floats, doubles, flags = words.split(sizes)
        flags = flags.view(torch.uint8)[: 2 * n].view(torch.bool)
        result = self._outputs.views((floats, doubles.view(torch.float64), flags), rows.device)
        rows.copy_(result["state", "agent", "position"])
        self._packed = (rows, rows._version, out)
        return result
