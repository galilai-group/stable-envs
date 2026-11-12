"""PushT batches stepped by the JAX contact solver, on CUDA or on the JAX CPU backend."""

import os

import torch

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.dlpack

from .batch import PushTBatchAdapter
from .jax_physics import advance

enable_x64 = jax.enable_x64 if hasattr(jax, "enable_x64") else jax.experimental.enable_x64


def from_torch(value):
    value = value.detach().contiguous()
    if value.data_ptr() % (16 if value.is_cuda else 64):
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


class PushTJaxAdapter(PushTBatchAdapter):
    """Step PushT lanes with batched JAX physics on the device of the State."""

    def __init__(self, batch, device="cpu", seed=0, **kwargs):
        device = torch.device(device)
        if device.type == "cuda":
            try:
                devices = jax.devices("gpu")
                devices[device.index if device.index is not None else torch.cuda.current_device()]
            except (RuntimeError, IndexError) as exc:
                raise RuntimeError(
                    f"PushT JAX cannot use {device}; install CUDA support with "
                    "pip install 'stable-envs[pusht-gpu]' and ensure JAX can see the selected GPU"
                ) from exc
        super().__init__(batch, device, seed, **kwargs)

    def _advance(self, target, active):
        with enable_x64():
            outputs = advance(
                from_torch(self._body),
                from_torch(self._bias),
                from_torch(self._impulse),
                from_torch(self._identifiers),
                tuple(from_torch(v) for v in self._geometry),
                from_torch(target),
                self._circle,
            )
            return tuple(torch.from_dlpack(v) for v in outputs)
