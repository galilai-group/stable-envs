"""Run a CPU simulator adapter for an accelerator device: inputs move to the host, outputs move to the device."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from .._core.tensordicts import build_tensordict, to_device

_CPU = torch.device("cpu")


def _to_host(value: Any) -> Any:
    if torch.is_tensor(value) or isinstance(value, TensorDict):
        return value.cpu() if value.device.type != "cpu" else value
    return value


class HostAdapter:
    """A CPU adapter whose tensor results move to ``device``.

    The adapter keeps its State on the host and steps there. Each call moves its tensor arguments to the host and its
    tensor results to ``device``. A TensorDict result, or a dict of tensors with one row per batch row, moves in one
    host-to-device copy. Other attributes are the adapter's own.
    """

    def __init__(self, adapter: Any, device: str | torch.device):
        self._adapter = adapter
        self.device = torch.device(device)

    @property
    def host_adapter(self) -> Any:
        """The wrapped CPU adapter."""
        return self._adapter

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, TensorDict):
            return to_device(value, self.device)
        if torch.is_tensor(value):
            return value.to(self.device)
        if type(value) is dict and value:
            n = getattr(self._adapter, "num_envs", None)
            if all(
                torch.is_tensor(v) and v.device.type == "cpu" and v.ndim and v.shape[0] == n for v in value.values()
            ):
                if all(type(k) is str for k in value):
                    return dict(to_device(build_tensordict(value, [n], _CPU), self.device).items())
            return {k: self._to_device(v) for k, v in value.items()}
        if type(value) in (tuple, list):
            return type(value)(self._to_device(v) for v in value)
        return value

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._adapter, name)
        if not callable(value) or isinstance(value, type):
            return self._to_device(value) if torch.is_tensor(value) or isinstance(value, TensorDict) else value

        def call(*args, **kwargs):
            args = tuple(_to_host(a) for a in args)
            kwargs = {k: _to_host(v) for k, v in kwargs.items()}
            return self._to_device(value(*args, **kwargs))

        call.__name__ = name
        return call
