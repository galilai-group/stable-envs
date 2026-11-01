"""DMC lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from ...._execution.process import ProcessAdapter, leaf_tensordict
from .factory import make_local

_ALIGN = 64


def _lanes(domain, count, seed):
    return make_local(domain, count, seed=seed)


class DMCProcessAdapter(ProcessAdapter):
    """Run DMC lanes in worker processes; physics stays on CPU when outputs use CUDA."""

    name = "DMC"

    def __init__(self, domain, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        self.domain = domain
        super().__init__(partial(_lanes, domain), num_envs, device, int(seed), processes=processes, cpus=cpus)

    def _to_device(self, leaves):
        """Copy ``{key: host array}`` to the CUDA output device in one transfer from a reused pinned buffer."""
        offsets, total = [], 0
        for value in leaves.values():
            offsets.append(total)
            total += -(-value.nbytes // _ALIGN) * _ALIGN
        if getattr(self, "_pinned", None) is None or self._pinned.numel() < total:
            self._pinned = torch.empty(max(total, 1), dtype=torch.uint8, pin_memory=True)
        host = self._pinned.numpy()
        for offset, value in zip(offsets, leaves.values(), strict=True):
            host[offset : offset + value.nbytes].view(value.dtype).reshape(value.shape)[...] = value
        moved = self._pinned[:total].to(self.device)
        return {
            key: moved[offset : offset + value.nbytes]
            .view(torch.from_numpy(value[:0]).dtype)
            .reshape(value.shape)
            .clone()
            for (key, value), offset in zip(leaves.items(), offsets, strict=True)
        }

    def _owned(self, group):
        if self.device.type != "cuda":
            return super()._owned(group)
        leaves = {key[1:]: view for key, view in self._views.items() if key[0] == group}
        return leaf_tensordict(self._to_device(leaves), self.num_envs, self.device)

    def _state_leaves(self):
        if self.device.type != "cuda" or self._static:
            return super()._state_leaves()
        return self._to_device({key[1:]: view for key, view in self._views.items() if key[0] == "state"})

    def configure_physics(self, values):
        self._check_idle()
        if self.physics_layout is None:
            raise ValueError(f"{self.domain} declares no physics Variation")
        values = np.asarray(values, np.float64).reshape(self.num_envs, self.physics_dim).copy()
        if not np.isfinite(values).all():
            raise ValueError("Configuration values must be finite")
        self.call("configure_physics", values)
