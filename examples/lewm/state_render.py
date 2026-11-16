"""Draw stable-envs frames from the state rows of a LeWM dataset with the public ``Env.render``."""

from __future__ import annotations

import torch

import stable_envs as snv
from stable_envs import get_field


class StateRenderer:
    """Write dataset state rows into full States and draw them with ``env.render(state)``.

    ``leaves`` holds ``(field, start, stop)``: a dotted State field name and the ``[start, stop)`` columns of a
    state row that hold it. The fields in ``zero`` are set to zero. Every other field keeps its value from
    ``env.reset(seed)``. ``get_field`` returns a field as a view, so writes to it change the State.
    """

    def __init__(self, env_id, leaves, zero=(), *, device="cpu", size=None, env_kwargs=None, seed=0):
        self.env = snv.make(env_id, num_envs=1, device=device, **dict(env_kwargs or {}))
        self.env.reset(seed=seed)
        self.size = size
        self.leaves = [(key, int(start), int(stop)) for key, start, stop in leaves]
        self.zero = list(zero)
        self._template = self.env.get_state()
        self._batch = None

    @property
    def device(self) -> torch.device:
        return self.env.device

    def states(self, rows: torch.Tensor):
        """Return a full State with one environment per row of ``(N, D)`` state rows."""
        rows = torch.as_tensor(rows, device=self.env.device)
        n = int(rows.shape[0])
        if self._batch is None or self._batch.batch_size[0] != n:
            self._batch = self._template.expand(n).clone()
        state = self._batch
        for key, start, stop in self.leaves:
            field = get_field(state, key)
            field.copy_(rows[:, start:stop].reshape(field.shape).to(field.dtype))
        for key in self.zero:
            get_field(state, key).zero_()
        return state

    @torch.no_grad()
    def __call__(self, rows) -> torch.Tensor:
        """Return ``(*lead, H, W, 3)`` uint8 frames for ``(*lead, D)`` state rows."""
        rows = torch.as_tensor(rows)
        lead = tuple(rows.shape[:-1])
        kwargs = {} if self.size is None else {"size": self.size}
        frames = self.env.render(self.states(rows.reshape(-1, rows.shape[-1])), **kwargs)
        return frames.reshape(*lead, *frames.shape[-3:])

    def close(self) -> None:
        self.env.close()


def from_config(env_cfg, *, device="cpu", size=None) -> StateRenderer:
    """Build a :class:`StateRenderer` from the ``env`` section of a config (a mapping)."""
    leaves = [(item["leaf"], *item["columns"]) for item in env_cfg["leaves"]]
    kwargs = dict(env_cfg.get("kwargs") or {})
    return StateRenderer(env_cfg["id"], leaves, list(env_cfg["zero"]), device=device, size=size, env_kwargs=kwargs)
