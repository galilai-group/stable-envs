"""PushT lanes in worker processes."""

from __future__ import annotations

from functools import partial

import numpy as np
import torch

from ...._execution.process import FRAME_FIELDS, ProcessAdapter, leaf_tensordict
from ...._execution.shared_memory import key_path
from .replay import ACTION_DTYPES, reset_options, validate_seed
from .replay_batch import PushTReplayAdapter, lane_options, reject_jax_state, state_columns

_ACTIONS, _CODES, _VALID = ("history", "actions"), ("history", "dtypes"), ("history", "valid")


def _lanes(discrete, config, count, seed):
    return PushTReplayAdapter(count, "cpu", seed, discrete=discrete, config=config)


class PushTProcessAdapter(ProcessAdapter):
    """Run PushT replay owners in worker processes.

    The parent records each lane's action history, so the shared buffers hold
    only fixed-size state fields. Restore replays the history in the workers.
    """

    name = "PushT"
    _pipe_leaves = frozenset({_ACTIONS, _CODES, _VALID})

    def __init__(self, batch, device="cpu", seed=0, *, discrete=False, config=None, processes=8, cpus=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("PushT batch must be a positive integer")
        validate_seed(seed)
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("PushT supports CPU physics with CPU/CUDA State storage")
        self._lengths = np.zeros(batch, np.int64)
        self._actions = np.zeros((batch, 0, 2), np.float64)
        self._codes = np.zeros((batch, 0), np.uint8)
        super().__init__(partial(_lanes, discrete, config), batch, device, seed, processes=processes, cpus=cpus)
        self._keys = [key_path(key) for key in self._metadata.get_state().keys(True, True)]
        self._fallback = None
        from .. import raster

        resolution = self._metadata.template.config["resolution"]
        if self.device.type == "cuda" and raster.HAVE_TRITON and raster.resize_supported(resolution):
            # Workers write the state leaves a step changes; the parent rasterizes them on the GPU.
            self.frame_mode = FRAME_FIELDS

    @property
    def template(self):
        """A lane whose configuration, variation space and state fields describe the batch."""
        self._check_open()
        return self._metadata.template

    def _state(self):
        fixed = self._state_leaves()
        width = int(self._lengths.max())
        history = {
            _ACTIONS: self._actions[:, :width].copy(),
            _CODES: self._codes[:, :width].copy(),
            _VALID: np.arange(width) < self._lengths[:, None],
        }
        return leaf_tensordict(
            {
                key: torch.as_tensor(history[key], device=self._output_device) if key in history else fixed[key]
                for key in self._keys
            },
            self.num_envs,
            self._output_device,
        )

    def _frame_fields(self):
        if self._static_cache is None:
            self._state_leaves()
        fields = {}
        for key, view in self._views.items():
            if key[0] == "state":
                cached = self._static_cache.get(key[1:])
                fields["/".join(key[1:])] = view.copy() if cached is None else cached.numpy()
        return fields

    def _fields_frame(self, fields):
        from .. import raster

        size = self._metadata.template.config["resolution"]
        return raster.render(fields, size, self.device, lambda rows: self._draw_pygame(rows, size))

    def _draw_pygame(self, fields, size):
        from ..projection import PushTProjection

        if self._fallback is None:
            self._fallback = PushTProjection()
        count = len(fields["recipe/config"])
        return self._fallback.render([{k: v[i] for k, v in fields.items()} for i in range(count)], size)

    def _abort(self):
        try:
            super()._abort()
        finally:
            if getattr(self, "_fallback", None) is not None:
                self._fallback.close()
                self._fallback = None

    def validate(self, state):
        self._check_open()
        reject_jax_state(state)
        state_columns(state, self._metadata.template.get_state(), self.num_envs)

    def set_state(self, state):
        super().set_state(state)
        host = state.cpu()
        self._lengths = host["history", "lengths"].numpy().astype(np.int64)
        self._actions = host[_ACTIONS].numpy().copy()
        self._codes = host[_CODES].numpy().copy()

    def reset(self, seed=None, options=None):
        self._check_idle()
        validate_seed(seed)
        space = self.template.env.variation_space
        lanes = (
            options
            if isinstance(options, list)
            else [lane_options(space, options, self.num_envs, lane) for lane in range(self.num_envs)]
        )
        for lane in lanes:
            reset_options(None, lane)
        self._lengths = np.zeros(self.num_envs, np.int64)
        self._actions = np.zeros((self.num_envs, 0, 2), np.float64)
        self._codes = np.zeros((self.num_envs, 0), np.uint8)
        return self._reset(seed, [(lanes[rows],) for rows in self._slices])

    def _submitted(self, actions, active):
        lanes = np.arange(self.num_envs) if active is None else np.flatnonzero(active)
        if not len(lanes):
            return
        positions = self._lengths[lanes]
        if int(positions.max()) >= self._actions.shape[1]:
            capacity = max(16, 2 * self._actions.shape[1])
            self._actions = np.concatenate(
                [self._actions, np.zeros((self.num_envs, capacity - self._actions.shape[1], 2), np.float64)], axis=1
            )
            self._codes = np.concatenate(
                [self._codes, np.zeros((self.num_envs, capacity - self._codes.shape[1]), np.uint8)], axis=1
            )
        self._actions[lanes, positions] = actions[lanes]
        self._codes[lanes, positions] = ACTION_DTYPES.index(actions.dtype)
        self._lengths[lanes] += 1
