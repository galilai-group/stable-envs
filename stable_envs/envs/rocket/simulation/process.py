"""Rocket lanes in worker processes."""

from __future__ import annotations

from functools import partial
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import torch

from ...._execution.process import ProcessAdapter
from .adapter import RocketAdapter, check_values, state_rows

#: Lanes one process may own. A staged restore or reset holds two PyBullet clients per lane, frames
#: use one more client, and PyBullet allows 1024 clients in one process.
LANES_PER_PROCESS = 511


def _lanes(config, frame, count, seed):
    return RocketAdapter(count, "cpu", seed, config, frame)


class RocketProcessAdapter(ProcessAdapter):
    """Run Rocket replay owners in worker processes with the whole State in shared buffers."""

    name = "Rocket"

    def __init__(self, batch, device, seed, config=None, *, processes=8, cpus=None, frame=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        if torch.device(device).type not in ("cpu", "cuda"):
            raise ValueError("Rocket supports CPU dynamics and CPU/CUDA State storage")
        if type(processes) is int and processes > 0 and -(-batch // processes) > LANES_PER_PROCESS:
            raise ValueError(f"Rocket processes may own at most {LANES_PER_PROCESS} lanes each")
        # Workers write ``draw`` frames here; it grows to the largest batch drawn.
        self._canvas = None
        super().__init__(partial(_lanes, config, frame), batch, device, seed, processes=processes, cpus=cpus)

    @property
    def template(self):
        """A lane whose configuration and Variation space describe the batch."""
        self._check_open()
        return self._metadata.template

    def validate(self, state):
        self._check_open()
        return state_rows(state, self.fields, [self._metadata.template] * self.num_envs)

    def rgba(self):
        return self.call("rgba").to(self.device)

    def configure(self, values):
        self._check_idle()
        check_values(self._metadata.template, values, self.num_envs)
        self.call("configure", values)

    def reset(self, seed=None, options=None, values=None):
        self._check_idle()
        template = self._metadata.template
        if values is not None and len(values) != self.num_envs:
            raise ValueError("Expected one Rocket Variation row per lane")
        for lane in range(self.num_envs):
            template.check_reset(
                None if seed is None else seed + lane, options, None if values is None else values[lane]
            )
        return self._reset(seed, [(options, None if values is None else values[rows]) for rows in self._slices])

    def draw(self, body, links, colors, size, ss):
        """Draw explicit poses with TinyRenderer, split across the workers.

        Worker ``k`` draws rows ``k::workers``, so neighbouring rows of similar cost spread across workers.
        While a step is in flight, or for one row, the parent draws them.
        """
        self._check_open()
        count = min(len(body), len(self._connections))
        if self._pending is not None or count < 2:
            return self._metadata.draw(body, links, colors, size, ss)
        shape = (len(body), size, size, 3)
        if self._canvas is None or self._canvas.size < int(np.prod(shape)):
            self._release_canvas()
            self._canvas = SharedMemory(create=True, size=int(np.prod(shape)))
        rows = [slice(k, None, count) for k in range(count)] + [slice(0, 0)] * (len(self._connections) - count)
        starts = np.cumsum([0] + [len(range(len(body))[r]) for r in rows])
        frame = size * size * 3
        self._request(
            "call",
            [
                ("draw", (body[r], links[r], colors[r], size, ss, (self._canvas.name, int(start) * frame)))
                for r, start in zip(rows, starts[:-1], strict=True)
            ],
        )
        canvas = np.ndarray(shape, np.uint8, buffer=self._canvas.buf)
        out = np.empty(shape, np.uint8)
        for r, start, stop in zip(rows, starts[:-1], starts[1:], strict=True):
            out[r] = canvas[start:stop]
        del canvas
        return out

    def _release_canvas(self):
        if self._canvas is not None:
            self._canvas.close()
            self._canvas.unlink()
            self._canvas = None

    def _abort(self):
        try:
            self._release_canvas()
        finally:
            super()._abort()
