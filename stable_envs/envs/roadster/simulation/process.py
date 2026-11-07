"""Roadster lanes in worker processes."""

from __future__ import annotations

from functools import partial

from ...._execution.process import FRAME_FIELDS, ProcessAdapter
from .adapter import RENDER_FIELDS, RoadsterAdapter, physics_rows


def _lanes(task, count, seed):
    return RoadsterAdapter(task, count, "cpu")


class RoadsterProcessAdapter(ProcessAdapter):
    """Run the lanes of one Roadster task in persistent spawned workers, with State in shared buffers.

    Each worker owns a local ``RoadsterAdapter``; lane ``i`` resets with ``seed + i`` as in one local batch.
    Workers write only the State fields a step changes; the fields only resets and restores change stay
    cached here. A step that needs a frame but no State hands the render-row fields to ``draw_fields``,
    which the env installs, so no snapshot is built for it (``FRAME_FIELDS``).
    """

    name = "Roadster"
    frame_mode = FRAME_FIELDS

    def __init__(self, task, num_envs, device="cpu", seed=0, *, processes=8, cpus=None):
        #: ``{State key: host array}`` render-row fields -> frames; installed by the env.
        self.draw_fields = None
        super().__init__(partial(_lanes, task), num_envs, device, seed, processes=processes, cpus=cpus)

    @property
    def live_frames(self) -> bool:
        """Step frames come from the shared State fields, so no worker frame buffer is needed."""
        return bool(self._static) and self.draw_fields is not None

    def _frame_fields(self) -> dict:
        """Copy just the render-row fields: the changing ones from the shared buffers, the rest from the cache."""
        if self._static_cache is None:
            self._state_leaves()  # fills the cache of the fields a step leaves unchanged
        return {
            key: (
                self._static_cache[key].numpy() if key in self._static_cache else self._views[("state", *key)].copy()
            )
            for key in RENDER_FIELDS
        }

    def _fields_frame(self, fields):
        return self.draw_fields(fields)

    def validate(self, state, *, batch=None):
        """The local adapter's checks on the whole State, before any worker sees it; any batch unless given."""
        self._check_open()
        RoadsterAdapter.validate(self, state, batch=batch)

    def set_state(self, state) -> None:
        self.validate(state, batch=self.num_envs)
        super().set_state(state)

    def configure_physics(self, values) -> None:
        """Install per-lane ``(wind speed, rain)`` rows for the next reset on every worker."""
        self._check_idle()
        self.call("configure_physics", physics_rows(values, self.num_envs))

    def advance(self, actions, *, active=None):
        return self.step(actions, active=active, capture_state=False)

    def close(self):
        self.draw_fields = None
        super().close()
