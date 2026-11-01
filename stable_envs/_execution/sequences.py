"""Run a sequence of actions on a local simulator adapter, as the step dispatcher runs them one at a time."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping

import numpy as np
import torch

#: Task outputs whose value ends a lane's sequence.
END_KEYS = (("terminated",), ("truncated",), ("done",))


def ended(arrays, count: int) -> np.ndarray:
    """Lanes whose ``terminated``, ``truncated`` or ``done`` output is set."""
    done = np.zeros(count, bool)
    for key in END_KEYS:
        if key in arrays:
            done |= np.asarray(arrays[key]).astype(bool).reshape(count, -1).any(-1)
    return done


def arrays_of(result) -> dict[tuple[str, ...], np.ndarray]:
    """Return the host leaves of a step result as ``{key tuple: array}``, without State."""
    return {
        key if isinstance(key, tuple) else (key,): value.detach().cpu().numpy()
        for key, value in result.exclude("state").items(True, True)
    }


def step_arrays(adapter) -> Callable:
    """Return ``f(actions, active) -> {key tuple: array}``: the task outputs of one step without State.

    ``active`` is None or a NumPy bool mask. Adapters that define ``step_arrays`` give their arrays
    directly; for other adapters, the outputs of ``step(..., capture_state=False)`` are converted.
    """
    own = getattr(adapter, "step_arrays", None)
    if own is not None:
        return own

    def convert(actions, active):
        active = None if active is None else torch.from_numpy(active)
        return arrays_of(adapter.step(actions, active=active, capture_state=False))

    return convert


def run_sequence(
    adapter,
    actions,
    active: np.ndarray | None,
    outputs: MutableMapping[tuple[str, ...], np.ndarray],
    valid: np.ndarray,
) -> np.ndarray:
    """Step ``adapter`` once per action of ``actions`` ``[n, T, ...]`` and return the lanes still active.

    A lane takes actions from the start, or from ``active`` when given, until a step ends it.
    ``valid[:, t]`` receives the lanes that took action ``t``. ``outputs`` maps each task output key to
    an ``[n, T, ...]`` array that receives the step outputs; an empty mapping is filled with new arrays
    at the first step. Slots of lanes that took no action are left unchanged.
    """
    count, length = actions.shape[:2]
    step = step_arrays(adapter)
    lanes = np.ones(count, bool) if active is None else np.array(active, bool)
    allocate = not outputs
    for t in range(length):
        valid[:, t] = lanes
        # Finished lanes stay finished; a step of only finished lanes changes no simulator.
        if not lanes.any():
            continue
        result = step(actions[:, t], None if lanes.all() else lanes.copy())
        if allocate:
            for key, value in result.items():
                outputs[key] = np.zeros((count, length, *value.shape[1:]), value.dtype)
            allocate = False
        for key, buffer in outputs.items():
            buffer[:, t] = result[key]
        lanes &= ~ended(result, count)
    return lanes
