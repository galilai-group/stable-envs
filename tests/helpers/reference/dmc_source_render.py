"""Render DM Control render rows through the pinned stable-worldmodel wrappers.

The wrappers compile their own appearance into the model: target colour, shape and visibility, and
the light. The DMC classes take their default appearance from them, so this is the reference for
what is drawn by default. Exact-pixel parity is checked against plain dm_control instead, because
the wrappers' scene edits change most pixels by a level or two.
"""

from __future__ import annotations

import importlib
import inspect

import numpy as np

from tests.helpers.reference.dmc_semantics import WRAPPERS
from tests.helpers.reference.profiles import PROFILES

# Classes registered as stable-worldmodel environments; the other DMC tasks follow plain dm_control.
# DMCManipulatorBringBall is aligned, but its declared default is the compiled model (target ball drawn)
# and reaches stable-worldmodel's hidden default through visibility.target_ball (see variation_maps).
ALIGNED_NAMES = frozenset(case.name for case in PROFILES["dmc"]) - {"DMCManipulatorBringBall"}

# Kept for the process lifetime: releasing a MuJoCo GL context while others are live corrupts them.
_WRAPPERS: dict = {}


def _wrapper(cls, size):
    key = (cls, int(size))
    if key not in _WRAPPERS:
        module = importlib.import_module(f"stable_worldmodel.envs.dmcontrol.{cls.DOMAIN}")
        wrapper_cls = getattr(module, WRAPPERS[cls.DOMAIN] + "DMControlWrapper")
        kwargs = {"task": cls.TASK} if "task" in inspect.signature(wrapper_cls.__init__).parameters else {}
        wrapper = wrapper_cls(seed=0, **kwargs)
        wrapper.reset(seed=0)
        _WRAPPERS[key] = wrapper
    return _WRAPPERS[key]


def render_rows(cls, wrapper, states, size):
    p = wrapper.env.physics
    m = p.model.ptr if hasattr(p.model, "ptr") else p.model
    slots = cls._slots()
    nq = int(m.nq)
    out = []
    for st in np.asarray(states, np.float64):
        for (arr, k), v in zip(slots, st[nq:]):
            getattr(m, arr).reshape(-1)[k] = v
        with p.reset_context():
            p.data.qpos[:] = st[:nq]
            p.data.qvel[:] = 0
        got = np.concatenate(
            [np.asarray(p.data.qpos, np.float64), [np.asarray(getattr(m, arr)).reshape(-1)[k] for arr, k in slots]]
        )
        if np.abs(got - st).max() > 1e-9:
            raise AssertionError(
                f"{cls.__name__}: the stable-worldmodel reference did not reach the requested state "
                f"(max {np.abs(got - st).max():.3e})"
            )
        out.append(wrapper.render(width=size, height=size).copy())
    return np.stack(out)


def appearance_reference(cls, size):
    """A ``states -> frames`` renderer for a stable-worldmodel-aligned DMC class, else None."""
    if cls.__name__ not in ALIGNED_NAMES:
        return None
    try:
        wrapper = _wrapper(cls, size)
    except ImportError:
        return None
    return lambda states: render_rows(cls, wrapper, states, int(size))
