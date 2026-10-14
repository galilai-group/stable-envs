"""Shared conventions for the GPU suite.

- :func:`make_or_skip` skips when an env's optional dependency is missing; any other build error fails.
- :func:`family_params` and :func:`resolve_family` make an empty family sweep skip when an env module
  failed to import and fail otherwise.
- :func:`assert_frames_identical` compares uint8 frames and reports the max difference and pixel count.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se

#: The parameter a family sweep is given when the family is empty. Never a real env name.
EMPTY = "__no_env__"


def make_or_skip(name: str, **kw):
    """``se.make(name, **kw)``, skipping when this interpreter lacks the env's optional dependency.

    Only ``ModuleNotFoundError`` skips: the registry lists every env name, installed or not, while any
    other construction error is a defect.
    """
    try:
        return se.make(name, **kw)
    except ModuleNotFoundError as e:
        pytest.skip(f"{name} needs {e.name}, which this interpreter does not have")


def family_params(names, *, family: str) -> list:
    """One ``pytest.param`` per env name, or a single ``EMPTY`` param when the family is empty.

    :func:`resolve_family` decides whether the ``EMPTY`` case skips or fails.
    """
    if names:
        return [pytest.param(n, id=n) for n in sorted(names)]
    return [pytest.param(EMPTY, id=f"{family}-EMPTY")]


def resolve_family(name: str, *, family: str, modules=(), subject: str) -> None:
    """Skip or fail an ``EMPTY`` family case; call it first in every family-swept test.

    Skips with the import error when one of ``modules`` (the env modules that would populate the
    family) failed to import. Fails otherwise, since the selection matched nothing.
    """
    if name != EMPTY:
        return
    errs = se.import_errors()
    blamed = {m: errs[m] for m in modules if m in errs}
    if blamed:
        pytest.skip(
            f"{family}: no env registered in this interpreter because "
            + "; ".join(f"{m} did not import ({e})" for m, e in blamed.items())
            + f"; {subject} is not checked here"
        )
    expected = ", ".join(modules) if modules else "any env module"
    pytest.fail(
        f"{family}: {subject} has no env to check it on, and {expected} imported cleanly. The registry "
        f"holds {len(se.registry.REGISTRY)} envs, so the family selection matches nothing."
    )


def sampled_states(env, n: int, seed: int):
    """``n`` states from an explicitly seeded generator, never the env's own ``rng``.

    The ``env`` fixtures here are module-scoped, so drawing from ``env.rng`` would make the states
    depend on test order.
    """
    return env.sample_states(n, rng=np.random.default_rng(seed))


def frame_diff(a, b) -> tuple[int, int]:
    """``(max abs difference in levels, number of pixels that differ)`` between two uint8 frames."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        a_ = torch.as_tensor(a).to(torch.int16)
        b_ = torch.as_tensor(b).to(torch.int16)
        d = (a_ - b_).abs()
        return int(d.max()), int((d > 0).any(-1).sum())
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return int(d.max()), int((d > 0).any(-1).sum())


def assert_frames_identical(got, want, *, what: str, why: str) -> None:
    """Assert ``got`` and ``want`` are the same bytes, reporting the max difference and pixel count.

    Both numbers distinguish a rounding change (small max, many pixels) from a shape change.
    """
    assert tuple(got.shape) == tuple(want.shape), (
        f"{what}: shapes differ, {tuple(got.shape)} against {tuple(want.shape)}. {why}"
    )
    mx, npx = frame_diff(got, want)
    assert mx == 0, f"{what}: differ by max {mx}/255 on {npx} pixels. {why}"
