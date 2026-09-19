"""The three conventions the GPU suite kept getting wrong, in one place because they were five places.

Nothing here is a helper for its own sake. Each of the three cost this project a real afternoon while it
was open-coded per file, and each failed in a way that looked like a code defect and was not.

**A missing optional dependency SKIPS, with a reason.** ``se.registry.REGISTRY`` is the UNION over interpreters:
an env module that cannot import is recorded by ``stable_envs.envs.import_errors()`` and its classes are
simply absent, so ``se.make(name)`` for a name the registry knows can still raise ``ModuleNotFoundError``
somewhere else. ``test_fused_cast.py`` forgave that; ``test_mover_window.py`` and
``test_raster_culling.py`` did not, and a run in an interpreter without ``dm_control`` turned a fact
about the interpreter into red marks against a kernel that is pure torch. Separately,
``test_gfxdraw_fused.py`` went red for four tests because it imported ``envs/carracing.py`` directly,
bypassing the guarded discovery, for a Box2D the code under test does not use. :func:`make_or_skip` and
:func:`import_or_skip` are now the only two ways to reach either.

**An EMPTY sweep is a FAILURE, not a pass.** The other side of the same coin, and the more expensive
one. Every family-parameterised file in this directory used to write ``@parametrize("name", NAMES or
["_none"])`` with a body that skipped on ``"_none"``, so a family with no registered env collected
exactly one green skip and the run said nothing. That is precisely how the DM Control family's 16 envs
— including the reference parity test, which is this repo's whole warrant — collected ZERO cases for a
week behind a green suite. :func:`family_params` and :func:`resolve_family` keep the distinction that
matters: an empty family with a recorded import error explaining it SKIPS with that error attached;
an empty family with nothing to explain it FAILS.

**One frame comparison, one message.** ``(a.to(int16) - b.to(int16)).abs().max() == 0`` was written out
ten times with ten different messages, several of which reported only the max and not how many pixels
moved — which is the number that says whether a difference is one rounding or a shape in the wrong
place. :func:`assert_frames_identical` reports both, always.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se

#: The parameter a family sweep is given when the family is empty. Never a real env name.
EMPTY = "__no_env__"


def import_or_skip(module: str, why: str):
    """Import a module the TEST needs, or skip naming what it was for.

    Distinct from ``pytest.importorskip`` only in that it insists on ``why``: "carracing's reference
    rasterizer is unavailable here; the fused kernel under test is pure torch and does not need it" is
    a useful line in a report and "ModuleNotFoundError: Box2D" is not.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as e:  # noqa: BLE001
        pytest.skip(f"{module} is not importable in this interpreter ({e}); {why}")


def make_or_skip(name: str, **kw):
    """``se.make(name, **kw)``, skipping when this interpreter lacks the env's optional dependency.

    ONLY ``ModuleNotFoundError`` is forgiven, and that is the whole design. The registry is the union
    over interpreters, so a name being registered here does not mean its package is installed here —
    but anything else that stops an env being built (a broken constructor, a missing asset, an EGL
    failure) is a real defect and must surface as one rather than being laundered into a skip.
    """
    try:
        return se.make(name, **kw)
    except ModuleNotFoundError as e:
        pytest.skip(f"{name} needs {e.name}, which this interpreter does not have")


def family_params(names, *, family: str, modules=()) -> list:
    """Parameters for a sweep over an env family. Empty is a parameter, not an absence.

    Returns one ``pytest.param`` per name, or — when the family is empty — a single ``EMPTY``
    parameter whose id says so, so the run reports "this family had nothing in it" rather than
    reporting nothing. :func:`resolve_family` then decides whether that is a skip or a failure.
    """
    if names:
        return [pytest.param(n, id=n) for n in sorted(names)]
    return [pytest.param(EMPTY, id=f"{family}-EMPTY")]


def resolve_family(name: str, *, family: str, modules=(), subject: str) -> None:
    """The zero-collection rule. Call it first in the body of every family-swept test.

    ``modules`` names the env modules whose failure to import would explain an empty family. If one of
    them is in ``import_errors()`` this skips and attaches the error, which is a fact about the
    interpreter. If none of them is, the family is empty for no stated reason and that FAILS — a sweep
    that quietly covers nothing is worse than one that is absent, because it is counted as evidence.
    """
    if name != EMPTY:
        return
    errs = se.import_errors()
    blamed = {m: errs[m] for m in modules if m in errs}
    if blamed:
        pytest.skip(
            f"{family}: no env registered in this interpreter because "
            + "; ".join(f"{m} did not import ({e})" for m, e in blamed.items())
            + f" — so {subject} is not checked here, and is checked wherever that module works"
        )
    expected = ", ".join(modules) if modules else "any env module"
    pytest.fail(
        f"{family}: {subject} has NO env to check it on, and {expected} imported cleanly, so nothing "
        f"explains that. The registry holds {len(se.registry.REGISTRY)} envs, which makes this a selection or a "
        f"naming change rather than a missing dependency — and a sweep that collects zero real cases "
        f"is GREEN and proves nothing, which is exactly how the DM Control family went uncovered for "
        f"a week."
    )


def sampled_states(env, n: int, seed: int):
    """``n`` states from an explicitly seeded generator. Never the env's own ``rng``.

    ``env.sample_states(n)`` with no generator consumes from ``env.rng``, and the ``env`` fixtures in
    this directory are MODULE-scoped, so the draw then depends on how many tests ran before it — i.e.
    on test ORDER, which ``pytest-randomly`` changes per run. A failure found that way cannot be
    reproduced and a pass is not evidence. See ``PARITY_STATE_SEED`` in ``test_conformance.py``.
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
    """``got`` and ``want`` must be the same bytes. Reports the max AND the pixel count.

    Both numbers, always, because they answer different questions: a max of 1 over 40,000 pixels is a
    rounding that moved, a max of 229 over 8 is a shape that moved, and a message carrying only one of
    them sends the reader to the wrong file.
    """
    assert tuple(got.shape) == tuple(want.shape), (
        f"{what}: shapes differ, {tuple(got.shape)} against {tuple(want.shape)} — {why}"
    )
    mx, npx = frame_diff(got, want)
    assert mx == 0, f"{what}: differ by max {mx}/255 on {npx} pixels. {why}"
