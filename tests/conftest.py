"""What is in this directory, how it differs from ``stable_envs/testing/``, and the empty-run guard.

``tests/`` is this repository's OWN suite: the assertions, the thresholds, and the per-env floors that
were measured before they were declared. ``stable_envs/testing/`` is the library that supplies the
measurements — public, importable, used by ``examples/`` and by env authors outside this repo — and it
deliberately asserts nothing. The dependency runs one way: everything here imports from there.
``stable_envs/testing/`` stays in the package (like ``numpy.testing``); it is not this directory.

    tests/
      unit/                       family/unit contracts. No GPU, no reference env, no checkpoint.
      integration/                several components together, including the CPU safety-net
                                  (``test_safety_net.py``) and the SWM interop suite.
      gpu/                        the release gate, parameterised over the registry. Needs CUDA by
                                  design — the product is a GPU renderer, so a suite that passes on
                                  CPU-only CI proves nothing about it.
        _shared.py                the three conventions the GPU files kept getting wrong. Read it
                                  before adding a file here.
        conftest.py               puts every shared env back at its default appearance per test.
      regression/                 frozen corpora, digests, declarations, provenance,
                                  and the support-matrix oracle. CPU unless also under ``gpu/``.
      fixtures/                   golden JSON the unit/regression readers assert against.
      packaging/                  wheel/sdist metadata, extras, and GitHub Actions safety.

Directory markers are applied below so ``pytest -m unit`` / ``-m integration`` /
``-m "regression and not gpu"`` / ``-m "regression and gpu"`` select the matching
bucket even when a file forgot ``pytestmark``. Existing per-file markers are kept.

THE EMPTY-RUN GUARD, below, and why it is a hook rather than a test. Twice now this suite has been
green while covering nothing: the DM Control family's 16 envs were absent from the suite interpreter
for a version skew, so every test parameterised over them collected ZERO cases and the run said so in
one line among a dozen other import notes. ``test_conformance.py`` closed that at the FAMILY level
(``SE_REQUIRE_ENV_MODULES``) and ``tests/gpu/_shared.py`` closes it at the SWEEP level
(``resolve_family``). This closes it at the RUN level, which neither of those can see.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set by CI profile jobs (``core``, ``dmcontrol``, ``ogbench``, ``fetch``, GPU profiles).
#: Tests already skip missing extras; the variable names the interpreter the job installed.
CI_PROFILE = os.environ.get("STABLE_ENVS_CI_PROFILE", "")

import pytest

try:
    import stable_worldmodel  # noqa: F401 — registers swm/* ids for certification oracles
except ImportError:
    pass

#: GPU files that are also the frozen regression oracle (parity / reproducibility / fast==slow).
_GPU_REGRESSION = {
    "test_conformance.py",
    "test_reproducibility.py",
    "test_fast_paths.py",
    "test_rollout_on_device.py",
}


def pytest_collection_modifyitems(config, items):
    """Stamp directory markers on every collected item.

    A file that already has ``pytestmark = pytest.mark.unit`` (or gpu, …) keeps that mark;
    adding the same mark again is a no-op for ``-m`` selection.
    """
    del config
    for item in items:
        path = Path(getattr(item, "path", item.fspath))
        parts = path.parts
        try:
            tests_i = parts.index("tests")
        except ValueError:
            continue
        if tests_i + 1 >= len(parts):
            continue
        bucket = parts[tests_i + 1]
        if bucket == "unit":
            item.add_marker(pytest.mark.unit)
        elif bucket == "integration":
            item.add_marker(pytest.mark.integration)
        elif bucket == "regression":
            item.add_marker(pytest.mark.regression)
        elif bucket == "gpu":
            item.add_marker(pytest.mark.gpu)
            if path.name in _GPU_REGRESSION:
                item.add_marker(pytest.mark.regression)


#: Set to anything to allow a run in which nothing passed or failed. For a deliberately narrow
#: selection on a machine that cannot service it — ``pytest tests/gpu -k RoboCasa`` in an interpreter
#: without robocasa — "everything skipped" is the honest answer rather than a fault.
ALLOW_EMPTY = "SE_ALLOW_EMPTY_RUN"

#: Set by the supervisor when it RESUMES after a crash. A resumed segment is handed only the tests the
#: previous segments did not finish, so a final segment made entirely of skips is normal and says
#: nothing about coverage — the run as a whole is what the guard is about, and the supervisor
#: aggregates that itself.
RESUME = "SE_TEST_DESELECT"

_COUNTS: dict[str, int] = {}


def pytest_runtest_logreport(report):
    """Count REAL outcomes — a pass, a failure, or an error — separately from skips.

    Counted from the reports rather than read off ``terminalreporter.stats`` so the answer is the same
    under ``-q``, under the supervisor's resumed segments and under xdist, none of which is guaranteed
    to leave the same stats dict behind.
    """
    if report.when == "call" or (report.when == "setup" and report.outcome == "skipped"):
        _COUNTS[report.outcome] = _COUNTS.get(report.outcome, 0) + 1
    elif report.failed:  # a setup or teardown error is a real outcome
        _COUNTS["error"] = _COUNTS.get("error", 0) + 1


def _verdict(session, exitstatus) -> str | None:
    """The one-line reason this run measured nothing, or None if it measured something."""
    if os.environ.get(ALLOW_EMPTY) or os.environ.get(RESUME):
        return None
    if getattr(session.config.option, "collectonly", False):
        return None
    # An exit for a node fault (78) or an internal error already carries its own diagnosis, and a run
    # that failed has by definition produced a real outcome. Only 0 and "nothing collected" can lie.
    if exitstatus not in (0, int(pytest.ExitCode.NO_TESTS_COLLECTED)):
        return None
    if _COUNTS.get("passed", 0) + _COUNTS.get("failed", 0) + _COUNTS.get("error", 0):
        return None

    n = len(getattr(session, "items", []) or [])
    if n == 0:
        return (
            "this run selected NO tests at all. That is not a pass: pytest exits 5 for it, a shell "
            "that only checks for zero reads 5 as failure and 0 as success, and both sbatch "
            "launchers pass their arguments straight through to pytest — so a mistyped -k or -m "
            "comes back looking like a run that finished. Check the selection, or set "
            f"{ALLOW_EMPTY}=1 if selecting nothing is genuinely the right answer here."
        )
    return (
        f"this run collected {n} test(s), {_COUNTS.get('skipped', 0)} of them SKIPPED, and NOTHING "
        f"passed, failed or errored. pytest reports that as success, and it is the exact shape of "
        f"both coverage incidents this suite has had: the tests existed, they were parameterised "
        f"over envs this interpreter does not have, and the run was green having measured nothing. "
        f"Run with -rs to see the reasons; set SE_REQUIRE_ENV_MODULES to make a named family's "
        f"absence fatal, or {ALLOW_EMPTY}=1 if an all-skipped run is what was wanted."
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Say it where a reader will see it. Written before the exit code is changed, not after.

    Split from ``pytest_sessionfinish`` on purpose: the terminal reporter emits its summary from
    within its own ``sessionfinish``, so a message produced there would land after the "== 0 passed =="
    line or not at all, and raising from there risks an INTERNALERROR that buries the sentence this
    exists to print.
    """
    why = _verdict(terminalreporter._session, exitstatus)
    if why:
        terminalreporter.write_sep("=", "EMPTY RUN", red=True, bold=True)
        terminalreporter.write_line(why)


def pytest_sessionfinish(session, exitstatus):
    if _verdict(session, exitstatus):
        session.exitstatus = 1
