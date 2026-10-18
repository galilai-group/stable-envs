"""Suite-wide hooks: directory and family markers, family-absence skips, and the empty-run guard.

``tests/`` holds this repository's own assertions and thresholds. ``stable_envs/testing/`` is the
public library that supplies the measurements and asserts nothing; tests import from it, never the
reverse.

    tests/
      unit/                       family/unit contracts. No GPU, no reference env, no checkpoint.
      integration/                several components together, including the CPU safety net
                                  (``test_safety_net.py``) and the SWM interop suite.
      gpu/                        the registry-wide GPU suite. Needs CUDA.
        _shared.py                skip/fail conventions for family sweeps and frame comparison.
        conftest.py               resets each shared env to its default appearance per test.
      regression/                 frozen corpora, digests, declarations, provenance,
                                  and the support-matrix oracle. CPU unless also under ``gpu/``.
      fixtures/                   golden JSON the unit/regression readers assert against.
      packaging/                  wheel/sdist metadata, extras, and GitHub Actions safety.

Directory markers are applied below so ``pytest -m unit`` / ``-m integration`` /
``-m "regression and not gpu"`` / ``-m "regression and gpu"`` select the matching
bucket even when a file has no ``pytestmark``. Existing per-file markers are kept.

Every test also gets one ``family_<name>`` marker from ``ci/families.toml`` (see
``tests/helpers/families.py``). A test that fails while its family is not installed here (missing,
or at a version the family code refuses) is reported as skipped with that cause; a test that passes
without the family still counts. A ``gl`` test on a machine without dm_control rendering is skipped
up front. The summary groups skips by cause. ``SE_REQUIRE_FAMILIES`` / ``SE_REQUIRE_CAPABILITIES``
(set by CI profile jobs) make the same absence a usage error before anything runs, and with ``swm``
required a required family's test that skips for want of stable-worldmodel fails instead.

The empty-run guard fails a run in which nothing passed, failed or errored (for example every test
skipped because its envs are not installed). It complements ``SE_REQUIRE_ENV_MODULES`` in
``tests/gpu/test_conformance.py`` and ``resolve_family`` in ``tests/gpu/_shared.py``.
"""

from __future__ import annotations

import importlib.abc
import os
import sys
from pathlib import Path

#: Set by CI profile jobs (``core``, ``dmcontrol``, ``ogbench``, ``fetch``, GPU profiles).
#: Tests already skip missing extras; the variable names the interpreter the job installed.
CI_PROFILE = os.environ.get("STABLE_ENVS_CI_PROFILE", "")

import pytest

from tests.helpers import families
from tests.helpers.reference.cases import ORIGINAL_GATES
from tests.helpers.reference.certification import install_on_classes

install_on_classes()


def pytest_configure(config):
    config.addinivalue_line("markers", "family(name): the ci/families.toml family this test needs installed")
    for name in families.FAMILIES:
        config.addinivalue_line("markers", f"family_{name}: stamped from ci/families.toml")
    for name, absent in families.CAPABILITIES.items():
        config.addinivalue_line("markers", f"{name}: skipped where {absent}")
    problems = families.check_required()
    if problems:
        raise pytest.UsageError("\n".join(problems))
    # An explicit source profile verifies and imports its requested checkout in its fixture;
    # importing an installed copy first would select another oracle before the pin is checked.
    # Otherwise import the installed copy so its IDs are registered for certification.
    if not config.getoption("--contract-profile", default=None):
        try:
            import stable_worldmodel  # noqa: F401
        except ImportError:
            pass


_FAMILY = pytest.StashKey[str]()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    report = (yield).get_result()
    if report.failed and report.when in ("setup", "call"):
        reason = families.absent(item.stash.get(_FAMILY, "core"))
        if reason:
            report.outcome = "skipped"
            report.longrepr = (report.location[0], report.location[1] or 0, f"Skipped: {reason}")
    elif report.skipped:
        cause = families.forbidden_skip(item.stash.get(_FAMILY, "core"), report)
        if cause:
            report.outcome = "failed"
            report.longrepr = f"{families.REQUIRE_CAPABILITIES} requires swm, and this skipped: {cause}"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    report = (yield).get_result()
    if report.failed and isinstance(collector, pytest.Module):
        reason = families.absent(families.family_of_file(collector.path.name))
        if reason:
            report.outcome = "skipped"
            report.longrepr = (str(collector.path), 0, f"Skipped: {reason}")
    elif report.skipped and isinstance(collector, pytest.Module):
        cause = families.forbidden_skip(families.family_of_file(collector.path.name), report)
        if cause:
            report.outcome = "failed"
            report.longrepr = f"{families.REQUIRE_CAPABILITIES} requires swm, and this module skipped: {cause}"


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
        family = families.family_of(item)
        item.stash[_FAMILY] = family
        item.add_marker(getattr(pytest.mark, f"family_{family}"))
        missing = [name for name in families.CAPABILITIES if item.get_closest_marker(name) and not families.have(name)]
        if missing:
            item.add_marker(pytest.mark.skip(reason=f"needs {missing[0]}: {families.CAPABILITIES[missing[0]]}"))
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


_ORIGINAL = frozenset(ORIGINAL_GATES.values())


class _NoStableWorldmodel(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.partition(".")[0] == "stable_worldmodel":
            raise ImportError(f"{name} is blocked: an ORIGINAL_GATES test must stand without stable-worldmodel")
        return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """An ``ORIGINAL_GATES`` test runs with stable_worldmodel unimportable."""
    if f"{item.nodeid.partition('::')[0]}::{getattr(item, 'originalname', item.name)}" not in _ORIGINAL:
        yield
        return
    hidden = {
        name: sys.modules.pop(name) for name in list(sys.modules) if name.partition(".")[0] == "stable_worldmodel"
    }
    blocker = _NoStableWorldmodel()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        sys.modules.update(hidden)


#: Set to anything to allow a run in which nothing passed or failed, e.g. a narrow ``-k`` selection
#: in an interpreter without that family installed.
ALLOW_EMPTY = "SE_ALLOW_EMPTY_RUN"

#: Set by the supervisor when it resumes after a crash. A resumed segment may be all skips; the
#: supervisor aggregates the whole run itself.
RESUME = "SE_TEST_DESELECT"

_COUNTS: dict[str, int] = {}


def pytest_runtest_logreport(report):
    """Count passes, failures and errors separately from skips.

    Counted from reports rather than ``terminalreporter.stats`` so the result is the same under ``-q``,
    resumed supervisor segments and xdist.
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
    # Other exit codes (node fault 78, internal error, failures) already carry their own diagnosis.
    if exitstatus not in (0, int(pytest.ExitCode.NO_TESTS_COLLECTED)):
        return None
    if _COUNTS.get("passed", 0) + _COUNTS.get("failed", 0) + _COUNTS.get("error", 0):
        return None

    n = len(getattr(session, "items", []) or [])
    if n == 0:
        return (
            "this run selected no tests at all; check the -k/-m selection, or set "
            f"{ALLOW_EMPTY}=1 if selecting nothing is intended."
        )
    return (
        f"this run collected {n} test(s), {_COUNTS.get('skipped', 0)} of them skipped, and nothing "
        f"passed, failed or errored, so it measured nothing. Run with -rs to see the reasons; set "
        f"SE_REQUIRE_ENV_MODULES to make a named family's absence fatal, or {ALLOW_EMPTY}=1 if an "
        f"all-skipped run is intended."
    )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print skips grouped by cause and the empty-run verdict.

    Printed here rather than in ``pytest_sessionfinish``, where the terminal summary has already been
    written.
    """
    skipped = terminalreporter.stats.get("skipped", [])
    if skipped:
        terminalreporter.write_sep("-", f"{len(skipped)} skipped, by cause")
        for line in families.grouped(skipped):
            terminalreporter.write_line(line)
    why = _verdict(terminalreporter._session, exitstatus)
    if why:
        terminalreporter.write_sep("=", "EMPTY RUN", red=True, bold=True)
        terminalreporter.write_line(why)


def pytest_sessionfinish(session, exitstatus):
    if _verdict(session, exitstatus):
        session.exitstatus = 1


@pytest.fixture
def loguru_to_caplog():
    """Route package log records to the stdlib logging tree, at DEBUG verbosity, for ``caplog``."""
    import logging

    from loguru import logger

    from stable_envs._core import config

    saved = config.get_config()["verbose"]
    config.set(verbose="DEBUG")

    def sink(message):
        r = message.record
        rec = logging.LogRecord(r["name"], r["level"].no, r["file"].path, r["line"], r["message"], None, None)
        rec.levelname = r["level"].name
        logging.getLogger(r["name"]).handle(rec)

    sink_id = logger.add(sink, filter=lambda r: r["extra"].get("package") == "stable_envs", level="DEBUG")
    yield
    logger.remove(sink_id)
    config.set(verbose=saved)
