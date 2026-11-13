"""Pytest hooks for crash recovery, node checks, and GL resource accounting.

Append flushed events for the external supervisor. Its resume file deselects
finished tests; deliberate crash tests remain opt-in. Smoke selection chooses
one representative per environment family and omits subprocess-heavy checks.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from tests._runner import node_health, resource_guard

# Fresh-interpreter checks remain in the full suite, outside the renderer smoke subset.
SMOKE_EXCLUDED_FIXTURES = frozenset({"fresh_process"})

# Harness subprocess tests remain in the full suite.
SMOKE_EXCLUDED_MODULES = frozenset({"test_safety_net"})

_EVENTS = None
_RUNDIR = None
_STARTED: dict = {}
_PHASES: dict = {}


# pytest-timeout bounds Python-level hangs; the external supervisor also watches native stalls.
DEFAULT_TIMEOUT = 900


def pytest_configure(config):
    global _EVENTS, _RUNDIR
    config.addinivalue_line("markers", "smoke: one env per family; the fast pre-commit subset")
    config.addinivalue_line("markers", "selftest: deliberately crashes the interpreter; opt-in only")

    # Set the default only when pytest-timeout is installed and the user has not supplied one.
    if hasattr(config.option, "timeout") and not config.option.timeout:
        config.option.timeout = DEFAULT_TIMEOUT

    # Install before collection so initial plate bakes are counted.
    patched = resource_guard.install()
    config._se_guard_patched = patched

    _RUNDIR = os.environ.get("SE_TEST_RUNDIR") or None
    path = os.environ.get("SE_TEST_EVENTS")
    if path:
        # Flush each event so a crash cannot lose the last started test.
        _EVENTS = open(path, "a", buffering=1)
        _emit(kind="session_start", pid=os.getpid(), host=node_health.hostname(), patched=len(patched))


def pytest_unconfigure(config):
    if _EVENTS is not None:
        _emit(kind="session_end", **resource_guard.totals())
        _EVENTS.close()


def _emit(**rec):
    if _EVENTS is None:
        return
    rec.setdefault("t", round(time.time(), 3))
    try:
        _EVENTS.write(json.dumps(rec, default=str) + "\n")
        _EVENTS.flush()
    except Exception:  # noqa: BLE001
        pass  # a broken breadcrumb must never fail a test


def _family_representatives() -> dict:
    """Choose one registered class per family, ranked by frame size and maze/body hints."""
    import stable_envs as se

    best = {}
    for name, cls in sorted(se.registry.REGISTRY.items()):
        fam = cls.__module__.rsplit(".", 1)[-1]
        maze = {"medium": 0, "teleport": 1, "large": 2, "giant": 3}.get(str(getattr(cls, "MAZE", "")).lower(), 0)
        loco = {"point": 0, "ant": 1, "humanoid": 2}.get(str(getattr(cls, "LOCO", "")).lower(), 0)
        key = (int(getattr(cls, "default_size", 0) or 0), maze, loco, name)
        if fam not in best or key < best[fam][0]:
            best[fam] = (key, name)
    return {f: n for f, (_, n) in best.items()}


def smoke_envs() -> set:
    override = os.environ.get("SE_SMOKE_ENVS", "").strip()
    if override:
        return {s for s in override.split(",") if s}
    return set(_family_representatives().values())


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # Apply resume exclusions before pytest reports marker/expression deselections.
    skip_file = os.environ.get("SE_TEST_DESELECT")
    if skip_file and os.path.exists(skip_file):
        with open(skip_file) as f:
            done = {ln.strip() for ln in f if ln.strip()}
        if done:
            keep = [i for i in items if i.nodeid not in done]
            dropped = [i for i in items if i.nodeid in done]
            if dropped:
                config.hook.pytest_deselected(items=dropped)
                items[:] = keep

    # Import the registry only for environment-parameterized tests; mark one representative per family.
    reps = smoke_envs() if any("env" in getattr(getattr(i, "callspec", None), "params", {}) for i in items) else set()
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        env_name = params.get("env")
        if env_name is not None and env_name not in reps:
            continue
        if SMOKE_EXCLUDED_FIXTURES & set(getattr(item, "fixturenames", ())):
            continue
        if item.nodeid.split("::")[0].rsplit("/", 1)[-1][:-3] in SMOKE_EXCLUDED_MODULES:
            continue  # by nodeid, not item.module: not every item type has one
        if "selftest" in item.keywords:
            continue
        item.add_marker(pytest.mark.smoke)

    # Deliberate native crashes require opt-in; otherwise they would abort ordinary test runs.
    if not os.environ.get("SE_SELFTEST_CRASH"):
        skip = pytest.mark.skip(
            reason="crash self-tests are opt-in: set SE_SELFTEST_CRASH=1 (they abort the interpreter by design)"
        )
        for item in items:
            if "selftest" in item.keywords:
                item.add_marker(skip)


def pytest_collection_finish(session):
    """Check host preconditions after collection identifies the reference renderers needed."""
    if session.config.option.collectonly or os.environ.get("SE_SKIP_NODE_GATE"):
        return
    if not session.items:
        return
    names = sorted(
        {p["env"] for i in session.items for p in [getattr(getattr(i, "callspec", None), "params", {})] if "env" in p}
    )
    # Probe rendering only when selected tests need an environment; always ask the scheduler.
    g = node_health.gate(candidates=names or None, check_renderer=bool(names))
    _emit(kind="node_gate", **g)
    session.config._se_node_gate = g
    if g["ok"]:
        return
    if _RUNDIR:
        try:
            with open(os.path.join(_RUNDIR, "node_fault.json"), "w") as f:
                json.dump(g, f, indent=2)
        except OSError:
            pass
    # End the session once when the host fails preconditions.
    pytest.exit(f"NODE FAULT on {g['host']}: {g['why']}", returncode=78)


@pytest.hookimpl(trylast=True)
def pytest_runtest_logstart(nodeid, location):
    # Re-arm before taking the snapshot so contexts from newly imported backends are counted.
    resource_guard.install()
    _STARTED[nodeid] = (time.time(), resource_guard.take())
    _emit(kind="start", nodeid=nodeid)


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    started = _STARTED.get(item.nodeid)
    if started is None or report.when not in ("call", "teardown"):
        return report
    if getattr(item, "_se_guard_reported", False):
        return report
    d = resource_guard.delta(started[1])
    item._se_guard_delta = d
    env = (item.funcargs or {}).get("env")
    bad = resource_guard.violations(d, batch=getattr(env, "B", None))
    if not bad:
        return report
    # Setup failures have no call report; inspect teardown too, without reporting twice.
    if report.when == "call" or report.when == "teardown":
        item._se_guard_reported = True
        names = ", ".join(n for n, _ in bad)
        body = "\n\n".join(f"{n}: {m}" for n, m in bad)
        _emit(kind="guard_violation", nodeid=item.nodeid, names=names, **{k: v for k, v in d.items() if k != "sites"})
        report.outcome = "failed"
        report.longrepr = (
            f"RESOURCE GUARD [{names}]\n\n{body}\n\n"
            f"counted during this test: created {d['context_created']} EGL context(s), released "
            f"{d['context_freed']}; renderers created {d['renderer_created']}, closed "
            f"{d['renderer_closed']}; {d['context_live']} live now.\n"
            f"The rule (stable_envs/testing/conformance.py): an EGL context is never released in this "
            f"process, and nothing may create them in proportion to the batch."
        )
    return report


def pytest_runtest_logreport(report):
    """Emit one outcome per test across setup, call, and teardown, retaining xfail/XPASS status."""
    phases = _PHASES.setdefault(report.nodeid, {})
    if hasattr(report, "wasxfail"):
        phases["outcome"] = "xpassed" if (report.when == "call" and report.passed) else "xfailed"
    elif report.failed:
        phases["outcome"] = "error" if report.when != "call" else "failed"
    elif report.skipped:
        phases.setdefault("outcome", "skipped")
    elif report.when == "call":
        phases.setdefault("outcome", "passed")
    if report.when != "teardown":
        return
    started = _STARTED.pop(report.nodeid, None)
    _emit(
        kind="end",
        nodeid=report.nodeid,
        outcome=phases.get("outcome", "passed"),
        dur=round(time.time() - started[0], 3) if started else None,
    )
    _PHASES.pop(report.nodeid, None)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    t = resource_guard.totals()
    gate = getattr(config, "_se_node_gate", {})
    line = (
        f"resources: {t['context_created']} EGL contexts created, {t['context_freed']} released, "
        f"{t['context_live']} live (peak {t['context_peak_live']}); "
        f"renderers {t['renderer_created']} created / {t['renderer_closed']} closed"
    )
    terminalreporter.write_sep("-", "stable-envs safety")
    terminalreporter.write_line(line)
    if not t["patched"]:
        terminalreporter.write_line("resource guard: mujoco not importable, EGL accounting NOT active for this run")
    if gate:
        terminalreporter.write_line(
            f"node: {gate.get('host')} slurm={gate.get('scheduler', {}).get('state') or '?'} "
            f"reason={gate.get('scheduler', {}).get('reason') or 'none'} "
            f"reference-renderer={gate.get('renderer', {})}"
        )
