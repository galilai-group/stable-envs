"""The pytest plugin that makes this suite safe to run: guards, gates, breadcrumbs, smoke selection.

The suite was green and still took ITSELF down four times in a week. A suite that can abort the
interpreter is a defect whatever its exit code says, because the failure it produces is not the failure
that happened: SIGABRT costs every test that had not run yet, reports none of them, and points at
nothing. Everything here exists to convert one class of that into a named, attributed, single-test
failure — and to make the run survive the rest.

Five mechanisms, each answering an incident this repo actually had:

  **resource guard** (``resource_guard.py``). EGL context creations and releases are counted per test.
  A test that releases a context, or that creates them in proportion to the batch, FAILS with a
  greppable name at the moment it does it — instead of corrupting an unrelated env's pixels twenty
  minutes later. Three of the four incidents were exactly this and all three were first diagnosed as
  renderer bugs.

  **node gate** (``node_health.py``). Before any test runs: what does the scheduler think of this host,
  and does the official renderer return the same pixels twice? One of the four "incidents" was a
  cluster-condemned node (``NHC: auto-reboot``, flagged 41 minutes before the run) and the suite was
  reporting, correctly, that the official renderer was broken. The output never named the host, so it
  cost hours. Now the run stops with the hostname, the SLURM state and the reason.

  **crash breadcrumbs**. One line per test into ``SE_TEST_EVENTS`` before it starts and after it ends,
  flushed. If the interpreter dies, the supervisor (``supervisor.py``) reads the last unmatched start,
  knows exactly which test killed it, and resumes the run past it. This is what makes a segfault cost
  one test instead of the run — see that module for why forking (``pytest-forked``) cannot be used
  here: torch refuses to initialise CUDA in a forked child, measured, even when the parent never
  touched it.

  **deselect file**. The other half of resume: the supervisor hands back the list of tests already
  finished, and they are dropped at collection so a resumed segment does not repeat work.

  **smoke selection**. One env per FAMILY, computed from the registry rather than listed, so a new env
  module is covered the day it lands. This is the subset that is cheap enough to run before a commit,
  and the point of it is that a suite nobody runs is worth nothing.

Everything is opt-out through environment variables (``SE_SKIP_NODE_GATE``, ``SE_GUARD_MODE=report``)
so that debugging a guard never requires editing the plugin.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from . import node_health, resource_guard

#: Fixtures whose cost is a fresh interpreter rather than a render. Excluded from ``-m smoke`` because
#: the cross-process check boots a second Python, imports torch and mujoco and rebuilds every selected
#: env in it — 60+ seconds that buys nothing a pre-commit run can act on, since it answers "does this
#: reproduce in a NEW process", which is a release question. It stays in the full run, where it is the
#: single most important test in the repo.
SMOKE_EXCLUDED_FIXTURES = frozenset({"fresh_process"})

#: Test modules whose cost is booting OTHER interpreters rather than rendering anything. Same argument
#: as above and the same threshold: ``test_safety_net`` launches about fifteen pytest processes to
#: prove the supervisor survives crashes, which is a minute and a half and answers a question about the
#: HARNESS. The smoke subset answers "did this change break a renderer", and the harness does not change
#: when a renderer does. Both stay in the full run.
SMOKE_EXCLUDED_MODULES = frozenset({"test_safety_net"})

_EVENTS = None
_RUNDIR = None
_STARTED: dict = {}
_PHASES: dict = {}


# ------------------------------------------------------------------ wiring

#: Per-test wall limit, applied when ``pytest-timeout`` is installed. Set from measurement, not from a
#: round number. Over a full 1833-test run (H200, 21.1 min): median 0.08s, p95 4.6s, p99 ~18s, and the
#: two slowest are 53.8s for the first test that needs ``fresh_process`` — it boots a second interpreter
#: and rebuilds all 49 envs in it — and 41.8s for ``test_per_env_variation_is_per_env[Roundabout]``,
#: which renders once per declared variation leaf on a 604-primitive scene. 900s is ~17x the slowest
#: test and 70% of the whole run, so it cannot fire on a healthy machine and still bounds a hang to a
#: fraction of a run. It is the FAST path only: a thread blocked inside a driver call never returns to
#: Python and cannot be signalled, which is why supervisor.py also watches from outside the process.
DEFAULT_TIMEOUT = 900


def pytest_configure(config):
    global _EVENTS, _RUNDIR
    config.addinivalue_line("markers", "smoke: one env per family; the fast pre-commit subset")
    config.addinivalue_line("markers", "selftest: deliberately crashes the interpreter; opt-in only")

    # Applied here rather than through ``addopts`` so that the suite still RUNS without pytest-timeout
    # installed. ``addopts = ["--timeout=900"]`` makes a bare `pytest tests/` die at startup with
    # "unrecognized arguments" on any environment that has the package's own dependencies but not the
    # dev extra, which is a confusing failure for something that is a safety margin, not a requirement.
    # ``hasattr`` on the parsed option is the check: the attribute exists exactly when the plugin does.
    if hasattr(config.option, "timeout") and not config.option.timeout:
        config.option.timeout = DEFAULT_TIMEOUT

    # Install BEFORE any env is built, so the first ogbench plate bake is counted like every other.
    patched = resource_guard.install()
    config._se_guard_patched = patched

    _RUNDIR = os.environ.get("SE_TEST_RUNDIR") or None
    path = os.environ.get("SE_TEST_EVENTS")
    if path:
        # Line-buffered append: the point of this file is to survive the process that writes it, so
        # every record has to reach the kernel before the next test starts. A buffered write would be
        # lost in exactly the case the file exists for.
        _EVENTS = open(path, "a", buffering=1)
        _emit(kind="session_start", pid=os.getpid(), host=node_health.hostname(),
              patched=len(patched))


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
    except Exception:                                                        # noqa: BLE001
        pass                                     # a broken breadcrumb must never fail a test


# ------------------------------------------------------------------ collection: resume + smoke

def _family_representatives() -> dict:
    """``family module -> the cheapest registered env in it``. Computed, not listed.

    The repo's env package auto-discovers modules, so a hand-written smoke list would be one more
    thing to forget when adding an env — and the first symptom of forgetting is a family with no fast
    coverage at all. Cheapness is ranked by output size first (a 96px PushT frame against a 224px
    maze), then by the maze/body hints the OGBench cells carry, since those 25 cells differ in cost by
    more than an order of magnitude and any one of them exercises the same code.
    """
    import stable_envs as se
    best = {}
    for name, cls in sorted(se.REGISTRY.items()):
        fam = cls.__module__.rsplit(".", 1)[-1]
        maze = {"medium": 0, "teleport": 1, "large": 2, "giant": 3}.get(
            str(getattr(cls, "MAZE", "")).lower(), 0)
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
    # 1. resume: drop what a previous segment of this run already finished. tryfirst so this happens
    #    before -m / -k filtering reports its own deselection counts.
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

    # 2. smoke: mark one env per family, plus every test that is not parameterised over envs at all.
    #    The registry is only imported when something is actually parameterised over it, so a run that
    #    touches no env (the safety-net tests, a unit run) does not pay for importing torch and mujoco.
    reps = smoke_envs() if any("env" in getattr(getattr(i, "callspec", None), "params", {})
                               for i in items) else set()
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        env_name = params.get("env")
        if env_name is not None and env_name not in reps:
            continue
        if SMOKE_EXCLUDED_FIXTURES & set(getattr(item, "fixturenames", ())):
            continue
        if item.nodeid.split("::")[0].rsplit("/", 1)[-1][:-3] in SMOKE_EXCLUDED_MODULES:
            continue                       # by nodeid, not item.module: not every item type has one
        if "selftest" in item.keywords:
            continue
        item.add_marker(pytest.mark.smoke)

    # 3. the deliberately-crashing self-tests are opt-in. They exist to prove the supervisor survives
    #    a SIGSEGV/SIGABRT, so running them by accident would abort a normal run on purpose.
    if not os.environ.get("SE_SELFTEST_CRASH"):
        skip = pytest.mark.skip(reason="crash self-tests are opt-in: set SE_SELFTEST_CRASH=1 "
                                       "(they abort the interpreter by design)")
        for item in items:
            if "selftest" in item.keywords:
                item.add_marker(skip)


def pytest_collection_finish(session):
    """The node gate. Runs after collection and before the first test — the only place it can.

    Before collection there is no way to know which envs the run will touch, and the probe should use
    one the run needs anyway so the reference env it builds is the one the parity tests then reuse from
    the process-wide cache. After the first test it would be too late to be a precondition.
    """
    if session.config.option.collectonly or os.environ.get("SE_SKIP_NODE_GATE"):
        return
    if not session.items:
        return
    names = sorted({p["env"] for i in session.items
                    for p in [getattr(getattr(i, "callspec", None), "params", {})]
                    if "env" in p})
    # The scheduler question is free and always worth asking. The renderer probe costs a MuJoCo build,
    # so it is only asked when the run contains something that renders — a unit or safety-net run has
    # no parity number to invalidate and should not pay for the answer.
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
    # pytest.exit rather than failing every test: nothing measured on this host means anything, and a
    # thousand identical failures would bury the one line that says why.
    pytest.exit(f"NODE FAULT on {g['host']}: {g['why']}", returncode=78)


# ------------------------------------------------------------------ per-test resource guard

@pytest.hookimpl(trylast=True)
def pytest_runtest_logstart(nodeid, location):
    # Re-arm before the snapshot, not only at ``pytest_configure``. That hook runs before any test
    # module is collected, so ``stable_envs`` — and therefore ``dm_control``, whose reference envs own
    # EGL contexts of a class mujoco does not export — has not been imported yet, and a one-shot
    # install permanently misses every context class an env family brings with it. ``install`` patches
    # only what is already in ``sys.modules``, so this is a few dict lookups per test.
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
    # A test that skipped or errored in setup never produces a "call" report, and a resource defect in
    # its setup is still a defect — so the check also runs on teardown for anything not already flagged.
    if report.when == "call" or report.when == "teardown":
        item._se_guard_reported = True
        names = ", ".join(n for n, _ in bad)
        body = "\n\n".join(f"{n}: {m}" for n, m in bad)
        _emit(kind="guard_violation", nodeid=item.nodeid, names=names, **{
            k: v for k, v in d.items() if k != "sites"})
        report.outcome = "failed"
        report.longrepr = (
            f"RESOURCE GUARD [{names}]\n\n{body}\n\n"
            f"counted during this test: created {d['context_created']} EGL context(s), released "
            f"{d['context_freed']}; renderers created {d['renderer_created']}, closed "
            f"{d['renderer_closed']}; {d['context_live']} live now.\n"
            f"The rule (stable_envs/testing/conformance.py): in this process an EGL context is created "
            f"and NEVER released, and nothing may create them in proportion to the batch.")
    return report


def pytest_runtest_logreport(report):
    """Accumulate the three phases into ONE outcome per test, and emit it when the test is done.

    The supervisor aggregates across segments from these lines, so the vocabulary has to match what a
    reader expects from pytest's own summary — ``xfailed`` and ``xpassed`` in particular, since this
    repo records confirmed defects as ``xfail(strict=True)`` and an XPASS is a real failure that says
    "the underlying bug is fixed, delete the entry". Collapsing those into pass/fail would erase the
    one signal those entries exist to produce.
    """
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
    _emit(kind="end", nodeid=report.nodeid, outcome=phases.get("outcome", "passed"),
          dur=round(time.time() - started[0], 3) if started else None)
    _PHASES.pop(report.nodeid, None)


# ------------------------------------------------------------------ summary

def pytest_terminal_summary(terminalreporter, exitstatus, config):
    t = resource_guard.totals()
    gate = getattr(config, "_se_node_gate", {})
    line = (f"resources: {t['context_created']} EGL contexts created, {t['context_freed']} released, "
            f"{t['context_live']} live (peak {t['context_peak_live']}); "
            f"renderers {t['renderer_created']} created / {t['renderer_closed']} closed")
    terminalreporter.write_sep("-", "stable-envs safety")
    terminalreporter.write_line(line)
    if not t["patched"]:
        terminalreporter.write_line(
            "resource guard: mujoco not importable, EGL accounting NOT active for this run")
    if gate:
        terminalreporter.write_line(
            f"node: {gate.get('host')} slurm={gate.get('scheduler', {}).get('state') or '?'} "
            f"reason={gate.get('scheduler', {}).get('reason') or 'none'} "
            f"reference-renderer={gate.get('renderer', {})}")
