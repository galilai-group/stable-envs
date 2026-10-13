"""The safety net, tested. A suite that cannot survive a crashing test has to prove that it can.

Everything in ``tests/_runner/supervisor.py`` and ``resource_guard.py`` exists to convert a class
of failure that kills the interpreter into one attributed test failure. That claim cannot be checked by
reading the code: the whole point is what happens when the process is gone, and the only honest way to
check it is to kill a process and look at what came back. So each test here generates a throwaway test
module, runs the real supervisor or the real plugin over it in a subprocess, and asserts on the result.

Deliberately CPU-only and dependency-free. The generated modules do not import torch, so a nested run
costs about a second and these can live in the default suite instead of being a thing someone runs by
hand once and then forgets. The GPU-side equivalents — a real env aborting with live EGL contexts, a
real ``mujoco.egl.GLContext`` being released — are in ``tests/gpu/test_crash_isolation.py``, marked
``selftest`` and opt-in, because running them by accident aborts the interpreter on purpose.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _clean_env(**extra) -> dict:
    """A child environment with this run's supervisor wiring stripped out.

    Inheriting ``SE_TEST_EVENTS`` would make the nested run append its breadcrumbs to the OUTER run's
    event log, and the outer supervisor would then attribute the nested crash to whatever it thought
    was in flight — a bug that reads as the safety net failing when in fact the test harness leaked.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("SE_TEST_")}
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    env["SE_SKIP_NODE_GATE"] = "1"  # no GPU is involved in any of these
    env.update(extra)
    return env


#: A module with tests on both sides of two different fatal events. ``ctypes.string_at(0)`` is a real
#: SIGSEGV in the C library and ``os.abort()`` a real SIGABRT — the two signals this suite has actually
#: died from — and the passing tests around them are what proves the run RESUMED rather than merely
#: reported. Names are ordered so pytest's file order is the execution order.
CRASHY = """
import ctypes, os

def test_a_before(): assert True
def test_b_segfault(): ctypes.string_at(0)
def test_c_middle(): assert True
def test_d_abort(): os.abort()
def test_e_after(): assert True
def test_f_ordinary_failure(): assert 1 == 2
"""


def _run_supervisor(tmp_path: Path, module_src: str, *extra_args, timeout=300) -> tuple[int, dict, str]:
    (tmp_path / "test_generated.py").write_text(module_src)
    rundir = tmp_path / "run"
    out_json = tmp_path / "result.json"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "tests._runner.supervisor",
        "--rundir",
        str(rundir),
        "--json",
        str(out_json),
        *extra_args,
        "--",
        str(tmp_path / "test_generated.py"),
        "-p",
        "tests._runner.safety_plugin",
        # The generated module lives in a tmp dir, so its rootdir is NOT this repo and it inherits
        # none of our addopts -- including `-p no:randomly`. These tests assert on which test the
        # crash is attributed to, which is an ORDER claim, so a shuffled child would fail them for
        # a reason that has nothing to do with the safety net.
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    p = subprocess.run(cmd, env=_clean_env(), capture_output=True, text=True, timeout=timeout, cwd=str(tmp_path))
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    return p.returncode, res, p.stdout + p.stderr


def test_the_supervisor_survives_a_segfault_and_an_abort_and_finishes_the_run(tmp_path):
    """Two fatal signals in one module must cost two tests. Everything else still runs and reports.

    This is the whole thesis of the supervisor in one assertion. Without it, ``test_b_segfault`` ends
    the interpreter and the four tests after it are never run, never reported, and the only artifact of
    the whole thing is a shell exit code. With it, the two crashes are named, the four survivors are
    reported with their real outcomes, and the ordinary failure at the end is still an ordinary
    failure — the net must not turn a normal red into something exotic.
    """
    rc, res, log = _run_supervisor(tmp_path, CRASHY)
    assert res, f"the supervisor produced no result json:\n{log}"
    counts, crashed = res["counts"], res["crashed"]
    assert counts["crashed"] == 2, f"expected both fatal tests to be recorded as crashed: {res}\n{log}"
    names = sorted(c[0].split("::")[-1] for c in crashed)
    assert names == ["test_b_segfault", "test_d_abort"], f"wrong tests blamed: {names}\n{log}"
    assert "SIGSEGV" in crashed[0][1] and "SIGABRT" in crashed[1][1], (
        f"the signal that killed each one is not reported: {crashed}"
    )
    assert counts["passed"] == 3, f"the survivors did not run: {res}\n{log}"
    assert counts["failed"] == 1, f"an ordinary failure must stay an ordinary failure: {res}\n{log}"
    assert res["segments"] == 3, f"expected one segment per crash plus the first: {res}"
    assert rc == 1, "a run containing crashes must not exit 0"


def test_a_crash_does_not_re_run_the_tests_that_already_passed(tmp_path):
    """Resuming must skip forward, not start over. Otherwise a crash near the end costs the run twice.

    The deselect file is what makes resume affordable on a 21-minute suite: without it, a crash in the
    last test would re-run the preceding twenty minutes, and the rational response to that would be to
    stop using the supervisor.
    """
    rc, res, log = _run_supervisor(tmp_path, CRASHY)
    ends = [
        json.loads(l)["nodeid"]
        for p in sorted((tmp_path / "run").glob("events.*.jsonl"))
        for l in p.read_text().splitlines()
        if l and json.loads(l).get("kind") == "end"
    ]
    assert ends, f"no test finished at all:\n{log}"
    assert len(ends) == len(set(ends)), f"a test ran in more than one segment: {ends}\n{log}"


def test_a_c_level_exit_mid_test_is_treated_as_a_crash_not_as_a_test_failure(tmp_path):
    """``os._exit(1)`` inside a test looks exactly like "some tests failed" to a shell. It is not.

    MuJoCo's own fatal path ends the process this way (``mju_error`` -> ``exit(1)``), so an exit code
    is not sufficient evidence of an orderly finish. The breadcrumb is: pytest always closes out every
    test it starts, so a start with no end means the process died inside that test whatever its status
    was. This asserts the supervisor uses that and not the number.
    """
    src = "import os\ndef test_a(): assert True\ndef test_b_exit(): os._exit(1)\ndef test_c(): assert True\n"
    rc, res, log = _run_supervisor(tmp_path, src)
    assert res["counts"]["crashed"] == 1, f"a mid-test exit(1) was not caught: {res}\n{log}"
    assert res["crashed"][0][0].endswith("test_b_exit"), res["crashed"]
    assert res["counts"]["passed"] == 2, f"the run did not continue past it: {res}\n{log}"


def test_the_stall_watchdog_kills_a_hung_test_and_carries_on(tmp_path):
    """A test that never returns must cost one test, and must not need anything in-process to notice.

    ``pytest-timeout`` handles the case where the test is hung in Python. It cannot handle a thread
    blocked inside a driver call, which is what a hung EGL or CUDA operation is, because the signal is
    only delivered when the C call returns. The supervisor watches the events file from OUTSIDE the
    process, so it does not care where the hang is.
    """
    src = "import time\ndef test_a(): assert True\ndef test_b_hangs(): time.sleep(600)\ndef test_c(): assert True\n"
    rc, res, log = _run_supervisor(tmp_path, src, "--stall-timeout", "12", timeout=300)
    assert res["counts"]["crashed"] == 1, f"the hang was not caught: {res}\n{log}"
    assert res["crashed"][0][0].endswith("test_b_hangs"), res["crashed"]
    assert "stall" in res["crashed"][0][1]
    assert res["counts"]["passed"] == 2, f"the run did not continue past the hang: {res}\n{log}"


# ------------------------------------------------------------------ the resource guard

#: Releases a context and creates one per batch row, through the guard's own accounting rather than
#: through EGL. The point being checked here is the POLICY and the plumbing — that a violation becomes
#: a named test failure at the moment it happens — which is independent of whether a real driver was
#: involved. That the patch is really attached to ``mujoco.egl.GLContext`` is asserted separately, on
#: the GPU, in tests/gpu/test_crash_isolation.py.
GUARD_PROBE = """
from tests._runner import resource_guard as G

def test_releases_a_context():
    G._record("context", "created")
    G._record("context", "freed")

def test_closes_a_renderer():
    G._record("renderer", "created")
    G._record("renderer", "closed")

def test_creates_one_context_per_row():
    for _ in range(16):
        G._record("context", "created")

def test_innocent():
    assert True
"""


def test_the_resource_guard_fails_the_test_that_broke_the_rule(tmp_path):
    """Each guarded rule violation fails the offending test under a greppable name (for example
    ``EGL_CONTEXT_RELEASED``), while an innocent test in the same module still passes."""
    (tmp_path / "test_guard.py").write_text(GUARD_PROBE)
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path / "test_guard.py"),
            "-p",
            "tests._runner.safety_plugin",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-q",
            "--no-header",
            "-rN",
        ],
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
    )
    out = p.stdout + p.stderr
    assert "EGL_CONTEXT_RELEASED" in out, out
    assert "MUJOCO_RENDERER_CLOSED" in out, out
    assert "EGL_CONTEXTS_PER_TEST" in out or "EGL_CONTEXTS_SCALE_WITH_BATCH" in out, out
    assert "3 failed, 1 passed" in out, f"the guard must fail exactly the three offenders:\n{out}"


def test_the_guard_can_be_put_in_report_only_mode_for_a_bisect(tmp_path):
    """The thresholds are environment-overridable, so investigating one does not require an edit."""
    (tmp_path / "test_guard.py").write_text(GUARD_PROBE)
    p = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tmp_path / "test_guard.py"),
            "-p",
            "tests._runner.safety_plugin",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-q",
            "--no-header",
            "-rN",
            "-k",
            "per_row",
        ],
        env=_clean_env(SE_MAX_CONTEXTS_PER_TEST="64"),
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
    )
    assert "1 passed" in p.stdout, p.stdout + p.stderr


# ------------------------------------------------------------------ the per-test timeout


def test_the_per_test_timeout_is_applied_but_never_required(tmp_path):
    """900s by default, an explicit ``--timeout`` still wins, and the suite runs without the plugin.

    Three properties and each one is a way the timeout could be worse than useless. Not applied: a hung
    test hangs the run, which is the thing it exists to stop. Not overridable: debugging the hang means
    editing the repo. Required: a bare ``pytest`` on an environment without pytest-timeout dies at
    startup with "unrecognized arguments", which is why the number is set from the plugin rather than
    from ``addopts`` — a safety margin must not become a dependency.
    """
    (tmp_path / "test_generated.py").write_text(
        "def test_timeout_is_set(request):\n"
        "    import os\n"
        "    want = int(os.environ.get('WANT_TIMEOUT', '900'))\n"
        "    got = getattr(request.config.option, 'timeout', None)\n"
        "    assert got == want, f'timeout is {got!r}, expected {want}'\n"
    )
    base = [
        sys.executable,
        "-m",
        "pytest",
        str(tmp_path / "test_generated.py"),
        "-p",
        "tests._runner.safety_plugin",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        "-q",
        "--no-header",
    ]
    for args, want, why in (
        ([], "900", "the plugin's default was not applied"),
        (["--timeout=123"], "123", "an explicit --timeout did not win over the default"),
    ):
        p = subprocess.run(
            base + args,
            env=_clean_env(WANT_TIMEOUT=want),
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(tmp_path),
        )
        assert "1 passed" in p.stdout, f"{why}:\n{p.stdout}\n{p.stderr}"

    # and with pytest-timeout switched off entirely, the run must still start
    p = subprocess.run(
        base + ["-p", "no:timeout"],
        env=_clean_env(WANT_TIMEOUT="None"),
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(tmp_path),
    )
    assert "1 failed" in p.stdout or "1 passed" in p.stdout, (
        f"the suite did not even start without pytest-timeout:\n{p.stdout}\n{p.stderr}"
    )
    assert "unrecognized arguments" not in p.stderr, p.stderr


# ------------------------------------------------------------------ the node gate


def test_a_condemned_node_is_refused_by_name_before_anything_is_measured():
    """The scheduler's own verdict, read and acted on. This is the incident that cost the most.

    Twenty-five OGBench cells failed with 69-90/255 differences and the interpreter then aborted; the
    node had been flagged ``NHC: auto-reboot`` 41 minutes before the job started. The suite was right
    and the output never said so, because nothing in it named the host. So the assertion here is not
    only that the gate closes but that the message carries the hostname and the scheduler's reason —
    those two strings are the entire difference between a one-line diagnosis and an afternoon.
    """
    from tests._runner import node_health as N

    real = N.slurm_node_status
    try:
        N.slurm_node_status = lambda host=None, timeout=20.0: {
            "state": "drain*",
            "reason": "NHC: auto-reboot [root@2026-08-09T22:00:00]",
            "source": "sinfo",
        }
        g = N.gate(check_renderer=False)
    finally:
        N.slurm_node_status = real
    assert not g["ok"]
    assert g["host"] in g["why"] and "NHC: auto-reboot" in g["why"], g["why"]
    assert "--exclude=" in g["why"], "the message must say what to do next, not only what is wrong"


def test_a_healthy_state_string_is_not_mistaken_for_a_condemned_one():
    """``mix-``, ``alloc``, ``idle~`` are healthy. Over-reading them would refuse every run.

    A gate that fires on a healthy node is worse than no gate: it gets disabled, and then the real one
    goes unnoticed. SLURM decorates states with ``*~#$`` suffixes, so the match is on the alphabetic
    stem — which is exactly the kind of thing that is obvious until someone writes ``state == "drain"``.
    """
    from tests._runner.node_health import state_is_condemned

    for good in ("mix", "mix-", "alloc", "alloc#", "idle", "idle~", "comp", "resv", "plnd"):
        assert not state_is_condemned(good), good
    for bad in ("drain", "drain*", "drng", "down*", "fail", "maint", "err"):
        assert state_is_condemned(bad), bad


def test_an_unstable_reference_renderer_closes_the_gate_even_on_a_healthy_looking_node():
    """The second question, and the one the scheduler cannot answer.

    A node can be ``mix`` and idle and still have an offscreen rasterizer that returns two different
    frames for the same scene — that is what produced 69-90/255 differences across 25 OGBench cells
    followed by SIGABRT. So the gate asks the renderer directly and, when it disagrees with itself, the
    message has to say that no stable-envs code was involved: otherwise the next reader spends the
    afternoon in ``raycast3d.py``.
    """
    from tests._runner import node_health as N

    real_sched, real_probe = N.slurm_node_status, N.reference_renderer_probe
    try:
        N.slurm_node_status = lambda host=None, timeout=20.0: {"state": "mix", "reason": "", "source": "sinfo"}
        N.reference_renderer_probe = lambda candidates=None: {
            "env": "PointmazeMediumNavigate",
            "size": 224,
            "max_abs_diff": 77,
        }
        g = N.gate()
    finally:
        N.slurm_node_status, N.reference_renderer_probe = real_sched, real_probe
    assert not g["ok"]
    assert "OFFICIAL" in g["why"] and "77/255" in g["why"], g["why"]
    assert "before any stable-envs code was involved" in g["why"], (
        "the message must say the fault is the node's, or it will be read as a renderer bug"
    )


def test_a_condemned_node_ends_the_whole_run_with_code_78_and_an_artifact(tmp_path):
    """End to end: the run stops before the first test, exits 78, and leaves the diagnosis on disk.

    78 rather than 1 on purpose. "The tests failed" and "this machine is not fit to measure on" demand
    completely different reactions — resubmit elsewhere versus go and read the code — and a CI that
    cannot tell them apart will eventually treat one as the other, which is how this went wrong the
    first time.
    """
    (tmp_path / "conftest.py").write_text(
        "from tests._runner import node_health\n"
        "node_health.slurm_node_status = lambda host=None, timeout=20.0: "
        "{'state': 'drain*', 'reason': 'NHC: auto-reboot', 'source': 'sinfo'}\n"
    )
    rc, res, log = _run_supervisor_env(tmp_path, "def test_a(): assert True\n", SE_SKIP_NODE_GATE="")
    assert rc == 78, f"a condemned node must not exit like a test failure:\n{log}"
    fault = json.loads((tmp_path / "run" / "node_fault.json").read_text())
    assert "NHC: auto-reboot" in fault["why"] and fault["host"] in fault["why"]
    assert "NODE FAULT" in log


def _run_supervisor_env(tmp_path: Path, module_src: str, **envextra):
    (tmp_path / "test_generated.py").write_text(module_src)
    rundir = tmp_path / "run"
    out_json = tmp_path / "result.json"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "tests._runner.supervisor",
        "--rundir",
        str(rundir),
        "--json",
        str(out_json),
        "--",
        str(tmp_path / "test_generated.py"),
        "-p",
        "tests._runner.safety_plugin",
        "-p",
        "no:randomly",
        "-p",
        "no:cacheprovider",
        "-q",
    ]
    p = subprocess.run(cmd, env=_clean_env(**envextra), capture_output=True, text=True, timeout=300, cwd=str(tmp_path))
    res = json.loads(out_json.read_text()) if out_json.exists() else {}
    return p.returncode, res, p.stdout + p.stderr


# ------------------------------------------------------------------ selection


def test_the_smoke_subset_covers_every_env_family_exactly_once():
    """One env per family, computed from the registry rather than listed.

    The failure this guards against is quiet: someone adds an env module, the fast subset silently does
    not cover it, and the first anyone hears is a full-run failure days later. Deriving the set from
    ``REGISTRY`` means a new family is covered the day it lands, and this asserts the derivation is
    total — every family gets exactly one representative, and every representative is a real env.
    """
    pytest.importorskip("torch")
    import stable_envs as se
    from tests._runner.safety_plugin import _family_representatives, smoke_envs

    reps = _family_representatives()
    families = {cls.__module__.rsplit(".", 1)[-1] for cls in se.registry.REGISTRY.values()}
    assert set(reps) == families, f"families without a smoke representative: {families - set(reps)}"
    assert set(reps.values()) <= set(se.registry.REGISTRY), "a representative is not a registered env"
    assert len(smoke_envs()) == len(families)
