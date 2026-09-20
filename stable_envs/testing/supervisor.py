"""Run the test suite so that a test which kills the interpreter costs ONE test, not the run.

THE PROBLEM. This suite has aborted itself four times in a week while being green. A segfault or a
``SIGABRT`` out of MuJoCo, EGL or a CUDA driver is not an exception: no ``try``/``except`` sees it, the
process is gone, pytest writes no report for the test that was running, and every test that had not
run yet is simply never run. The information the run produces about the failure is a shell exit code.
That is the defect this module fixes, and it is a defect independently of whether the suite is passing
today.

WHY NOT ``pytest-forked``/``--forked``. It is the obvious answer and it does not work here, measured
rather than assumed. On this node (H200, torch 2.11+cu130), a forked child cannot use CUDA at all::

    torch.cuda.is_available()      -> True     (NVML path; does not initialise a context)
    torch.cuda.is_initialized()    -> False
    pid = os.fork()
    # in the child:
    torch.zeros(4, device="cuda")
    RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing,
                  you must use the 'spawn' start method

That is with CUDA *never touched* in the parent — torch installs a fork handler when ``torch.cuda`` is
imported and refuses in the child regardless. Since every test in ``tests/gpu`` renders on CUDA,
``--forked`` would turn 1221 passes into 1221 errors. It is also the wrong shape twice over: it would
re-run every module-scoped fixture per test (an OGBench plate bake apiece), and the EGL contexts a
child creates die with the child, which is precisely the release pattern this repo forbids.

WHAT THIS DOES INSTEAD. pytest runs in one subprocess, at full speed, with no per-test cost. The
plugin writes one line per test into an events file *before* the test starts and *after* it ends, and
flushes. If the interpreter dies, the supervisor reads the last start with no matching end — that is
the test that killed it, by name — records it as CRASHED with the signal, adds it and everything
already finished to a deselect file, and launches the next segment. The run continues from where it
died. A run with no crashes pays exactly one extra process and nothing else.

A stall is handled the same way. ``pytest-timeout`` interrupts a test that is hung in Python, but
nothing in-process can interrupt a thread blocked inside a C call in the driver — which is what a hung
EGL or CUDA call is. So the supervisor watches the events file: if no test has started or finished for
``--stall-timeout`` seconds, it kills the process group and treats the in-flight test as a crash. The
per-test timeout is the fast path; this is the one that cannot be defeated.

Exit codes: 0 all good, 1 tests failed or crashed, 78 NODE FAULT (see ``node_health.py``), 2 the
supervisor itself could not proceed.

    python -m stable_envs.testing.supervisor -- tests/ -v -rs
    python -m stable_envs.testing.supervisor --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

#: Exit code the plugin uses for ``pytest.exit`` when the node itself is unfit. Distinct from "tests
#: failed" on purpose: those two demand completely different reactions and conflating them is how a
#: condemned node once got diagnosed as a repo bug.
NODE_FAULT_RC = 78

#: A crash loop has to end somewhere. Twelve is far more than any real run needs and small enough that
#: a systematically-crashing suite reports rather than spins.
MAX_SEGMENTS = 12

_OUTCOME_ORDER = ("error", "failed", "crashed", "xpassed", "xfailed", "skipped", "passed")


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue  # a line half-written when the process died
    return out


def _in_flight(events: list[dict]) -> str | None:
    """The nodeid that was running when the process died: the last start with no matching end."""
    open_ids = []
    for e in events:
        if e.get("kind") == "start":
            open_ids.append(e["nodeid"])
        elif e.get("kind") == "end" and e.get("nodeid") in open_ids:
            open_ids.remove(e["nodeid"])
    return open_ids[-1] if open_ids else None


def _signal_name(rc: int) -> str:
    if rc < 0:
        try:
            return signal.Signals(-rc).name
        except ValueError:
            return f"signal {-rc}"
    if rc > 128:
        try:
            return signal.Signals(rc - 128).name
        except ValueError:
            return f"exit {rc}"
    return f"exit {rc}"


def _run_segment(cmd: list[str], env: dict, events: Path, stall: float, max_wall: float) -> tuple[int, str]:
    """Launch one pytest process, watching the events file for a stall. Returns (rc, note)."""
    started = time.time()
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    last_size, last_change = -1, time.time()
    note = ""
    while True:
        try:
            rc = p.wait(timeout=5.0)
            return rc, note
        except subprocess.TimeoutExpired:
            pass
        size = events.stat().st_size if events.exists() else 0
        now = time.time()
        if size != last_size:
            last_size, last_change = size, now
        if now - last_change > stall:
            note = (
                f"no test started or finished for {stall:.0f}s — killed as a stall. Nothing "
                f"in-process can interrupt a thread blocked in a driver call, which is why this "
                f"watchdog exists alongside the per-test timeout"
            )
            _kill_tree(p)
            return p.wait(), note
        if max_wall and now - started > max_wall:
            note = f"segment exceeded --max-wall {max_wall:.0f}s"
            _kill_tree(p)
            return p.wait(), note


def _kill_tree(p: subprocess.Popen) -> None:
    """SIGTERM the whole process group, then SIGKILL. The group matters: pytest spawns the
    cross-process render worker, and a supervisor that killed only pytest would leave a GPU-holding
    orphan behind for the next segment to collide with."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(p.pid), sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rundir", default="", help="where events/deselect land; default a temp dir")
    ap.add_argument("--smoke", action="store_true", help="shorthand for -m smoke")
    ap.add_argument(
        "--stall-timeout",
        type=float,
        default=900.0,
        help="seconds with no test starting or finishing before the segment is killed",
    )
    ap.add_argument("--max-wall", type=float, default=0.0, help="0 disables")
    ap.add_argument("--max-segments", type=int, default=MAX_SEGMENTS)
    ap.add_argument("--json", default="", help="also write the aggregated result here")
    ap.add_argument("pytest_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    args = [x for x in a.pytest_args if x != "--"]
    # Default to tests/ when the caller named no target. Detected by shape rather than by "the first
    # token that is not a flag": `-k PushT` and `-m smoke` both have a bare word in them, and treating
    # that word as a path would leave pytest collecting from the repo root.
    # and examples/, hundreds of probe scripts that were never meant to be collected.
    if not any(x.endswith(".py") or "/" in x or x == "tests" for x in args):
        args = args + ["tests/"]
    if a.smoke and "-m" not in args:
        args = ["-m", "smoke"] + args

    rundir = Path(a.rundir) if a.rundir else Path(os.environ.get("TMPDIR", "/tmp")) / f"se-tests-{os.getpid()}"
    rundir.mkdir(parents=True, exist_ok=True)
    deselect = rundir / "deselect.txt"
    deselect.write_text("")
    for stale in rundir.glob("events.*.jsonl"):
        stale.unlink()

    base_env = dict(os.environ)
    base_env["SE_TEST_RUNDIR"] = str(rundir)
    base_env["SE_TEST_DESELECT"] = str(deselect)
    base_env.setdefault("SDL_VIDEODRIVER", "dummy")

    # No core dumps. A crash-isolating runner exists to make crashes cheap, and a Python process with
    # torch and CUDA mapped in writes a multi-gigabyte core — per crash, on shared storage, in a job
    # that is already having a bad day. Children inherit the limit.
    try:
        import resource as _rlimit

        _rlimit.setrlimit(_rlimit.RLIMIT_CORE, (0, 0))
    except Exception:  # noqa: BLE001
        pass

    crashed: list[tuple[str, str]] = []
    t0 = time.time()
    rc = 0
    for seg in range(1, a.max_segments + 1):
        # ONE EVENT FILE PER SEGMENT, and the reason is a bug this cost: with a single cumulative file,
        # the start record of the test that crashed in segment 1 stays unmatched forever — it is
        # deselected, so it never runs again and never gets an end — and every later segment therefore
        # looks like it died in that same test. Observed as a run that "crashed" twelve times in a row
        # on one hung test and then gave up. In-flight is a question about ONE process, so it has to be
        # asked of one process's records.
        events = rundir / f"events.{seg}.jsonl"
        events.write_text("")
        base_env["SE_TEST_EVENTS"] = str(events)
        cmd = [sys.executable, "-u", "-m", "pytest", *args]
        print(f"\n=== segment {seg}: {' '.join(cmd)}", flush=True)
        rc, note = _run_segment(cmd, base_env, events, a.stall_timeout, a.max_wall)
        ev = _read_events(events)
        if rc == NODE_FAULT_RC:
            break
        # A test left in flight is the definition of a crash, and it is checked BEFORE the exit code
        # rather than after. An exit code alone is not sufficient evidence: MuJoCo's own fatal path
        # (``mju_error`` -> ``mjERROR`` -> ``exit(1)``) ends the process with status 1, which is
        # indistinguishable from "some tests failed" if all you look at is the number. The breadcrumb
        # is not ambiguous — pytest always closes out every test it started.
        who = _in_flight(ev)
        if who is None:
            # 0 passed, 1 tests failed, 2 interrupted, 3 internal error, 4 usage error, 5 no tests
            if 0 <= rc <= 5:
                break
            print(
                f"\nSUPERVISOR: the run died with {_signal_name(rc)} and no test was in flight — "
                f"this is a collection/import/session-level crash, which no per-test isolation can "
                f"survive. Nothing to resume past.",
                flush=True,
            )
            rc = 1
            break
        why = note or f"the interpreter died mid-test: {_signal_name(rc)}"
        crashed.append((who, why))
        print(f"\nSUPERVISOR: {who}\n  {why}\n  recorded as CRASHED; resuming after it", flush=True)
        done = {e["nodeid"] for e in _all_events(rundir) if e.get("kind") == "end"}
        done |= {c[0] for c in crashed}
        deselect.write_text("\n".join(sorted(done)) + "\n")
    else:
        print(f"\nSUPERVISOR: gave up after {a.max_segments} segments", flush=True)

    return _summarise(rundir, crashed, time.time() - t0, rc, a.json)


def _all_events(rundir: Path) -> list[dict]:
    """Every segment's records, in segment order. Numeric sort: ``events.10`` follows ``events.9``."""
    files = sorted(
        rundir.glob("events.*.jsonl"), key=lambda p: int(p.name.split(".")[1]) if p.name.split(".")[1].isdigit() else 0
    )
    return [e for f in files for e in _read_events(f)]


def _summarise(rundir: Path, crashed, wall: float, rc: int, json_out: str) -> int:
    ev = _all_events(rundir)
    counts: dict[str, int] = {}
    slowest = []
    for e in ev:
        if e.get("kind") != "end":
            continue
        counts[e.get("outcome", "?")] = counts.get(e.get("outcome", "?"), 0) + 1
        if e.get("dur") is not None:
            slowest.append((e["dur"], e["nodeid"]))
    counts["crashed"] = len(crashed)
    res = {k: counts.get(k, 0) for k in _OUTCOME_ORDER}
    # Summed over segments, not read off the last one: a run that crashed twice has three session_end
    # records and the interesting number is what the whole run did. Peak live is a max for the same
    # reason -- each segment starts from zero live contexts, so the peak is per-segment by construction.
    ends = [e for e in ev if e.get("kind") == "session_end"]
    totals = {
        k: sum(int(e.get(k, 0) or 0) for e in ends)
        for k in ("context_created", "context_freed", "renderer_created", "renderer_closed")
    }
    totals["context_peak_live"] = max((int(e.get("context_peak_live", 0) or 0) for e in ends), default=0)
    gate = next((e for e in reversed(ev) if e.get("kind") == "node_gate"), {})
    segs = sum(1 for e in ev if e.get("kind") == "session_start")
    slowest.sort(reverse=True)

    node_fault = rundir / "node_fault.json"
    print("\n" + "=" * 100)
    if rc == NODE_FAULT_RC or node_fault.exists():
        g = json.loads(node_fault.read_text()) if node_fault.exists() else gate
        print(f"NODE FAULT on {g.get('host')}: {g.get('why')}")
        print("=" * 100, flush=True)
        return NODE_FAULT_RC
    parts = [f"{res[k]} {k}" for k in _OUTCOME_ORDER if res[k]] or ["0 tests"]
    print(
        f"SUMMARY: {', '.join(parts)}"
        f" | {segs} segment(s) | {wall / 60:.1f} min"
        f" | EGL contexts {totals.get('context_created', '?')} created /"
        f" {totals.get('context_freed', '?')} released, peak live"
        f" {totals.get('context_peak_live', '?')}"
        f" | node {gate.get('host', '?')} {gate.get('scheduler', {}).get('state', '?')}"
    )
    if slowest:
        print("slowest: " + ", ".join(f"{d:.0f}s {n.split('::')[-1]}" for d, n in slowest[:5]))
    for who, why in crashed:
        print(f"CRASHED  {who}\n         {why}")
    print(f"artifacts: {rundir}")
    print("=" * 100, flush=True)

    if json_out:
        Path(json_out).write_text(
            json.dumps(
                {
                    "counts": res,
                    "segments": segs,
                    "wall_s": wall,
                    "crashed": crashed,
                    "resources": totals,
                    "node": gate,
                    "slowest": [{"dur": d, "nodeid": n} for d, n in slowest[:50]],
                },
                indent=2,
            )
        )
    # xpassed is NOT counted as bad here: every xfail in this repo is strict, and a strict XPASS is
    # already reported by pytest as a failure. Adding it again would make the supervisor's verdict
    # differ from pytest's on a non-strict xpass for no reason a reader could predict.
    bad = res["failed"] + res["error"] + res["crashed"]
    return 1 if (bad or rc not in (0, 5)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
