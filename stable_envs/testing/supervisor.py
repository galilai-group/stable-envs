"""Run pytest in restartable subprocesses and aggregate failures across crashes.

Use python -m stable_envs.testing.supervisor -- [pytest arguments]. Finished
and crashed tests are deselected on restart; crashes remain failures in the
combined report. A bounded restart count prevents an endless crash loop.
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

# Distinguish a host-precondition failure from a failing test suite.
NODE_FAULT_RC = 78

# Bound restart attempts when successive tests crash.
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
    """Return the last started test without a matching completion event."""
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
    """Run one pytest subprocess with a stall watchdog; return its exit code and diagnostic."""
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
    """Terminate the process group, then kill survivors, including render-worker children."""
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
    # Recognize path-shaped targets so values after -k or -m do not suppress the tests/ default.
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

    # Children inherit this limit; native crashes must not fill shared storage with core dumps.
    try:
        import resource as _rlimit

        _rlimit.setrlimit(_rlimit.RLIMIT_CORE, (0, 0))
    except Exception:  # noqa: BLE001
        pass

    crashed: list[tuple[str, str]] = []
    t0 = time.time()
    rc = 0
    for seg in range(1, a.max_segments + 1):
        # Each process needs its own event file; an earlier crash leaves an unmatched start forever.
        events = rundir / f"events.{seg}.jsonl"
        events.write_text("")
        base_env["SE_TEST_EVENTS"] = str(events)
        cmd = [sys.executable, "-u", "-m", "pytest", *args]
        print(f"\n=== segment {seg}: {' '.join(cmd)}", flush=True)
        rc, note = _run_segment(cmd, base_env, events, a.stall_timeout, a.max_wall)
        ev = _read_events(events)
        if rc == NODE_FAULT_RC:
            break
        # Check unfinished tests before exit status: MuJoCo fatal errors can exit 1, like pytest failures.
        who = _in_flight(ev)
        if who is None:
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
    """Read segment events in numeric segment order."""
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
    # Sum counts across restarted processes, but take the maximum live-resource peak.
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
    # Strict XPASS already counts as failed; do not additionally reject non-strict XPASS.
    bad = res["failed"] + res["error"] + res["crashed"]
    return 1 if (bad or rc not in (0, 5)) else 0


if __name__ == "__main__":
    raise SystemExit(main())
