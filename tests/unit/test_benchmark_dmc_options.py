"""CPU-only checks for benchmark planning and memory/error reporting; no envs or CUDA."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import dmc_mjwarp as benchmark


def _unexpected(*args, **kwargs):
    pytest.fail("a dry run must not initialize CUDA, construct an environment or launch a worker")


@pytest.mark.parametrize(
    "options,batches",
    [
        ([], [1, 16, 64, 256, 1024]),
        (["--extended-batches"], [1, 16, 64, 256, 1024, 2048, 4096, 8192]),
        (["--batches", "1024,4096", "--extended-batches"], [1024, 4096, 2048, 8192]),
        (["--batches", "2048,8192"], [2048, 8192]),
    ],
)
def test_dry_run_expands_matrix_without_starting_work(tmp_path, monkeypatch, options, batches):
    monkeypatch.setattr(benchmark, "CELLS", ("DMCPendulumSwingup", "DMCCheetahRun"))
    monkeypatch.setattr(benchmark.common, "checked_device", _unexpected)
    monkeypatch.setattr(benchmark.common, "public_env", _unexpected)
    monkeypatch.setattr(benchmark, "MemoryProbe", _unexpected)
    monkeypatch.setattr(benchmark, "isolated_record", _unexpected)
    monkeypatch.setattr(benchmark.torch.cuda, "set_device", _unexpected)
    output = tmp_path / "plan.json"
    assert benchmark.main(["--dry-run", "--output", str(output), *options]) == 0
    plan = json.loads(output.read_text())
    assert plan["dry_run"] is True
    assert plan["record_count"] == 2 * len(batches) * 2
    assert plan["records"] == [
        {"env": env, "batch": batch, "simulation_backend": backend}
        for env in benchmark.CELLS
        for batch in batches
        for backend in ("native", "mjwarp")
    ]


@pytest.mark.parametrize(
    "options",
    [
        ["--envs", "missing"],
        ["--simulation-backends", "mjx"],
        ["--batches", "1024,0"],
        ["--_worker"],
    ],
)
def test_invalid_preview_never_writes_a_plan(tmp_path, monkeypatch, options):
    monkeypatch.setattr(benchmark, "CELLS", ("DMCPendulumSwingup",))
    monkeypatch.setattr(benchmark.common, "checked_device", _unexpected)
    output = tmp_path / "plan.json"
    with pytest.raises(SystemExit) as error:
        benchmark.main(["--dry-run", "--output", str(output), *options])
    assert error.value.code == 2
    assert not output.exists()


def test_preview_works_without_dmc_but_measurement_still_requires_it(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "REGISTRY", {})
    monkeypatch.setattr(benchmark.common, "checked_device", _unexpected)
    output = tmp_path / "plan.json"
    args = ["--envs", "DMCPendulumSwingup", "--batches", "8192", "--output", str(output)]
    assert benchmark.main(["--dry-run", *args]) == 0
    assert json.loads(output.read_text())["record_count"] == 2
    with pytest.raises(SystemExit) as error:
        benchmark.main(args)
    assert error.value.code == 2


@pytest.mark.parametrize("platform,expected", [("linux", 123 * 1024), ("darwin", 123), ("win32", None)])
def test_peak_rss_units_are_explicit(monkeypatch, platform, expected):
    monkeypatch.setattr(benchmark.sys, "platform", platform)
    assert benchmark._maxrss_bytes(123) == expected


def test_memory_sample_distinguishes_lifetime_peak_from_current_rss(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(NoSuchProcess=ProcessLookupError))
    monkeypatch.setattr(benchmark, "_self_peak_rss_bytes", lambda: 300)
    child = SimpleNamespace(pid=2, memory_info=lambda: SimpleNamespace(rss=10))
    probe = benchmark.MemoryProbe.__new__(benchmark.MemoryProbe)
    probe.process = SimpleNamespace(
        pid=1, memory_info=lambda: SimpleNamespace(rss=20), children=lambda recursive: [child]
    )
    probe.handle = object()
    probe.nvml = SimpleNamespace(
        nvmlDeviceGetMemoryInfo=lambda handle: SimpleNamespace(used=100),
        nvmlDeviceGetComputeRunningProcesses=lambda handle: [SimpleNamespace(pid=1, usedGpuMemory=50)],
    )
    sample = probe.sample()
    assert sample["rss_peak_self_bytes"] == 300
    assert sample["rss_parent_and_workers_bytes"] == 30
    assert sample["rss_bytes_by_pid"] == {"1": 20, "2": 10}
    assert sample["nvml_parent_and_workers_used_bytes"] == 50


class FakeWorker:
    pid = 123
    returncode = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.mark.parametrize("status,returncode", [(0, 0), (7 << 8, 7), (int(signal.SIGKILL), -signal.SIGKILL)])
def test_worker_collects_kernel_peak_and_bounded_logs_on_every_exit(monkeypatch, status, returncode):
    worker = FakeWorker()

    def start(command, *, cwd, stdout, stderr):
        assert command == ["benchmark-worker"]
        stdout.write(b"x" * 5000 + b"end")
        stderr.write(b"worker diagnostic")
        return worker

    def wait(pid, options):
        assert (pid, options) == (worker.pid, 0)
        return pid, status, SimpleNamespace(ru_maxrss=123)

    monkeypatch.setattr(benchmark.subprocess, "Popen", start)
    monkeypatch.setattr(benchmark.os, "wait4", wait, raising=False)
    monkeypatch.setattr(benchmark.sys, "platform", "linux")
    code, peak, stdout, stderr = benchmark._run_worker(["benchmark-worker"])
    assert code == returncode
    assert peak == 123 * 1024
    assert stdout == "x" * 3997 + "end"
    assert stderr == "worker diagnostic"


def test_worker_reports_unavailable_peak_without_wait4(monkeypatch):
    worker = FakeWorker()

    def wait():
        worker.returncode = 7

    worker.wait = wait
    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: worker)
    monkeypatch.delattr(benchmark.os, "wait4", raising=False)
    assert benchmark._run_worker(["benchmark-worker"]) == (7, None, "", "")


def test_interrupted_wait_cleans_up_worker(monkeypatch):
    worker = FakeWorker()
    calls = []
    worker.kill = lambda: calls.append("kill")
    worker.wait = lambda: calls.append("wait")

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(benchmark.subprocess, "Popen", lambda *args, **kwargs: worker)
    monkeypatch.setattr(benchmark.os, "wait4", interrupt, raising=False)
    with pytest.raises(KeyboardInterrupt):
        benchmark._run_worker(["benchmark-worker"])
    assert calls == ["kill", "wait"]


@pytest.mark.parametrize(
    "result,returncode,expected_status",
    [(None, -9, "error"), (None, -35, "error"), ({"status": "ok"}, -9, "error"), ({"status": "ok"}, 0, "ok")],
)
def test_isolated_record_keeps_peak_even_without_worker_json(monkeypatch, result, returncode, expected_status):
    args = benchmark._parser().parse_args(["--extended-batches"])

    def run(command):
        # Only the resolved cell goes to the worker; it must not expand the matrix again.
        assert command[command.index("--batches") + 1] == "8192"
        assert "--extended-batches" not in command
        if result is not None:
            Path(command[command.index("--output") + 1]).write_text(json.dumps(result))
        return returncode, 52_000_000_000, "output", "diagnostic"

    monkeypatch.setattr(benchmark, "_run_worker", run)
    row = benchmark.isolated_record("DMCPendulumSwingup", 8192, "mjwarp", args)
    assert row["status"] == expected_status
    assert row["worker_peak_rss_bytes"] == 52_000_000_000
    assert row["worker_exit"]["returncode"] == returncode
    if expected_status == "error":
        assert row["worker_stderr_tail"] == "diagnostic"
        assert row["error_type"] == "BenchmarkWorkerError"
