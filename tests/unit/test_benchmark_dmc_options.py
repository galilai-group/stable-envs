"""CPU-only checks for benchmark planning and memory/error reporting; no envs or CUDA."""

from __future__ import annotations

import hashlib
import json
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

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
    assert benchmark.main(["--dry-run", "--seed", "0", "--output", str(output), *options]) == 0
    plan = json.loads(output.read_text())
    assert plan["dry_run"] is True
    assert plan["record_count"] == 2 * len(batches) * 2
    assert plan["records"] == [
        {"env": env, "batch": batch, "seed": 0, "simulation_backend": backend}
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
        ["--seed", "1", "--seeds", "2,3"],
        ["--seeds", "1,1"],
        ["--seeds", "1,-1"],
        ["--seeds", ""],
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
    args = ["--envs", "DMCPendulumSwingup", "--batches", "8192", "--seed", "0", "--output", str(output)]
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
    args = benchmark._parser().parse_args(["--extended-batches", "--seed", "91010"])

    def run(command):
        # Only the resolved cell goes to the worker; it must not expand the matrix again.
        assert command[command.index("--batches") + 1] == "8192"
        assert command[command.index("--seed") + 1] == "91010"
        assert "--seeds" not in command
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


@pytest.mark.parametrize("options,seeds", [([], [91000, 91010, 91020]), (["--seeds", "0,9"], [0, 9])])
def test_seed_sweep_defaults_and_backend_order_are_in_preview(tmp_path, monkeypatch, options, seeds):
    monkeypatch.setattr(benchmark.common, "checked_device", _unexpected)
    output = tmp_path / "plan.json"
    assert (
        benchmark.main(
            ["--dry-run", "--envs", "DMCPendulumSwingup", "--batches", "1024", "--output", str(output), *options]
        )
        == 0
    )
    plan = json.loads(output.read_text())
    assert plan["protocol"]["seeds"] == seeds
    assert plan["protocol"]["warmup_calls"] == 20
    assert plan["protocol"]["timed_calls"] == 100
    assert plan["record_count"] == len(seeds) * 2
    assert [(cell["seed"], cell["simulation_backend"]) for cell in plan["records"]] == [
        (seed, backend)
        for index, seed in enumerate(seeds)
        for backend in (("native", "mjwarp") if index % 2 == 0 else ("mjwarp", "native"))
    ]


def test_random_actions_match_frozen_pr28_recipe_and_batch_prefix():
    actions = benchmark._action_array(3, 2, 91000, 100)
    # Golden digest from the original November 6 PR harness, seed 91000, B=3, A=2.
    assert hashlib.sha256(actions.tobytes()).hexdigest() == (
        "de56224f4117f53ee354c9ef5532173a5c2cc89bb311fe7a8e056087a3b3128d"
    )
    assert actions.dtype == np.float64 and actions.flags.c_contiguous
    np.testing.assert_array_equal(actions, benchmark._action_array(5, 2, 91000, 100)[:, :3])
    assert not np.array_equal(actions, benchmark._action_array(3, 2, 91010, 100))


@pytest.mark.parametrize("warmup", [0, 2, 5])
def test_rollout_restarts_action_sequence_and_checkpoint_after_warmup(monkeypatch, warmup):
    calls = []
    state = [99]
    actions = tuple(range(max(warmup, 3)))
    args = SimpleNamespace(warmup=warmup, repeats=3)

    def restore():
        state[0] = 0

    def step(action):
        calls.append((state[0], action))
        state[0] += 1

    def metric(fn, args, device, *, transitions, warmup, memory_fn):
        assert warmup == 0  # The measured helper must not add a second warmup.
        assert state[0] == 0
        for _ in range(args.repeats):
            fn()
        memory_fn()  # Must not exhaust the timed-action iterator.
        return {"memory": {}}

    monkeypatch.setattr(benchmark, "_metric", metric)
    benchmark._rollout_metric(step, restore, actions, args, "cpu", transitions=1)
    assert calls == [(i, i) for i in range(warmup)] + [(0, 0), (1, 1), (2, 2), (3, 0)]


def test_checkpoint_hash_ignores_leaf_order_but_covers_rng_dtype_and_values():
    state = TensorDict({"qpos": torch.zeros(2, 1), "rng": {"keys": torch.ones(2, 3, dtype=torch.int64)}}, [2])
    reordered = TensorDict({"rng": state["rng"].clone(), "qpos": state["qpos"].clone()}, [2])
    expected = benchmark._state_sha256(state)
    assert benchmark._state_sha256(reordered) == expected
    reordered["rng", "keys"][0, 0] += 1
    assert benchmark._state_sha256(reordered) != expected
    assert benchmark._state_sha256(state.to(torch.float64)) != expected


@pytest.mark.parametrize("mismatch", [None, "actions_sha256", "initial_state_sha256"])
def test_paired_measurements_require_identical_inputs(mismatch):
    def row(backend):
        return {
            "env": "DMCPendulumSwingup",
            "batch": 1024,
            "seed": 91000,
            "simulation_backend": backend,
            "status": "ok",
            "metadata": {"inputs": {"actions_sha256": "actions", "initial_state_sha256": "state"}},
        }

    native, warp = row("native"), row("mjwarp")
    if mismatch:
        warp["metadata"]["inputs"][mismatch] = "different"
    benchmark._check_paired_inputs([native], warp)
    assert native["paired_inputs_match"] is (mismatch is None)
    assert warp["paired_inputs_match"] is (mismatch is None)
    assert warp["status"] == ("error" if mismatch else "ok")
    if mismatch:
        assert warp["error_type"] == "BenchmarkInputMismatch"
        assert mismatch in warp["error"]


@pytest.mark.parametrize("backend", ["native", "mjwarp"])
def test_measurement_phases_replay_the_same_inputs_with_fake_environment(monkeypatch, backend):
    actions_seen, controls_seen = [], []

    class FakeEnvironment:
        action_dim = 1
        simulation_backend = backend
        execution_mode = "fake"
        size, ss = 32, 1
        closed = False

        def __init__(self):
            self.backend = "auto"
            self.count = 0
            self._simulator = SimpleNamespace(
                b=SimpleNamespace(nthread=1),
                _controls=lambda action: action.to(torch.float32),
                _runtime=SimpleNamespace(advance=lambda control: controls_seen.append(control.clone())),
            )

        def reset(self, seed):
            self.count = 0
            return TensorDict({"qpos": torch.zeros(2, 1, dtype=torch.float64)}, [2])

        def set_state(self, saved):
            self.count = 0

        def step(self, action, *, states, frames):
            assert states is None
            actions_seen.append((frames is not None, self.count, action.clone()))
            self.count += 1
            return {"state": self.count}

        def render(self, saved):
            return None

        def close(self):
            self.closed = True

    def profile(fn, device):
        fn()
        return {}

    candidate = FakeEnvironment()
    args = benchmark._parser().parse_args(["--seed", "91000", "--warmup", "2", "--repeats", "3"])
    args.memory_probe = SimpleNamespace(sample=lambda: {}, profile=profile)
    monkeypatch.setattr(benchmark.common, "public_env", lambda name, kwargs: candidate)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_allocated", lambda device: 0)
    monkeypatch.setattr(benchmark.torch.cuda, "max_memory_reserved", lambda device: 0)
    row = benchmark.measure_cell("DMCPendulumSwingup", 2, backend, args, torch.device("cpu"))
    expected = benchmark._action_array(2, 1, 91000, 3)
    public = [(count, action) for pixels, count, action in actions_seen if not pixels]
    pixels = [(count, action) for pixels, count, action in actions_seen if pixels]
    assert [count for count, _ in public] == [0, 1, 0, 1, 0, 1, 2, 3]
    assert [count for count, _ in pixels] == [0, 1, 0, 1, 2, 3]
    for seen, indices in ((public, [0, 0, 0, 1, 0, 1, 2, 0]), (pixels, [0, 1, 0, 1, 2, 0])):
        np.testing.assert_array_equal(np.stack([action.numpy() for _, action in seen]), expected[indices])
    if backend == "mjwarp":
        np.testing.assert_array_equal(
            np.stack([control.numpy() for control in controls_seen]),
            expected[[0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 0, 0]].astype(np.float32),
        )
    assert row["metadata"]["inputs"]["actions_sha256"] == hashlib.sha256(expected).hexdigest()
    assert len(row["metrics"]["public_step_no_pixels"]["samples_ms"]) == 3
    assert candidate.closed
