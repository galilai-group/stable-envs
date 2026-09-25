"""Benchmark accounting guards; no changes to simulator semantics."""

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts/benchmarks/state_ops.py"
spec = importlib.util.spec_from_file_location("state_ops", PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_preparation_and_warmup_are_excluded(monkeypatch):
    clock = [0]
    calls = []
    monkeypatch.setattr(bench, "synchronize", lambda x=None: x)
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])

    def prepare():
        clock[0] += 1000000
        calls.append("prepare")

    def operation():
        clock[0] += 100
        calls.append("operation")
        return 42

    result = bench.measure(operation, prepare=prepare, warmup=2, iterations=5, minimum_seconds=0)
    assert result["samples_ns"] == [100] * 5
    assert result["cold_first_call_ns"] == 100
    assert result["operations_per_second"] == 1e7
    assert calls == ["prepare", "operation"] * 8


def test_percentiles_come_from_retained_samples(monkeypatch):
    clock = [0]
    durations = iter([999, 200, 10, 20, 30, 40])
    monkeypatch.setattr(bench, "synchronize", lambda x=None: x)
    monkeypatch.setattr(bench.time, "perf_counter_ns", lambda: clock[0])

    def operation():
        clock[0] += next(durations)

    result = bench.measure(operation, warmup=1, iterations=4, minimum_seconds=0)
    assert result["median_ns"] == 25
    assert result["p95_ns"] == pytest.approx(38.5)
    assert result["cold_first_call_ns"] == 999


def test_all_accepted_registrations_are_separate_cases():
    cases = bench.catalog()
    assert len(cases) == 39
    assert sum("Fetch" in name for name in cases) == 12
    assert sum("Craftax" in name for name in cases) == 4
    assert not any("PushT" in name or "Rocket" in name for name in cases)
