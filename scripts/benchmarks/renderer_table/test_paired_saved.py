"""Check fair workload accounting without requiring GPU hardware."""

import pytest

from scripts.benchmarks.renderer_table import paired_saved


def test_paired_timer_runs_equal_work_and_alternates_order(monkeypatch):
    ticks = iter(range(0, 100_000_000, 1_000_000))
    monkeypatch.setattr(paired_saved.time, "perf_counter_ns", lambda: next(ticks))
    calls = []
    result = paired_saved.sample_pairs(
        lambda: calls.append("ours"), lambda: calls.append("source"), 64, lambda: None, minimum=4, maximum=4
    )
    assert calls == ["ours", "source", "source", "ours"] * 2
    assert result["ours"]["count"] == result["source"]["count"] == 4
    assert result["ours"]["frames_per_second"] == 64000
    assert result["source"]["mean_ms_per_frame"] == pytest.approx(1 / 64)


def test_batch_time_normalization_and_sample_sd():
    result = paired_saved.statistics_for([2.0, 4.0], 64)
    assert result["mean_ms_per_frame"] == pytest.approx(3 / 64)
    assert result["sd_ms_per_frame"] == pytest.approx(2**0.5 / 64)


def test_quality_analysis_is_outside_both_render_timers(monkeypatch):
    clock = [0]
    monkeypatch.setattr(paired_saved.time, "perf_counter_ns", lambda: clock[0])

    def ours():
        clock[0] += 2_000_000
        return "ours image"

    def source():
        clock[0] += 4_000_000
        return "source image"

    def quality(left, right):
        assert (left, right) == ("ours image", "source image")
        clock[0] += 1_000_000_000

    result = paired_saved.sample_pairs(ours, source, 64, lambda: None, minimum=3, maximum=3, quality=quality)
    assert result["ours"]["samples_ms"] == [2.0] * 3
    assert result["source"]["samples_ms"] == [4.0] * 3
