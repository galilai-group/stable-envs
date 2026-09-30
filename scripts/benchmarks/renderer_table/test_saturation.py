import pytest

from scripts.benchmarks.renderer_table.saturation import select_batch, should_stop, stats


def rows(rates):
    return [dict(batch=64 * 2**i, status="COMPLETE", timing={"frames_per_second": v}) for i, v in enumerate(rates)]


def test_select_smallest_batch_within_five_percent_of_peak():
    assert select_batch(rows([100, 195, 200, 198])) == 128


def test_never_claim_plateau_before_256():
    assert not should_stop(rows([100, 101]))
    assert should_stop(rows([100, 101, 102]))
    assert not should_stop(rows([100, 160, 190]))


def test_failed_sizes_do_not_win_selection():
    r = rows([100, 180])
    r.append(dict(batch=256, status="OOM"))
    assert select_batch(r) == 128


def test_per_frame_sd_and_total_fps():
    s = stats([2.0, 4.0], 64)
    assert s["frames_per_second"] == pytest.approx(64000 / 3)
    assert s["sd_ms_per_frame"] == pytest.approx(2**0.5 / 64)
