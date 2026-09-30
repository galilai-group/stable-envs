"""Harness checks, not environment parity or timing acceptance evidence."""

import json

import pytest
import torch

from scripts.benchmarks.renderer_table.common import ROOT
from scripts.benchmarks.renderer_table.profile_all import batch_kwargs, choose_frames, measure, projection
from scripts.benchmarks.renderer_table.profile_campaign import export, plan


def test_timer_retains_outlier_and_reports_sample_sd(monkeypatch):
    # cold call, then 1, 1, 10 ms: outlier must not be dropped.
    ticks = iter([0, 2_000_000, 2_000_000, 3_000_000, 3_000_000, 4_000_000, 4_000_000, 14_000_000])
    monkeypatch.setattr("time.perf_counter_ns", lambda: next(ticks))
    timing, value = measure(lambda: 42, "cpu", minimum=3, maximum=3, warmups=0)
    assert value == 42 and timing["samples_ms"] == [1.0, 1.0, 10.0]
    assert timing["mean_ms"] == 4
    assert timing["sd_ms"] == pytest.approx(27**0.5)
    assert timing["first_call_ms"] == 2


def test_selection_never_mixes_simulator_configs():
    frames = [dict(case=i % 2, index=i) for i in range(20)]
    selected, config = choose_frames(
        dict(frames=frames, recipes=[dict(simulator_config={"height": 1}), dict(simulator_config={"height": 2})]), 6
    )
    assert config == {"height": 1}
    assert len(selected) == 6
    assert all(f["case"] == 0 for f in selected)
    assert selected[-1]["index"] == 18


def test_style_batch_matches_pose_indices():
    kwargs = [
        {"style": torch.tensor([[1.0, 2.0]]), "backend": "auto"},
        {"style": torch.tensor([[3.0, 4.0]]), "backend": "auto"},
    ]
    result = batch_kwargs(kwargs, [1, 0, 1])
    assert result["style"].tolist() == [[3, 4], [1, 2], [3, 4]]


def test_private_projection_validates_before_packing():
    calls = []

    class Fake:
        def _validate_state(self, state):
            calls.append("validate")

        def _pack_render_state(self, state):
            calls.append("pack")
            return state

        def _render(self, state):
            return state

    prepare, _, label = projection(Fake())
    assert prepare(7) == 7 and calls == ["validate", "pack"]
    assert "private" in label


def test_plan_all_42_device_corpora_and_no_false_completion(tmp_path):
    value = plan(ROOT / "docs/research/semantic_recovery_evidence/renderer_parity_1000_20260928/manifest-v6.json")
    assert len({t["registration"] for t in value["tasks"]}) == 42
    assert len(value["jobs"]) == 84
    assert sorted(i for job in value["jobs"] for i in job["tasks"]) == list(range(len(value["tasks"])))
    for job in value["jobs"]:
        assert all(
            value["tasks"][i]["device"] == job["device"] and value["tasks"][i]["registration"] == job["registration"]
            for i in job["tasks"]
        )
    for t in value["tasks"]:
        if "CraftaxPixels" in t["registration"] and t["device"] == "cuda":
            assert "source-gpu-craftax" in t["corpus"]
        if "Fetch" in t["registration"] and t["device"] == "cpu":
            assert t["backend"] == "opengl"
    manifest = tmp_path / "plan.json"
    manifest.write_text(json.dumps(value))
    export(manifest, tmp_path / "absent", tmp_path / "export")
    result = json.loads((tmp_path / "export/summary.json").read_text())
    assert result["status"] == "INCOMPLETE"
    assert result["statuses"]["MISSING"] == len(value["tasks"])
