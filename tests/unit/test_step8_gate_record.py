"""Regression coverage for truthful historical Gate-A reporting."""
import importlib.util
from pathlib import Path

import numpy as np

from stable_envs.testing.conformance import resolution_consistency_applicability


ROOT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_exclusion_only_applies_to_all_states_outside_exclusion():
    class Env:
        state_names = ["light"]
        resolution_consistency_applicability = {"light < 0.5": "not_applicable: night"}
    result = resolution_consistency_applicability(Env(), np.array([[0.5], [1.0]]))
    assert result["applicable"] is True
    assert result["indices"] is None
    result = resolution_consistency_applicability(Env(), np.array([[0.2], [0.8]]))
    assert result["indices"] == [1]


def test_empty_explicit_inclusion_does_not_invent_complement():
    class Env:
        state_names = ["light"]
        resolution_consistency_applicability = {
            "light >= 0.8": "applicable", "light < 0.2": "not_applicable: dark"}
    result = resolution_consistency_applicability(Env(), np.array([[0.1], [0.5]]))
    assert result["applicable"] is False
    assert result["indices"] == []


def test_historical_family_failure_reaches_support_matrix():
    rejudge = module("step8_rejudge_test", "scratch/step8/rejudge_gate_a.py")
    report = module("step8_report_test", "scratch/step8/report.py")
    evidence = rejudge.build()
    rocket = next(r for r in evidence["records"]
                  if r["cell"] == "PFRocketLanding" and r["device"] == "cuda")
    assert rocket["previous_mandatory_only_verdict"] == "PASS"
    assert "check_colors_apply_exactly" in rocket["judgment"]["failures"]
    row = {"stable_envs_envs": ["PFRocketLanding"], "gate_a": {"result": "PASS"}}
    joined = report._gate_a(row, evidence)
    assert joined["status"] == "FAIL"
    assert "check_colors_apply_exactly" in joined["failing_checks"]
    assert joined["per_device"]["cuda"] == "FAIL"
