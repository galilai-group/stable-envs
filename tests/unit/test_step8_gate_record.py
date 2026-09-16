"""Regression coverage for truthful historical Gate-A reporting.

Two things are being defended here and they pull against each other, which is why both get tests:

* a family_specific check that IS applicable must reach the Gate-A verdict -- the Step-7 harness
  judged only the mandatory group, so a cell could be recorded PASS while carrying a failing Gate-A
  check;
* a check whose bound the CLASS declares inapplicable, with the official renderer's own evidence
  attached, must come out ``not_applicable`` with that evidence -- neither FAIL (which would judge a
  renderer against a relation its reference does not satisfy) nor a silent PASS (which would hide
  that nothing was asserted).

The loophole tests matter as much as the two above: an applicability declaration that costs nothing
to write is a way to silence a check, so a declaration missing its evidence must be REFUSED.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from stable_envs.testing.conformance import (color_exactness_applicability,
                                             partition_colour_exactness,
                                             resolution_consistency_applicability)


ROOT = Path(__file__).resolve().parents[2]

#: The five leaves PFRocketLanding declares, and nothing else: the ruling is scoped by name.
ROCKET_LEAVES = ("pad.color", "rocket.body_color", "rocket.booster_color", "rocket.fin_color",
                 "rocket.leg_color")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture(scope="module")
def rejudged():
    return module("step8_rejudge_test", "scratch/step8/rejudge_gate_a.py").build()


@pytest.fixture(scope="module")
def report():
    return module("step8_report_test", "scratch/step8/report.py")


def record(rejudged, cell, device):
    return next(r for r in rejudged["records"] if r["cell"] == cell and r["device"] == device)


# --------------------------------------------------------------- resolution applicability reader

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


# --------------------------------------------------------------- colour applicability reader

def test_no_colour_declaration_leaves_every_leaf_judged():
    class Env:
        pass
    result = color_exactness_applicability(Env, ["a.color->[17, 33, 200]"])
    assert result["honoured"] is False
    assert result["not_applicable_probes"] == []
    assert result["undeclared_leaves_still_judged"] == ["a.color->[17, 33, 200]"]


@pytest.mark.parametrize("missing", ["leaves", "mechanism", "max_shading_factor",
                                     "official_renderer_fails_the_same_check", "evidence"])
def test_a_declaration_missing_its_evidence_is_refused(missing):
    """Every required key is load-bearing: drop any one and the check stays judged."""
    full = {"leaves": ["a.color"], "mechanism": "a law", "max_shading_factor": 0.99,
            "official_renderer_fails_the_same_check": True, "evidence": ["somewhere.json"]}
    Env = type("Env", (), {"color_exactness_applicability": {k: v for k, v in full.items()
                                                             if k != missing}})
    result = color_exactness_applicability(Env, ["a.color->[17, 33, 200]"])
    assert result["honoured"] is False
    assert missing in result["rejected_because"]


def test_a_law_that_reaches_unit_gain_is_refused():
    """The whole argument is that the ceiling is below 1. A ceiling of 1 has no case to make."""
    Env = type("Env", (), {"color_exactness_applicability": {
        "leaves": ["a.color"], "mechanism": "a law", "max_shading_factor": 1.0,
        "official_renderer_fails_the_same_check": True, "evidence": ["somewhere.json"]}})
    assert color_exactness_applicability(Env, ["a.color->[17, 33, 200]"])["honoured"] is False


def test_an_undeclared_leaf_is_not_covered_by_a_declared_one():
    Env = type("Env", (), {"color_exactness_applicability": {
        "leaves": ["a.color"], "mechanism": "a law", "max_shading_factor": 0.9,
        "official_renderer_fails_the_same_check": True, "evidence": ["somewhere.json"]}})
    result = color_exactness_applicability(Env, ["a.color->[17, 33, 200]", "b.color->[17, 33, 200]"])
    assert result["honoured"] is True
    assert result["not_applicable_probes"] == ["a.color->[17, 33, 200]"]
    assert result["undeclared_leaves_still_judged"] == ["b.color"]


def test_the_rocket_declares_exactly_the_five_lit_leaves():
    """Read off the CLASS, with no env constructed and no PyFlyt installed."""
    from stable_envs.envs.rocket_landing import PFRocketLanding
    decl = PFRocketLanding.color_exactness_applicability
    assert tuple(decl["leaves"]) == ROCKET_LEAVES
    assert decl["max_shading_factor"] == 0.993567
    assert decl["official_renderer_fails_the_same_check"] is True
    assert any("colour_law_ceiling.json" in e for e in decl["evidence"])
    for cited in decl["evidence"]:
        assert (ROOT / cited.split(":")[0]).exists(), cited


# --------------------------------------------------------------- the rejudged Gate-A record

def test_the_rocket_colour_check_is_not_applicable_with_evidence(rejudged):
    """Not FAIL, and not a silent PASS: declared, with the mechanism and the artifacts attached."""
    rocket = record(rejudged, "PFRocketLanding", "cuda")
    check = rocket["judgment"]["checks"]["check_colors_apply_exactly"]
    assert check["status"] == "not_applicable"
    assert "check_colors_apply_exactly" not in rocket["judgment"]["failures"]
    assert "check_colors_apply_exactly" in rocket["judgment"]["not_applicable"]
    assert check["declared_by"] == "PFRocketLanding.color_exactness_applicability"
    assert "0.993567" in check["why"] and "colour_law_ceiling.json" in check["why"]
    # The measurement survives the ruling: the ten probes are still in the record, so a wiring
    # regression is still visible even though no bound is asserted on them.
    kept = check["measured_evidence"]["colors_not_applied"]
    assert len(kept) == 2 * len(ROCKET_LEAVES)
    assert {p.split("->")[0] for p in kept} == set(ROCKET_LEAVES)


def test_the_rocket_declaration_is_recorded_as_a_join_not_invented(rejudged):
    join = next(j for j in rejudged["applicability_joins"]
                if j["cell"] == "PFRocketLanding" and j["check"] == "check_colors_apply_exactly")
    assert join["result"].startswith("NOT_APPLICABLE")
    assert join["declared_leaves"] == sorted(ROCKET_LEAVES)
    assert join["undeclared_leaves_still_judged"] == []
    # The Step-8 re-sweep measured the distances, so the declaration is not the only thing in the
    # record: every declared probe carries a number a wiring regression would move. Every distance
    # is > 0, which is the whole claim -- a leaf that could reach its colour would not need the
    # declaration, and a zero here would mean the declaration had gone stale.
    best = join["measured_best_abs_diff"]
    assert best is not None and len(best) == 2 * len(ROCKET_LEAVES)
    assert all(v > 0 for v in best.values()), best
    assert join["best_abs_diff_note"] is None


def test_a_record_without_the_distance_says_so_rather_than_showing_a_blank(rejudged):
    """An older shard has no best_abs_diff. The join must name the gap, not imply zero."""
    rejudge = module("step8_rejudge_nobest", "scratch/step8/rejudge_gate_a.py")
    from stable_envs.envs.rocket_landing import PFRocketLanding
    res = {"status": "measured", "evidence": {"colors_not_applied": [
        f"{leaf}->[17, 33, 200]" for leaf in ROCKET_LEAVES]}}
    _, join = rejudge._colour_join("PFRocketLanding", "cuda", res, PFRocketLanding)
    assert join["measured_best_abs_diff"] is None
    assert "records it from Step 8 onward" in join["best_abs_diff_note"]


def test_craftax_classic_is_judged_on_its_applicable_subset(rejudged):
    """The frozen 1.000000 floor, asserted on the band it was recorded for, holds exactly."""
    for device in ("cpu", "cuda"):
        check = record(rejudged, "CraftaxClassic", device)["judgment"]["checks"][
            "check_resolution_consistency"]
        assert check["status"] == "pass"
        assert check["observed"]["exact_frac"] == 1.0
        assert check["judged_on_subset"] == 6
        assert check["applicability"]["bands"] == {"light_level >= 0.5": "applicable",
                                                   "light_level < 0.5": "not_applicable"}
        # The mixed-band number is kept beside it, recorded and never judged.
        assert check["all_bands_recorded_not_judged"]["exact_frac"] == pytest.approx(0.816382137)
        assert check["bound"] == ">= 1.0 - 0.005"


def test_the_craftax_subset_join_cites_the_artifact_it_read(rejudged):
    join = next(j for j in rejudged["applicability_joins"]
                if j["cell"] == "CraftaxClassic" and j["device"] == "cuda"
                and j["check"] == "check_resolution_consistency")
    assert join["result"] == "JOINED"
    assert join["source"] == "examples/results/step8/certification/oscar_paired_tests.json"
    assert len(join["source_sha256"]) == 64
    assert join["n_judged"] == 6
    assert rejudged["sources"]["subset_judgment_sha256"] == join["source_sha256"]


def test_a_subset_join_whose_bands_are_not_the_class_s_is_refused():
    rejudge = module("step8_rejudge_refuse", "scratch/step8/rejudge_gate_a.py")
    cls = type("Env", (), {"resolution_consistency_applicability": {"a >= 1": "applicable"}})
    paired = {"applicability_on_gpu_hardware": {"cells": {"X/cuda": {
        "bands": {"a >= 1": "not_applicable"}, "applicable": True,
        "judged": {"exact_frac": 1.0}, "n_judged": 6}}}}
    evidence, join = rejudge._resolution_join("X", "cuda", {"exact_frac": 0.5}, cls, paired)
    assert join["result"].startswith("REFUSED")
    assert evidence == {"exact_frac": 0.5}


def test_a_subset_join_for_a_different_measurement_is_refused():
    """The joined subset has to belong to the same draw as the shard it is joined onto."""
    rejudge = module("step8_rejudge_refuse2", "scratch/step8/rejudge_gate_a.py")
    cls = type("Env", (), {"resolution_consistency_applicability": {"a >= 1": "applicable"}})
    paired = {"applicability_on_gpu_hardware": {"cells": {"X/cuda": {
        "bands": {"a >= 1": "applicable"}, "applicable": True,
        "all_bands": {"exact_frac": 0.9, "mean_abs": 1.0, "max_abs": 3},
        "judged": {"exact_frac": 1.0}, "n_judged": 6}}}}
    shard = {"exact_frac": 0.5, "mean_abs": 1.0, "max_abs": 3}
    evidence, join = rejudge._resolution_join("X", "cuda", shard, cls, paired)
    assert join["result"].startswith("REFUSED")
    assert evidence == shard


# --------------------------------------------------------------- the support-matrix join

def test_an_applicable_family_failure_still_reaches_the_support_matrix(rejudged, report):
    """The inconsistency this whole change exists to fix, on a cell with no declaration."""
    fake = {"records": [{"cell": "X", "device": "cuda", "source": "s.json", "source_sha256": "0" * 64,
                         "judgment": {"gate_a": "FAIL", "failures": ["check_colors_apply_exactly"],
                                      "blocked": [], "not_applicable": [], "checks": {}}}]}
    joined = report._gate_a({"stable_envs_envs": ["X"], "gate_a": {"result": "PASS"}}, fake)
    assert joined["status"] == "FAIL"
    assert joined["failing_checks"] == ["check_colors_apply_exactly"]


def test_the_rocket_row_passes_with_the_ruling_recorded(rejudged, report):
    row = {"stable_envs_envs": ["PFRocketLanding"],
           "gate_a": {"result": "PASS", "note": "the Step-7 record",
                      "per_device": {"cpu": {"verdict": "REFUSED_BY_DESIGN"}}}}
    joined = report._gate_a(row, rejudged)
    assert joined["status"] == "PASS"
    assert joined["failing_checks"] == []
    # The Step-8 re-sweep records the declared reference versions, so nothing is blocked any more.
    assert joined["blocked_checks"] == []
    assert joined["per_device"] == {"cpu": "UNSUPPORTED_BY_DESIGN", "cuda": "PASS"}
    assert any(d["check"] == "check_colors_apply_exactly" and d["kind"] == "declared_not_applicable"
               for d in joined["applicability"])
    assert "the Step-7 record" in joined["note"]


def test_the_rocket_row_is_judged_on_the_resweep_and_cites_what_it_supersedes(rejudged):
    rocket = record(rejudged, "PFRocketLanding", "cuda")
    assert rocket["measurement"] == "step8_resweep"
    assert rocket["supersedes"]["source"] == "examples/results/step7/closure2/gate_a/cuda_rocket.json"
    assert len(rocket["supersedes"]["source_sha256"]) == 64
    # Same corpus as the shard it supersedes: a re-sweep on a different draw would not be comparable.
    assert (rocket["judgment"]["config_id"], rocket["device"]) == ("PFRocketLanding", "cuda")
    # The two checks that were blocked on an unrecorded reference version are now judged.
    assert rocket["judgment"]["blocked"] == []
    assert rocket["judgment"]["checks"]["check_visual_fidelity"]["status"] == "pass"
    assert rocket["judgment"]["gate_a"] == "PASS"
    # And the closure2 row it replaces is not also present, which would double-count the cell.
    assert sum(1 for r in rejudged["records"]
               if r["cell"] == "PFRocketLanding" and r["device"] == "cuda") == 1


def test_the_craftax_classic_row_passes_with_the_band_recorded(rejudged, report):
    row = {"stable_envs_envs": ["CraftaxClassic"], "gate_a": {"result": "PASS", "note": "kept"}}
    joined = report._gate_a(row, rejudged)
    assert joined["status"] == "PASS"
    assert joined["failing_checks"] == [] and joined["blocked_checks"] == []
    band = next(d for d in joined["applicability"]
                if d["check"] == "check_resolution_consistency" and d["device"] == "cuda")
    assert band["judged_on_subset"] == 6
    assert band["all_bands_recorded_not_judged"]["exact_frac"] == pytest.approx(0.816382137)


# ------------------------------------------------- the helper the GPU assertion uses (CPU-runnable)

def _probe(leaf, target="[17, 33, 200]"):
    return f"{leaf}->{target}"


def test_partition_without_a_declaration_asserts_everything():
    """The pre-Step-8 behaviour, exactly: no declaration means every failure is asserted on."""
    part = partition_colour_exactness({"colors_not_applied": [_probe("a.color"), _probe("b.color")]})
    assert part["honoured"] is False
    assert part["asserted"] == [_probe("a.color"), _probe("b.color")]
    assert part["not_applicable"] == []


def test_partition_asserts_the_undeclared_leaves_and_only_those():
    result = {"colors_not_applied": [_probe("a.color"), _probe("b.color")],
              "best_abs_diff": {_probe("a.color"): 4, _probe("b.color"): 9},
              "applicability": {"honoured": True, "why": "a law",
                                "not_applicable_probes": [_probe("a.color")]}}
    part = partition_colour_exactness(result)
    assert part["asserted"] == [_probe("b.color")]
    assert part["not_applicable"] == [{"probe": _probe("a.color"), "best_abs_diff": 4}]


def test_partition_keeps_the_measurement_beside_the_excused_leaf():
    """The declaration stops the assertion, never the measurement: the number has to come through."""
    result = {"colors_not_applied": [_probe("a.color")],
              "best_abs_diff": {_probe("a.color"): 7},
              "applicability": {"honoured": True, "not_applicable_probes": [_probe("a.color")]}}
    part = partition_colour_exactness(result)
    assert part["asserted"] == []
    assert part["not_applicable"][0]["best_abs_diff"] == 7


def test_partition_on_a_passing_env_is_empty_either_way():
    for result in ({"colors_not_applied": []},
                   {"colors_not_applied": [], "applicability": {"honoured": True}}):
        part = partition_colour_exactness(result)
        assert part["asserted"] == [] and part["not_applicable"] == []


def test_the_rocket_declaration_drives_the_partition_end_to_end():
    """The class declaration, the real probe names, and the helper the GPU suite calls."""
    from stable_envs.envs.rocket_landing import PFRocketLanding
    probes = [_probe(leaf, t) for leaf in ROCKET_LEAVES
              for t in ("[17, 33, 200]", "[200, 17, 33]")]
    result = {"colors_not_applied": probes + [_probe("environment.sky_color")],
              "best_abs_diff": {p: 3 for p in probes + [_probe("environment.sky_color")]},
              "applicability": color_exactness_applicability(PFRocketLanding, probes
                                                             + [_probe("environment.sky_color")])}
    part = partition_colour_exactness(result)
    # The five declared leaves are excused; a sixth leaf nobody declared still fails.
    assert part["asserted"] == [_probe("environment.sky_color")]
    assert len(part["not_applicable"]) == len(probes)


# ------------------------------------------------------------- the variation column's taxonomy

def test_a_partial_leaf_is_never_reported_as_a_complete_map(report):
    """`partial` means the upstream effect is reproduced only IN PART. Rolling it into `complete`
    would say the mapping is done while a leaf is knowingly approximate."""
    class Map:
        @staticmethod
        def classification():
            return {"a.color": "mapped", "background.color": "partial", "b.size": "unsupported"}
    row = {"stable_envs_envs": ["X"], "variation_space": {"variation_map_status": "missing"}}
    v = report._variation(row, {"X": Map})
    assert v["map_status"] == "partial"
    assert v["status"] == "DOCUMENTED_LIMITATION"      # known, measured, bounded — not PASS, not UNMEASURED
    assert v["partial_leaves"] == ["background.color"]


def test_a_map_with_no_partial_leaf_is_complete_and_passes(report):
    class Map:
        @staticmethod
        def classification():
            return {"a.color": "mapped", "b.size": "unsupported", "c.x": "dead_upstream"}
    v = report._variation({"stable_envs_envs": ["X"]}, {"X": Map})
    assert (v["map_status"], v["status"]) == ("complete", "PASS")
    assert v["partial_leaves"] == []
    assert v["unsupported_or_dead_leaves"] == ["b.size", "c.x"]


def test_an_unregistered_cell_keeps_the_coverage_matrix_answer(report):
    """No installed map means the column reports what the artifact said, never a guess."""
    row = {"stable_envs_envs": ["X"], "variation_space": {"variation_map_status": "missing"}}
    v = report._variation(row, {})
    assert v["map_status"] == "missing" and v["status"] == "UNMEASURED"
    assert v["source"] == "examples/results/step7/closure2/coverage_matrix.json"
