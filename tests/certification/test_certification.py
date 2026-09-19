"""The certification spec is two files, and they must not drift apart.

``docs/research/certification.md`` is the normative prose; ``scratch/certification.json`` is the
machine-readable copy that a later benchmarking step reads to turn a result into a label. Two copies
of the same number is exactly the arrangement that rots, and the failure mode is silent: a threshold
gets tuned in the JSON because a run missed it, the document still says the old number, and the
"frozen before measurement" property the whole spec rests on is gone with nothing to show for it.

So every threshold is asserted to appear VERBATIM in the prose, and the internal shape of the ladder
is asserted too — monotonic tiers, disjoint backend classes, 4096 in the required sweep. These are
cheap checks whose only job is to make an edit to one file fail until the other one agrees.

Nothing here runs a benchmark or imports stable_envs; it is a documentation contract.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "scratch" / "certification.json"
DOC = ROOT / "docs" / "research" / "certification.md"


@pytest.fixture(scope="module")
def spec() -> dict:
    if not SPEC.is_file():
        pytest.skip(f"{SPEC.relative_to(ROOT)} is not present in this checkout")
    return json.loads(SPEC.read_text())


@pytest.fixture(scope="module")
def doc() -> str:
    if not DOC.is_file():
        pytest.skip(f"{DOC.relative_to(ROOT)} is not present in this checkout")
    return DOC.read_text()


def test_every_frozen_threshold_appears_in_the_prose(spec, doc):
    """Each number below is quoted in the document; a change to one file must fail until both agree.

    The token, not a regex over the JSON, is what is checked: a threshold that appears in the prose
    only as part of a derivation ("approx p10 (2.27)") is not the same as one stated as the bar.
    """
    gpu, cpu = spec["gpu"], spec["cpu"]
    tokens = {
        # GPU tiers, class S
        "gpu.S.GPU_PARITY":       "`1 ≤ R < 2`",
        "gpu.S.GPU_FASTER":       "`2 ≤ R < 4`",
        "gpu.S.GPU_FAST":         "`R ≥ 4`",
        "gpu.S.GPU_EXCEPTIONAL":  "`R ≥ 40`",
        # GPU tiers, class R
        "gpu.R.GPU_PARITY":       "`1 ≤ R < 10`",
        "gpu.R.GPU_FASTER":       "`10 ≤ R < 500`",
        "gpu.R.GPU_FAST":         "`R ≥ 500`",
        "gpu.R.GPU_EXCEPTIONAL":  "`R ≥ 5000`",
        # CPU tiers
        "cpu.CPU_REGRESSION":     "`S_cpu < 1`",
        "cpu.CPU_PARITY":         "`1 ≤ S_cpu < 2`",
        "cpu.CPU_FASTER":         "`2 ≤ S_cpu < 5`",
        "cpu.CPU_FAST":           "`S_cpu ≥ 5`",
        "cpu.CPU_EXCEPTIONAL":    "`S_cpu ≥ 20`",
        # qualifiers
        "gpu.BATCH_SCALES":       "`fps(4096) ≥ 20 × fps(1)`",
        "gpu.NO_LARGE_REGRESS":   "`fps(4096) ≥ 0.9 × fps(256)`",
        "gpu.HOST_BOUND":         "`cpu_frac ≥ 1.5`",
        # device scaling
        "device_factor":          "5.13",
        # representation tripwire
        "repr.parity":            "`A ≤ 0.10`",
        "repr.visual":            "`A ≤ 0.25`",
        # physics
        "phys.near_exact.first":  "`first_step_max_err ≤ 1e-12`",
        "phys.near_exact.max":    "`max_err ≤ 1e-9`",
        "phys.approx.control":    "`control_max_err < 1e-3`",
        "phys.approx.gate":       "`block_median_err < 2.0`",
        "phys.approx.touched":    "75 %",
        # downstream
        "down.n":                 "`n ≥ 300`",
        "down.margin":            "±5 points",
        "down.control_delta":     "`Δ ≤ −20`",
        "down.control_p":         "`p < 0.01`",
        "down.discordance":       "0.19",
        # NO_REFERENCE fallbacks -- both devices, since CPU_FAST is mandatory for ENVIRONMENT_DONE
        "gpu.noref.FASTER":       "`cores ≥ 1`",
        "gpu.noref.FAST":         "`cores ≥ 10`",
        "gpu.noref.EXCEPTIONAL":  "`cores ≥ 100`",
        "cpu.noref.FASTER":       "`cores_cpu ≥ 1`",
        "cpu.noref.FAST":         "`cores_cpu ≥ 3`",
        "cpu.noref.EXCEPTIONAL":  "`cores_cpu ≥ 10`",
    }
    missing = sorted(k for k, tok in tokens.items() if tok not in doc)
    assert not missing, (
        "threshold(s) present in scratch/certification.json but not quoted verbatim in "
        "docs/research/certification.md — the two copies have drifted:\n  "
        + "\n  ".join(f"{k}: expected {tokens[k]!r}" for k in missing))

    # and the tokens above must actually match the JSON they claim to mirror
    assert gpu["tiers"]["S"]["GPU_FAST"]["R_gte"] == 4
    assert gpu["tiers"]["S"]["GPU_EXCEPTIONAL"]["R_gte"] == 40
    assert gpu["tiers"]["R"]["GPU_FAST"]["R_gte"] == 500
    assert gpu["tiers"]["R"]["GPU_EXCEPTIONAL"]["R_gte"] == 5000
    assert cpu["tiers"]["CPU_FAST"]["S_cpu_gte"] == 5
    assert cpu["tiers"]["CPU_EXCEPTIONAL"]["S_cpu_gte"] == 20
    assert gpu["qualifiers"]["BATCH_SCALES"]["factor"] == 20
    assert gpu["qualifiers"]["HOST_BOUND"]["threshold"] == 1.5
    assert spec["device_scaling"]["factor_h200_to_rtx3090"] == 5.13
    assert spec["representation_tripwire"]["bounds"] == {"parity": 0.10, "visual": 0.25}
    assert spec["downstream"]["equivalence_margin_points"] == 5.0
    assert spec["downstream"]["min_paired_episodes"] == 300
    assert gpu["no_reference_fallback"]["GPU_FAST"]["cores_gte"] == 10
    assert cpu["no_reference_fallback"]["CPU_FAST"]["cores_cpu_gte"] == 3


def test_both_devices_have_a_no_reference_escape(spec):
    """Making CPU_FAST mandatory for completion creates a trap this closes.

    Craftax, MiniHack and Merge have no installable official renderer in any interpreter, so neither
    ``R`` nor ``S_cpu`` can be formed. Before ENVIRONMENT_DONE required CPU, that only cost them a GPU
    tier; now, without a CPU escape, "the upstream package cannot be installed" would silently mean
    "this env can never be finished" -- a conclusion about packaging masquerading as one about the
    renderer.
    """
    for dev, key, floors in (("gpu", "cores_gte", (1, 10, 100)),
                             ("cpu", "cores_cpu_gte", (1, 3, 10))):
        fb = spec[dev]["no_reference_fallback"]
        assert fb["may_contribute_to_parity_claim"] is False
        got = tuple(fb[t][key] for t in
                    (f"{dev.upper()}_FASTER", f"{dev.upper()}_FAST", f"{dev.upper()}_EXCEPTIONAL"))
        assert got == floors, f"{dev} NO_REFERENCE floors moved: {got} != {floors}"
    # the CPU ladder must be the gentler one: one core against one core, not a device against a core
    assert (spec["cpu"]["no_reference_fallback"]["CPU_FAST"]["cores_cpu_gte"]
            < spec["gpu"]["no_reference_fallback"]["GPU_FAST"]["cores_gte"])


@pytest.mark.parametrize("cls", ["S", "R"])
def test_gpu_tiers_are_monotonic(spec, cls):
    """A ladder whose rungs overlap or invert cannot produce a deterministic label."""
    t = spec["gpu"]["tiers"][cls]
    floors = [t["GPU_PARITY"]["R_gte"], t["GPU_FASTER"]["R_gte"],
              t["GPU_FAST"]["R_gte"], t["GPU_EXCEPTIONAL"]["R_gte"]]
    assert floors == sorted(floors) and len(set(floors)) == len(floors), floors
    assert t["GPU_PARITY"]["R_lt"] == t["GPU_FASTER"]["R_gte"], "a gap between PARITY and FASTER"
    assert t["GPU_FASTER"]["R_lt"] == t["GPU_FAST"]["R_gte"], "a gap between FASTER and FAST"
    assert t["GPU_REGRESSION"]["R_lt"] == t["GPU_PARITY"]["R_gte"] == 1


def test_cpu_tiers_are_monotonic(spec):
    t = spec["cpu"]["tiers"]
    floors = [t["CPU_PARITY"]["S_cpu_gte"], t["CPU_FASTER"]["S_cpu_gte"],
              t["CPU_FAST"]["S_cpu_gte"], t["CPU_EXCEPTIONAL"]["S_cpu_gte"]]
    assert floors == sorted(floors) and len(set(floors)) == len(floors), floors
    assert t["CPU_PARITY"]["S_cpu_lt"] == t["CPU_FASTER"]["S_cpu_gte"]
    assert t["CPU_FASTER"]["S_cpu_lt"] == t["CPU_FAST"]["S_cpu_gte"]


def test_backend_classes_are_disjoint(spec):
    """Class assignment has to be a function of the declared backends, or the label is ambiguous."""
    s = set(spec["backend_classes"]["S"]["backends"])
    r = set(spec["backend_classes"]["R"]["backends"])
    assert s and r and not (s & r), f"backend appears in both classes: {sorted(s & r)}"


def test_the_headline_batch_is_in_the_required_sweep(spec):
    """B=4096 is where batched economics and loader overhead both become visible."""
    assert spec["gpu"]["headline_batch"] == 4096
    assert 4096 in spec["gpu"]["required_batches"]
    assert spec["gpu"]["required_batches"] == sorted(spec["gpu"]["required_batches"])
    assert 1 in spec["gpu"]["required_batches"], "B=1 is where launch overhead shows"
    assert spec["gpu"]["never_certify_on_batch"] == 1


def test_cpu_support_can_never_imply_a_cpu_tier(spec):
    """The one rule Step 1 spent the most effort establishing, restated so it cannot be edited away."""
    assert spec["cpu"]["no_stable_envs_cpu_measurement_exists"] is True
    assert "CPU_UNMEASURED" in spec["cpu"]["critical_rule"]
    assert "CPU_UNMEASURED" in spec["ladder"]["badges"]


def test_a_missing_variation_map_blocks_nothing(spec):
    assert spec["variations"]["blocks_other_gates"] is False
    assert "VARIATIONS_UNMAPPED" in spec["ladder"]["badges"]


def test_environment_done_requires_both_devices_at_their_fast_bar(spec, doc):
    """CPU is a first-class target, so there is no way to finish an env by not advertising CPU.

    The escape hatch this replaces read "an env can be certified while carrying CPU_REGRESSION,
    provided CPU is not among its advertised targets" — which is exactly how a first-class target
    becomes a fallback: nothing in the state model ever forces the CPU number to exist, so it never
    does. Per-backend certification is still expressible, and still useful; it just is not DONE.
    """
    lad = spec["ladder"]
    req = lad["environment_done_requires"]
    assert "BACKEND_CERTIFIED(cpu)" in req and "BACKEND_CERTIFIED(gpu)" in req
    assert lad["environment_done_min_tiers"] == {"cpu": "CPU_FAST", "gpu": "GPU_FAST"}
    assert lad["advertised_target_exemption"] is False
    assert lad["forks_by_device_at"] == "PERFORMANCE_MEASURED"
    assert "PERFORMANCE_MEASURED" in lad["shared_rungs"]
    assert "ENVIRONMENT_DONE" not in lad["shared_rungs"], "ENVIRONMENT_DONE is not a shared rung"

    # a failing tier still satisfies PERFORMANCE_MEASURED -- measuring a regression is progress
    assert lad["performance_measured_accepts_a_failing_tier"] is True

    # the documented legal state must be per-backend, and must NOT be environment-done
    assert lad["legal_example_is_environment_done"] is False
    assert "CPU_REGRESSION" in lad["legal_example"] and "BACKEND_CERTIFIED(gpu)" in lad["legal_example"]
    assert "ENVIRONMENT_DONE" not in lad["legal_example"]

    for tok in ("`ENVIRONMENT_DONE`", "`BACKEND_CERTIFIED(cpu)`", "`BACKEND_CERTIFIED(gpu)`",
                "No advertised-target exemption"):
        assert tok in doc or tok.strip("`") in doc, f"{tok!r} is not stated in the prose"


def test_partial_coverage_can_never_reach_environment_done(spec, doc):
    """The invariant, asserted from both directions.

    A partial renderer is genuinely useful -- benchmarkable, conformance-tested, usable in the SWM
    workflow, and entitled to any performance tier. The one thing it must not do is let a throughput
    number retire a declared exclusion. swm/CraftaxClassicPixels-v1 draws the world layer of the
    63x63 observation and leaves out the inventory strip, night frames and the sleeping-player
    desaturation; all three are part of what the registered default actually shows an agent.
    """
    lad = spec["ladder"]
    assert lad["invariant"] == "Gate A = partial can never produce ENVIRONMENT_DONE."
    assert lad["partial_coverage_exemption"] is False
    assert "registered_default_coverage == full" in lad["environment_done_requires"]

    # BACKEND_CERTIFIED requires full coverage on BOTH devices; the PARTIAL variant requires partial
    for dev in ("cpu", "gpu"):
        assert "registered_default_coverage == full" in lad["backend_certified_requires"][dev]
        part = lad["backend_certified_partial_requires"][dev]
        assert "registered_default_coverage == partial" in part
        assert "registered_default_coverage == full" not in part
        # the only difference between the two states is coverage -- everything else must match
        assert ([x for x in lad["backend_certified_requires"][dev] if "coverage" not in x]
                == [x for x in part if "coverage" not in x]), (
            f"{dev}: the PARTIAL state has drifted from BACKEND_CERTIFIED on something other than coverage")

    # it is a PASSING state, and it is terminal
    p = lad["backend_certified_partial_requires"]
    assert p["is_a_passing_state"] is True and p["is_terminal"] is True

    for tok in ("Gate A = `partial` can never produce `ENVIRONMENT_DONE`",
                "`BACKEND_CERTIFIED_PARTIAL(device)`", "No partial-coverage exemption"):
        assert tok in doc, f"{tok!r} is not stated in the prose"


def test_the_coverage_target_adds_up(spec, doc):
    """33 + 1 + 6 = 40, and the 2 symbolic ids are outside the denominator, not inside it at zero.

    Recomputed rather than trusted: a denominator that is asserted but never summed is exactly the
    kind of number that survives a change to one of its parts.
    """
    t = spec["coverage_target"]
    c = t["components"]
    assert sum(c.values()) == t["denominator"] == 40, (c, t["denominator"])
    assert t["excluded_from_denominator"]["symbolic_ids"] == 2
    assert t["excluded_from_denominator"]["terminal_state"] == "NOT_APPLICABLE_SYMBOLIC"
    assert t["every_row_in_the_denominator_must_be"] == "full"
    assert len(t["partial_row_exclusions"]) == 3
    assert "40  of 40 pixel-producing swm ids" in doc or "40 / 40" in doc or "40/40" in doc

    # and the components must still match the live Step 1 matrix
    matrix = ROOT / "scratch" / "swm_matrix.json"
    if not matrix.is_file():
        pytest.skip("scratch/swm_matrix.json is not present in this checkout")
    m = json.loads(matrix.read_text())["summary"]
    assert m["renderer_gaps"] == c["renderer_gaps_to_implement"], "gap count drifted from the matrix"
    assert m["symbolic_no_frame"] == t["excluded_from_denominator"]["symbolic_ids"]
    cov = m["by_registered_default_render_covered"]
    assert cov["True"] == c["fully_covered_today"], "fully-covered count drifted from the matrix"
    assert m["by_headline_status"]["partial_observation_coverage"] == c["partial_to_upgrade"]


def test_cpu_support_is_only_claimed_for_rows_that_have_a_renderer(spec, doc):
    """Support, absence and not-applicable stay three different things.

    An earlier draft said all 42 Step 1 rows carry a CPU-support value, which quietly folded the 6
    renderer gaps and 2 symbolic ids into a population where "we have not measured this yet" is the
    right reading. There is no renderer there to run on a CPU at all.
    """
    assert "34 matched renderer rows" in doc
    matrix = ROOT / "scratch" / "swm_matrix.json"
    if not matrix.is_file():
        pytest.skip("scratch/swm_matrix.json is not present in this checkout")
    rows = json.loads(matrix.read_text())["rows"]
    matched = [r for r in rows if r["stable_envs"]["envs"]]
    assert len(matched) == 34
    from collections import Counter
    got = Counter(r["axes"]["cpu_support"]["render_executes"] for r in matched)
    assert got == {"yes": 18, "unknown": 12, "unknown_dependency_absent_here": 4}, got
    for n, v in got.items():
        assert f"`{n}` on {v}" in doc or f"on {v}" in doc, f"{n}={v} not reflected in the prose"
    assert all(r["axes"]["cpu_support"]["throughput"] == "needs_measurement" for r in matched)

    # the 8 unmatched rows carry no stable-envs env at all, so there is nothing to time -- n/a, which
    # is a third thing from "supported" and from "we have not got to it yet"
    unmatched = [r for r in rows if not r["stable_envs"]["envs"]]
    assert len(unmatched) == 8
    assert all(not r["speed"]["per_env"] for r in unmatched), (
        "an unmatched row grew a speed entry; the n/a wording in the doc would become wrong")

    # and no passage may re-widen the CPU-throughput population back to all 42 rows
    import re
    for m in re.finditer(r"(?:all )?42 (?:Step 1 )?rows", doc):
        raise AssertionError(
            f"docs/research/certification.md claims a CPU/throughput fact over 42 rows near "
            f"{doc[max(0, m.start() - 120):m.end() + 40]!r}; only 34 carry a renderer")


def test_the_spec_assigns_no_work_to_an_unbriefed_step(spec, doc):
    """Only Step 3 is named by the Step 2 brief; the rest is evidence, not a numbered plan.

    An earlier draft put "VariationMap work" against a "Step 4" that nobody had briefed, which turns
    a certification spec into a project plan it has no standing to write.
    """
    for n in (4, 5, 6):
        assert f"Step {n}" not in doc, (
            f"docs/research/certification.md assigns work to 'Step {n}', which this specification has "
            f"no standing to define; list it as unscheduled required evidence instead")


def test_reference_only_physics_is_certifiable(spec):
    assert spec["physics"]["classes"]["reference_only"]["certifiable"] is True
    assert spec["physics"]["renderer_certification_never_waits_on_a_fast_step"] is True
    assert set(spec["physics"]["classes"]) == {
        "n/a", "reference_only", "exact_fast_step", "near_exact_fast_step", "approximate_fast_step"}


def test_the_downstream_rule_passes_the_evidence_that_already_exists(spec):
    """A rule written after the fact that cannot certify the one result the project has is not a rule.

    Recomputed here rather than trusted from the JSON, so the recorded verdict cannot drift from the
    arithmetic it claims.
    """
    import math
    d = spec["downstream"]
    v = d["validated_against_existing_evidence"]
    b, c, n = v["b"], v["c"], v["n"]
    delta = (c - b) / n * 100
    half = d["ci_z"] * math.sqrt(b + c) / n * 100
    lo, hi = delta - half, delta + half
    assert n >= d["min_paired_episodes"]
    assert abs(lo - v["ci"][0]) < 0.01 and abs(hi - v["ci"][1]) < 0.01, (lo, hi, v["ci"])
    m = d["equivalence_margin_points"]
    assert -m < lo and hi < m, f"the frozen rule would not certify the PushT evidence: [{lo}, {hi}]"
    assert v["control"]["delta_points"] <= d["control_max_delta_points"]
    assert v["control"]["mcnemar_p"] < d["control_max_p"]


def test_the_power_guard_is_consistent_with_the_margin(spec):
    """The largest discordance at which the +/-5 CI can still close at n=300."""
    import math
    d = spec["downstream"]
    n, m, z = d["min_paired_episodes"], d["equivalence_margin_points"], d["ci_z"]
    worst = d["power_guard"]["max_discordance_at_n300"]
    half = z * math.sqrt(worst * n) / n * 100
    assert half < m, f"at discordance {worst} the half-width is {half:.2f} pts, not inside +/-{m}"
    # and one notch past it must NOT fit, or the guard is not where it says it is
    half_next = z * math.sqrt((worst + 0.02) * n) / n * 100
    assert half_next > m, f"the guard is loose: discordance {worst + 0.02} still fits in +/-{m}"


def test_representative_environments_are_live_swm_ids(spec):
    """They must be ids the Step 1 matrix actually carries, not aspirational names."""
    matrix = ROOT / "scratch" / "swm_matrix.json"
    if not matrix.is_file():
        pytest.skip("scratch/swm_matrix.json is not present in this checkout")
    live = {r["swm_id"] for r in json.loads(matrix.read_text())["rows"]}
    named = list(spec["representative_environments"]["required"])
    named.append(spec["representative_environments"]["conditional"]["env"])
    missing = [i for i in named if i not in live]
    assert not missing, f"representative id(s) not in the live swm registry: {missing}"


def test_the_document_states_that_no_benchmark_was_run(spec, doc):
    """The property that makes the thresholds worth freezing, asserted rather than assumed."""
    assert spec["no_benchmark_was_run"] is True
    assert re.search(r"No benchmark was run", doc), (
        "docs/research/certification.md must say, in the document, that its thresholds were chosen "
        "without running a fresh benchmark")
