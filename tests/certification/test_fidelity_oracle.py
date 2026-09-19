"""The Step 4 merge, judging and universe machinery, on synthetic shards.

Nothing here renders anything. Every fixture is a hand-written shard, because what is being tested is
how a MEASUREMENT becomes a VERDICT, and that has to be checkable without a GPU, without an Oscar
allocation and without an hour of wall time. The campaign's real numbers are evidence; this file is
about the arithmetic that turns them into claims.

Two invariants run under all of it:

  * **the certifier contains no bound.** Every number it applies is extracted by
    ``scratch/step4/bounds.py`` from the test module that declares it, or read off the env's own
    declaration, or read out of the frozen Step 2 spec. ``test_no_bound_is_hardcoded`` enforces that,
    and the extraction tests below fail if a source file's shape changes so that an extraction goes
    silently blind.
  * **a dependency failure is never a fidelity failure.** It is the rule that lets the shards be
    merged at all, and the one a hurried refactor is most likely to lose.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEP4 = ROOT / "scratch" / "step4"


def _load(name: str):
    p = STEP4 / f"{name}.py"
    if not p.is_file():
        pytest.skip(f"{p} is not present in this checkout")
    spec = importlib.util.spec_from_file_location(f"step4_{name}", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def C():
    return _load("certify")


@pytest.fixture(scope="module")
def BOUNDS():
    return _load("bounds")


@pytest.fixture(scope="module")
def U():
    return _load("universe")


@pytest.fixture(scope="module")
def SPEC():
    return json.loads((ROOT / "scratch" / "certification.json").read_text())


# ------------------------------------------------------------------ fixtures

CLEAN = {
    "check_state_roundtrip": {"max_abs_err": 0.0},
    "check_state_dims_matter": {"dead_dims": []},
    "check_batch_invariance": {"max_abs_diff": 0},
    "check_batch_size_invariance": {"max_abs_diff": 0},
    "check_order_independence": {"permutation_max_abs_diff": 0, "duplicate_max_abs_diff": 0},
    "check_determinism": {"identical": True},
    "check_render_is_pure": {
        "state_changed_by_render": 0.0,
        "style_changed_by_render": 0.0,
        "state_changed_by_double_set": 0.0,
        "frame_changed_by_double_set": 0,
    },
    "check_chunked_render_matches": {"max_abs_diff": 0},
    "check_input_form_invariance": {"disagreements": {}},
    "check_set_state_copies": {"aliased_caller_buffer": 0.0, "state_tensor_rebound": False},
    "check_style_is_state": {"max_abs_diff": 0},
    "check_default_style_is_official": {"max_abs_diff": 0},
    "check_per_env_variation": {"collapsed_variations": []},
    "check_fidelity_declaration": {
        "tier": "parity",
        "describe_reports_tier": "parity",
        "agrees": True,
        "parity_interior_min": 1.0,
        "parity_all_exact_min": None,
        "visual_fidelity": None,
        "render_decoder": "renderer",
    },
}
GOOD_PARITY = {
    "interior_exact_frac": 1.0,
    "exact_frac": 0.95,
    "mean_abs": 1.2,
    "ssim": 0.99,
    "max_abs": 40,
    "digest_ours": "aa",
    "digest_official": "bb",
}


def config(cid="Widget", tier="parity", coverage="full", mandatory=None, family=None, status="measured", **kw):
    m = {k: {"status": "measured", "evidence": v} for k, v in (mandatory or CLEAN).items()}
    f = {"render_parity_with_official_renderer": {"status": "measured", "evidence": GOOD_PARITY}}
    f.update(family or {})
    row = {
        "config_id": cid,
        "env": cid,
        "swm_ids": [f"swm/{cid}-v0"],
        "role": "registered_default",
        "siblings": [],
        "registered_default_coverage": coverage,
        "declared_tier": tier,
        "backends": ["scene"],
        "status": status,
        "mandatory": m,
        "family_specific": f,
        "physics": {"status": "not_applicable", "why": "reference_only"},
        "oracle": {"digest_ours": "aa", "frames_shape": [64, 96, 96, 3]},
        "corpus": {"sha256": "cc", "kept": 64, "pool": 256, "state_dim": 5},
    }
    row.update(kw)
    return row


def shard(tmp_path, tag, configs, **prov):
    d = tmp_path / "conformance"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}.json").write_text(
        json.dumps(
            {
                "shard": tag,
                "provenance": {"host": "nodeA", "device": "cuda", "torch": "2.14", "python": "3.12", **prov},
                "configs": {c["config_id"]: c for c in configs},
            }
        )
    )
    return tmp_path


def verdicts(C, tmp_path):
    merged = C.merge(tmp_path)
    return merged, C.certify(merged)["configs"]


# ------------------------------------------------------------------ merge


def test_merge_is_deterministic_and_order_independent(C, tmp_path):
    shard(tmp_path, "b", [config("Beta")])
    shard(tmp_path, "a", [config("Alpha")])
    first = json.dumps(C.merge(tmp_path), sort_keys=True)
    assert first == json.dumps(C.merge(tmp_path), sort_keys=True)
    assert list(C.merge(tmp_path)["configs"]) == ["Alpha", "Beta"], "configs must be sorted by id"


def test_a_config_seen_twice_keeps_the_richer_run_and_records_both(C, tmp_path):
    thin = config("Widget", mandatory={"check_determinism": {"identical": True}})
    thin["family_specific"] = {}
    shard(tmp_path, "thin", [thin])
    shard(tmp_path, "full", [config("Widget")])
    merged = C.merge(tmp_path)
    assert merged["configs"]["Widget"]["shard"] == "full"
    assert merged["configs"]["Widget"]["measured_in_shards"] == ["full", "thin"], (
        "a config measured in two interpreters must say so, not silently keep one"
    )


def test_a_blocked_run_never_supersedes_a_measured_one(C, tmp_path):
    shard(tmp_path, "ok", [config("Widget")])
    shard(tmp_path, "zz", [config("Widget", status="blocked_environment", why="no dm_control here")])
    merged = C.merge(tmp_path)
    assert merged["configs"]["Widget"]["shard"] == "ok"


# ------------------------------------------------------------------ gate A


def test_a_clean_configuration_passes(C, tmp_path):
    shard(tmp_path, "s", [config()])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "PASS", v["Widget"]["failures"]


def test_one_bad_mandatory_check_fails_the_configuration(C, tmp_path):
    bad = dict(CLEAN, check_batch_invariance={"max_abs_diff": 3})
    shard(tmp_path, "s", [config(mandatory=bad)])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "FAIL"
    assert v["Widget"]["failures"] == ["check_batch_invariance"]


def test_a_missing_dependency_is_blocked_never_a_failure(C, tmp_path):
    fam = {
        "render_parity_with_official_renderer": {
            "status": "blocked_environment",
            "why": "ImportError: no module named gym_pusht",
        }
    }
    shard(tmp_path, "s", [config(family=fam)])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "BLOCKED"
    assert v["Widget"]["failures"] == []
    assert v["Widget"]["blocked"] == ["render_parity_with_official_renderer"]


def test_a_real_failure_outranks_a_blocked_check(C, tmp_path):
    """Otherwise an interpreter losing a package would HIDE a renderer that is measurably wrong."""
    bad = dict(CLEAN, check_determinism={"identical": False})
    fam = {"render_parity_with_official_renderer": {"status": "blocked_environment", "why": "ImportError"}}
    shard(tmp_path, "s", [config(mandatory=bad, family=fam)])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "FAIL"


def test_a_config_that_could_not_be_built_is_blocked(C, tmp_path):
    shard(tmp_path, "s", [config(status="blocked_environment", why="not registered here")])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "BLOCKED"
    assert "not registered" in v["Widget"]["why"]


def test_a_missing_mandatory_check_is_a_failure_not_a_pass(C, tmp_path):
    """A check that was never run is not a check that passed. Section 2: never silently omit one."""
    shard(tmp_path, "s", [config(mandatory={"check_determinism": {"identical": True}})])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "FAIL"
    assert "check_state_roundtrip" in v["Widget"]["missing_mandatory"]


def test_partial_coverage_passes_gate_a_and_is_reported_as_partial(C, tmp_path, SPEC):
    shard(tmp_path, "s", [config(coverage="partial")])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "PARTIAL", "partial is a PASSING gate A state -- spec section 4.2"
    assert v["Widget"]["failures"] == []
    assert SPEC["ladder"]["partial_coverage_exemption"] is False


def test_a_check_with_no_declared_bound_neither_passes_nor_fails(C, tmp_path):
    fam = {
        "check_resolution_consistency": {
            "status": "measured",
            "evidence": {"exact_frac": 0.3, "mean_abs": 9.0, "max_abs": 200},
        }
    }
    shard(tmp_path, "s", [config(cid="NoFloorEnv", family=fam)])
    _, v = verdicts(C, tmp_path)
    rec = v["NoFloorEnv"]["checks"]["check_resolution_consistency"]
    assert rec["status"] == "no_declared_bound"
    assert v["NoFloorEnv"]["gate_a"] == "PASS", "an unbounded diagnostic must not fail a renderer"


def test_a_recorded_downsample_floor_is_applied_when_there_is_one(C, BOUNDS, tmp_path):
    floors = BOUNDS.load()["downsample_exact_frac"]["value"]
    margin = BOUNDS.load()["downsample_margin"]["value"]
    name, floor = sorted(floors.items())[0]
    ok = {"check_resolution_consistency": {"status": "measured", "evidence": {"exact_frac": floor, "mean_abs": 1.0}}}
    bad = {
        "check_resolution_consistency": {
            "status": "measured",
            "evidence": {"exact_frac": floor - 10 * margin, "mean_abs": 1.0},
        }
    }
    shard(tmp_path, "s", [config(cid=name, family={**ok}), config(cid=name + "X", family={**bad})])
    _, v = verdicts(C, tmp_path)
    assert v[name]["checks"]["check_resolution_consistency"]["status"] == "pass"
    assert v[name + "X"]["checks"]["check_resolution_consistency"]["status"] == "no_declared_bound", (
        "the floor is keyed on the env name; an env with no recorded floor must not inherit one"
    )


# ------------------------------------------------------------------ the two tiers stay apart


def test_the_downsample_verdict_records_which_states_it_was_measured_on(C, BOUNDS, tmp_path):
    """A shard measured on the bound's own protocol says so and carries the corpus number beside it;
    an older shard measured on the corpus is marked as such rather than read as the protocol."""
    name, floor = next(iter(BOUNDS.load()["downsample_exact_frac"]["value"].items()))
    proto = {
        "n": 8,
        "state_seed": 20260809,
        "draw": "render_worker.sample_inputs(env, n)",
        "style": "default appearance",
        "states_sha256": "ab" * 32,
    }
    fam = {
        "check_resolution_consistency": {
            "status": "measured",
            "evidence": {
                "exact_frac": floor,
                "mean_abs": 1.0,
                "max_abs": 9,
                "protocol": proto,
                "frozen_corpus": {"exact_frac": floor - 0.02, "mean_abs": 1.1, "max_abs": 9, "n_states": 64},
            },
        }
    }
    shard(tmp_path, "s", [config(cid=name, family=fam)])
    _, v = verdicts(C, tmp_path)
    c = v[name]["checks"]["check_resolution_consistency"]
    assert c["status"] == "pass" and c["protocol"] == proto
    assert c["frozen_corpus"]["exact_frac"] == floor - 0.02 and c["frozen_corpus"]["n_states"] == 64
    fam = {
        "check_resolution_consistency": {
            "status": "measured",
            "evidence": {"exact_frac": floor, "mean_abs": 1.0, "max_abs": 9},
        }
    }
    shard(tmp_path, "s", [config(cid=name, family=fam)])
    _, v = verdicts(C, tmp_path)
    assert v[name]["checks"]["check_resolution_consistency"]["protocol"].startswith("not recorded")


def test_a_visual_env_is_never_held_to_a_parity_threshold(C, tmp_path):
    decl = dict(
        CLEAN["check_fidelity_declaration"],
        tier="visual",
        describe_reports_tier="visual",
        visual_fidelity={"mean_abs_max": 30.0, "within8_min": 0.2, "ssim_min": 0.5},
    )
    fam = {
        "render_parity_with_official_renderer": {"status": "not_applicable", "why": "declares the VISUAL tier"},
        "check_visual_fidelity": {
            "status": "measured",
            "evidence": {"mean_abs": 25.0, "within8_frac": 0.3, "ssim": 0.6, "interior_exact_frac": 0.0},
        },
    }
    shard(
        tmp_path,
        "s",
        [config(cid="Vis", tier="visual", mandatory=dict(CLEAN, check_fidelity_declaration=decl), family=fam)],
    )
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["gate_a"] == "PASS", (
        "a visual env at interior_exact_frac 0.0 must still pass -- it does not claim pixel identity"
    )
    assert v["Vis"]["checks"]["check_visual_fidelity"]["status"] == "pass"


def test_a_visual_env_that_misses_its_own_declaration_fails(C, tmp_path):
    decl = dict(
        CLEAN["check_fidelity_declaration"],
        tier="visual",
        describe_reports_tier="visual",
        visual_fidelity={"mean_abs_max": 10.0, "within8_min": 0.9, "ssim_min": 0.9},
    )
    fam = {
        "render_parity_with_official_renderer": {"status": "not_applicable", "why": "visual"},
        "check_visual_fidelity": {
            "status": "measured",
            "evidence": {"mean_abs": 25.0, "within8_frac": 0.3, "ssim": 0.6},
        },
    }
    shard(
        tmp_path,
        "s",
        [config(cid="Vis", tier="visual", mandatory=dict(CLEAN, check_fidelity_declaration=decl), family=fam)],
    )
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["gate_a"] == "FAIL"
    assert len(v["Vis"]["checks"]["check_visual_fidelity"]["violations"]) == 3


def _visual_row(reference_versions=None, observed=None):
    """A visual env whose measurement MISSES its declaration -- what Step 4 recorded for the DM Control
    trackcom cells -- with the declared reference versions in the declaration evidence."""
    vf = {"mean_abs_max": 6.5, "within8_min": 0.84, "ssim_min": 0.62}
    if reference_versions:
        vf["reference_versions"] = dict(reference_versions)
    decl = dict(
        CLEAN["check_fidelity_declaration"],
        tier="visual",
        describe_reports_tier="visual",
        parity_interior_min=None,
        visual_fidelity=vf,
        reference_versions=dict(reference_versions or {}) or None,
    )
    fam = {
        "render_parity_with_official_renderer": {"status": "not_applicable", "why": "visual"},
        "check_visual_fidelity": {
            "status": "measured",
            "evidence": observed or {"mean_abs": 6.54, "within8_frac": 0.796, "ssim": 0.58},
        },
    }
    return config("Vis", tier="visual", mandatory=dict(CLEAN, check_fidelity_declaration=decl), family=fam)


def test_a_visual_bound_judged_under_the_wrong_reference_version_is_blocked_not_failed(C, tmp_path):
    """The measured case (examples/results/step6/texture-fidelity/findings.md): declared under mujoco
    3.11.0, judged under 3.12.0, whose finite-plane texture re-anchoring had moved the reference's
    floor. A dependency at the wrong version is a comparison that could not be made -- the same rule
    as a missing dependency -- so the check is blocked, the mismatch is written down, and the
    measurement is kept beside it."""
    shard(
        tmp_path,
        "dmc",
        [_visual_row({"mujoco": "3.11.0", "mujoco_warp": "3.11.0"})],
        reference_packages={"mujoco": "3.12.0", "mujoco_warp": "3.12.0", "dm_control": "1.0.45"},
    )
    _, v = verdicts(C, tmp_path)
    c = v["Vis"]["checks"]["check_visual_fidelity"]
    assert c["status"] == "blocked_environment"
    assert c["reference_version_mismatch"] == [
        "mujoco: declared 3.11.0, interpreter had 3.12.0",
        "mujoco_warp: declared 3.11.0, interpreter had 3.12.0",
    ]
    assert c["judged_against_wrong_reference"] == "fail"  # what it WOULD have been
    assert c["observed"]["mean_abs"] == 6.54  # the measurement is not discarded
    assert v["Vis"]["gate_a"] == "BLOCKED"
    assert v["Vis"]["failures"] == [] and v["Vis"]["blocked"] == ["check_visual_fidelity"]


def test_a_visual_bound_judged_under_its_declared_reference_is_judged(C, tmp_path):
    shard(
        tmp_path,
        "dmc",
        [_visual_row({"mujoco": "3.11.0", "mujoco_warp": "3.11.0"})],
        reference_packages={"mujoco": "3.11.0", "mujoco_warp": "3.11.0"},
    )
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["checks"]["check_visual_fidelity"]["status"] == "fail"
    assert v["Vis"]["gate_a"] == "FAIL"
    shard(
        tmp_path,
        "dmc",
        [_visual_row({"mujoco": "3.11.0"}, observed={"mean_abs": 4.99, "within8_frac": 0.882, "ssim": 0.661})],
        reference_packages={"mujoco": "3.11.0"},
    )
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["checks"]["check_visual_fidelity"]["status"] == "pass"
    assert v["Vis"]["gate_a"] == "PASS"


def test_the_certifier_reads_reference_versions_with_the_same_precedence_as_describe(C, tmp_path):
    """visual_fidelity's entry overrides the class attribute (``reference_versions_of``),
    so a class-level 3.10.0 under a visual_fidelity 3.11.0 is judged against 3.11.0."""
    row = _visual_row({"mujoco": "3.11.0"}, observed={"mean_abs": 4.99, "within8_frac": 0.882, "ssim": 0.661})
    row["mandatory"]["check_fidelity_declaration"]["evidence"]["reference_versions"] = {
        "mujoco": "3.10.0",
        "gymnasium": "1.3.0",
    }
    shard(tmp_path, "dmc", [row], reference_packages={"mujoco": "3.11.0", "gymnasium": "1.3.0"})
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["checks"]["check_visual_fidelity"]["status"] == "pass"
    shard(tmp_path, "dmc", [row], reference_packages={"mujoco": "3.10.0", "gymnasium": "1.3.0"})
    _, v = verdicts(C, tmp_path)
    c = v["Vis"]["checks"]["check_visual_fidelity"]
    assert c["status"] == "blocked_environment"
    assert c["reference_version_mismatch"] == ["mujoco: declared 3.11.0, interpreter had 3.10.0"]


def test_a_declaration_without_reference_versions_is_judged_as_before(C, tmp_path):
    """The pre-field state of the world: every existing declaration keeps its verdict."""
    shard(tmp_path, "dmc", [_visual_row(None)], reference_packages={"mujoco": "3.12.0"})
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["checks"]["check_visual_fidelity"]["status"] == "fail"
    assert v["Vis"]["gate_a"] == "FAIL"


def test_a_reference_version_the_interpreter_did_not_record_cannot_confirm_a_match(C, tmp_path):
    shard(tmp_path, "dmc", [_visual_row({"mujoco": "3.11.0"})])  # no reference_packages
    _, v = verdicts(C, tmp_path)
    c = v["Vis"]["checks"]["check_visual_fidelity"]
    assert c["status"] == "blocked_environment" and "not recorded" in c["reference_version_mismatch"][0]


def test_a_wrong_reference_version_blocks_only_the_checks_that_read_the_reference(C, tmp_path):
    """A real self-consistency failure under the wrong reference is still a failure: the wrong version
    says nothing about determinism, and hiding it would be the other misreading."""
    row = _visual_row({"mujoco": "3.11.0"})
    row["mandatory"]["check_determinism"] = {"status": "measured", "evidence": {"identical": False}}
    shard(tmp_path, "dmc", [row], reference_packages={"mujoco": "3.12.0"})
    _, v = verdicts(C, tmp_path)
    assert v["Vis"]["checks"]["check_visual_fidelity"]["status"] == "blocked_environment"
    assert v["Vis"]["checks"]["check_determinism"]["status"] == "fail"
    assert v["Vis"]["gate_a"] == "FAIL"


def test_a_parity_env_uses_its_own_declared_interior_floor(C, tmp_path):
    """A relaxed floor is the env's to declare; the certifier reads it rather than assuming 1.0."""
    decl = dict(CLEAN["check_fidelity_declaration"], parity_interior_min=0.99)
    ev = dict(GOOD_PARITY, interior_exact_frac=0.995)
    fam = {"render_parity_with_official_renderer": {"status": "measured", "evidence": ev}}
    shard(
        tmp_path,
        "s",
        [
            config(cid="Relaxed", mandatory=dict(CLEAN, check_fidelity_declaration=decl), family=fam),
            config(cid="Strict", family=fam),
        ],
    )
    _, v = verdicts(C, tmp_path)
    assert v["Relaxed"]["checks"]["render_parity_with_official_renderer"]["status"] == "pass"
    assert v["Strict"]["checks"]["render_parity_with_official_renderer"]["status"] == "fail"


# ------------------------------------------------------------------ physics stays on its own axis


def test_reference_only_physics_is_not_a_renderer_failure(C, tmp_path):
    shard(tmp_path, "s", [config()])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["gate_a"] == "PASS"
    assert v["Widget"]["physics"]["status"] == "not_applicable"


def test_a_physics_failure_does_not_fail_gate_a(C, BOUNDS, tmp_path):
    b = BOUNDS.load()
    row = config()
    row["physics"] = {
        "status": "measured",
        "evidence": {
            "agent_max_err": b["step_agent_max_err"]["value"] * 10,
            "block_median_err": b["step_block_median_err"]["value"] * 10,
        },
    }
    shard(tmp_path, "s", [row])
    _, v = verdicts(C, tmp_path)
    assert v["Widget"]["physics"]["status"] == "fail"
    assert v["Widget"]["gate_a"] == "PASS", "physics is a separate axis -- brief section 3"


# ------------------------------------------------------------------ bounds come from elsewhere


def test_every_bound_is_extracted_and_none_is_none(BOUNDS):
    b = BOUNDS.load()
    missing = [k for k, v in b.items() if v.get("value") in (None, {}, "")]
    assert not missing, (
        f"these bounds could not be extracted from the files that declare them: {missing}. An "
        f"extraction that goes blind must fail here rather than silently drop a check's bar."
    )


def test_extracted_bounds_match_the_files_that_declare_them(BOUNDS):
    """The whole point of extracting rather than copying: this is what catches a drift."""
    b = BOUNDS.load()
    conf = ast.parse((ROOT / "tests" / "gpu" / "test_conformance.py").read_text())
    repro = ast.parse((ROOT / "tests" / "gpu" / "test_reproducibility.py").read_text())
    want = {}
    for tree in (conf, repro):
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                t = node.targets[0] if isinstance(node, ast.Assign) else node.target
                if isinstance(t, ast.Name) and node.value is not None:
                    try:
                        want[t.id] = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        pass
    assert b["parity_interior_min_default"]["value"] == want["INTERIOR_EXACT_MIN"]
    assert b["parity_all_exact_min_default"]["value"] == want["ALL_EXACT_MIN"]
    assert b["parity_mean_abs_max"]["value"] == want["MEAN_ABS_MAX"]
    assert b["downsample_margin"]["value"] == want["DOWNSAMPLE_MARGIN"]
    assert b["downsample_exact_frac"]["value"] == want["DOWNSAMPLE_EXACT_FRAC"]
    src = (ROOT / "tests" / "gpu" / "test_conformance.py").read_text()
    assert f">= {b['backend_agreement_exact_frac_min']['value']}" in src


def test_every_bound_cites_where_it_came_from(BOUNDS):
    for k, v in BOUNDS.load().items():
        assert v.get("from"), f"{k} has no citation; a bound nobody can trace is a bound nobody can argue with"


def test_no_bound_is_hardcoded_in_the_certifier(C):
    """The judging functions may not contain a numeric literal. Every bar comes from somewhere else.

    Scoped to the functions that DECIDE a verdict, so a table width or a string slice in the report
    cannot trip it -- the same scoping ``tests/certification/test_semantic_equivalence.py`` settled on, and for the same
    reason: an over-broad AST scan gets suppressed rather than fixed.
    """
    src = Path(C.__file__).read_text()
    judged = {"_parity_judge", "_visual_judge", "_downsample_judge", "_step_judge", "_cmp_ge", "judges", "verdict"}
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.FunctionDef) and node.name in judged):
            continue
        for n in ast.walk(node):
            if (
                isinstance(n, ast.Constant)
                and isinstance(n.value, (int, float))
                and not isinstance(n.value, bool)
                and n.value not in (0, 1)
            ):
                bad.append(f"{node.name}: {n.value}")
    assert not bad, (
        "the Step 4 certifier must not contain a threshold -- Step 4 may not move a bar it is "
        f"measuring against: {bad}"
    )


# ------------------------------------------------------------------ the tripwire


def test_the_tripwire_uses_the_frozen_step2_bounds(C, SPEC, tmp_path):
    (tmp_path / "dino" / "run").mkdir(parents=True)
    lim = SPEC["representation_tripwire"]["bounds"]["parity"]
    (tmp_path / "dino" / "run" / "dino_agreement.json").write_text(
        json.dumps(
            {
                "model": "m",
                "n": 32,
                "state_seed": 1,
                "envs": {
                    "Widget": {
                        "env": "Widget",
                        "tier": "parity",
                        "dino_cos_ours_vs_official": lim * 0.5,
                        "dino_cos_blank_vs_official": 1.0,
                        "dino_cos_state_vs_state": 0.2,
                    },
                    "Broken": {
                        "env": "Broken",
                        "tier": "parity",
                        "dino_cos_ours_vs_official": lim * 2,
                        "dino_cos_blank_vs_official": 1.0,
                        "dino_cos_state_vs_state": 0.2,
                    },
                },
            }
        )
    )
    shard(tmp_path, "s", [config(), config(cid="Broken")])
    merged = C.merge(tmp_path)
    t = C.tripwire(tmp_path, merged)
    assert t["bounds"] == SPEC["representation_tripwire"]["bounds"]
    assert t["envs"]["Widget"]["pass"] is True
    assert t["envs"]["Broken"]["pass"] is False


def test_the_tripwire_needs_its_blank_control(C, tmp_path):
    """``A`` is a RATIO to the broken-renderer floor. Without the control there is no number."""
    (tmp_path / "dino" / "run").mkdir(parents=True)
    (tmp_path / "dino" / "run" / "dino_agreement.json").write_text(
        json.dumps(
            {
                "envs": {
                    "Widget": {
                        "env": "Widget",
                        "tier": "parity",
                        "dino_cos_ours_vs_official": 0.001,
                        "dino_cos_blank_vs_official": 0.0,
                    }
                }
            }
        )
    )
    shard(tmp_path, "s", [config()])
    t = C.tripwire(tmp_path, C.merge(tmp_path))
    assert t["envs"]["Widget"]["status"] == "no_control"
    assert t["envs"]["Widget"]["pass"] is None


def test_the_tripwire_is_not_a_downstream_claim(SPEC):
    assert "never a substitute for the downstream gate" in SPEC["representation_tripwire"]["purpose"]


# ------------------------------------------------------------------ the universe


def test_the_universe_covers_every_matched_swm_row(U):
    matrix = json.loads((ROOT / "scratch" / "swm_matrix.json").read_text())
    matched = {r["swm_id"] for r in matrix["rows"] if r["stable_envs"]["envs"]}
    covered = {i for c in U.configurations() for i in c["swm_ids"]}
    assert covered == matched, f"rows with a renderer that Step 4 does not validate: {matched - covered}"
    assert len(matched) == matrix["summary"]["with_stable_envs_match"] == 34


def test_the_universe_excludes_the_gaps_and_the_symbolic_ids(U):
    matrix = json.loads((ROOT / "scratch" / "swm_matrix.json").read_text())
    unmatched = {r["swm_id"] for r in matrix["rows"] if not r["stable_envs"]["envs"]}
    covered = {i for c in U.configurations() for i in c["swm_ids"]}
    assert not (covered & unmatched), "a row with no renderer cannot be validated"
    assert len(unmatched) == 8


def test_every_matched_env_is_validated_or_stands_behind_a_representative(U):
    matrix = json.loads((ROOT / "scratch" / "swm_matrix.json").read_text())
    matched = {e for r in matrix["rows"] for e in r["stable_envs"]["envs"]}
    cs = U.configurations()
    reached = {c["env"] for c in cs} | {s for c in cs for s in c["siblings"]}
    assert matched <= reached, f"matched envs that no configuration reaches: {matched - reached}"


def test_only_the_measured_ogbench_collapse_is_applied(U):
    """Every other family is one configuration per env. A collapse nobody measured is an assumption."""
    collapsed = [c for c in U.configurations() if c["siblings"]]
    assert collapsed, "the OGBench cells should collapse"
    for c in collapsed:
        assert (c["reference_id"] or "").startswith(U.CELL_COLLAPSE_PREFIX), (
            f"{c['config_id']} collapses siblings without being an OGBench cell: {c['siblings']}"
        )


def test_configurations_sharing_a_reference_id_are_not_assumed_equal(U):
    """``Piecewise``/``PiecewiseZones`` and ``TwoRoom``/``TwoRoomTarget`` share an official id.

    They are separately validated on purpose. Grouping by ``reference_id`` would have folded a env
    that draws target zones into one that does not and reported one result for both.
    """
    ids = {c["config_id"] for c in U.configurations()}
    for a, b in (("Piecewise", "PiecewiseZones"), ("TwoRoom", "TwoRoomTarget")):
        assert {a, b} <= ids, f"{a} and {b} must both be validated"


def test_the_universe_is_deterministic(U):
    a = json.dumps(U.configurations(), sort_keys=True)
    assert a == json.dumps(U.configurations(), sort_keys=True)
    assert [c["config_id"] for c in U.configurations()] == sorted(c["config_id"] for c in U.configurations())


def test_every_configuration_lands_in_exactly_one_shard(U):
    for c in U.configurations():
        assert U.shard_of(c) in {"dmc", "fetch", "ogb", "raster", "swm"}, c["config_id"]


def test_craftax_stays_partial(U):
    c = [x for x in U.configurations() if x["config_id"] == "CraftaxClassic"]
    assert c and c[0]["registered_default_coverage"] == "partial", (
        "Craftax Classic draws a documented subset of the registered default observation; no result "
        "in Step 4 may promote it to full"
    )


# ------------------------------------------------------------------ Step 2 is not touched


def test_step4_changes_no_step2_threshold(SPEC):
    frozen = json.loads((ROOT / "scratch" / "certification.json").read_text())
    assert frozen == SPEC
    assert frozen["no_benchmark_was_run"] is True
    assert frozen["representation_tripwire"]["bounds"] == {"parity": 0.10, "visual": 0.25}


def test_the_certifier_reads_the_frozen_spec_rather_than_restating_it(C):
    src = Path(C.__file__).read_text()
    assert 'SPEC = json.loads((ROOT / "scratch" / "certification.json").read_text())' in src
    assert "representation_tripwire" in src


# ------------------------------------------------------------------ the floors are per-env


def test_the_report_prints_the_floor_each_row_was_actually_held_to(C, tmp_path):
    """The report may not assert a blanket parity floor. It did, and it was wrong for 16 rows.

    The prose used to read "every env in this universe declares ``1.0``" while the Fetch family
    declares ``parity_interior_min = 0.99`` with a long ``parity_note`` naming two structural
    residuals, and passed at 0.9988. The certifier was right -- it reads the env's own declaration --
    and the sentence beside its output was not, which is the worse failure of the two: the numbers
    were correct and the page said they meant something else.
    """
    decl = dict(CLEAN["check_fidelity_declaration"], parity_interior_min=0.99)
    ev = dict(GOOD_PARITY, interior_exact_frac=0.9988)
    shard(
        tmp_path,
        "s",
        [
            config(
                cid="Relaxed",
                mandatory=dict(CLEAN, check_fidelity_declaration=decl),
                family={"render_parity_with_official_renderer": {"status": "measured", "evidence": ev}},
            )
        ],
    )
    merged = C.merge(tmp_path)
    cert = C.certify(merged)
    md = C.report(merged, cert, C.tripwire(tmp_path, merged))
    assert "0.99" in md, "the declared floor must appear on the page, not just in the JSON"
    assert "every env in this universe declares" not in md, (
        "a blanket claim about the parity floor is exactly the sentence that was wrong; the floor is "
        "per-env and the page has to show it per row"
    )


def test_a_relaxed_floor_without_a_stated_mechanism_is_reported(C, tmp_path):
    """A tolerance with no ``parity_note`` is indistinguishable from a bug being tolerated.

    ``tests/gpu/test_conformance.py::test_relaxed_parity_floors_are_all_explained`` already asserts
    this on the class over the whole registry, so Step 4 does not re-assert it -- it SHOWS it, beside
    the number the tolerance bought. This checks the showing does not silently stop happening.
    """
    decl = dict(CLEAN["check_fidelity_declaration"], parity_interior_min=0.99)
    shard(
        tmp_path,
        "s",
        [
            config(
                cid="Relaxed",
                mandatory=dict(CLEAN, check_fidelity_declaration=decl),
                family={
                    "render_parity_with_official_renderer": {
                        "status": "measured",
                        "evidence": dict(GOOD_PARITY, interior_exact_frac=0.9988),
                    }
                },
            )
        ],
    )
    merged = C.merge(tmp_path)
    cert = C.certify(merged)
    v = cert["configs"]["Relaxed"]
    assert v["checks"]["render_parity_with_official_renderer"]["status"] == "pass"
    assert v["checks"]["render_parity_with_official_renderer"]["bound"]["interior_exact_frac_min"] == 0.99, (
        "the judged bound must be the env's declaration, never the shared default"
    )
    md = C.report(merged, cert, C.tripwire(tmp_path, merged))
    assert "declared mechanism" in md, (
        "a relaxed floor has to be listed with the mechanism its class states, or the page reports a "
        "pass against a bar it never shows"
    )
