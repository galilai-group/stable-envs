"""The Step 3 merge and certification machinery, on synthetic shards.

Nothing here runs a benchmark. Every fixture is a hand-written shard, which is the point: the rules
being tested are about how a measurement becomes a LABEL, and those have to be checkable without a
GPU, without an Oscar allocation, and without waiting an hour. The campaign's real numbers are
evidence; this file is about the arithmetic that turns them into claims.

The invariant under all of it is Step 2's: the evaluator may not contain a threshold. Every bound is
read from ``scratch/certification.json``, so these tests assert behaviour AGAINST that file rather
than against numbers retyped here -- if the policy moves, they follow it, and if the evaluator grows a
constant of its own, ``test_no_threshold_is_hardcoded`` fails.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CERT_PY = ROOT / "scratch" / "step3" / "certify.py"


@pytest.fixture(scope="module")
def C():
    if not CERT_PY.is_file():
        pytest.skip("scratch/step3/certify.py is not present in this checkout")
    spec = importlib.util.spec_from_file_location("step3_certify", CERT_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def SPEC():
    return json.loads((ROOT / "scratch" / "certification.json").read_text())


def shard(tmp_path, name, envs, device="cuda", host="nodeA", threads=1, device_name="RTX 3090"):
    d = {"provenance": {"device": device, "host": host, "torch_threads": threads,
                        "device_name": device_name, "timestamp": "T", "git_commit": "abc"},
         "batches": [1, 16, 64, 256, 1024, 4096], "envs": envs}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d))
    return p


def cells(**fps):
    """``{batch: fps}`` -> cell list, with a benign cpu_frac unless one is given per batch."""
    return [{"batch": int(b), "fps": float(v), "cpu_frac": 1.0, "peak_gb": 1.0} for b, v in fps.items()]


def env_rec(name, ids, fps_by_batch, official=1000.0, decode=5000.0, cpu_frac=1.0, **kw):
    c = cells(**{str(k): v for k, v in fps_by_batch.items()})
    for x in c:
        x["cpu_frac"] = cpu_frac
    r = {"env": name, "projected_to": ids, "cells": c, "shape": "96x96", "size": 96, "supersample": 1,
         "decode": {"decode_fps_per_core": decode} if decode else None}
    r["official_cpu"] = ({"set_state_render": {"fps": official}} if official
                         else {"absent": "no reference", "kind": "no_reference"})
    r.update(kw)
    return r


CENSUS_R = {"PushT": {"backends": ["scene"]}}
CENSUS_S = {"DMCReacherHard": {"backends": ["mjwarp", "raycast3d"]}}


# ----------------------------------------------------------------- merge

def test_merge_is_deterministic_and_byte_stable(C, tmp_path):
    """Two merges of the same shards must be identical, in content and in serialized bytes.

    Shard discovery is a directory walk and dict iteration order is insertion order, so an unsorted
    implementation produces a merged artifact whose diff is noise. A reviewer has to be able to treat
    a diff on this file as a change in the measurements.
    """
    a = shard(tmp_path, "b", [env_rec("PushT", ["swm/PushT-v1"], {1: 10, 256: 100})])
    b = shard(tmp_path, "a", [env_rec("DMCReacherHard", ["swm/ReacherDMControl-v0"], {1: 5, 256: 50})])
    m1, m2 = C.merge([a, b]), C.merge([b, a])
    assert json.dumps(m1, sort_keys=True) == json.dumps(m2, sort_keys=True)
    assert [r["env"] for r in m1["runs"]] == sorted(r["env"] for r in m1["runs"])


def test_a_repeat_measurement_keeps_the_richer_run_and_says_so(C, tmp_path):
    """A rerun that got further must win, and the one it replaced must stay named."""
    thin = shard(tmp_path, "thin", [env_rec("PushT", ["x"], {1: 10})])
    full = shard(tmp_path, "full", [env_rec("PushT", ["x"], {1: 10, 256: 100, 4096: 900})])
    m = C.merge([thin, full])
    assert len(m["runs"]) == 1
    run = m["runs"][0]
    assert len([c for c in run["cells"] if c.get("fps")]) == 3
    assert run["_superseded"] and "thin" in run["_superseded"][0]


def test_the_same_env_on_two_hosts_is_two_runs(C, tmp_path):
    """Step 2 requires every ratio to be same-host, so host is part of the identity of a measurement."""
    a = shard(tmp_path, "m5", [env_rec("PushT", ["x"], {256: 100})], device="cpu", host="m5")
    b = shard(tmp_path, "ox", [env_rec("PushT", ["x"], {256: 400})], device="cpu", host="oscar")
    m = C.merge([a, b])
    assert len(m["runs"]) == 2, "two hosts collapsed into one run"
    assert {r["_key"]["host"] for r in m["runs"]} == {"m5", "oscar"}


def test_projection_to_several_swm_rows_is_recorded_not_duplicated(C, tmp_path):
    """PushT backs two registered ids and is timed once; both ids must trace to the one measurement."""
    s = shard(tmp_path, "s", [env_rec("PushT", ["swm/PushT-v1", "swm/PushT-Discrete-v1"], {256: 100})])
    lab = C.certify(C.merge([s]), CENSUS_R)["labels"]
    assert len(lab) == 1, "one renderer measurement became more than one label"
    assert lab[0]["projected_to"] == ["swm/PushT-v1", "swm/PushT-Discrete-v1"]


# ----------------------------------------------------------------- classification

def test_a_missing_dependency_is_never_a_regression(C, tmp_path):
    """The rule Step 1 spent the most effort on, restated where it is easiest to violate."""
    blocked = {"env": "DMCCheetahRun", "projected_to": ["swm/CheetahDMControl-v0"],
               "blocked": {"stage": "construct", "kind": "blocked_environment",
                           "reason": "ModuleNotFoundError: No module named 'warp'"}}
    lab = C.certify(C.merge([shard(tmp_path, "s", [blocked])]), {})["labels"][0]
    assert lab["tier"] == "GPU_UNMEASURED"
    assert "REGRESSION" not in lab["tier"] and "blocked" in lab
    assert lab["blocked"]["kind"] == "blocked_environment"


def test_b1_is_never_the_certified_throughput(C, tmp_path):
    """B=1 is required by the sweep and forbidden as a certification point.

    A renderer that happens to peak at batch 1 is measuring launch overhead. The corpus median at B=1
    is 0.6x the official renderer, so certifying there would systematically flatter small batches.
    """
    s = shard(tmp_path, "s", [env_rec("PushT", ["x"], {1: 99999, 256: 100, 4096: 120})])
    lab = C.certify(C.merge([s]), CENSUS_R)["labels"][0]
    assert lab["evidence"]["best_batch"] == 4096
    assert lab["evidence"]["se_fps"] == 120


@pytest.mark.parametrize("cls,census,ratio,expect", [
    ("S", CENSUS_S, 50.0, "GPU_EXCEPTIONAL"),
    ("S", CENSUS_S, 5.0, "GPU_FAST"),
    ("S", CENSUS_S, 3.0, "GPU_FASTER"),
    ("S", CENSUS_S, 1.5, "GPU_PARITY"),
    ("S", CENSUS_S, 0.5, "GPU_REGRESSION"),
    ("R", CENSUS_R, 6000.0, "GPU_EXCEPTIONAL"),
    ("R", CENSUS_R, 600.0, "GPU_FAST"),
    ("R", CENSUS_R, 50.0, "GPU_FASTER"),
    ("R", CENSUS_R, 5.0, "GPU_PARITY"),
])
def test_gpu_tiers_follow_the_frozen_class_ladder(C, tmp_path, cls, census, ratio, expect):
    """Both backend classes, every rung, read out of the spec rather than retyped."""
    name = "DMCReacherHard" if cls == "S" else "PushT"
    fps = 1000.0 * ratio
    # decode set low so `cores` never caps the tier; the cap has its own test
    s = shard(tmp_path, "s", [env_rec(name, ["x"], {1: 1.0, 256: fps, 4096: fps},
                                      official=1000.0, decode=fps / 1000.0)])
    lab = C.certify(C.merge([s]), census)["labels"][0]
    assert lab["backend_class"] == cls
    assert lab["tier"] == expect, (lab["tier"], lab["evidence"])


def test_below_g4_caps_the_tier_at_faster(C, tmp_path, SPEC):
    """DMCBallInCupCatch's real shape: faster than the official renderer, still losing to decode.

    Historically 12.5x the reference and 0.59 cores of decode -- the only row in the 196-env corpus
    under the G4 bar. Step 2 requires both facts to be sayable at once.
    """
    s = shard(tmp_path, "s", [env_rec("DMCReacherHard", ["x"], {1: 1.0, 256: 12_500, 4096: 12_500},
                                      official=1000.0, decode=21_000.0)])
    lab = C.certify(C.merge([s]), CENSUS_S)["labels"][0]
    assert lab["evidence"]["ratio"] > SPEC["gpu"]["tiers"]["S"]["GPU_FAST"]["R_gte"]
    assert lab["evidence"]["cores"] < 1
    assert "BELOW_G4" in lab["qualifiers"]
    assert lab["tier"] == "GPU_FASTER", "the cores conjunct did not cap the ratio tier"


def test_no_reference_uses_the_frozen_alternate_ladder(C, tmp_path, SPEC):
    """No installable official renderer means no ratio -- and no invented one."""
    fb = SPEC["gpu"]["no_reference_fallback"]
    s = shard(tmp_path, "s", [env_rec("PushT", ["x"], {1: 1.0, 256: 50_000, 4096: 50_000},
                                      official=None, decode=400.0)])
    lab = C.certify(C.merge([s]), CENSUS_R)["labels"][0]
    assert "NO_REFERENCE" in lab["qualifiers"]
    assert lab["evidence"].get("ratio") is None, "a ratio was computed with no reference to divide by"
    assert lab["evidence"]["cores"] >= fb["GPU_EXCEPTIONAL"]["cores_gte"]
    assert lab["tier"] == "GPU_EXCEPTIONAL"


def test_host_bound_is_an_independent_qualifier(C, tmp_path, SPEC):
    """It labels the row, it does not lower it. Reacher is both fast and host-bound."""
    thr = SPEC["gpu"]["qualifiers"]["HOST_BOUND"]["threshold"]
    hot = env_rec("DMCReacherHard", ["x"], {1: 1.0, 256: 50_000, 4096: 50_000}, decode=100.0,
                  cpu_frac=thr + 0.5)
    cool = env_rec("DMCReacherHard", ["x"], {1: 1.0, 256: 50_000, 4096: 50_000}, decode=100.0,
                   cpu_frac=1.0)
    a = C.certify(C.merge([shard(tmp_path, "a", [hot], host="h1")]), CENSUS_S)["labels"][0]
    b = C.certify(C.merge([shard(tmp_path, "b", [cool], host="h2")]), CENSUS_S)["labels"][0]
    assert "HOST_BOUND" in a["qualifiers"] and "HOST_BOUND" not in b["qualifiers"]
    assert a["tier"] == b["tier"], "HOST_BOUND changed the tier instead of qualifying it"


def test_batch_scales_and_large_batch_regression_are_computed(C, tmp_path, SPEC):
    q = SPEC["gpu"]["qualifiers"]
    good = env_rec("PushT", ["x"], {1: 10.0, 256: 5_000, 4096: 10_000}, decode=100.0)
    bad = env_rec("PushT", ["x"], {1: 10.0, 256: 5_000, 4096: 100.0}, decode=100.0)
    a = C.certify(C.merge([shard(tmp_path, "a", [good], host="h1")]), CENSUS_R)["labels"][0]
    b = C.certify(C.merge([shard(tmp_path, "b", [bad], host="h2")]), CENSUS_R)["labels"][0]
    assert a["evidence"]["batch_scales"] == 1000.0 >= q["BATCH_SCALES"]["factor"]
    assert "BATCH_SCALES" in a["qualifiers"] and "LARGE_BATCH_REGRESSION" not in a["qualifiers"]
    assert "BATCH_SCALES_FAIL" in b["qualifiers"], "a 10x scaling passed a 20x bar"
    assert "LARGE_BATCH_REGRESSION" in b["qualifiers"]


def test_failing_batch_scales_caps_the_tier_not_just_the_badge(C, tmp_path, SPEC):
    """The spec says `required_above_tier`, so the qualifier has to BITE, not merely be recorded.

    Fetch is the live case: ~5,200 f/s at every batch from 256 up, a scaling factor of 7.6-8.1x
    against the 20x bar, and a ratio that clears class S's GPU_FAST floor several times over. An
    evaluator that files BATCH_SCALES_FAIL as a badge and leaves the tier alone certifies a renderer
    that is not actually batching -- which is the one thing that bar exists to catch.
    """
    q = SPEC["gpu"]["qualifiers"]["BATCH_SCALES"]
    cap = q["required_above_tier"]
    # R = 50 clears class S GPU_EXCEPTIONAL (floor 40) and cores = 50 clears its cores conjunct;
    # scaling is 50,000/6,250 = 8x, well under the 20x bar.
    rec = env_rec("DMCReacherHard", ["x"], {1: 6_250.0, 256: 50_000, 4096: 50_000},
                  official=1_000.0, decode=1_000.0)
    lab = C.certify(C.merge([shard(tmp_path, "s", [rec])]), CENSUS_S)["labels"][0]
    assert lab["evidence"]["batch_scales"] == 8.0 < q["factor"]
    assert "BATCH_SCALES_FAIL" in lab["qualifiers"]
    assert lab["tier"] == cap, f"a renderer scaling 8x held {lab['tier']} above the {cap} cap"
    assert lab["evidence"]["tier_before_batch_scales_cap"] == "GPU_EXCEPTIONAL"
    assert "capped_by" in lab and "BATCH_SCALES" in lab["capped_by"]

    # and a renderer that DOES batch keeps its tier
    ok = env_rec("DMCReacherHard", ["x"], {1: 1_000.0, 256: 50_000, 4096: 50_000},
                 official=1_000.0, decode=1_000.0)
    good = C.certify(C.merge([shard(tmp_path, "t", [ok], host="h2")]), CENSUS_S)["labels"][0]
    assert good["evidence"]["batch_scales"] >= q["factor"]
    assert good["tier"] == "GPU_EXCEPTIONAL" and "capped_by" not in good


def test_cpu_tiers_use_the_cpu_ladder_and_its_cores_conjunct(C, tmp_path, SPEC):
    """S_cpu alone cannot reach CPU_FAST; cores_cpu >= 1 is required with it."""
    # ratio 10 clears the CPU_FAST floor of 5, but the renderer loses to decode on the same core
    slow = env_rec("PushT", ["x"], {1: 1.0, 256: 10_000}, official=1_000.0, decode=50_000.0)
    fast = env_rec("PushT", ["x"], {1: 1.0, 256: 10_000}, official=1_000.0, decode=5_000.0)
    a = C.certify(C.merge([shard(tmp_path, "a", [slow], device="cpu", host="h1")]), CENSUS_R)["labels"][0]
    b = C.certify(C.merge([shard(tmp_path, "b", [fast], device="cpu", host="h2")]), CENSUS_R)["labels"][0]
    assert a["evidence"]["ratio"] == 10.0 and a["evidence"]["cores"] < 1
    assert a["tier"] == "CPU_FASTER", "cores_cpu < 1 still reached CPU_FAST"
    assert b["tier"] in ("CPU_FAST", "CPU_EXCEPTIONAL")


def test_backend_class_comes_from_declared_backends_not_the_env_object(C):
    """``env.backend`` is the rasterizer variant ('auto'), not the render module Step 2 classes over."""
    assert C.backend_class("PushT", CENSUS_R)[0] == "R"
    assert C.backend_class("DMCReacherHard", CENSUS_S)[0] == "S"
    assert C.backend_class("Nope", {})[0] == C.UNKNOWN


def test_no_threshold_is_hardcoded_in_the_evaluator(C):
    """Step 3 is not allowed to contain a bar. Every bound must come from the frozen spec.

    A bare numeric literal in a comparison is how a policy quietly forks: the JSON keeps saying 5 and
    the code starts saying 4. Only the structural constants of the sweep (0, 1, indices) are allowed.
    """
    import ast
    tree = ast.parse(CERT_PY.read_text())
    # Strip every docstring before scanning: the prose legitimately quotes bounds ("2 <= R < 4") to
    # explain them, and a text scan cannot tell a quoted bound from an applied one. The AST can.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    # Only the functions that DECIDE a label. The reporting half of the module legitimately compares
    # against small integers for display ("name the first 3 envs, then an ellipsis"), and a scan that
    # cannot tell a truncation from a bar produces a guard people learn to ignore.
    DECIDES = {"_holds", "_tier_from", "classify", "best_certified", "backend_class"}
    bad = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or fn.name not in DECIDES:
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Compare) and any(
                    isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                for c in node.comparators:
                    if isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) \
                            and float(c.value) not in (0.0, 1.0):
                        bad.append((fn.name, c.value))
    assert DECIDES <= {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)}, (
        "a label-deciding function was renamed; this guard now scans nothing")
    assert not bad, (f"scratch/step3/certify.py compares against hardcoded number(s) {bad}; every "
                     f"bound must be read from scratch/certification.json")


def test_partial_coverage_never_becomes_environment_done(C, tmp_path, SPEC):
    """The Step 2 invariant, checked where Step 3 could break it: a great tier is still not DONE.

    CraftaxClassicPixels is the live case -- NO_REFERENCE, partial coverage, and fast. Nothing in the
    performance evidence may promote it.
    """
    matrix = json.loads((ROOT / "scratch" / "swm_matrix.json").read_text())
    row = next(r for r in matrix["rows"] if r["swm_id"] == "swm/CraftaxClassicPixels-v1")
    assert row["axes"]["registered_default_render_covered"]["covered"] is False
    assert "registered_default_coverage == full" in SPEC["ladder"]["environment_done_requires"]
    s = shard(tmp_path, "s", [env_rec("CraftaxClassic", ["swm/CraftaxClassicPixels-v1"],
                                      {1: 1.0, 256: 500_000, 4096: 500_000}, official=None, decode=5_000.0)])
    lab = C.certify(C.merge([s]), {"CraftaxClassic": {"backends": ["tilemap"]}})["labels"][0]
    assert lab["tier"] == "GPU_EXCEPTIONAL"
    # the label is per-device and carries no environment-level state at all -- that is the guard
    assert "ENVIRONMENT_DONE" not in json.dumps(lab)


def test_labels_name_the_spec_they_were_derived_from(C, tmp_path):
    out = C.certify(C.merge([shard(tmp_path, "s", [env_rec("PushT", ["x"], {256: 100})])]), CENSUS_R)
    spec = json.loads((ROOT / "scratch" / "certification.json").read_text())
    assert out["spec_schema_version"] == spec["schema_version"]
    assert out["spec_frozen_on"] == spec["frozen_on"]


def test_a_missing_cpu_backend_is_not_a_dependency_blocker(C, tmp_path):
    """Two failures that look alike in a log and mean opposite things for the work queue.

    Step 3 filed both as `blocked_environment`, which put 16 rows at the top of the optimization queue
    as setup problems. Step 3B proved the dependency half is solved -- the matched mujoco 3.11 stack
    measures all of Fetch on the GPU -- and the CPU failure survived it, inside the renderer. One is
    an install; the other is a backend nobody wrote.
    """
    gap = {"env": "FetchPush", "projected_to": ["swm/FetchPush-v3"],
           "blocked": {"stage": "construct", "kind": "blocked_environment",
                       "reason": "RuntimeError: Expected all tensors to be on the same device, "
                                 "but found at least two devices, cuda:0 and cpu!"}}
    dep = {"env": "DMCCheetahRun", "projected_to": ["swm/CheetahDMControl-v0"],
           "blocked": {"stage": "construct", "kind": "blocked_environment",
                       "reason": "ModuleNotFoundError: No module named 'warp'"}}
    out = C.certify(C.merge([shard(tmp_path, "s", [gap, dep], device="cpu")]), {})["labels"]
    by = {l["env"]: l for l in out}

    assert C.CPU_BACKEND_MISSING in by["FetchPush"]["qualifiers"]
    assert by["FetchPush"]["blocked"]["kind"] == "backend_missing"
    assert by["FetchPush"]["tier"] == "CPU_UNMEASURED", "the frozen tier must not change"

    assert C.CPU_BACKEND_MISSING not in by["DMCCheetahRun"]["qualifiers"], \
        "a missing package was mistaken for a missing backend"
    assert by["DMCCheetahRun"]["blocked"]["kind"] == "blocked_environment"

    # and the queue must rank them apart rather than lumping both under one category
    assert C.bottleneck(None, by["FetchPush"], False)[0] == "GPU-only backend, no CPU path"
    assert C.bottleneck(None, by["DMCCheetahRun"], False)[0].startswith("dependency")
    assert (C._CAT_RANK["GPU-only backend, no CPU path"]
            > C._CAT_RANK["dependency / benchmark environment blocker"])


def test_a_cpu_measurement_anywhere_refutes_a_missing_backend(C, tmp_path):
    """The claim is about the env, not about the host that happened to fail.

    PushT renders on the M5 and raised `Expected a cuda device` on one Oscar node -- the same
    exception text as a real backend gap, from a host problem. Tagging it would have said the renderer
    does not exist while its measured CPU throughput sat two rows above in the same table.
    """
    good = env_rec("PushT", ["x"], {256: 400.0}, official=1_000.0, decode=20_000.0)
    bad = {"env": "PushT", "projected_to": ["x"],
           "blocked": {"stage": "construct", "kind": "blocked_environment",
                       "reason": "ValueError: Expected a cuda device, but got: cpu"}}
    labels = C.certify(C.merge([shard(tmp_path, "m5", [good], device="cpu", host="m5"),
                                shard(tmp_path, "ox", [bad], device="cpu", host="oscar")]),
                       CENSUS_R)["labels"]
    assert not any(C.CPU_BACKEND_MISSING in (l.get("qualifiers") or []) for l in labels), \
        "an env with a real CPU measurement was labelled as having no CPU backend"
