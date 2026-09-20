"""The committed performance accounting is exactly what its generator produces.

Two things are asserted and they are different things:

  * ``tests/fixtures/regression/join/step8/performance/tiers.json`` is BYTE-identical to a regeneration from the
    committed Step-6/Step-7 artifacts. A generated artifact that has been hand-edited is a report
    that no longer matches its evidence, and this is the only cheap way to catch that.
  * every one of the 42 registered ``swm/`` ids carries a CPU and a GPU entry whose ``status`` is
    one of the four words the closure criteria use, with the numbers a MEASURED row must
    have and no numbers on a row that claims not to be measured.

The test runs on CPU, imports no renderer and reaches no network.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TIERS = ROOT / "tests" / "fixtures" / "regression" / "performance_tiers.json"
BUILDER = ROOT / "scripts" / "regression" / "support" / "build_performance.py"

STATUSES = {"MEASURED", "UNMEASURED", "MISSING-UNSUPPORTED_BY_DESIGN", "UNSUPPORTED_BY_DESIGN"}


@pytest.fixture(scope="module")
def doc() -> dict:
    if not TIERS.is_file():
        pytest.fail(f"{TIERS} is missing: run {BUILDER.relative_to(ROOT)}")
    return json.loads(TIERS.read_text())


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("step8_build_performance", BUILDER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tiers_json_is_byte_identical_to_a_regeneration(builder):
    regenerated = json.dumps(builder.build(), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    assert TIERS.read_text() == regenerated, (
        "tests/fixtures/regression/join/step8/performance/tiers.json differs from a regeneration; re-run "
        "scripts/regression/support/build_performance.py instead of editing the artifact"
    )


def test_every_registered_id_has_a_cpu_and_gpu_status(doc):
    assert doc["counts"]["rows"] == 42
    assert len({r["swm_id"] for r in doc["rows"]}) == 42
    for row in doc["rows"]:
        for dev in ("cpu", "gpu"):
            entry = row[dev]
            assert entry["status"] in STATUSES, (row["swm_id"], dev, entry["status"])
            assert entry["tier"], (row["swm_id"], dev)


def test_symbolic_ids_are_unsupported_by_design_not_unmeasured(doc):
    sym = [r for r in doc["rows"] if r["pixel_or_symbolic"] == "symbolic"]
    assert len(sym) == 2
    for row in sym:
        for dev in ("cpu", "gpu"):
            assert row[dev]["status"] == "UNSUPPORTED_BY_DESIGN", row["swm_id"]
            assert row[dev]["numbers"] == {}


def test_measured_rows_carry_the_numbers_the_tier_rests_on(doc):
    for row in doc["rows"]:
        for dev, ratio_key, cores_key in (("cpu", "S_cpu", "cores_cpu"), ("gpu", "R", "cores")):
            entry = row[dev]
            if entry["status"] != "MEASURED":
                continue
            n = entry["numbers"]
            assert n.get("fps_best"), (row["swm_id"], dev, "a MEASURED row must have a throughput")
            assert entry["provenance"].get("artifact"), (row["swm_id"], dev)
            # Either the ratio against an official renderer, or the NO_REFERENCE cores ladder.
            if entry["reference"]["label"] == "NO_REFERENCE":
                assert n.get(cores_key) is not None, (row["swm_id"], dev)
            else:
                assert n.get(ratio_key) is not None, (row["swm_id"], dev)


def test_unmeasured_and_missing_rows_make_no_throughput_claim(doc):
    for row in doc["rows"]:
        for dev in ("cpu", "gpu"):
            entry = row[dev]
            if entry["status"] == "MEASURED":
                continue
            assert not entry["numbers"].get("fps_best"), (row["swm_id"], dev)
            assert entry.get("blocked") or entry["status"] == "UNSUPPORTED_BY_DESIGN", (
                row["swm_id"],
                dev,
                "an unmeasured row must say what blocked it",
            )


def test_cpu_raises_by_design_rows_are_named_and_flagged(doc):
    raises = {r["swm_id"] for r in doc["rows"] if r["cpu"]["status"] == "MISSING-UNSUPPORTED_BY_DESIGN"}
    assert raises == {"swm/OGBCube-v0", "swm/OGBScene-v0", "swm/PFRocketLanding-v0"}
    for row in doc["rows"]:
        if row["swm_id"] in raises:
            assert "RAISES_BY_DESIGN" in row["cpu"]["flags"]
            assert row["cpu"]["tier"] == "CPU_MISSING"


def test_quadruped_cpu_stays_load_quarantined(doc):
    row = next(r for r in doc["rows"] if r["swm_id"] == "swm/QuadrupedDMControl-v0")
    assert row["cpu"]["tier"] == "CPU_UNMEASURED"
    assert "LOAD_QUARANTINED" in row["cpu"]["flags"]
    assert row["cpu"]["numbers"] == {}


def test_batch_scales_failures_are_kept_not_relabelled(doc):
    """The frozen 20x bar is applied as Step 2 froze it; Rocket and Fetch keep GPU_PARITY."""
    for sid in ("swm/PFRocketLanding-v0", "swm/FetchPickAndPlace-v3", "swm/OGBCube-v0", "swm/OGBScene-v0"):
        row = next(r for r in doc["rows"] if r["swm_id"] == sid)
        assert "BATCH_SCALES_FAIL" in row["gpu"]["flags"], sid
        assert row["gpu"]["tier"] == "GPU_PARITY", sid
        n = row["gpu"]["numbers"]
        assert n["batch_scales_ratio"] < 20, sid


def test_every_source_is_named_with_its_sha256(doc):
    assert doc["sources"]
    for s in doc["sources"]:
        assert len(s["sha256"]) == 64
        assert (ROOT / s["path"]).is_file(), s["path"]


def test_no_absolute_oscar_paths_in_the_committed_artifact():
    assert "/oscar/" not in TIERS.read_text()
