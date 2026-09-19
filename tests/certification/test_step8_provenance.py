"""Step 8 Workstream F: the reference-version declarations, and the record that reads them.

Three things are asserted, and the middle one is the load-bearing one:

1. the four DM Control trackcom cells DECLARE a reference version, and it is the reference side
   only -- this is what makes "certified against 3.11" a machine-readable fact rather than prose;
2. the committed ``reference_arm.json`` REGENERATES from the committed inputs, so the record cannot
   drift from the shards it cites without this test going red;
3. the support matrix reads the authoritative arm AND still carries the drifted one, because the
   plan forbids both reporting a version drift as a renderer failure and erasing the drift.

Everything here is a file read plus one class attribute. Nothing renders and no GPU is required.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ARM = ROOT / "examples/results/step8/provenance/reference_arm.json"
MATRIX = ROOT / "examples/results/step8/support_matrix.json"

TRACKCOM = ("DMCCheetahRun", "DMCHopperHop", "DMCHumanoidStand", "DMCWalkerWalk")

#: The reference the four cells are certified against. Hard-coded here on purpose: this test exists
#: to notice a change to it, so reading it from the class would make it vacuous.
DECLARED = {"mujoco": "3.11.0"}

#: Package names that are OURS, not the reference's. A declared reference version naming one of
#: these would mean the certifier blocks on a bump of our own renderer, which is a regression to
#: see rather than a reason to refuse to judge.
OURS = ("mujoco_warp", "mujoco-warp", "warp", "nvdiffrast", "torch", "triton")


@pytest.fixture(scope="module")
def arm() -> dict:
    assert ARM.is_file(), f"{ARM} is missing; run scratch/step8/provenance/reference_arm.py"
    return json.loads(ARM.read_text())


@pytest.mark.parametrize("cell", TRACKCOM)
def test_the_cell_declares_the_reference_it_was_measured_against(cell: str) -> None:
    from stable_envs.env import reference_versions_of
    import stable_envs

    cls = stable_envs.registry.REGISTRY.get(cell)
    assert cls is not None, f"{cell} is not registered in this interpreter"
    declared = dict(reference_versions_of(cls) or {})
    assert declared == DECLARED, (
        f"{cell} declares {declared}; the Step-8 certification of this cell rests on "
        f"{DECLARED}. Changing it is a re-declaration, not a test fix.")
    assert not any(p in declared for p in OURS), (
        f"{cell} declares one of OUR packages as a REFERENCE version: {declared}. "
        f"measured_with is the field for our side, and it is recorded, never compared.")


def test_the_reference_arm_record_regenerates(arm: dict) -> None:
    """Re-run the generator and compare. Only the HEAD sha may differ between runs."""
    spec = importlib.util.spec_from_file_location(
        "_ref_arm", ROOT / "scratch/step8/provenance/reference_arm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fresh = mod.build()
    volatile = "generated_at_head_sha"
    assert json.dumps({k: v for k, v in fresh.items() if k != volatile}, sort_keys=True) == \
           json.dumps({k: v for k, v in arm.items() if k != volatile}, sort_keys=True), (
        "the committed reference_arm.json is not what the generator produces from the committed "
        "inputs; regenerate it rather than editing it")


@pytest.mark.parametrize("cell", TRACKCOM)
def test_the_authoritative_arm_is_the_declared_one_and_it_passes(arm: dict, cell: str) -> None:
    rec = arm["cells"][cell]
    assert rec["declared_reference_versions"] == DECLARED
    auth = next(a for a in rec["arms"] if a["arm"] == rec["authoritative_arm"])
    assert auth["reference_version_mismatch"] == [], (
        f"{cell}'s authoritative arm does not satisfy its own declaration: "
        f"{auth['reference_version_mismatch']}")
    assert (auth["provenance"]["reference_packages"] or {})["mujoco"] == DECLARED["mujoco"]
    assert auth["gate_a"] == "PASS", f"{cell} on the declared reference: {auth['gate_a']}"
    assert auth["failures"] == [] and auth["blocked"] == []
    # Provenance a reader can re-run: which job, which host, which tree.
    for field in ("host", "slurm_job_id", "gpu", "driver", "git_commit"):
        assert auth["provenance"].get(field), f"{cell}: the authoritative arm records no {field}"
    assert len(auth["source_sha256"]) == 64


@pytest.mark.parametrize("cell", TRACKCOM)
def test_the_drifted_arm_is_kept_not_erased(arm: dict, cell: str) -> None:
    """The 3.12 measurement is what a user on a newer MuJoCo gets. Losing it is the real damage."""
    rec = arm["cells"][cell]
    drift = [a for a in rec["arms"] if a["arm"] != rec["authoritative_arm"]]
    assert drift, f"{cell} records only one arm; the reference-drift finding would be invisible"
    for d in drift:
        assert d["reference_version_mismatch"], "a non-authoritative arm with no mismatch recorded"
        assert (d["provenance"]["reference_packages"] or {})["mujoco"] == "3.12.0"
        vf = d["checks"]["check_visual_fidelity"]
        # blocked_environment, never fail: a reference at the wrong version is a dependency
        # failure, and Step 4's own rule is that a dependency failure is not a fidelity failure.
        assert vf["status"] == "blocked_environment", vf["status"]
        assert vf.get("judged_against_wrong_reference"), (
            "the verdict the drifted arm WOULD have received is the finding; it must be recorded")


@pytest.mark.parametrize("cell", TRACKCOM)
def test_the_support_matrix_reads_the_authoritative_arm_and_reports_the_drift(cell: str) -> None:
    rows = [r for r in json.loads(MATRIX.read_text())["rows"]
            if cell in (r.get("stable_envs_cells") or [])]
    assert len(rows) == 1, f"{cell} appears in {len(rows)} matrix rows"
    row = rows[0]
    assert row["gate_a"]["status"] == "PASS", row["gate_a"]["status"]
    assert row["gate_a"]["source"] == "examples/results/step8/provenance/reference_arm.json"
    assert any(a["cell"] == cell for a in row["gate_a"]["authoritative_arm"])
    kinds = {(x["kind"], x["what"]) for x in row["limitations"]}
    assert ("DOCUMENTED_LIMITATION", "reference/upstream drift") in kinds, (
        f"{cell}'s row passes without carrying the 3.12 drift as a limitation; both facts are "
        f"required to stay in the row")


def test_no_committed_provenance_file_carries_an_absolute_cluster_path() -> None:
    """A committed path into somebody's scratch is not reproducible by anyone else. ``$SCR``."""
    bad = []
    for d in (ROOT / "scratch/step8/provenance", ROOT / "examples/results/step8/provenance"):
        for p in sorted(d.rglob("*")):
            if not p.is_file() or p.name == "sanitize.py":
                continue                   # the scrubber's own patterns, not a path into a home
            try:
                text = p.read_text()
            except UnicodeDecodeError:
                continue
            for m in re.finditer(r"/oscar/\S*|/users/\S*|/user/\S*", text):
                # A HOSTNAME is provenance and must stay; a filesystem path must not.
                if m.group(0).startswith("/oscar") or m.group(0).startswith("/user"):
                    bad.append(f"{p.relative_to(ROOT)}: {m.group(0)[:80]}")
    assert not bad, "absolute cluster paths in committed files (write $SCR/...):\n" + "\n".join(bad)
