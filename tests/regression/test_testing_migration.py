"""Saved-evidence verifiers and required profiles reject defects instead of passing them."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from stable_envs.testing.evidence import verify_seal
from tests.helpers.reference.saved import native_corpus, verify_rocket_render

ROOT = Path(__file__).resolve().parents[2]
ROCKET = ROOT / "tests/fixtures/saved/rocket_public_cpu_coverage"
ROCKET_REVISION = "ebf82e6eeed8b2b9c7560e4dff6f9a6b5c95e95b"


@pytest.fixture(scope="module")
def recorded_tree(tmp_path_factory):
    """Materialize the code the report recorded, read from Git without checking it out."""
    root = tmp_path_factory.mktemp("recorded-rocket")
    report = json.loads((ROCKET / "result.json").read_text())
    for name in report["production_hashes"]:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "show", f"{ROCKET_REVISION}:{name}"], cwd=ROOT))
    return root


def test_rocket_fixture_passes_coverage_and_code_identity(tmp_path):
    """Control: the unmodified fixture gets past coverage, then stops for lack of frames."""
    shutil.copy(ROCKET / "result.json", tmp_path / "result.json")
    with pytest.raises(FileNotFoundError, match="frames.npz"):
        verify_rocket_render(tmp_path, ROOT, revision=ROCKET_REVISION)


@pytest.mark.parametrize("defect", ["duplicate", "missing_group", "sibling"])
def test_new_rocket_coverage_rejects_duplicate_or_missing_groups_before_pixels(tmp_path, defect):
    report = json.loads((ROCKET / "result.json").read_text())
    if defect == "duplicate":
        report["records"][1] = report["records"][0]
    elif defect == "sibling":
        report["records"][0]["label"] = "another_registration"
    else:
        report["groups"].pop(next(iter(report["groups"])))
    (tmp_path / "result.json").write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        verify_rocket_render(tmp_path, ROOT, revision=ROCKET_REVISION)


def test_historical_report_does_not_certify_changed_code(recorded_tree, tmp_path):
    candidate = tmp_path / "candidate"
    shutil.copytree(recorded_tree, candidate)
    report = json.loads((ROCKET / "result.json").read_text())
    changed = candidate / next(iter(report["production_hashes"]))
    changed.write_bytes(changed.read_bytes() + b"\n# Deliberate mismatch\n")
    with pytest.raises(ValueError, match="code hash mismatch"):
        verify_rocket_render(ROCKET, candidate)


@pytest.mark.parametrize("defect", ["missing", "hash", "schema"])
def test_native_saved_corpus_rejects_missing_or_changed_inputs(tmp_path, defect):
    source = ROOT / "tests/fixtures/semantics/native_classic"
    manifest = json.loads((source / "manifest.json").read_text())
    entry = manifest["domains"]["cartpole"]
    if defect != "missing":
        shutil.copyfile(source / entry["file"], tmp_path / entry["file"])
    if defect == "hash":
        entry["sha256"] = "0" * 64
    elif defect == "schema":
        del entry["arrays"]["state/rng/state"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((AssertionError, FileNotFoundError)):
        native_corpus(tmp_path, "cartpole")


@pytest.mark.parametrize("selection", ["frozen_source", "independent_source_trajectory"])
def test_partial_or_blocked_source_profile_cannot_pass(tmp_path, selection):
    run = tmp_path / "evidence"
    # No source checkout supplied. Unrelated passing frozen tests must not mask it.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/contracts",
            "--contract-profile",
            "native",
            "--contract-output",
            str(run),
            "-k",
            selection,
        ],
        cwd=ROOT,
        env={**os.environ, "SE_ALLOW_EMPTY_RUN": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    verify_seal(run)
    report = json.loads((run / "result.json").read_text())
    assert report["status"] == "BLOCKED" and report["passed"] == 0 and report["required"] == 128


def test_required_profile_does_not_import_unverified_installed_source(tmp_path):
    package = tmp_path / "stable_worldmodel"
    package.mkdir()
    marker = tmp_path / "unverified-import"
    (package / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise RuntimeError('Unverified installed source executed')\n"
    )
    run = tmp_path / "evidence"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/contracts/test_contract_state.py",
            "--contract-profile",
            "native",
            "--contract-output",
            str(run),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert not marker.exists(), result.stdout + result.stderr
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads((run / "result.json").read_text())["status"] == "BLOCKED"
