"""Run explicit MJWarp CUDA validation; any skip is a failure.

SE_DMC_MJWARP_SOURCE_ROOT and SE_DMC_MJWARP_EVIDENCE are needed for full
acceptance. --smoke runs actual CUDA stepping, exact State and pixel contracts
without a source checkout or calibration, and is deliberately not acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class NoSkippedTests:
    def __init__(self):
        self.skipped = []
        self.deselected = []
        self.passed = 0
        self.results = []

    def pytest_runtest_logreport(self, report):
        self.results.append(
            dict(nodeid=report.nodeid, phase=report.when, outcome=report.outcome, seconds=report.duration)
        )
        if report.skipped:
            self.skipped.append(report.nodeid)
        if report.when == "call" and report.passed:
            self.passed += 1

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_deselected(self, items):
        self.deselected.extend(item.nodeid for item in items)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.skipped or self.deselected or not self.passed:
            session.exitstatus = 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--source-root")
    parser.add_argument("--evidence")
    parser.add_argument(
        "--allow-implementation-change",
        action="store_true",
        help="keep frozen oracle/recipe/caps while evaluating changed implementation code",
    )
    parser.add_argument(
        "--report", help="new JSON report path; full acceptance defaults to a sibling of the evidence directory"
    )
    args = parser.parse_args()
    os.environ["SE_REQUIRE_FAMILIES"] = "dmc"
    if args.source_root:
        os.environ["SE_DMC_MJWARP_SOURCE_ROOT"] = args.source_root
    if args.evidence:
        os.environ["SE_DMC_MJWARP_EVIDENCE"] = args.evidence
    os.environ["SE_DMC_MJWARP_ALLOW_IMPLEMENTATION_CHANGE"] = "1" if args.allow_implementation_change else "0"
    from stable_envs.testing.evidence import sha256
    from stable_envs.testing.provenance import verify_recorded_code
    from tests.helpers.reference.dmc_mjwarp import candidate_identity, machine_identity, recipe_identity, require_cuda

    require_cuda()
    calibration = None
    evaluated_files, evaluated_recipe = candidate_identity(), recipe_identity()
    # Prepare the pinned namespace before pytest_configure imports an installed
    # stable_worldmodel, avoiding an accidental switch to another source copy.
    if not args.smoke:
        from tests.helpers.reference.dmc_mjwarp import load_evidence, prepare_source

        prepare_source(os.environ.get("SE_DMC_MJWARP_SOURCE_ROOT"))
        calibration = load_evidence(
            os.environ.get("SE_DMC_MJWARP_EVIDENCE"), allow_implementation_change=args.allow_implementation_change
        )
    import pytest

    suite = str(ROOT / "tests/gpu/test_dmc_mjwarp.py")
    command = [
        suite + "::test_cuda_stepping_snapshot_ownership_and_pixels" if args.smoke else suite,
        str(ROOT / "tests/gpu/test_dmc_mjwarp_streams.py"),
        str(ROOT / "tests/unit/test_dmc_mjwarp_contacts.py"),
        str(ROOT / "tests/unit/test_dmc_sparse_outputs.py"),
        str(ROOT / "tests/unit/test_dmc_capture_selection.py"),
        str(ROOT / "tests/unit/test_dmc_mjwarp_runtime.py")
        + ("::test_cuda_capture_restore_and_alternate_caller_streams" if args.smoke else ""),
        "-ra",
    ]
    if not args.smoke:
        command.append(str(ROOT / "tests/unit/test_dmc_mjwarp_tasks.py"))
    plugin = NoSkippedTests()
    status = pytest.main(command, plugins=[plugin])
    verify_recorded_code(evaluated_files, ROOT)
    verify_recorded_code(evaluated_recipe, ROOT)
    if not args.smoke or args.report:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        evidence_root = Path(os.environ["SE_DMC_MJWARP_EVIDENCE"]).resolve() if calibration else None
        path = Path(args.report) if args.report else evidence_root.parent / f"dmc-mjwarp-acceptance-{timestamp}.json"
        report = dict(
            schema="stable-envs-dmc-mjwarp-acceptance/v1",
            smoke=args.smoke,
            exit_status=int(status),
            passed=plugin.passed,
            skipped=plugin.skipped,
            deselected=plugin.deselected,
            results=plugin.results,
            allow_implementation_change=args.allow_implementation_change,
            candidate_files=evaluated_files,
            recipe_files=evaluated_recipe,
            machine=machine_identity(),
            calibration=None
            if calibration is None
            else {
                "directory": str(evidence_root),
                "file_sha256": sha256(evidence_root / "calibration.json"),
                "seal_sha256": sha256(evidence_root / "SHA256SUMS"),
                "candidate_files": calibration["candidate_files"],
            },
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        print(f"Acceptance report: {path}")
    print(
        f"DMC MJWarp {'smoke' if args.smoke else 'acceptance'}: {plugin.passed} passed, "
        f"{len(plugin.skipped)} skipped, {len(plugin.deselected)} deselected"
    )
    return int(status)


if __name__ == "__main__":
    raise SystemExit(main())
