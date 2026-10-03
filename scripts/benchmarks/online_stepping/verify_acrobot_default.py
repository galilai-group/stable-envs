"""Verify the independent Acrobot default gate before enabling its route."""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.evidence / "results/acrobot.json").read_text())
    terminal = json.loads((args.evidence / "terminal.json").read_text())
    assert terminal["state"] == "COMPLETED" and terminal["exit_code"] == "0:0"
    assert terminal["job"] == report["job"]
    assert report["status"] == "PASS" and report["env"] == "swm/AcrobotControl-v1"
    assert len(report["affinity"]) == 8 and "3090" in report["gpu"]
    tests = ET.parse(args.evidence / "tests.xml").getroot().findall(".//testcase")
    assert len(tests) == 132 and all(not list(t) for t in tests)
    names = {t.get("name") for t in tests}
    for device in ("cpu", "cuda"):
        for pipeline in (False, True):
            assert f"test_public_chunks_slices_early_stop_and_current_render[{device}-{pipeline}]" in names
    gates = report["gates"]
    assert len(gates) == 2 and {g["device"] for g in gates} == {"cpu", "cuda"}
    for g in gates:
        assert g["cases"] == 25 and g["overlaps"] > 0
        assert g["batch"] == 64 and g["processes"] == 8
        assert g["unpatched_constructor"] is True and g["native_resolution"] == [500, 500]
        assert g["frame_mse_vs_local"] == 0
        for field in ("complete_state_task_rng_lifecycle_ownership", "variation_and_device_actions",
                      "failed_restore_keeps_owners"):
            assert g[field] == "PASS"
    manifest = json.loads((args.evidence / "source_manifest.json").read_text())["files"]
    for hashes in (manifest, report["source_sha256"]):
        for rel, digest in hashes.items():
            assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
    print(f"PASS: 132 tests, independent CPU/CUDA default gates, {len(manifest)} frozen files")


if __name__ == "__main__":
    main()
