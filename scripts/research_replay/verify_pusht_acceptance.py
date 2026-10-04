"""Verify per-registration PushT closure and preserve accepted-family evidence."""

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/semantic_recovery_evidence"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    edges = BASE / "final_three_20261004/pusht/edges"
    report = json.loads((edges / "result.json").read_text())
    assert report["status"] == "PASS_PUBLIC_ACCEPTANCE_EDGES"
    assert sha(edges / report["artifact"]) == report["artifact_sha256"]
    assert sha(Path(__file__).with_name("pusht_acceptance_edges.py")) == report["script_sha256"]
    for path, digest in report["production_files"].items():
        assert sha(ROOT / path) == digest, path
    for path, digest in report["source_files"].items():
        assert sha(Path("/Users/samib/stable-worldmodel") / path) == digest, path
    with np.load(edges / report["artifact"], allow_pickle=False) as raw:
        keys = [k[:-7] for k in raw if k.endswith("/source")]
        assert len(keys) == report["raw_comparisons"] == 2094
        for k in keys:
            np.testing.assert_array_equal(raw[k + "/source"], raw[k + "/actual"], err_msg=k)
    for folder in ("pusht_public_state_20261003", "pusht_public_cuda_20261004"):
        root = BASE / folder
        for line in (root / "SHA256SUMS").read_text().splitlines():
            digest, path = line.split("  ", 1)
            assert sha(root / path) == digest, (folder, path)
    # Recompute source continuations, including both independently certified IDs.
    for device, directory in [
        ("cpu", BASE / "pusht_public_state_20261003/cpu"),
        ("cuda", BASE / "pusht_public_cuda_20261004/public"),
    ]:
        out = args.output.with_name(device + "-raw-reverification.json")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("verify_pusht_public.py")),
                "--evidence",
                str(directory),
                "--output",
                str(out),
            ],
            check=True,
        )
    unit = ET.parse(BASE / "pusht_public_cuda_20261004/unit.xml").getroot().find("testsuite")
    assert unit.attrib["tests"] == "473"
    assert all(unit.attrib[k] == "0" for k in ("failures", "errors", "skipped"))
    scheduler = (BASE / "pusht_public_cuda_20261004/scheduler.txt").read_text()
    assert "6993756|COMPLETED|0:0|" in scheduler
    terminal = BASE / "pusht_public_state_20261003/cpu-terminal"
    t = json.loads((terminal / "result.json").read_text())
    assert t["status"] == "PASS_PUBLIC_MIXED_TERMINAL_SOURCE_GATE"
    counts = {}
    for case in t["cases"]:
        assert sha(terminal / case["artifact"]) == case["sha256"]
        with np.load(terminal / case["artifact"], allow_pickle=False) as raw:
            for field in ("valid", "task", "frames"):
                np.testing.assert_array_equal(raw["source_" + field], raw["actual_" + field])
        counts[case["registration"]] = counts.get(case["registration"], 0) + 1
    assert set(counts.values()) == {12} and len(counts) == 2
    # Earlier task/reset proofs apply only to unchanged task/quantizer/space code.
    historical = BASE / "noncraftax_completion_20261003/pusht_task/result.json"
    history = json.loads(historical.read_text())
    assert history["status"] == "PASS_PRIVATE_TASK_MIGRATION" and history["quantizer_centers"] == 256
    for name in ("pusht_task", "pusht_quantizer", "pusht_spaces", "pusht_draw"):
        path = "stable_envs/_adapters/" + name + ".py"
        assert sha(ROOT / path) == history["production_files"][path], path
    previous = json.loads((ROOT / "docs/research/semantic_recovery_implementation_map.json").read_text())
    preserved = {}
    for row in previous["rows"]:
        if row["current_status"] != "VERIFIED" or row["family"] in ("craftax", "pusht"):
            continue
        for a in row["evidence_manifest"]["artifacts"]:
            assert sha(ROOT / a["path"]) == a["sha256"], a["path"]
            preserved[a["path"]] = a["sha256"]
    # No production/shared/test-fixture bytes changed from the completed GPU gate.
    changed = subprocess.check_output(
        ["git", "diff", "8451e7ac", "--name-only", "--", "stable_envs", "tests/fixtures"], cwd=ROOT, text=True
    )
    assert not changed, changed
    registrations = {}
    for name in ("swm/PushT-v1", "swm/PushT-Discrete-v1"):
        gates = [r for r in report["records"] if r["registration"] == name]
        by_name = {r["gate"]: r for r in gates}
        assert by_name["task_boundaries"]["cases"] == 48
        assert by_name["render_configuration"]["cases"] == 4
        assert by_name["persistent_variation"]["cases"] == 17
        if "Discrete" in name:
            assert by_name["all_polar_bins_and_continuous_equivalence"]["cases"] == 256
        registrations[name] = dict(
            status="ACCEPTED_SCOPED_PUBLIC_CPU_AND_CUDA_STORAGE",
            source_cases_per_device=74,
            frame_pairs_per_device=2424,
            normalized_mse=0.0,
            mixed_terminal_cpu_cases=12,
            edge_gates=gates,
        )
    result = dict(
        status="PASS_PER_REGISTRATION_PUSHT_ACCEPTANCE",
        registrations=registrations,
        preserved_prior_artifacts=preserved,
        production_files=report["production_files"],
        base_commit="8451e7accc2925456d5573c3d8fabd8d1c361b85",
        renderer="CPU_PYGAME_ON_BOTH_STORAGE_DEVICES",
        limitations=report["limitations"]
        + [
            "Replay checkpoint storage and restoration grow with history.",
            "No throughput acceptance or fresh all-family rendering run is claimed.",
        ],
        verifier_sha256=sha(Path(__file__)),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
