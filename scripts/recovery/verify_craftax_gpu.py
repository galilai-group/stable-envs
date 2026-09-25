"""Independently verify downloaded Craftax GPU evidence; never run GPU work locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREPARED = ROOT / "docs/research/semantic_recovery_evidence/craftax_gpu_prepared/bundle.json"
CPU = ROOT / "docs/research/semantic_recovery_evidence/craftax_cpu/final"


def verify_hashes(manifest):
    root = manifest.parent.resolve()
    entries = []
    for line in manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = (root / relative.lstrip("*")).resolve()
        assert path.is_relative_to(root), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
        entries.append(str(path.relative_to(root)))
    assert entries and len(entries) == len(set(entries)), "empty or duplicate manifest entries"
    return entries


def verify_registration(evidence, name, count, job, log_lines, bundle):
    assert f"CRAFTAX_REGISTRATION_ACCEPTANCE_PASS {name} {job}" in log_lines
    cases = ET.parse(evidence / f"{name}.xml").findall(".//testcase")
    assert len(cases) == count, (name, len(cases), count)
    assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
    for case in cases:
        test = case.attrib["name"]
        assert name in test or name == "CraftaxSymbolic" and "full_symbolic_preserves" in test, test
    folder = evidence / name
    hashed = verify_hashes(folder / "SHA256SUMS")
    expected_files = set(bundle["expected_per_registration_arrays"])
    expected_files.update(bundle["expected_per_registration_other"])
    assert expected_files - {"SHA256SUMS"} == set(hashed)
    report = json.loads((folder / "result.json").read_text())
    assert report["name"] == name and report["registration"] == f"swm/{name}-v1"
    marker = f"CRAFTAX_{name.upper()}_GPU_GATE_PASS"
    assert report["status"] == "PASS" and report["marker"] == marker
    assert marker in (evidence / f"{name}-arrays.log").read_text().splitlines()
    profile = report["profile"]
    assert profile == json.loads((folder / "profile.json").read_text())
    assert profile["backend"] == "gpu" and profile["gpu"] and profile["devices"]
    assert profile["python"].startswith("3.12.")
    assert profile["x64"] is False and profile["prng"] == "threefry2x32"
    assert profile["threefry_partitionable"] is True
    baseline = json.loads((CPU / name / "profile.json").read_text())
    for package, expected in baseline["packages"].items():
        actual = profile["packages"][package]
        assert (actual.split("+")[0] if package == "torch" else actual) == expected, (package, actual)
    assert profile["packages"]["torch"].endswith("+cu126")
    schema = json.loads((CPU / name / "schema.json").read_text())["state"]
    assert json.loads((folder / "schema.json").read_text())["state"] == schema
    state_keys = {"state/" + "/".join(field["key"]) for field in schema}
    engine_keys = {k.removeprefix("state/") for k in state_keys if k.startswith("state/engine/")}
    engine_keys = {"source_" + k for k in engine_keys}
    assert len(report["cases"]) == 11
    assert {c["array_file"] for c in report["cases"]} == set(bundle["expected_per_registration_arrays"])
    for case in report["cases"]:
        assert case["state_exact"] and case["required_output_exact"] and case["nonmutation"]
        with np.load(folder / case["array_file"], allow_pickle=False) as arrays:
            assert {k for k in arrays.files if k.startswith("state/")} == state_keys
            assert {k for k in arrays.files if k.startswith("source_engine/")} == engine_keys
            for field in schema:
                value = arrays["state/" + "/".join(field["key"])]
                assert value.shape == (1, *field["shape"])
                assert value.dtype == np.dtype(field["dtype"].removeprefix("torch."))
                assert np.isfinite(value).all()
            for key in engine_keys:
                np.testing.assert_array_equal(arrays[key], arrays["state/" + key.removeprefix("source_")][0])
            np.testing.assert_array_equal(arrays["source_wrapper_key"], arrays["state/rng/wrapper_key"][0])
            if "Pixels" in name:
                expected_shape = (63, 63, 3) if "Classic" in name else (130, 110, 3)
                for key in ["source_frame", "full_frame", "render_frame"]:
                    assert arrays[key].shape == expected_shape and arrays[key].dtype == np.uint8
                np.testing.assert_array_equal(arrays["source_frame"], arrays["full_frame"])
                np.testing.assert_array_equal(arrays["full_frame"], arrays["render_frame"])
            else:
                expected_shape = (1345,) if "Classic" in name else (8268,)
                assert arrays["candidate_output"].shape == expected_shape
                assert arrays["candidate_output"].dtype == np.float32
                np.testing.assert_array_equal(arrays["source_output"], arrays["candidate_output"])
    return {"status": "PASS", "tests": count, "exact_array_cases": 11, "profile": profile}


def main():
    if sys.flags.optimize:
        raise RuntimeError("Verification requires enabled assertions; do not use Python -O")
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--slurm-log", type=Path, required=True)
    parser.add_argument("--sacct", type=Path, required=True, help="JobIDRaw|State|ExitCode|... from Oscar")
    parser.add_argument("--job", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Refusing to overwrite an existing evidence result"
    bundle = json.loads(PREPARED.read_text())
    assert json.loads((args.evidence / "bundle.json").read_text()) == bundle
    assert (args.evidence / "SHA256SUMS").read_bytes() == PREPARED.with_name("SHA256SUMS").read_bytes()
    files = verify_hashes(args.evidence / "GPU-SHA256SUMS")
    actual_files = {
        str(p.relative_to(args.evidence))
        for p in args.evidence.rglob("*")
        if p.is_file() and not p.is_symlink() and p.name != "GPU-SHA256SUMS"
    }
    assert actual_files == set(files), "GPU manifest does not cover the actual evidence tree"
    rows = [line.split("|") for line in args.sacct.read_text().splitlines()]
    job_rows = [row for row in rows if row[0] == str(args.job)]
    assert len(job_rows) == 1
    state, exit_code = job_rows[0][1:3]
    assert state in {"COMPLETED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"}, state
    log_lines = args.slurm_log.read_text().splitlines()
    results = {}
    for name, count in bundle["expected_registration_test_counts"].items():
        try:
            results[name] = verify_registration(args.evidence, name, count, args.job, log_lines, bundle)
        except (AssertionError, KeyError, ValueError, OSError, ET.ParseError) as error:
            results[name] = {"status": "NOT_ACCEPTED", "reason": repr(error)}
    if all(r["status"] == "PASS" for r in results.values()):
        assert state == "COMPLETED" and exit_code == "0:0"
        assert f"CRAFTAX_FAMILY_GPU_GATE_PASS {args.job}" in log_lines
    report = {
        "job": args.job,
        "slurm_state": state,
        "exit_code": exit_code,
        "verified_hashes": len(files),
        "registrations": results,
        "source_archive_sha256": bundle["archive_sha256"]["source.tar.gz"],
        "cross_device_equality": "NOT_CLAIMED",
        "promotion_performed": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["status"] for k, v in results.items()}))


if __name__ == "__main__":
    main()
