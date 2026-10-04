"""Verify completed bounded Rocket online reports, coverage and raw pixels."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.input / "result.json").read_text())
    assert report["status"] == "PASS_BOUNDED_ROCKET_ONLINE_SOURCE"
    assert f"{report['job_id']}|COMPLETED|0:0|" in args.terminal.read_text()
    assert report["batch"] == 2 and report["size"] == 96 and report["frames"] == 224
    assert report["source_sha256"] == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    assert (
        report["script_sha256"]
        == hashlib.sha256(Path(__file__).with_name("online_source_gate.py").read_bytes()).hexdigest()
    )
    for path, digest in report["production_hashes"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    cases = report["cases"]
    assert len(cases) == 64
    assert {(c["scenario"], c["dtype"], c["length"], c["selection"]) for c in cases} == set(
        itertools.product(
            ("flight", "contact", "success", "horizon"),
            ("float32", "float64"),
            (1, 8),
            ("none", "empty", "all", "reverse_sparse"),
        )
    )
    assert any(c["length"] == 8 and c["valid_counts"][0] != c["valid_counts"][1] for c in cases)
    assert any(c["length"] == 8 and max(c["valid_counts"]) < 8 for c in cases)
    with np.load(args.input / "frames.npz", allow_pickle=False) as raw:
        ours, source = raw["ours"], raw["source"]
    assert ours.shape == source.shape == (224, 96, 96, 3)
    assert ours.dtype == source.dtype == np.uint8
    errors = np.mean(((ours.astype(np.float64) - source) / 255) ** 2, axis=(1, 2, 3))
    assert np.all(errors <= 0.001)
    offset = 0
    for case in cases:
        assert case["source_never_restored"] and case["exact_state_task_lifecycle"] and case["owned_snapshots"]
        count = case["frames"]
        if count:
            selected = errors[offset : offset + count]
            assert np.isclose(selected.mean(), case["mse_mean"], rtol=1e-12, atol=0)
            assert np.isclose(selected.max(), case["mse_max"], rtol=1e-12, atol=0)
        else:
            assert case["mse_mean"] is None and case["mse_max"] is None
        offset += count
    assert offset == 224
    result = dict(
        status="PASS_RECOMPUTED_BOUNDED_ONLINE_SOURCE",
        device=report["device"],
        job=report["job_id"],
        cases=64,
        frames=224,
        mse_mean=float(errors.mean()),
        mse_sd=float(errors.std(ddof=1)),
        mse_max=float(errors.max()),
        scope=report["scope"],
        performance_claim=False,
        report_sha256=hashlib.sha256((args.input / "result.json").read_bytes()).hexdigest(),
        frames_sha256=hashlib.sha256((args.input / "frames.npz").read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
