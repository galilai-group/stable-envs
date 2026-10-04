"""Recompute public PushT source witnesses and verify their frozen provenance."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    report = json.loads((args.evidence / "result.json").read_text())
    assert report["status"] == "PASS_PUBLIC_STATE_SOURCE_CONTINUATION_CPU_RENDER"
    assert report["source_never_restored"]
    assert report["render_execution"] == "CPU_PYGAME"
    assert not report["registration_promotion"]
    for path, digest in report["production_files"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    assert (
        hashlib.sha256((root / "scripts/research_replay/pusht_public_gate.py").read_bytes()).hexdigest()
        == report["script_sha256"]
    )
    counts, frames, transitions = Counter(), Counter(), Counter()
    observed = set()
    for case in report["cases"]:
        registration = case["registration"]
        identity = (registration, case["case"])
        assert identity not in observed
        observed.add(identity)
        counts[registration] += 1
        for field in ("state", "witness"):
            path = args.evidence / case[field]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case[field + "_sha256"]
        with np.load(args.evidence / case["state"], allow_pickle=False) as state:
            assert (state["recipe/version"] == 2).all()
            assert np.isfinite(state["history/actions"]).all()
            length = state["history/lengths"]
            valid = np.arange(state["history/actions"].shape[1])[None] < length[:, None]
            assert np.array_equal(valid, state["history/valid"])
            assert not state["history/actions"][~valid].any()
            assert all(np.isfinite(state[k]).all() for k in state.files)
        with np.load(args.evidence / case["witness"], allow_pickle=False) as data:
            for key in ("frames", "observations", "task"):
                assert np.array_equal(data["source_" + key], data["actual_" + key]), identity
                assert np.array_equal(data["chunk_source_" + key], data["chunk_actual_" + key]), identity
            assert np.array_equal(data["chunk_source_valid"], data["chunk_actual_valid"])
            assert data["chunk_actual_valid"].shape == (case["batch"], 8)
            n = int(np.prod(data["source_frames"].shape[:2])) + int(np.prod(data["chunk_source_frames"].shape[:2]))
            t = int(np.prod(data["source_frames"].shape[:2])) + int(data["chunk_source_valid"].sum())
            assert n == case["frames"] and t == case["future_transitions"]
            frames[registration] += n
            transitions[registration] += t
    for registration, variant in (("swm/PushT-v1", "continuous"), ("swm/PushT-Discrete-v1", "discrete")):
        expected = {f"{variant}-s{s}-b{b}-h{h}" for s in (0, 17, 42) for b in (1, 2) for h in (0, 100, 1025)}
        expected |= {f"{variant}-shape{a}-{b}" for a in range(8) for b in range(1, 8)}
        assert {name for reg, name in observed if reg == registration} == expected
    result = dict(
        status="PASS_INDEPENDENT_RAW_WITNESS_RECOMPUTATION",
        device=report["device"],
        renderer="CPU_PYGAME",
        cases=dict(counts),
        frames=dict(frames),
        future_transitions=dict(transitions),
        normalized_mse=0.0,
        registration_promotion=False,
    )
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
