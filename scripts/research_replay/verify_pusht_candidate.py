"""Verify private PushT evidence hashes and raw numeric witnesses without engines."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cohorts = {
        "pusht_replay": (72, "PASS_BOUNDED_SEEDED_REPLAY", "pusht_replay_gate.py"),
        "pusht_reset_anchors": (24, "PASS_BOUNDED_RESET_ANCHORS", "pusht_reset_anchor_gate.py"),
        "pusht_task": (96, "PASS_PRIVATE_TASK_MIGRATION", "pusht_task_gate.py"),
        "pusht_adapter": (72, "PASS_PRIVATE_REPLAY_ADAPTER_SOURCE_GATE", "pusht_adapter_gate.py"),
        "pusht_adapter_variation": (
            112,
            "PASS_PRIVATE_REPLAY_SHAPE_AND_RESET_GATE",
            "pusht_adapter_variation_gate.py",
        ),
    }
    summaries = {}
    hashes = {}
    for cohort, (count, status, script) in cohorts.items():
        directory = args.evidence / cohort
        report_path = directory / "result.json"
        report = json.loads(report_path.read_text())
        assert report["status"] == status and len(report["cases"]) == count
        assert not report.get("public_acceptance", False)
        if "script_sha256" in report:
            assert digest(Path(__file__).with_name(script)) == report["script_sha256"]
        for path, expected in report.get("production_files", {}).items():
            assert digest(ROOT / path) == expected, path
        hashes[str(report_path.relative_to(args.evidence))] = digest(report_path)
        frame_pairs = transitions = 0
        for case in report["cases"]:
            path = directory / case["artifact"]
            expected = case.get("artifact_sha256")
            if cohort == "pusht_task":
                path = args.evidence / case["cohort"] / case["artifact"]
                expected = case["source_artifact_sha256"]
            assert digest(path) == expected, path
            hashes[str(path.relative_to(args.evidence))] = expected
            with np.load(path, allow_pickle=False) as raw:
                assert all(raw[key].dtype.kind in "biuf" for key in raw)
                other = "dest" if "dest_frames" in raw else "replay"
                for field in ("frames", "observations", "tasks"):
                    if f"{other}_{field}" in raw:
                        np.testing.assert_array_equal(raw[f"source_{field}"], raw[f"{other}_{field}"], strict=True)
                frames = raw["source_frames"]
                frame_pairs += int(np.prod(frames.shape[:-3]))
                transitions += int(np.prod(raw["source_observations"].shape[:-1]))
            for key, hash_key in (("anchor", "anchor_sha256"), ("state_artifact", "state_sha256")):
                if key not in case:
                    continue
                file = directory / case[key]
                assert digest(file) == case[hash_key]
                hashes[str(file.relative_to(args.evidence))] = case[hash_key]
                if key == "state_artifact":
                    with np.load(file, allow_pickle=False) as state:
                        assert all(state[k].dtype.kind in "biuf" and np.isfinite(state[k]).all() for k in state)
                        actions = state["history/actions"]
                        lengths = state["history/lengths"]
                        valid = state["history/valid"]
                        assert actions.shape == (case["batch"], case["max_prefix"], 2)
                        np.testing.assert_array_equal(valid, np.arange(actions.shape[1])[None] < lengths[:, None])
                        assert not np.any(actions[~valid])
                        assert not np.any(state["history/dtypes"][~valid])
        summaries[cohort] = dict(
            cases=count,
            raw_frame_pairs=frame_pairs,
            raw_transition_pairs=transitions,
            report_sha256=digest(report_path),
        )
    adapter = json.loads((args.evidence / "pusht_adapter/result.json").read_text())
    assert {(c["variant"], c["batch"], c["seed"], c["max_prefix"]) for c in adapter["cases"]} == set(
        itertools.product(("continuous", "discrete"), (1, 16), (0, 17, 42), (0, 1, 100, 300, 1025, 10240))
    )
    variation = json.loads((args.evidence / "pusht_adapter_variation/result.json").read_text())
    assert {(c["variant"], c["agent_shape"], c["block_shape"]) for c in variation["cases"]} == set(
        itertools.product(("continuous", "discrete"), range(8), range(1, 8))
    )
    migration = json.loads((args.evidence / "pusht_task/result.json").read_text())
    assert migration["quantizer_centers"] == 256
    result = dict(
        status="PASS_RAW_PRIVATE_PUSHT_EVIDENCE",
        public_acceptance=False,
        gpu_renderer_acceptance=False,
        cohorts=summaries,
        artifacts=hashes,
        verifier_sha256=digest(Path(__file__)),
        note="Migration reuses replay/reset witnesses; counts across cohorts overlap. Shape-gate task equality is asserted live; its raw file records source tasks only.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(status=result["status"], cohorts=summaries), indent=2))


if __name__ == "__main__":
    main()
