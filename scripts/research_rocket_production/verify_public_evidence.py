"""Recompute the public Rocket visual decision from immutable downloaded frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from stable_envs.envs.rocket_landing import PyFlytRocketLanding
from stable_envs.testing.conformance import pixel_agreement, ssim

ROOT = Path(__file__).resolve().parents[2]
START = "ccb997c51f7faa56036b0502c50f233647754f52"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_equal(actual, expected):
    return (
        actual is expected
        if actual is None or expected is None
        else bool(np.isclose(actual, expected, rtol=0, atol=1e-12))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh output file; never overwrite sealed evidence."
    submission = json.loads(args.submission.read_text())
    report_path = args.evidence / "results/result.json"
    report = json.loads(report_path.read_text())
    assert report["status"] in {"PUBLIC_GPU_PASS", "FROZEN_RENDERER_ACCEPTANCE_BLOCKER"}
    assert report["production_hashes"] == submission["production_hashes"]
    for rel, sha in report["production_hashes"].items():
        assert digest(ROOT / rel) == sha, rel
    frozen_path = "stable_envs/envs/rocket_landing.py"
    assert (
        subprocess.check_output(["git", "show", START + ":" + frozen_path], cwd=ROOT)
        == (ROOT / frozen_path).read_bytes()
    )
    assert report["oracle_source"]["sha256"] == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    assert report["versions"]["PyFlyt"] == "0.29.0"
    assert report["versions"]["pybullet"] == "3.2.7"
    records = report["records"]
    assert len(records) == 204
    assert {r["seed"] for r in records} == {0, 17, 42}
    assert {r["dtype"] for r in records} == {"float32", "float64"}
    assert all(
        r[k] for r in records for k in ["exact_source_numeric", "independent_restore_exact", "immediate_pixels_exact"]
    )
    frame_path = args.evidence / "results/frames.npz"
    with np.load(frame_path, allow_pickle=False) as frames:
        ours, oracle = frames["ours"], frames["oracle"]
    assert ours.shape == oracle.shape == (204, 480, 480, 3)
    assert ours.dtype == oracle.dtype == np.uint8
    limits = {k: PyFlytRocketLanding.visual_fidelity[k] for k in ["mean_abs_max", "within8_min", "ssim_min"]}
    per_frame = []
    for i, row in enumerate(records):
        metrics = pixel_agreement(ours[i : i + 1], oracle[i : i + 1])
        metrics["ssim"] = ssim(ours[i : i + 1], oracle[i : i + 1])
        for key, value in metrics.items():
            assert metric_equal(value, row["metrics"][key]), (i, key)
        delta = ours[i].astype(np.float64) - oracle[i].astype(np.float64)
        raw_mse = float(np.mean(delta * delta))
        per_frame.append(
            dict(
                index=i,
                label=row["label"],
                seed=row["seed"],
                dtype=row["dtype"],
                mse_uint8=raw_mse,
                mse_normalized=raw_mse / 255**2,
                mae_rgb_uint8=float(np.mean(np.abs(delta))),
                metrics=metrics,
                ours_sha256=hashlib.sha256(ours[i].tobytes()).hexdigest(),
                oracle_sha256=hashlib.sha256(oracle[i].tobytes()).hexdigest(),
            )
        )
    groups = {}
    for label in sorted({r["label"] for r in records}):
        idx = [i for i, r in enumerate(records) if r["label"] == label]
        metrics = pixel_agreement(ours[idx], oracle[idx])
        metrics["ssim"] = ssim(ours[idx], oracle[idx])
        for key, value in metrics.items():
            assert metric_equal(value, report["groups"][label]["metrics"][key]), (label, key)
        passed = (
            metrics["mean_abs"] <= limits["mean_abs_max"]
            and metrics["within8_frac"] >= limits["within8_min"]
            and metrics["ssim"] >= limits["ssim_min"]
        )
        assert passed == report["groups"][label]["passed"]
        groups[label] = dict(count=len(idx), metrics=metrics, passed=passed)
    all_pass = all(g["passed"] for g in groups.values())
    assert all_pass == (report["status"] == "PUBLIC_GPU_PASS")
    terminal = json.loads((args.evidence / "exit.json").read_text())
    if all_pass:
        assert terminal["exit_code"] == 0
        for name, status, count in [
            ("persistent-variation", "PERSISTENT_VARIATION_PASS", 90),
            ("source-suffixes", "INDEPENDENT_SOURCE_SUFFIXES_PASS", 108),
        ]:
            extra = json.loads((args.evidence / name / "result.json").read_text())
            assert extra["status"] == status and len(extra["records"]) == count
            for rel, sha in extra["production_hashes"].items():
                assert sha == submission["production_hashes"][rel]
    else:
        assert terminal["exit_code"] == 2
    mse = np.array([r["mse_normalized"] for r in per_frame])
    result = dict(
        status="PUBLIC_GATE_INTEGRITY_VERIFIED" if all_pass else "RENDERER_BLOCKER_VERIFIED",
        source_commit=submission["commit"],
        job_id=report["job_id"],
        frozen_limits=limits,
        reports_sha256={str(p.relative_to(args.evidence)): digest(p) for p in [report_path, frame_path]},
        per_frame=per_frame,
        groups=groups,
        normalized_mse=dict(
            mean=float(mse.mean()),
            median=float(np.median(mse)),
            p95=float(np.percentile(mse, 95)),
            maximum=float(mse.max()),
        ),
        promotion=False,
        scope="Independent artifact verification. Performance and repository closure remain separate gates.",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"], len(records), "frames", len(groups), "groups")


if __name__ == "__main__":
    main()
