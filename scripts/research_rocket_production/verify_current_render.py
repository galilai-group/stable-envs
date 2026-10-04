"""Recompute current public Rocket frame metrics; does not promote semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads((args.input / "result.json").read_text())
    assert report["status"] in ("PUBLIC_CPU_RENDER_PASS", "PUBLIC_GPU_PASS")
    assert len(report["records"]) == 204 and report["per_frame_mse_pass"]
    assert report["oracle_source"]["sha256"] == "5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23"
    assert report["versions"]["PyFlyt"] == "0.29.0" and report["versions"]["pybullet"] == "3.2.7"
    for name, digest in report["production_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    with np.load(args.input / "frames.npz", allow_pickle=False) as frames:
        ours, oracle = frames["ours"], frames["oracle"]
    assert ours.shape == oracle.shape == (204, 480, 480, 3)
    assert ours.dtype == oracle.dtype == np.uint8
    errors = []
    for i, record in enumerate(report["records"]):
        assert all(record[k] for k in ("exact_source_numeric", "independent_restore_exact", "immediate_pixels_exact"))
        delta = (ours[i].astype(np.float64) - oracle[i]) / 255
        error = float(np.mean(delta**2))
        assert error == record["metrics"]["normalized_mse"] and error <= 0.001
        errors.append(error)
    # Recompute the frozen group pixel criteria. The reported SSIM is also
    # recomputed when pixels differ; equality itself proves SSIM=1 otherwise.
    from stable_envs.envs.rocket_landing import PyFlytRocketLanding
    from stable_envs.testing.conformance import ssim

    limits = PyFlytRocketLanding.visual_fidelity
    for label, group in report["groups"].items():
        indices = [i for i, r in enumerate(report["records"]) if r["label"] == label]
        assert len(indices) == group["count"] and group["passed"]
        a, b = ours[indices], oracle[indices]
        delta = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
        mean_abs, within8 = float(delta.mean()), float((delta <= 8).mean())
        score = 1.0 if not np.any(delta) else ssim(a, b)
        assert mean_abs == group["metrics"]["mean_abs"] and within8 == group["metrics"]["within8_frac"]
        assert abs(score - group["metrics"]["ssim"]) < 1e-12
        assert mean_abs <= limits["mean_abs_max"] and within8 >= limits["within8_min"] and score >= limits["ssim_min"]
    result = dict(
        status="PASS_RECOMPUTED_PUBLIC_RENDER_WITNESSES",
        device=report["device"],
        job=report["job_id"],
        checkpoints=204,
        mse_mean=float(np.mean(errors)),
        mse_sd=float(np.std(errors, ddof=1)),
        mse_max=max(errors),
        groups=len(report["groups"]),
        production_files=len(report["production_hashes"]),
        public_semantic_promotion=False,
        scope="Current public render/restore gate only; companion Variation/suffix/unit and other-device gates remain separate.",
        evidence_hashes={
            name: hashlib.sha256((args.input / name).read_bytes()).hexdigest()
            for name in ("result.json", "frames.npz")
        },
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
