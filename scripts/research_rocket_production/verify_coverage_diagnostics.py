"""Independently recompute retained Rocket pixel diagnostics and script hashes."""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np


def verify(root):
    reports = []
    sequence = root / "rocket-sequence-v1-20261004/results/6995655/probe/witness.npz"
    source = np.load(sequence, allow_pickle=False)["source"]
    for path in sorted(root.rglob("result.json")):
        data = json.loads(path.read_text())
        if "trials" not in data:
            continue
        run = path.parent.parent.parent.parent
        manifest = json.loads((run / "patch_manifest.json").read_text())
        with tarfile.open(run / "patch.tar.gz") as archive:
            for name, digest in manifest["files"].items():
                assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest
        assert data["script_sha256"] in manifest["files"].values()
        assert data["input_sha256"] == hashlib.sha256(sequence.read_bytes()).hexdigest()
        with np.load(path.parent / "witness.npz", allow_pickle=False) as raw:
            np.testing.assert_array_equal(source, raw["source"])
            for trial in data["trials"]:
                frames = raw[trial["kind"]]
                assert frames.shape == source.shape and frames.dtype == np.uint8
                mse = np.sum((frames.astype(np.int64) - source) ** 2, axis=(-3, -2, -1)) / (
                    255**2 * np.prod(source.shape[-3:])
                )
                np.testing.assert_allclose(mse, trial["mse"], rtol=1e-14, atol=0)
                reports.append(
                    dict(job=data["job"], kind=trial["kind"], frames=int(mse.size), max_mse=float(mse.max()))
                )
    assert len(reports) == 10
    with np.load(sequence, allow_pickle=False) as raw:
        np.testing.assert_array_equal(raw["source"], raw["cpu"])
        for name in ("render_restored", "render_chunk1", "single", "reversed"):
            np.testing.assert_array_equal(raw["online"], raw[name])
    a = np.load(root / "rocket-exhaustive-v1-20261004/results/6995863/probe/witness.npz", allow_pickle=False)
    b = np.load(root / "rocket-triangle-v1-20261004/results/6995878/probe/witness.npz", allow_pickle=False)
    np.testing.assert_array_equal(a["exhaustive_coverage"], b["triangle_coverage"])
    return dict(
        status="VERIFIED_DIAGNOSTICS_NOT_ACCEPTANCE",
        recomputed_frame_comparisons=sum(r["frames"] for r in reports),
        reports=reports,
    )


def verify_online_prototypes(root):
    results = []
    for version, job in [("v5", "6995956"), ("v6", "6995996")]:
        run = root / f"rocket-triangle-gate-{version}-20261004"
        path = run / "results" / job / "probe/online"
        report = json.loads((path / "result.json").read_text())
        assert report["status"] == "PASS_BOUNDED_ROCKET_ONLINE_SOURCE"
        assert len(report["cases"]) == 64 and report["frames"] == 224
        with tarfile.open(run / "patch.tar.gz") as archive:
            script = archive.extractfile("scripts/research_rocket_production/online_source_gate.py").read()
        assert hashlib.sha256(script).hexdigest() == report["script_sha256"]
        with np.load(path / "frames.npz", allow_pickle=False) as raw:
            assert raw["ours"].shape == raw["source"].shape == (224, 96, 96, 3)
            assert raw["ours"].dtype == raw["source"].dtype == np.uint8
            errors = np.sum((raw["ours"].astype(np.int64) - raw["source"]) ** 2, axis=(1, 2, 3)) / (255**2 * 96**2 * 3)
        assert np.all(errors <= 0.001)
        offset = 0
        for case in report["cases"]:
            assert case["source_never_restored"] and case["exact_state_task_lifecycle"] and case["owned_snapshots"]
            count = case["frames"]
            if count:
                part = errors[offset : offset + count]
                np.testing.assert_allclose(
                    [part.mean(), part.max()], [case["mse_mean"], case["mse_max"]], rtol=1e-12, atol=0
                )
            offset += count
        assert offset == 224
        results.append(
            dict(job=job, cases=64, frames=224, mse_max=float(errors.max()), scope="PROTOTYPE_NOT_ACCEPTANCE")
        )
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    args = p.parse_args()
    result = verify(args.root)
    prototypes = args.root.parent / "passing-prototypes"
    if prototypes.exists():
        result["online_prototypes"] = verify_online_prototypes(prototypes)
    print(json.dumps(result, indent=2))
