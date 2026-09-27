"""Fail closed on incomplete, unpaired or mislabeled renderer-table evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from .common import METRICS, ROOT, catalog, distribution, write

UNAVAILABLE = {"UNSUPPORTED", "NONCOMPARABLE", "NOT IMPLEMENTED"}


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), "NaN/Inf"
    elif isinstance(value, dict):
        for v in value.values():
            finite(v)
    elif isinstance(value, list):
        for v in value:
            finite(v)


def cell(value, *, gpu=False, original=False):
    if value.get("status") in UNAVAILABLE:
        assert value.get("reason", "").strip(), "unexplained unavailable cell"
        return
    assert value["status"] == "MEASURED"
    assert len(value["samples_ms"]) >= 100, "fewer than 100 measured calls"
    assert value["minimum_warmups"] >= 10
    assert value["statistics_ms"] == distribution(value["samples_ms"]), "latency aggregation mismatch"
    assert value["includes_reset_step_get_set_state"] is False
    if gpu:
        assert "synchronize" in value["synchronization"]
        assert value["output_location"] == "cuda", "GPU primary must retain native output"
    else:
        assert value["output_location"] == "cpu"
        assert value["execution_device"] == "cpu", "GPU renderer mislabeled CPU"
        if value["renderer_backend"].lower().find("opengl") >= 0:
            assert value["software_cpu_gl_proven"] is True
            assert "llvmpipe" in value["gl_renderer"].lower()


def validate(summary):
    finite(summary)
    assert summary["metric_definitions"] == METRICS, "missing or changed metric definitions"
    rows = summary["registrations"]
    assert len(rows) == 42, "expected exactly 42 rows"
    expected = catalog()
    names = [r["registration"] for r in rows]
    assert len(set(names)) == 42, "duplicate registration"
    assert set(names) == set(expected), "missing or unexpected registration"
    for row in rows:
        assert row["status"] == expected[row["registration"]]["status"]
        for backend in ["cpu", "gpu", "original"]:
            cell(row["timings"][backend], gpu=backend == "gpu", original=backend == "original")
        frames = row["frames"]
        assert row["matched_frames"] == len(frames)
        if not frames:
            assert row["noncomparison_evidence"] and all(
                row["timings"][b]["status"] in UNAVAILABLE for b in row["timings"]
            )
            continue
        assert len(frames) >= 16
        assert {0, 17, 42} <= {f["seed"] for f in frames}
        assert len({f["index"] for f in frames}) == len(frames)
        assert row["profiles"]["cpu"]["host"] == row["profiles"]["gpu"]["host"]
        for f in frames:
            proof = f["proof"]
            assert proof["equal"] is True
            assert proof["candidate_state_hash"] == f["state_hash"], "State/proof pairing"
            if proof["kind"] == "EXACT_NUMERIC_STATE":
                assert proof["candidate_state_hash"] == proof["oracle_state_hash"], "unequal numeric States"
                assert proof["compared_leaves"] > 0
            else:
                assert proof["kind"] == "ACCEPTED_SEMANTIC_WITNESS" and proof["helper"] and proof["compared"]
            assert f["oracle_render_nonmutation"] and f["candidate_render_nonmutation"]
            assert len(f["prefix_hash"]) == 64 and len(f["state_hash"]) == 64
            assert f["camera"] == row["camera"]
            assert f["shape"] == row["shape"]
            if "gpu" in f:
                g = f["gpu"]
                assert g["state_hash"] == f["state_hash"], "GPU State/frame pairing"
                assert g["camera"] == f["camera"], "camera mismatch"
                assert g["shape"] == f["shape"], "resolution mismatch"
                assert g["supplied_and_live_state_nonmutation"]
                if row["name"].startswith("Craftax"):
                    device_replay(g["device_replay_proof"], f, comparable="gpu_quality" in g)
            for backend in ["cpu", "gpu"]:
                metric = f.get("cpu_quality") if backend == "cpu" else f.get("gpu", {}).get("gpu_quality")
                if metric is None:
                    cell_ = f["gpu"]["gpu_timing"] if backend == "gpu" else row["timings"][backend]
                    assert cell_["status"] in UNAVAILABLE and cell_["reason"]
                    continue
                assert set(METRICS) <= metric.keys()
                assert metric["oracle_sha256"] == f["oracle_sha256"], "oracle frame pairing"
                assert metric["shape"] == f["shape"]
                assert len(metric["candidate_sha256"]) == 64
                assert abs(metric["mse_uint8"] / 255**2 - metric["mse_normalized"]) < 1e-12
        for backend, aggregate in row["quality"].items():
            if aggregate.get("status") in UNAVAILABLE:
                assert aggregate["reason"]
                continue
            records = frames if backend == "cpu" else [f["gpu"] for f in frames]
            metrics = [f[backend + "_quality"] for f in records if backend + "_quality" in f]
            for key in METRICS:
                assert aggregate[key] == distribution([m[key] for m in metrics]), "metric aggregation/count mismatch"
    return dict(status="PASS", registrations=42, matrix_unchanged=True)


def device_replay(proof, frame, comparable=True):
    assert proof["kind"] == "EXACT_REPLAY_EXCEPT_PLATFORM_SIGNATURE"
    assert proof["excluded_leaf"] == "meta/signature" and proof["reason"]
    a, b = proof["cpu_leaf_hashes"], proof["gpu_leaf_hashes"]
    assert a.keys() == b.keys() and len(a) > 1
    differences = {k for k in a if k != "meta/signature" and a[k] != b[k]}
    if comparable:
        assert not differences
    else:
        assert differences and differences == set(proof["differences"])
    assert proof["action_prefix_hash"] == frame["prefix_hash"]
    assert hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest() == frame["state_hash"]
    assert hashlib.sha256(json.dumps(b, sort_keys=True).encode()).hexdigest() == proof["actual_gpu_state_hash"]


def preserved():
    checkpoint = json.loads(
        (ROOT / "docs/research/semantic_recovery_evidence/randall_renderer_start/start.json").read_text()
    )
    for name, digest in checkpoint["immutable_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    return len(checkpoint["immutable_hashes"])


def retained(evidence):
    """Bind compact evidence to source State files and exact action-prefix recipes."""
    import numpy as np

    from .common import array_hash, load_state, state_hash

    manifest = json.loads((evidence / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((evidence / name).read_bytes()).hexdigest() == digest, name
    frames = 0
    for folder in evidence.iterdir():
        if not folder.is_dir() or not (folder / "cpu.json").exists():
            continue
        c = json.loads((folder / "cpu.json").read_text())
        g = json.loads((folder / "gpu.json").read_text())
        assert g["corpus_result_sha256"] == hashlib.sha256((folder / "cpu.json").read_bytes()).hexdigest()
        recipes = {r["case"]: r for r in c.get("recipes", [])}
        for f in c["frames"]:
            assert state_hash(load_state(folder / "corpus" / f["state_file"])) == f["state_hash"]
            r = recipes[f["case"]]
            actions = np.load(folder / "corpus" / r["actions_file"], allow_pickle=False)
            assert array_hash(actions) == r["actions_hash"]
            assert array_hash(actions[: f["step"]]) == f["prefix_hash"]
            gpu = g["frames"][f["index"]]
            if "device_replay_proof" in gpu:
                proof = gpu["device_replay_proof"]
                device_replay(proof, f, comparable="gpu_quality" in gpu)
                state = load_state(folder / "corpus" / proof["actual_gpu_state_file"])
                assert state_hash(state) == proof["actual_gpu_state_hash"]
            frames += 1
    return dict(
        files=len(manifest["files"]),
        matched_state_action_pairs=frames,
        independently_recomputed_backend_frames=manifest["independent_frames_recomputed"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--evidence", type=Path)
    args = p.parse_args()
    result = validate(json.loads(args.summary.read_text()))
    summary = json.loads(args.summary.read_text())
    for name, digest in summary["selected_artifact_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    result["preserved_files"] = preserved()
    if args.evidence:
        from .assemble import build

        selection = json.loads((args.evidence / "selection.json").read_text())
        assert summary == build(selection), "summary differs from selected retained measurements"
        result["retained_artifacts"] = retained(args.evidence)
    result["summary_sha256"] = hashlib.sha256(args.summary.read_bytes()).hexdigest()
    write(args.output, result)


if __name__ == "__main__":
    main()
