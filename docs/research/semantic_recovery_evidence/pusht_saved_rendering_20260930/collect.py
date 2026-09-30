"""Validate completed PushT renderer evidence and export the two-row comparison."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent
rows = []
for index, registration in enumerate(["swm/PushT-v1", "swm/PushT-Discrete-v1"]):
    run = root / f"run-{index}"
    meta = json.loads((run / "corpus/manifest.json").read_text())
    assert meta["registration"] == registration
    assert meta["count"] == 1024 and meta["distinct_poses"] >= 1000
    assert hashlib.sha256((run / "corpus/corpus.npz").read_bytes()).hexdigest() == meta["corpus_sha256"]
    record = dict(
        env=registration,
        frames=meta["count"],
        distinct_poses=meta["distinct_poses"],
        batch=256,
        output="cpu",
        scope="Renderer only; default appearance; full State/dynamics INCOMPLETE",
    )
    for device in ["cpu", "cuda"]:
        result = json.loads((run / device / "result.json").read_text())
        assert result["status"] == "COMPLETE" and result["renderer_nonmutation_passed"]
        assert result["semantic_state_acceptance"] == "INCOMPLETE"
        assert result["registration"] == registration and result["device"] == device
        assert result["source_render_device"] == "cpu" and result["source_gpu_availability"] == "CPU_ONLY"
        assert result["corpus_sha256"] == meta["corpus_sha256"]
        assert result["source_hashes"] == json.loads((root / "oracle-sha256.json").read_text())
        expected = {
            hashlib.sha256((root / f"harness/pusht_saved-{version}.py").read_bytes()).hexdigest()
            for version in ["v2", "v3"]
        }
        assert result["harness_sha256"] in expected
        assert len(result["frame_mse"]) == 1024
        if index == 1 and device == "cuda":
            original_quality = json.loads((root / "discrete-v2-partial/result-before-cancel.json").read_text())
            assert result["frame_mse"] == original_quality["frame_mse"], "v3 changed a per-frame MSE"
        assert abs(statistics.mean(result["frame_mse"]) - result["paired_mse"]["mean"]) < 1e-15
        assert abs(statistics.stdev(result["frame_mse"]) - result["paired_mse"]["sd"]) < 1e-15
        for measurement in result["measurements"]:
            assert measurement["frames_per_repeat"] >= 1024
            for side in ["ours", "source"]:
                timing = measurement[side]
                assert timing["count"] == 7 and len(timing["samples_ms"]) == 7
                assert (
                    abs(
                        timing["frames_per_second"]
                        - 1000 * measurement["frames_per_repeat"] / statistics.mean(timing["samples_ms"])
                    )
                    < 1e-7
                )
        m = next(m for m in result["measurements"] if m["batch"] == 256 and m["output_device"] == "cpu")
        label = "gpu" if device == "cuda" else "cpu"
        record[f"{label}_host"] = result["profile"]["host"]
        record[f"{label}_hardware"] = result["profile"]["gpu" if device == "cuda" else "cpu"]
        for k in ["mean", "sd", "max"]:
            record[f"{label}_mse_{k}"] = result["paired_mse"][k]
        for side in ["ours", "source"]:
            for k in ["frames_per_second", "fps_sd", "mean_ms_per_frame", "sd_ms_per_frame"]:
                record[f"{label}_{side}_{k}"] = m[side][k]
        record[f"{label}_speedup_vs_cpu_source"] = m["speedup"]
        if device == "cuda":
            best = max(
                (m for m in result["measurements"] if m["output_device"] == "cpu"),
                key=lambda m: m["ours"]["frames_per_second"],
            )
            record["best_gpu_batch_cpu_output"] = best["batch"]
            record["best_gpu_fps_cpu_output"] = best["ours"]["frames_per_second"]
    record["source_gpu_fps"] = ""
    record["source_gpu_status"] = "UNAVAILABLE: pinned source uses CPU Pygame"
    rows.append(record)
with (root / "pusht_comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "validation.json").write_text(
    json.dumps(
        dict(
            status="PASS", registrations=[r["env"] for r in rows], semantic_promotions=0, frames_per_registration=1024
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(rows, indent=2))

panels = []
for index, row in enumerate(rows):
    for device in ["cpu", "cuda"]:
        result = json.loads((root / f"run-{index}/{device}/result.json").read_text())
        worst = result["worst_frame"]
        panels.append(
            f"<article><h2>{row['env']} — {device}</h2>"
            f"<p>Worst frame: seed {worst['seed']}, step {worst['step']}; "
            f"normalized MSE {worst['mse']:.9f}. Source left, ours right.</p>"
            f'<a href="run-{index}/{device}/worst-pair.png">'
            f'<img src="run-{index}/{device}/worst-pair.png" width="448" height="224"></a></article>'
        )
(root / "images.html").write_text(
    '<!doctype html><html lang="en"><meta charset="utf-8"><title>PushT matched-source images</title>'
    "<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:16px;}"
    "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(448px,1fr));gap:20px;}"
    "h2{font-size:18px}img{max-width:100%;height:auto}</style>"
    "<h1>PushT matched-source rendering</h1><p>1,024 source frames per registration; "
    "default appearance, 224×224. Renderer-only evidence; full State/dynamics remain incomplete.</p><main>"
    + "".join(panels)
    + "</main></html>\n"
)
