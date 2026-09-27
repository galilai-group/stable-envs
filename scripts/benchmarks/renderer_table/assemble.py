"""Assemble a reviewed explicit selection of retained measurement artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .common import METRICS, catalog, distribution, write
from .verify import validate


def timings(records, key, backend, gl):
    values = [r[key] for r in records]
    unavailable = [v for v in values if "status" in v]
    values = [v for v in values if "status" not in v]
    if not values:
        assert len({v["status"] for v in unavailable}) == 1
        return dict(status=unavailable[0]["status"], reason="; ".join(dict.fromkeys(v["reason"] for v in unavailable)))
    raw = [x for v in values for x in v["samples_ms"]]
    result = dict(
        status="MEASURED",
        samples_ms=raw,
        statistics_ms=distribution(raw),
        minimum_warmups=min(v["warmups"] for v in values),
        first_call_per_checkpoint_ms=[v["first_call_ms"] for v in values],
        synchronization=values[0]["synchronization"],
        output_location=values[0]["output_location"],
        includes_reset_step_get_set_state=False,
        execution_device="cuda" if backend == "gpu" else "cpu",
        renderer_backend=backend,
        measured_checkpoints=len(values),
        unavailable_checkpoints=len(unavailable),
    )
    if backend == "original":
        result["renderer_backend"] = records[0]["original_backend"]
        if gl:
            result["renderer_backend"] = "Mesa llvmpipe software OpenGL"
    if backend == "cpu" and gl:
        result["renderer_backend"] = "CPU public renderer (software OpenGL for any plate bake)"
    if gl:
        result.update(
            software_cpu_gl_proven=gl.get("software_cpu_gl_proven", False), gl_renderer=gl.get("gl_renderer")
        )
    return result


def build(selection):
    rows = []
    artifacts = {}

    def read(path):
        p = Path(path)
        artifacts[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())

    for registration, meta in catalog().items():
        chosen = selection[registration]
        if "probe" in chosen:
            probe = read(chosen["probe"])
            rows.append(
                dict(
                    meta,
                    matched_frames=0,
                    shape=None,
                    camera=None,
                    frames=[],
                    timings=probe["cells"],
                    quality={b: probe["cells"][b] for b in ["cpu", "gpu"]},
                    profiles={"cpu": probe["profile"]},
                    noncomparison_evidence=probe,
                    notes=probe["cells"]["cpu"]["reason"],
                )
            )
            continue
        cpu, gpu = read(chosen["cpu"]), read(chosen["gpu"])
        assert cpu["status"] == gpu["status"] == "COMPLETE"
        assert gpu["corpus_result_sha256"] == artifacts[chosen["cpu"]]
        assert cpu["registration"] == gpu["registration"] == registration
        frames = cpu["frames"]
        if not frames:
            cells = {**cpu["unavailable"], **gpu["unavailable"]}
            rows.append(
                dict(
                    meta,
                    matched_frames=0,
                    shape=None,
                    camera=None,
                    frames=[],
                    timings=cells,
                    quality={b: cells[b] for b in ["cpu", "gpu"]},
                    profiles={"cpu": cpu["profile"], "gpu": gpu["profile"]},
                    noncomparison_evidence=cpu["semantic_witness"],
                    notes=cells["original"]["reason"],
                )
            )
            continue
        assert len(frames) == len(gpu["frames"])
        for f, g in zip(frames, gpu["frames"], strict=True):
            assert f["index"] == g["index"]
            f["gpu"] = g
        gl = read(chosen["gl"]) if chosen.get("gl") else None
        cells = {
            "original": timings(frames, "original_timing", "original", gl),
            "cpu": timings(frames, "cpu_timing", "cpu", gl),
            "gpu": timings(gpu["frames"], "gpu_timing", "gpu", None),
        }
        aggregates = {}
        for backend in ["cpu", "gpu"]:
            if cells[backend]["status"] != "MEASURED":
                aggregates[backend] = cells[backend]
            else:
                metrics = [
                    r[backend + "_quality"]
                    for r in (frames if backend == "cpu" else gpu["frames"])
                    if backend + "_quality" in r
                ]
                aggregates[backend] = {k: distribution([m[k] for m in metrics]) for k in METRICS}
        equality = (
            all(f["cpu_quality"]["candidate_sha256"] == f["gpu"]["gpu_quality"]["candidate_sha256"] for f in frames)
            if "status" not in aggregates["cpu"] and all("gpu_quality" in f["gpu"] for f in frames)
            else None
        )
        notes = []
        if meta["name"].startswith("Craftax"):
            notes.append(
                f"GPU matched {sum('gpu_quality' in f['gpu'] for f in frames)}/{len(frames)} CPU checkpoints; native platform signatures differ; no signature or engine field edits"
            )
        if equality is not None:
            notes.append("CPU/GPU RGB exactly equal" if equality else "CPU/GPU RGB differ; MSE kept separate")
        if meta["status"] == "INCOMPLETE":
            notes.append("Measurement does not imply semantic acceptance")
        if meta["name"] == "PyFlytRocketLanding":
            notes.append("Includes retained high-ceiling horizon-600 renderer failure")
        if cpu.get("sampling_limitation"):
            notes.append(cpu["sampling_limitation"]["scope"] + ": " + cpu["sampling_limitation"]["reason"])
        for backend, c in cells.items():
            if c["status"] != "MEASURED":
                notes.append(backend + ": " + c["reason"])
        rows.append(
            dict(
                meta,
                matched_frames=len(frames),
                shape=frames[0]["shape"],
                camera=frames[0]["camera"],
                frames=frames,
                timings=cells,
                quality=aggregates,
                cpu_gpu_rgb_equal=equality,
                profiles={"cpu": cpu["profile"], "gpu": gpu["profile"]},
                recipes=cpu["recipes"],
                sampling_limitation=cpu.get("sampling_limitation"),
                supplemental_gpu_to_cpu=timings(gpu["frames"], "gpu_to_cpu_timing", "gpu", None),
                notes="; ".join(notes),
            )
        )
    return dict(schema=1, metric_definitions=METRICS, registrations=rows, selected_artifact_hashes=artifacts)


def display(row):
    def latency(b):
        c = row["timings"][b]
        return f"{c['statistics_ms']['median']:.5g}" if c["status"] == "MEASURED" else c["status"]

    def mse(b):
        c = row["quality"][b]
        return (
            c["status"] if "status" in c else f"{c['mse_normalized']['mean']:.7g} / {c['mse_normalized']['p95']:.7g}"
        )

    return [
        row["registration"],
        row["status"],
        row["matched_frames"],
        "×".join(map(str, row["shape"][:2])) if row["shape"] else row["timings"]["original"]["status"],
        mse("cpu"),
        mse("gpu"),
        latency("cpu"),
        latency("gpu"),
        latency("original"),
        row["notes"],
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("selection", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = build(json.loads(args.selection.read_text()))
    validate(summary)
    write(args.output / "summary.json", summary)
    columns = [
        "Registration",
        "Status",
        "Matched frames",
        "Resolution H×W",
        "CPU MSE vs original CPU (mean / p95)",
        "GPU MSE vs original CPU (mean / p95)",
        "Ours CPU ms",
        "Ours GPU ms",
        "Original CPU ms",
        "Notes",
    ]
    rows = [display(r) for r in summary["registrations"]]
    with (args.output / "summary.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)
    text = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n"
    text += "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows) + "\n"
    (args.output / "table.md").write_text(text)


if __name__ == "__main__":
    main()
