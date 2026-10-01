"""Rebuild only this checkpoint's summary; accepted earlier evidence is untouched."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
repo = ROOT.parents[3]
rows = []
for kind in ("cpu", "gpu"):
    data = json.loads((ROOT / "runs/v3" / f"fetch-{kind}.json").read_text())
    assert data["status"] == "PASS"
    assert len([g for g in data["gates"] if g.get("state_task_thread_and_endpoint") == "PASS"]) == 12
    for timing in data["timings"]:
        frames = timing.get("frames_per_second", [])
        rows.append(
            dict(
                env="FetchPush",
                simulator_device="cpu",
                output_device=data["device"],
                batch=timing["batch"],
                horizon=8,
                workers=timing["workers"],
                stage=timing["stage"],
                unit=timing["unit"],
                rate_mean=timing["rate_mean"],
                rate_sd=timing["rate_sd"],
                frames_s_mean=statistics.mean(frames) if frames else "",
                frames_s_sd=statistics.stdev(frames) if frames else "",
                resolution="64x64",
                render_backend="opengl",
                repeats=timing["repeats"],
            )
        )
with (ROOT / "fetch_execution.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
for kind in ("cpu", "gpu"):
    accepted = json.loads((ROOT / "runs/v4" / f"fetch-{kind}.json").read_text())
    assert accepted["status"] == "PASS"
    gates = [g for g in accepted["gates"] if "default_workers" in g]
    assert len(gates) == 12 and all(g["batch"] == 64 and g["default_workers"] == 4 for g in gates)
manifest = json.loads((ROOT / "runs/v4/manifest.json").read_text())
production = {p: h for p, h in manifest.items() if p.startswith("stable_envs/")}
for path, expected in production.items():
    assert hashlib.sha256((repo / path).read_bytes()).hexdigest() == expected, path
summary = dict(
    baseline_commit="d20190261bec77f2ad68dd94ed4403f86841344b",
    production_matches_gpu_stage=True,
    jobs={"fetch_timing": 6911735, "native_gpu": 6911954, "automatic_worker_acceptance": 6912594},
    local_api="215 passed, 4 CUDA-only skips; those 4 passed on Oscar",
    local_oracle="50 passed (includes 12 DMC/Reacher oracle cases and API/State checks)",
    oscar_api="51 passed, 1 optional Matplotlib skip on final source",
    semantic_harness="2 passed",
    wheel="Built in a fresh source directory; installed-wheel API smoke passed using existing dependencies",
    fetch_registrations_independently_checked=12,
    acceptance_batch=64,
    default_fetch_workers=4,
    semantic_matrix="39 VERIFIED / 3 INCOMPLETE; unchanged",
    fetch_capture_mse_vs_repeated_single_step=0.0,
    native_gpu_capture_checks=["CartPole", "Pendulum", "TwoRoom", "Piecewise"],
    dependency_versions={
        "torch": "2.14.0+cu126",
        "numpy": "2.5.3",
        "mujoco": "3.11.0",
        "gymnasium": "1.3.0",
        "gymnasium-robotics": "1.4.2",
    },
    limitations=[
        "Known-action chunks, not an image-policy loop or the 42-row advisor renderer table.",
        "Rendering timings use the existing OpenGL backend at 64x64.",
        "Large Fetch batches select up to four persistent workers; other families retain existing executors.",
        "No process executor, producer/consumer overlap, or bounded VRAM queue is promoted.",
        "Capture MSE compares new sequence capture to existing single-step rendering, not a new upstream image oracle.",
        "Broad historical renderer-State tests already fail on the baseline; representative reproductions are retained.",
    ],
)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
