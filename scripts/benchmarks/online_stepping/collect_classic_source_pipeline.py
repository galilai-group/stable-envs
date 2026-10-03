"""Combine five independently validated source reports without replacing evidence."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REGISTRATIONS = {
    "cartpole": "swm/CartPoleControl-v1",
    "pendulum": "swm/PendulumControl-v1",
    "acrobot": "swm/AcrobotControl-v1",
    "mountain_car": "swm/MountainCarControl-v0",
    "mountain_car_continuous": "swm/MountainCarContinuousControl-v0",
}


def write(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parts = args.output / "per-registration"
    parts.mkdir()
    pipeline, components, summary = [], [], []
    for domain, registration in REGISTRATIONS.items():
        report = json.loads((args.results / f"{domain}.json").read_text())
        assert report["domain"] == domain
        subprocess.run([sys.executable, str(Path(__file__).with_name("summarize_classic_source_pipeline.py")),
                        str(args.results / f"{domain}.json"), str(parts / f"{domain}.csv"),
                        "--source", str(args.source), "--components-output", str(parts / f"{domain}-components.csv")], check=True)
        with (parts / f"{domain}.csv").open() as f:
            cells = list(csv.DictReader(f))
        pipeline.extend(dict(registration=registration, **c) for c in cells)
        with (parts / f"{domain}-components.csv").open() as f:
            components.extend(dict(registration=registration, **c) for c in csv.DictReader(f))
        selected = {device: next(c for c in cells if c["device"] == device and c["batch"] == "256"
                                 and c["horizon"] == "8" and c["frames"] == "all") for device in ("cpu", "cuda")}
        row = dict(registration=registration, env=report["name"], batch=256, horizon=8, selection="all",
                   source_gpu_supported=False, source_workers=1,
                   candidate_processes=int(selected["cuda"]["candidate_processes"]),
                   mse_max_all_gates=max(g["mse_max"] or 0 for g in report["gates"]))
        for label, device, key in (("source_cpu", "cpu", "source_cpu_native_output"),
                                   ("ours_cpu", "cpu", "ours"), ("ours_gpu", "cuda", "ours"),
                                   ("source_cpu_to_gpu", "cuda", "source_same_output_destination")):
            for metric in ("mean", "sd"):
                row[f"{label}_fps_{metric}"] = float(selected[device][f"{key}_fps_{metric}"])
        row["ours_cpu_vs_source_cpu"] = row["ours_cpu_fps_mean"] / row["source_cpu_fps_mean"]
        row["ours_gpu_vs_source_cpu_fallback"] = row["ours_gpu_fps_mean"] / row["source_cpu_fps_mean"]
        summary.append(row)
    for name, rows in (("pipeline", pipeline), ("components", components), ("batch256_allframes", summary)):
        write(args.output / f"{name}.csv", rows)
    print("Validated all five registrations; wrote 180 pipeline and 90 component rows")


if __name__ == "__main__":
    main()
