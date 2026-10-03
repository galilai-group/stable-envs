"""Validate every sustained baseline/candidate pair without dropping samples."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

CELLS = ((1, "none", "final"), (8, "all", "final"), (8, "none", "none"), (8, "none", "final"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, reference, pids = [], None, set()
    for horizon, states, frames in CELLS:
        reports = {}
        for mode in ("baseline", "candidate"):
            report = json.loads((args.results / f"t{horizon}-{states}-{frames}-{mode}.json").read_text())
            assert report["status"] == "PASS" and report["mode"] == mode
            assert report["batch"] == args.batch and report["transitions_per_lane"] == 128
            assert report["horizon"] == horizon and report["states"] == states and report["frames"] == frames
            assert report["device"] == args.device
            assert "3090" in report["gpu"] if args.device == "cuda" else report["gpu"] is None
            assert len(report["affinity"]) == 8 and report["gc_enabled"]
            assert report["domain"] in ("pendulum", "acrobot")
            assert report["workers"] == (8 if report["domain"] == "acrobot" else 0)
            assert report["renderer_backend"] == "auto"
            assert report["pid"] not in pids
            pids.add(report["pid"])
            if reference is None:
                reference = report
                for rel, expected in report["source_sha256"].items():
                    assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == expected, rel
            for key in ("host", "job", "affinity", "gpu", "domain", "versions", "source_sha256", "gc_thresholds"):
                assert report[key] == reference[key], key
            samples = report["samples"]
            assert len(samples) == 6 and [s["index"] for s in samples] == list(range(6))
            assert len(report["trajectory_sha256"]) == 128 // horizon
            for sample in samples:
                assert sample["seconds"] > 0 and math.isfinite(sample["seconds"])
                assert sample["reset_seconds"] > 0 and math.isfinite(sample["reset_seconds"])
                assert sample["valid_steps"] == args.batch * 128 and sample["calls"] == 128 // horizon
                assert sample["frames"] == (args.batch * 128 // horizon if frames == "final" else 0)
                assert sample["steps_per_s"] == sample["valid_steps"] / sample["seconds"]
                assert sample["fps"] == (sample["frames"] / sample["seconds"] if frames == "final" else None)
                assert sample["final_state_sha256"] == report["gate_final_state_sha256"]
                assert len(sample["gc_collections"]) == 3 and min(sample["gc_collections"]) >= 0
            reports[mode] = report
        assert reports["baseline"]["trajectory_sha256"] == reports["candidate"]["trajectory_sha256"]
        assert reports["baseline"]["gate_final_state_sha256"] == reports["candidate"]["gate_final_state_sha256"]
        row = dict(domain=reference["domain"], device=args.device, host=reference["host"], job=reference["job"],
                   batch=args.batch, horizon=horizon, states=states, frames=frames,
                   transitions_per_window=128, samples=6, exact_trajectory="PASS")
        for mode, report in reports.items():
            samples = report["samples"]
            rates = [s["steps_per_s"] for s in samples]
            row[f"{mode}_steps_per_s_mean"] = statistics.mean(rates)
            row[f"{mode}_steps_per_s_sd"] = statistics.stdev(rates)
            row[f"{mode}_fps_mean"] = statistics.mean(s["fps"] for s in samples) if frames == "final" else None
            row[f"{mode}_fps_sd"] = statistics.stdev(s["fps"] for s in samples) if frames == "final" else None
            row[f"{mode}_reset_seconds_mean"] = statistics.mean(s["reset_seconds"] for s in samples)
            row[f"{mode}_generation2_collections"] = sum(s["gc_collections"][2] for s in samples)
        row["speedup_vs_baseline"] = row["candidate_steps_per_s_mean"] / row["baseline_steps_per_s_mean"]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("Validated", reference["domain"], "four sustained pairs; every trajectory exact")


if __name__ == "__main__":
    main()
