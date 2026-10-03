"""Require exact independent-process outputs and all retained slow-cell timings."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

CELLS = (
    ("mountain_car_continuous", "cpu", 256, 1, "none", "final"),
    ("mountain_car_continuous", "cpu", 1024, 8, "all", "final"),
    ("mountain_car_continuous", "cuda", 1024, 8, "all", "final"),
    ("acrobot", "cpu", 256, 8, "all", "none"),
    ("acrobot", "cuda", 1024, 8, "all", "none"),
    ("acrobot", "cuda", 1024, 1, "all", "final"),
    ("acrobot", "cpu", 64, 1, "all", "none"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected = CELLS[args.cell]
    reports = {}
    fields = ("domain", "device", "batch", "horizon", "states", "frames")
    row = dict(zip(fields, expected))
    for mode in ("baseline", "candidate"):
        r = json.loads((args.results / f"{mode}.json").read_text())
        assert r["status"] == "PASS" and r["mode"] == mode and r["cell"] == args.cell and not r["smoke"]
        assert tuple(r[k] for k in fields) == expected
        assert len(r["affinity"]) == 8 and r["gc_enabled"]
        assert "3090" in r["gpu"] if r["device"] == "cuda" else r["gpu"] is None
        assert r["workers"] == (8 if r["domain"] == "acrobot" else 0)
        assert len(r["windows"]) == 6
        for rel, h in r["source_sha256"].items():
            assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == h, rel
        rates = []
        for index, window in enumerate(r["windows"]):
            assert window["index"] == index and window["exact_output"] == "PASS"
            assert len(window["seconds"]) == len(window["restore_seconds"]) == 32
            assert all(t > 0 and math.isfinite(t) for t in [*window["seconds"], *window["restore_seconds"]])
            assert window["valid_steps"] == r["batch"] * r["horizon"] * 32
            assert window["emitted_frames"] == (r["batch"] * 32 if r["frames"] == "final" else 0)
            rates.append(window["valid_steps"] / sum(window["seconds"]))
        row[f"{mode}_steps_per_s_mean"] = statistics.mean(rates)
        row[f"{mode}_steps_per_s_sd"] = statistics.stdev(rates)
        reports[mode] = r
    for key in ("host", "job", "affinity", "versions", "source_sha256", "gc_thresholds", "initial_sha256", "output_sha256"):
        assert reports["baseline"][key] == reports["candidate"][key], key
    assert reports["baseline"]["pid"] != reports["candidate"]["pid"]
    row.update(host=reports["baseline"]["host"], job=reports["baseline"]["job"],
               calls_per_strategy=192, exact_output="PASS",
               speedup_vs_baseline=row["candidate_steps_per_s_mean"] / row["baseline_steps_per_s_mean"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print("Validated retained cell", args.cell, "ratio", row["speedup_vs_baseline"])


if __name__ == "__main__":
    main()
