"""Validate independent production-memory diagnostics and retain every phase."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--array", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    terminal = {row[0]: row for line in (args.results / "terminal.txt").read_text().splitlines()
                if line.strip() for row in [line.split("|")]}
    memory, timing = [], []
    for index, family in enumerate(("Cube", "Scene")):
        report = json.loads((args.results / f"{family}.json").read_text())
        job = terminal[f"{args.array}_{index}"]
        assert job[2:4] == ["COMPLETED", "0:0"]
        assert report["job"] == job[1] and report["host"].split(".")[0] == job[5]
        assert report["status"] == "PASS" and report["family"] == family
        assert report["batch"] == 64 and len(report["affinity"]) == len(report["workers"]) == 8
        assert "3090" in report["gpu"]
        for rel, digest in report["source_sha256"].items():
            assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
        common = dict(family=family, batch=report["batch"], device="cuda", host=report["host"], job=report["job"])
        phases = ["before_construction", "constructed", "initial_state_owned"]
        phases += [f"T{h}_{s}_{phase}" for h, s, phase in itertools.product(
            (1, 8), ("none", "final", "all"), ("result_owned", "outputs_released"))]
        phases += ["closed"]
        assert [s["phase"] for s in report["samples"]] == phases
        for sample in report["samples"]:
            expected_count = 1 if sample["phase"] in ("before_construction", "closed") else 9
            assert len(sample["processes"]) == expected_count
            for metric in ("pss_bytes", "rss_bytes"):
                assert sample["total_" + metric] == sum(p[metric] for p in sample["processes"])
            memory.append(dict(**common, phase=sample["phase"], state_bytes=report["state_bytes"],
                               **{k: v for k, v in sample.items() if k not in ("phase", "processes")}))
        cells = set()
        for cell in report["timings"]:
            key = cell["horizon"], cell["frames"]
            assert key not in cells and cell["exact_replay"] == "PASS"
            cells.add(key)
            expected = report["batch"] * (cell["horizon"] if cell["frames"] == "all" else int(cell["frames"] == "final"))
            assert cell["frame_count"] == expected
            seconds = cell["seconds"]
            assert len(seconds) == 6 and all(math.isfinite(s) and s > 0 for s in seconds)
            row = dict(**common, horizon=cell["horizon"], frames=cell["frames"])
            for label, count in (("steps_per_s", report["batch"] * cell["horizon"]), ("fps", expected)):
                rates = [count / s for s in seconds]
                row[label + "_mean"] = statistics.mean(rates) if count else None
                row[label + "_sd"] = statistics.stdev(rates) if count else None
            timing.append(row)
        assert cells == set(itertools.product((1, 8), ("none", "final", "all")))
    args.output.mkdir(parents=True)
    for name, rows in (("memory", memory), ("pipeline", timing)):
        with (args.output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("Validated both independent jobs; 32 phase snapshots and 12 pipeline cells; no source comparison")


if __name__ == "__main__":
    main()
