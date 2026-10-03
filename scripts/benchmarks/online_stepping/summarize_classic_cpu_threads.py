"""Validate the full CPU thread sweep and retain every configuration's rates."""

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    assert report["status"] == "PASS" and report["device"] == "cpu" and len(report["affinity"]) == 8
    for rel, digest in report["source_sha256"].items():
        assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
    ops = [("render_saved", 0, None)] + [("online", h, s) for h in (1, 8) for s in ("none", "final", "all")]
    expected = {(b, *op, t) for b in (64, 256, 1024) for op in ops for t in (1, 2, 4, 8)}
    keys = ("batch", "operation", "horizon", "selection", "threads")
    gates = {}
    for gate in report["gates"]:
        key = tuple(gate[k] for k in keys)
        assert key not in gates and gate["exact_state_task_frame"] == "PASS"
        gates[key] = gate
    assert set(gates) == expected
    assert set(report["profiles"]) == {"1", "8"}
    cells = {}
    for cell in report["timings"]:
        key = tuple(cell[k] for k in keys)
        assert key not in cells and len(cell["samples"]) == 6
        assert cell["workers"] == (8 if report["domain"] == "acrobot" else 0)
        assert len(set(cell["pids"])) == cell["workers"] + 1
        count = cell["batch"] * (cell["horizon"] if cell["operation"] == "online" else 1)
        frames = (cell["batch"] if cell["operation"] == "render_saved" else
                  cell["batch"] * (cell["horizon"] if cell["selection"] == "all" else int(cell["selection"] == "final")))
        assert cell["count"] == count and cell["frame_count"] == frames
        for sample in cell["samples"]:
            assert sample["seconds"] > 0 and math.isfinite(sample["seconds"])
            assert set(sample["cpu_seconds"]) == {str(pid) for pid in cell["pids"]}
            assert all(v >= 0 and math.isfinite(v) for v in sample["cpu_seconds"].values())
            assert sample["average_busy_cores"] == sum(sample["cpu_seconds"].values()) / sample["seconds"]
        cells[key] = cell
    assert set(cells) == expected
    rows = []
    for key, cell in cells.items():
        seconds = [s["seconds"] for s in cell["samples"]]
        rates = [cell["count"] / s for s in seconds]
        baseline = cells[(*key[:-1], 1)]
        baseline_rate = statistics.mean(baseline["count"] / s["seconds"] for s in baseline["samples"])
        utilization = [s["average_busy_cores"] for s in cell["samples"]]
        rows.append(dict(domain=report["domain"], host=report["host"], job=report["job"],
            **{k: cell[k] for k in keys}, workers=cell["workers"],
            rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates),
            rate_unit="frames/s" if cell["operation"] == "render_saved" else "steps/s",
            fps_mean=statistics.mean(cell["frame_count"] / s for s in seconds) if cell["frame_count"] else None,
            fps_sd=statistics.stdev(cell["frame_count"] / s for s in seconds) if cell["frame_count"] else None,
            average_busy_cores_mean=statistics.mean(utilization), average_busy_cores_sd=statistics.stdev(utilization),
            ratio_vs_one_thread=statistics.mean(rates) / baseline_rate,
            exact_state_task_frame="PASS"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("Validated", report["domain"], "84 thread-sweep rows including CPU counters and exact gates")


if __name__ == "__main__":
    main()
