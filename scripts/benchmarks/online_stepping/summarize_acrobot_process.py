"""Validate the complete Acrobot process sweep and independent CPU/CUDA tests."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("tests", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    tests = ET.parse(args.tests).getroot().findall(".//testcase")
    assert len(tests) == 10 and all(not list(test) for test in tests)
    names = {t.get("name") for t in tests}
    for device, pipeline in itertools.product(("cpu", "cuda"), (False, True)):
        assert f"test_public_chunks_slices_early_stop_and_current_render[{device}-{pipeline}]" in names
    report = json.loads(args.report.read_text())
    assert report["status"] == "PASS" and report["env"] == "swm/AcrobotControl-v1"
    assert len(report["affinity"]) == 8 and "3090" in report["gpu"]
    for rel, digest in report["source_sha256"].items():
        assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
    cells = {}
    for cell in report["timings"]:
        key = tuple(cell[k] for k in ("device", "batch", "horizon", "processes"))
        assert key not in cells and cell["exact_state_task_gate"] == "PASS"
        assert len(cell["seconds"]) == 6 and all(math.isfinite(x) and x > 0 for x in cell["seconds"])
        assert cell["valid_transitions"] == cell["batch"] * cell["horizon"]
        cells[key] = cell
    assert set(cells) == set(itertools.product(("cpu", "cuda"), (64, 256, 1024), (1, 8), (0, 4, 8)))
    rows = []
    for device, batch, horizon, processes in itertools.product(("cpu", "cuda"), (64, 256, 1024), (1, 8), (4, 8)):
        row = dict(env=report["env"], device=device, batch=batch, horizon=horizon, processes=processes,
                   host=report["host"], gpu=report["gpu"], job=report["job"])
        for name, count in (("local", 0), ("process", processes)):
            cell = cells[device, batch, horizon, count]
            rates = [cell["valid_transitions"] / t for t in cell["seconds"]]
            row[name + "_steps_per_s_mean"] = statistics.mean(rates)
            row[name + "_steps_per_s_sd"] = statistics.stdev(rates)
        row["speedup_vs_local"] = row["process_steps_per_s_mean"] / row["local_steps_per_s_mean"]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Validated ten tests without skips and 36 timing cells; wrote", len(rows), "paired rows")


if __name__ == "__main__":
    main()
