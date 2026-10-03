"""Reconcile completed baseline cells without rewriting timed-out raw reports."""

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
    p.add_argument("original", type=Path)
    p.add_argument("missing64", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes, cells = None, {}
    for family in ("Maze", "Cube", "Scene"):
        paths = [args.original / f"{family}.json"]
        if family != "Maze":
            paths += [args.missing64 / f"{family}-{device}.json" for device in ("cpu", "cuda")]
        for path in paths:
            report = json.loads(path.read_text())
            assert report["family"] == family, path
            original = path.parent == args.original
            # Those two original allocations timed out after four complete cells.
            # Their RUNNING status is retained, not relabeled as success.
            assert report["status"] == ("RUNNING" if original and family != "Maze" else "PASS"), path
            expected_cells = set(itertools.product((1, 16, 64) if family == "Maze" else (1, 16), ("cpu", "cuda"))) if original else {(64, path.stem.split("-")[-1])}
            assert {(c["batch"], c["device"]) for c in report["cells"]} == expected_cells, path
            assert len(report["cells"]) == len(expected_cells), path
            if hashes is None:
                hashes = report["source_sha256"]
                for rel, digest in hashes.items():
                    assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
            assert report["source_sha256"] == hashes, path
            for cell in report["cells"]:
                key = family, cell["batch"], cell["device"]
                assert key not in cells and cell["restore_step_replay"] == "PASS", path
                assert set(cell["timings"]) == {"step_with_state", "step_without_export_DIAGNOSTIC", "get_state", "set_state"}, path
                cells[key] = report, cell, path
    assert set(cells) == set(itertools.product(("Maze", "Cube", "Scene"), (1, 16, 64), ("cpu", "cuda")))
    rows = []
    for (family, batch, device), (report, cell, path) in sorted(cells.items()):
        for operation, timing in cell["timings"].items():
            samples = timing["seconds"]
            assert len(samples) == 6 and all(math.isfinite(t) and t > 0 for t in samples), path
            rates = [batch / t for t in samples]
            rows.append(dict(family=family, batch=batch, device=device, operation=operation,
                             seconds_mean=statistics.mean(samples), seconds_sd=statistics.stdev(samples),
                             lanes_per_s_mean=statistics.mean(rates), lanes_per_s_sd=statistics.stdev(rates),
                             state_bytes=cell["state_bytes"], state_leaves=cell["state_leaves"],
                             host=report["host"], gpu=report["gpu"], job=report["job"], report=str(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Validated 18 replay-gated baseline cells; wrote 72 component rows")


if __name__ == "__main__":
    main()
