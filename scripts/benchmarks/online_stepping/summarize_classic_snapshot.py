"""Validate the complete paired classic snapshot sweep before writing its table."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def rates(seconds, count):
    assert len(seconds) == 6 and all(math.isfinite(s) and s > 0 for s in seconds)
    values = [count / s for s in seconds]
    return statistics.mean(values), statistics.stdev(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    assert report["status"] == "PASS"
    assert len(report["affinity"]) == 8 and "3090" in report["gpu"]
    for rel, digest in report["source_sha256"].items():
        assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
    gates = {}
    for gate in report["gates"]:
        assert gate["device"] not in gates
        gates[gate["device"]] = gate
        assert gate["cases"] == 14
        assert (gate["batch"], gate["workers"]) == ((64, 8) if report["domain"] == "acrobot" else (4, 0))
        for key in ("complete_state_task_rng_lifecycle_ownership", "reset_variation_device_actions", "current_render"):
            assert gate[key] == "PASS"
        assert gate["frame_mse_vs_baseline"] == 0
    assert set(gates) == {"cpu", "cuda"}
    cells = {}
    for cell in report["timings"]:
        key = tuple(cell[k] for k in ("device", "batch", "horizon", "states", "frames", "mode"))
        assert key not in cells
        assert cell["valid_steps"] == cell["batch"] * cell["horizon"]
        assert cell["frame_count"] == (cell["batch"] if cell["frames"] == "final" else 0)
        assert cell["exact_state_task_frame"] == "PASS"
        mean, sd = rates(cell["seconds"], cell["valid_steps"])
        assert math.isclose(mean, cell["steps_per_s_mean"], rel_tol=1e-12)
        assert math.isclose(sd, cell["steps_per_s_sd"], rel_tol=1e-12, abs_tol=1e-12)
        cells[key] = cell
    combinations = list(itertools.product(("cpu", "cuda"), (1, 64, 256, 1024), (1, 8), ("none", "all"), ("none", "final")))
    assert set(cells) == {(*key, mode) for key in combinations for mode in ("baseline", "candidate")}
    rows = []
    for device, batch, horizon, states, frames in combinations:
        row = dict(domain=report["domain"], device=device, batch=batch, horizon=horizon, states=states,
                   frames=frames, host=report["host"], gpu=report["gpu"], job=report["job"], frame_mse_vs_baseline=0)
        for mode in ("baseline", "candidate"):
            cell = cells[device, batch, horizon, states, frames, mode]
            for metric, count in (("steps_per_s", cell["valid_steps"]), ("fps", cell["frame_count"])):
                mean, sd = rates(cell["seconds"], count) if count else (None, None)
                row[f"{mode}_{metric}_mean"], row[f"{mode}_{metric}_sd"] = mean, sd
        row["speedup_vs_previous_adapter"] = row["candidate_steps_per_s_mean"] / row["baseline_steps_per_s_mean"]
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Validated", report["domain"], "both devices; wrote", len(rows), "paired rows; not an upstream comparison")


if __name__ == "__main__":
    main()
