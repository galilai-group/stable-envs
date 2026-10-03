"""Validate classic-control exporter gates and paired timing cells."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, candidate_hashes, reference_hashes, script_hash = [], None, None, None
    names = {"cartpole": "CartPole", "pendulum": "Pendulum", "acrobot": "Acrobot",
             "mountain_car": "MountainCar", "mountain_car_continuous": "MountainCarContinuous"}
    for family, device in itertools.product(names, args.devices):
        path = args.results / f"{family}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["domain"] == family and report["env"] == names[family], path
        assert {g["device"] for g in report["gates"]} == set(args.devices), path
        assert len(report["gates"]) == len(args.devices), path
        gate = next(g for g in report["gates"] if g["device"] == device)
        assert gate["device"] == device and gate["cases"] == 10, path
        assert gate["state_task_rng_lifecycle_ownership"] == "PASS" and gate["frame_mse_vs_legacy"] == 0, path
        if device == "cuda":
            assert report["gpu"], path
        if candidate_hashes is None:
            candidate_hashes, reference_hashes, script_hash = report["candidate_hashes"], report["reference_hashes"], report["script_sha256"]
            for name, digest in {**candidate_hashes, **reference_hashes,
                                 "scripts/benchmarks/online_stepping/classic_state_export.py": script_hash}.items():
                assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest, name
        assert (report["candidate_hashes"], report["reference_hashes"], report["script_sha256"]) == (candidate_hashes, reference_hashes, script_hash), path
        cells = {}
        for cell in report["timings"]:
            if cell["device"] != device:
                continue
            key = (cell["batch"], cell["horizon"], cell["operation"], cell["mode"])
            assert key not in cells and cell["device"] == device, path
            assert len(cell["seconds"]) == 6 and all(math.isfinite(t) and t > 0 for t in cell["seconds"]), path
            cells[key] = cell
        assert set(cells) == set(itertools.product(args.batches, (1, 8), ("get_state", "public_step_states_all"), ("legacy", "column"))), path
        for batch, horizon, operation in itertools.product(args.batches, (1, 8), ("get_state", "public_step_states_all")):
            count = batch * (horizon if operation == "public_step_states_all" else 1)
            rates = {mode: [count / t for t in cells[batch, horizon, operation, mode]["seconds"]] for mode in ("legacy", "column")}
            rows.append(dict(env=report["env"], device=device, batch=batch, horizon=horizon, operation=operation,
                             legacy_rate_mean=statistics.mean(rates["legacy"]), legacy_rate_sd=statistics.stdev(rates["legacy"]),
                             column_rate_mean=statistics.mean(rates["column"]), column_rate_sd=statistics.stdev(rates["column"]),
                             speedup=statistics.mean(rates["column"]) / statistics.mean(rates["legacy"]),
                             frame_mse_vs_legacy=0, host=report["host"], gpu=report["gpu"], job=report["job"], report=str(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {5 * len(args.devices)} independent gates; wrote {len(rows)} paired rows")


if __name__ == "__main__":
    main()
