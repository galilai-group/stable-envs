"""Validate independent native step gates and summarize the paired dispatcher sweep."""

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
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, hashes = [], None
    for name in ("TwoRoom", "Piecewise", "SimplePointMaze"):
        path = args.results / f"{name}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["env"] == name, path
        assert len(report["gates"]) == len(args.devices), path
        assert {g["device"] for g in report["gates"]} == set(args.devices), path
        if hashes is None:
            hashes = report["source_sha256"]
            for source, digest in hashes.items():
                assert hashlib.sha256((args.source / source).read_bytes()).hexdigest() == digest, source
        assert report["source_sha256"] == hashes, path
        for gate in report["gates"]:
            assert gate["cases"] == 16 and gate["exact_state_task_frames"] == "PASS", path
            assert gate["frame_mse_vs_legacy"] == 0, path
            if gate["device"] == "cuda":
                assert report["gpu"], path
        cells = {}
        for cell in report["timings"]:
            key = tuple(cell[k] for k in ("device", "batch", "horizon", "operation", "mode"))
            assert key not in cells, path
            assert len(cell["seconds"]) == len(cell["valid_transitions"]) == 6, path
            assert all(math.isfinite(t) and t > 0 for t in cell["seconds"]), path
            assert all(0 < n <= cell["batch"] * cell["horizon"] for n in cell["valid_transitions"]), path
            cells[key] = cell
        workloads = list(itertools.product(args.devices, (1, 16, 64), (1, 8),
                                            ("states_all", "states_none", "pipeline_final_frame")))
        assert set(cells) == {(*w, mode) for w in workloads for mode in ("legacy", "candidate")}, path
        for device, batch, horizon, operation in workloads:
            pair = {mode: cells[device, batch, horizon, operation, mode] for mode in ("legacy", "candidate")}
            assert pair["legacy"]["valid_transitions"] == pair["candidate"]["valid_transitions"], path
            row = dict(env=name, device=device, batch=batch, horizon=horizon, operation=operation,
                       frame_mse_vs_legacy=0, host=report["host"], gpu=report["gpu"], job=report["job"])
            for mode, cell in pair.items():
                rates = [n / t for n, t in zip(cell["valid_transitions"], cell["seconds"], strict=True)]
                row[mode + "_steps_per_s_mean"] = statistics.mean(rates)
                row[mode + "_steps_per_s_sd"] = statistics.stdev(rates)
                fps = [batch / t for t in cell["seconds"]] if operation == "pipeline_final_frame" else None
                row[mode + "_frames_per_s_mean"] = statistics.mean(fps) if fps else ""
                row[mode + "_frames_per_s_sd"] = statistics.stdev(fps) if fps else ""
            row["speedup_vs_legacy"] = row["candidate_steps_per_s_mean"] / row["legacy_steps_per_s_mean"]
            row["report"] = str(path)
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {3 * len(args.devices)} independent gates; wrote {len(rows)} paired rows")


if __name__ == "__main__":
    main()
