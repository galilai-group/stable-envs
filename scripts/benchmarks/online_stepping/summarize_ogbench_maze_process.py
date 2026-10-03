"""Require both Maze process device gates and complete paired throughput cells."""

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
    p.add_argument("--production", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, hashes = [], None
    for family, device in itertools.product(("Maze",), ("cpu", "cuda")):
        path = args.results / f"{device}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["domain"] == family, path
        assert len(report["gates"]) == 1, path
        gate = report["gates"][0]
        assert gate["device"] == device and gate["cases"] == 22 and gate["overlaps"] > 0, path
        assert gate["state_task_rng_lifecycle_ownership"] == gate["failed_restore_keeps_owners"] == "PASS", path
        assert gate["frame_mse_vs_local"] == 0, path
        assert report["opengl"]["software"] == (device == "cpu"), path
        assert ("mesa" if device == "cpu" else "nvidia") in report["opengl"]["vendor"].lower(), path
        assert len(report["affinity"]) >= 8, path
        if hashes is None:
            hashes = report["source_sha256"]
            for rel, digest in hashes.items():
                assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
        assert report["source_sha256"] == hashes, path
        if args.production:
            assert gate["production_default"] and gate["batch"] == 64 and gate["processes"] == 8, path
            assert report["timings"] == [], path
            rows.append(dict(env="OGBMaze", device=device, batch=64, processes=8, cases=gate["cases"],
                             frame_mse_vs_local=0, overlaps=gate["overlaps"],
                             host=report["host"], gpu=report["gpu"], job=report["job"], report=str(path)))
            continue
        cells = {}
        for cell in report["timings"]:
            key = tuple(cell[k] for k in ("batch", "horizon", "processes"))
            assert key not in cells and cell["device"] == device, path
            assert len(cell["seconds"]) == len(cell["valid_transitions"]) == 6, path
            assert all(math.isfinite(t) and t > 0 for t in cell["seconds"]), path
            assert all(0 < n <= cell["batch"] * cell["horizon"] for n in cell["valid_transitions"]), path
            cells[key] = cell
        assert set(cells) == set(itertools.product((64, 256), (1, 8), (0, 4, 8))), path
        for batch, horizon, processes in itertools.product((64, 256), (1, 8), (4, 8)):
            local, remote = cells[batch, horizon, 0], cells[batch, horizon, processes]
            assert local["valid_transitions"] == remote["valid_transitions"], path
            rates = {name: [n / t for n, t in zip(cell["valid_transitions"], cell["seconds"], strict=True)]
                     for name, cell in (("local", local), ("process", remote))}
            row = dict(env="OGB" + family, device=device, batch=batch, horizon=horizon, processes=processes,
                       frame_mse_vs_local=0, host=report["host"], gpu=report["gpu"], job=report["job"])
            for name, samples in rates.items():
                row[name + "_steps_per_s_mean"] = statistics.mean(samples)
                row[name + "_steps_per_s_sd"] = statistics.stdev(samples)
            row["speedup_vs_local"] = statistics.mean(rates["process"]) / statistics.mean(rates["local"])
            row["report"] = str(path)
            rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Validated two independent gates; wrote", len(rows), "rows")


if __name__ == "__main__":
    main()
