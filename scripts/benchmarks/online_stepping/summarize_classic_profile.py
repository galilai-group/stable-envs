"""Validate complete classic component profiles without relabeling diagnostic rates."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

REGISTRATIONS = {
    "cartpole": "swm/CartPoleControl-v1",
    "pendulum": "swm/PendulumControl-v1",
    "acrobot": "swm/AcrobotControl-v1",
    "mountain_car": "swm/MountainCarControl-v0",
    "mountain_car_continuous": "swm/MountainCarContinuousControl-v0",
}
OPERATIONS = {"step_with_state", "step_without_export_DIAGNOSTIC", "get_state", "set_state"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, hashes = [], None
    for family, registration in REGISTRATIONS.items():
        path = args.results / f"{family}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["family"] == family, path
        assert len(report["affinity"]) == 8 and "3090" in report["gpu"], path
        if hashes is None:
            hashes = report["source_sha256"]
            for rel, digest in hashes.items():
                assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
        assert report["source_sha256"] == hashes, path
        assert len(report["cells"]) == 8, path
        assert {(c["batch"], c["device"]) for c in report["cells"]} == set(
            itertools.product((1, 64, 256, 1024), ("cpu", "cuda"))), path
        for cell in report["cells"]:
            assert cell["restore_step_replay"] == "PASS" and cell["step_cprofile"], path
            assert cell["state_bytes"] > 0 and cell["state_leaves"] > 0, path
            assert set(cell["timings"]) == OPERATIONS, path
            for operation, timing in cell["timings"].items():
                seconds = timing["seconds"]
                assert len(seconds) == 6 and all(math.isfinite(t) and t > 0 for t in seconds), path
                rates = [cell["batch"] / t for t in seconds]
                assert math.isclose(timing["seconds_mean"], statistics.mean(seconds)), path
                assert math.isclose(timing["seconds_sd"], statistics.stdev(seconds)), path
                assert math.isclose(timing["lanes_per_s_mean"], statistics.mean(rates)), path
                assert math.isclose(timing["lanes_per_s_sd"], statistics.stdev(rates)), path
                rows.append(dict(registration=registration, batch=cell["batch"], output_device=cell["device"],
                                 operation=operation, batch_ms_mean=1000 * statistics.mean(seconds),
                                 batch_ms_sd=1000 * statistics.stdev(seconds),
                                 lanes_per_s_mean=statistics.mean(rates), lanes_per_s_sd=statistics.stdev(rates),
                                 state_bytes=cell["state_bytes"], state_leaves=cell["state_leaves"],
                                 construction_s=cell["construction_s"], host=report["host"], gpu=report["gpu"],
                                 job=report["job"], report=str(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("Validated 40 replay-gated profile cells; wrote 160 component rows")


if __name__ == "__main__":
    main()
