"""Require independent CPU/CUDA gates and every paired validator/pipeline cell."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def rates(cell):
    seconds, count = cell["seconds"], cell["count"]
    assert len(seconds) == 6 and all(math.isfinite(s) and s > 0 for s in seconds)
    values = [count / s for s in seconds]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    assert math.isclose(mean, cell["rate_mean"], rel_tol=1e-12)
    assert math.isclose(sd, cell["rate_sd"], rel_tol=1e-12, abs_tol=1e-12)
    return mean, sd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    hashes, family, components, pipelines = None, None, [], []
    for device in ("cpu", "cuda"):
        report = json.loads((args.results / f"{device}.json").read_text())
        assert report["status"] == "PASS" and len(report["affinity"]) == 8
        if device == "cuda":
            assert "3090" in report["gpu"]
        assert report["opengl"]["software"] == (device == "cpu")
        assert ("mesa" if device == "cpu" else "nvidia") in report["opengl"]["vendor"].lower()
        if hashes is None:
            hashes, family = report["source_sha256"], report["family"]
            for rel, digest in hashes.items():
                assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
        assert report["source_sha256"] == hashes and report["family"] == family
        assert len(report["gates"]) == 1
        gate = report["gates"][0]
        assert (gate["device"], gate["batch"], gate["workers"], gate["cases"]) == (device, 64, 8, 12)
        for key in ("complete_state_task_rng_lifecycle_ownership", "rejected_render_restore",
                    "reset_variation_device_actions", "validation_during_inflight"):
            assert gate[key] == "PASS"
        assert gate["frame_mse_vs_baseline"] == 0
        common = dict(family=family, device=device, host=report["host"], job=report["job"])
        for collection, fields, dimensions, output in (
                ("components", ("batch", "operation"), ((64, 128), ("validate_full", "project_full", "render_saved_full")), components),
                ("pipeline", ("batch", "horizon", "selection"), ((64, 128), (1, 8), ("none", "final", "all")), pipelines)):
            cells = {}
            for cell in report[collection]:
                key = tuple(cell[k] for k in (*fields, "mode"))
                assert key not in cells and cell["device"] == device
                assert cell["count"] == cell["batch"] * (cell["horizon"] if collection == "pipeline" else 1)
                if collection == "pipeline":
                    expected = cell["batch"] * (cell["horizon"] if cell["selection"] == "all" else int(cell["selection"] == "final"))
                    assert cell["emitted_frames"] == expected and cell["exact_state_task_frames"] == "PASS"
                rates(cell)
                cells[key] = cell
            assert set(cells) == set(itertools.product(*dimensions, ("baseline", "candidate")))
            for key in itertools.product(*dimensions):
                row = dict(**common, **dict(zip(fields, key)))
                for mode in ("baseline", "candidate"):
                    cell = cells[(*key, mode)]
                    row[mode + "_rate_mean"], row[mode + "_rate_sd"] = rates(cell)
                    if collection == "pipeline":
                        fps = [cell["emitted_frames"] / s for s in cell["seconds"]]
                        row[mode + "_fps_mean"] = statistics.mean(fps) if cell["emitted_frames"] else None
                        row[mode + "_fps_sd"] = statistics.stdev(fps) if cell["emitted_frames"] else None
                row["speedup_vs_previous_validator"] = row["candidate_rate_mean"] / row["baseline_rate_mean"]
                output.append(row)
    args.output.mkdir(parents=True)
    for name, rows in (("components", components), ("pipeline", pipelines)):
        with (args.output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("Validated", family, "both devices:", len(components), "component and", len(pipelines), "pipeline pairs")


if __name__ == "__main__":
    main()
