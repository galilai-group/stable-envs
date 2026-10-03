"""Require complete default-renderer pipeline and component cells on both devices."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path


def stats(seconds, count):
    assert len(seconds) == 6 and all(math.isfinite(t) and t > 0 for t in seconds)
    rates = [count / t for t in seconds]
    return statistics.mean(rates), statistics.stdev(rates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    component_rows, pipeline_rows, source_hashes, domain = [], [], None, None
    for device in ("cpu", "cuda"):
        report = json.loads((args.results / f"{device}.json").read_text())
        assert report["status"] == "PASS"
        assert len(report["affinity"]) == 8
        if device == "cuda":
            assert "3090" in report["gpu"]
        gl = report["opengl"]
        assert gl["software"] == (device == "cpu")
        assert ("mesa" if device == "cpu" else "nvidia") in gl["vendor"].lower()
        if source_hashes is None:
            source_hashes, domain = report["source_sha256"], report["domain"]
            for rel, digest in source_hashes.items():
                assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
        assert report["source_sha256"] == source_hashes and report["domain"] == domain
        common = dict(domain=domain, device=device, host=report["host"], gpu=report["gpu"], job=report["job"])
        gates = {}
        for gate in report["gates"]:
            key = tuple(gate[k] for k in ("batch", "horizon", "selection"))
            assert key not in gates and gate["device"] == device and gate["complete_state_task"] == "PASS"
            assert gate["frame_mse_vs_deferred"] == (None if gate["selection"] == "none" else 0)
            gates[key] = gate
        assert set(gates) == set(itertools.product(args.batches, (1, 8), ("none", "final", "all")))
        components = set()
        for cell in report["components"]:
            key = tuple(cell[k] for k in ("batch", "horizon", "operation"))
            assert key not in components and cell["device"] == device
            components.add(key)
            count = cell["batch"] * (cell["horizon"] if cell["operation"] in ("task_no_state_with_ipc", "render_saved_all") else 1)
            assert cell["count_per_sample"] == count
            mean, sd = stats(cell["seconds"], count)
            component_rows.append(dict(**common, batch=cell["batch"], horizon=cell["horizon"],
                                       operation=cell["operation"], rate_mean=mean, rate_sd=sd))
        assert components == set(itertools.product(args.batches, (1, 8),
            ("task_no_state_with_ipc", "state_export_with_ipc_transfer", "render_saved_final", "render_saved_all")))
        cells = {}
        for cell in report["pipeline"]:
            key = tuple(cell[k] for k in ("batch", "horizon", "selection", "mode"))
            assert key not in cells and cell["device"] == device
            count = cell["batch"] * (cell["horizon"] if cell["selection"] == "all" else int(cell["selection"] == "final"))
            assert cell["emitted_frames"] == count and cell["steps"] == cell["batch"] * cell["horizon"]
            stats(cell["seconds"], cell["steps"])
            cells[key] = cell
        assert set(cells) == set(itertools.product(args.batches, (1, 8), ("none", "final", "all"), ("deferred", "pipeline")))
        for batch, horizon, selection in itertools.product(args.batches, (1, 8), ("none", "final", "all")):
            row = dict(**common, batch=batch, horizon=horizon, selection=selection)
            for mode in ("deferred", "pipeline"):
                cell = cells[batch, horizon, selection, mode]
                for name, count in (("steps_per_s", cell["steps"]), ("fps", cell["emitted_frames"])):
                    mean, sd = stats(cell["seconds"], count) if count else (None, None)
                    row[f"{mode}_{name}_mean"], row[f"{mode}_{name}_sd"] = mean, sd
            row["speedup_vs_deferred"] = row["pipeline_steps_per_s_mean"] / row["deferred_steps_per_s_mean"]
            pipeline_rows.append(row)
    args.output.mkdir(parents=True)
    for name, rows in (("components", component_rows), ("pipeline", pipeline_rows)):
        with (args.output / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    print("Validated", domain, "on both devices; wrote", len(component_rows), "component and", len(pipeline_rows), "pipeline rows")


if __name__ == "__main__":
    main()
