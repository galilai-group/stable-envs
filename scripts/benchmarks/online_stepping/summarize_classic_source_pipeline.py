"""Validate direct-source classic measurements without inventing a GPU source."""

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
    p.add_argument("report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.report.read_text())
    assert report["status"] == "PASS"
    assert "3090" in report["gpu"] and len(report["affinity"]) == 8
    assert report["source_gpu_supported"] is False and report["source_active_simulation_workers"] == 1
    assert report["source_state_export_in_timing"] is False
    for rel, h in report["candidate_files"].items():
        assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == h, rel
    expected = set(itertools.product(("cpu", "cuda"), args.batches, (1, 8), ("none", "final", "all")))
    gates, cells = {}, {}
    for g in report["gates"]:
        key = tuple(g[k] for k in ("device", "batch", "horizon", "selection"))
        assert key not in gates and g["state_task"] == "PASS"
        frames = g["batch"] * (g["horizon"] if g["selection"] == "all" else 1) if g["selection"] != "none" else 0
        assert g["frames_checked"] == frames
        if frames:
            assert all(math.isfinite(g[k]) and g[k] >= 0 for k in ("mse_mean", "mse_sd", "mse_max"))
            assert g["mse_le_0_001"] == (g["mse_max"] <= .001)
        gates[key] = g
    assert set(gates) == expected
    for c in report["timings"]:
        key = tuple(c[k] for k in ("device", "batch", "horizon", "selection", "mode"))
        assert key not in cells
        assert len(c["seconds"]) == 6 and all(math.isfinite(s) and s > 0 for s in c["seconds"])
        assert c["count"] == c["batch"] * c["horizon"]
        assert c["candidate_processes"] == (8 if report["domain"] == "acrobot" and c["batch"] >= 64 else 0)
        cells[key] = c
    assert set(cells) == {(*k, mode) for k in expected for mode in ("source_serial", "ours")}
    components = {(c["device"], c["batch"], c["operation"]): c for c in report["components"]}
    assert len(components) == len(report["components"])
    assert set(components) == set(itertools.product(("cpu", "cuda"), args.batches,
                                                    ("ours_state_export", "ours_render_saved", "source_render_saved")))
    for c in components.values():
        assert len(c["seconds"]) == 6 and all(math.isfinite(s) and s > 0 for s in c["seconds"])
    rows = []
    for key in sorted(expected):
        device, batch, horizon, selection = key
        ours, source = cells[(*key, "ours")], cells[(*key, "source_serial")]
        source_cpu = cells["cpu", batch, horizon, selection, "source_serial"]
        row = dict(env=report["name"], device=device, batch=batch, horizon=horizon, frames=selection,
                   source_simulation="serial_cpu", source_gpu_fps=None,
                   candidate_processes=ours["candidate_processes"], source_workers=1,
                   allocated_cpu_count=len(report["affinity"]),
                   mse_mean=gates[key]["mse_mean"], mse_sd=gates[key]["mse_sd"],
                   mse_max=gates[key]["mse_max"], mse_le_0_001=gates[key]["mse_le_0_001"])
        for label, cell in (("ours", ours), ("source_same_output_destination", source), ("source_cpu_native_output", source_cpu)):
            for unit, count in (("steps_per_s", cell["count"]), ("fps", cell["frames"])):
                rates = [count / s for s in cell["seconds"]]
                row[f"{label}_{unit}_mean"] = statistics.mean(rates) if count else None
                row[f"{label}_{unit}_sd"] = statistics.stdev(rates) if count else None
        row["ratio_vs_serial_source_same_destination"] = row["ours_steps_per_s_mean"] / row["source_same_output_destination_steps_per_s_mean"]
        row.update(host=report["host"], job=report["job"], scope="No parallel-source speedup claim; ours additionally returns complete final State")
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("Validated both devices; wrote", len(rows), "direct-serial-source comparison rows")


if __name__ == "__main__":
    main()
