"""Require complete Navigation source comparisons, valid-step counts and exact task outputs."""

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
    p.add_argument("--components-output", type=Path, required=True)
    p.add_argument("--cpu-threads", type=int, default=8)
    args = p.parse_args()
    for path in (args.output, args.components_output):
        if path.exists():
            raise FileExistsError(path)
    assert args.output.resolve() != args.components_output.resolve()
    report = json.loads(args.report.read_text())
    assert report["status"] == "PASS" and not report["gate_only"]
    assert len(report["affinity"]) == report["source_workers"] == 8
    assert "3090" in report["gpu"] and report["source_gpu_supported"] is False
    assert report["source_render_device"] == "cpu" and report["source_state_export_in_timing"] is False
    assert report["ours_mandatory_final_state"] is True and report["ours_state_history"] is False
    assert report["source_pin"] == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
    configured = report["parent_torch_threads"]
    assert report["source_native_batch"] is False and report["source_device_api"] is False
    assert report["source_threads_per_worker"] == 1
    assert configured == {"cpu": args.cpu_threads, "cuda": args.cpu_threads}
    assert 1 <= args.cpu_threads <= len(report["affinity"])
    assert len(report["source_files"]) == 6 and len(report["source_runtime_files"]) >= 2
    for rel, digest in report["candidate_files"].items():
        assert hashlib.sha256((args.source / rel).read_bytes()).hexdigest() == digest, rel
    components, component_rows = set(), []
    for cell in report["components"]:
        key = (cell["device"], cell["batch"], cell["operation"])
        assert key not in components
        assert len(cell["seconds"]) == 6 and all(t > 0 and math.isfinite(t) for t in cell["seconds"])
        rates = [cell["batch"] / t for t in cell["seconds"]]
        assert cell["rate_mean"] == statistics.mean(rates)
        assert cell["rate_sd"] == statistics.stdev(rates)
        component_rows.append(dict(domain=report["domain"], device=cell["device"], batch=cell["batch"],
            operation=cell["operation"], host=report["host"], job=report["job"],
            rate_unit="frames/s" if cell["operation"] == "render_saved_state" else
                      ("states/s" if cell["operation"] == "get_state" else "steps/s"),
            rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates),
            seconds_mean=statistics.mean(cell["seconds"]), seconds_sd=statistics.stdev(cell["seconds"])))
        components.add(key)
    assert components == set(itertools.product(("cpu", "cuda"), (16, 64),
        ("raw_task_step_with_state", "get_state", "render_saved_state")))
    expected = set(itertools.product(("cpu", "cuda"), (16, 64), (1, 8), ("none", "final", "all")))
    gates, cells = {}, {}
    for gate in report["gates"]:
        key = tuple(gate[k] for k in ("device", "batch", "horizon", "selection"))
        assert key not in gates
        assert 0 < gate["valid_steps"] <= gate["batch"] * gate["horizon"]
        assert all(gate[k] == "PASS" for k in ("complete_state_task", "parallel_source_exact", "source_output_ownership"))
        count = gate["batch"] * (gate["horizon"] if gate["selection"] == "all" else int(gate["selection"] == "final"))
        assert gate["frames_checked"] == count
        if count:
            assert 0 <= gate["mse_mean"] <= gate["mse_max"] <= .001
            assert gate["mse_sd"] >= 0 and math.isfinite(gate["mse_sd"])
        else:
            assert all(gate[k] is None for k in ("mse_mean", "mse_sd", "mse_max"))
        gates[key] = gate
    assert set(gates) == expected
    for cell in report["timings"]:
        key = tuple(cell[k] for k in ("device", "batch", "horizon", "selection", "mode"))
        assert key not in cells and len(cell["seconds"]) == 6
        assert all(math.isfinite(t) and t > 0 for t in cell["seconds"])
        assert cell["steps"] == gates[key[:-1]]["valid_steps"]
        assert cell["frames"] == gates[key[:-1]]["frames_checked"]
        assert cell["ours_workers"] == 0
        cells[key] = cell
    assert set(cells) == {(*k, mode) for k in expected for mode in ("source_serial", "source_parallel", "ours")}
    rows = []
    for key in sorted(expected):
        gate = gates[key]
        row = dict(domain=report["domain"], output_device=key[0], batch=key[1], horizon=key[2], selection=key[3],
                   source_render_device="cpu", source_gpu_supported=False, source_parallel_workers=8,
                   host=report["host"], job=report["job"],
                   mse_mean=gate["mse_mean"], mse_sd=gate["mse_sd"], mse_max=gate["mse_max"],
                   frames_checked=gate["frames_checked"], valid_steps=gate["valid_steps"])
        if "parent_torch_threads" in report:
            row["parent_torch_threads"] = configured[key[0]]
        for mode in ("source_serial", "source_parallel", "ours"):
            cell = cells[(*key, mode)]
            rates = [cell["steps"] / s for s in cell["seconds"]]
            row[f"{mode}_steps_per_s_mean"] = statistics.mean(rates)
            row[f"{mode}_steps_per_s_sd"] = statistics.stdev(rates)
            row[f"{mode}_fps_mean"] = statistics.mean(cell["frames"] / s for s in cell["seconds"]) if cell["frames"] else None
            row[f"{mode}_fps_sd"] = statistics.stdev(cell["frames"] / s for s in cell["seconds"]) if cell["frames"] else None
        row["ours_vs_serial"] = row["ours_steps_per_s_mean"] / row["source_serial_steps_per_s_mean"]
        row["ours_vs_parallel"] = row["ours_steps_per_s_mean"] / row["source_parallel_steps_per_s_mean"]
        row["ours_vs_best_source"] = min(row["ours_vs_serial"], row["ours_vs_parallel"])
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    args.components_output.parent.mkdir(parents=True, exist_ok=True)
    with args.components_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(component_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(component_rows)
    print("Validated", report["domain"], "24 source-comparison rows, 72 six-sample timing cells, 12 component rows")


if __name__ == "__main__":
    main()
