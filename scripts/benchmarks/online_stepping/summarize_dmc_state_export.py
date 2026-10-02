"""Validate all DMC export gates and summarize their paired public-API timings."""

import argparse
import csv
import itertools
import json
import statistics
from pathlib import Path

from dmc_profile import DOMAINS
from dmc_state_export import NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    candidate_hashes = reference_hashes = None
    for domain, device in itertools.product(DOMAINS, ("cpu", "cuda")):
        path = args.results / domain / f"{device}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["domain"] == domain and report["env"] == NAMES[domain], path
        assert len(report["gates"]) == 1
        gate = report["gates"][0]
        assert gate["device"] == device and gate["cases"] == 10
        assert gate["state_task_rng_lifecycle_ownership"] == "PASS" and gate["frame_mse_vs_legacy"] == 0
        assert report["opengl"]["software"] == (device == "cpu")
        if device == "cuda":
            assert "nvidia" in report["opengl"]["vendor"].lower()
        if candidate_hashes is None:
            candidate_hashes, reference_hashes = report["candidate_hashes"], report["reference_hashes"]
        assert candidate_hashes == report["candidate_hashes"] and reference_hashes == report["reference_hashes"]
        cells = {}
        for cell in report["timings"]:
            key = (cell["batch"], cell["horizon"], cell["operation"], cell["mode"])
            assert key not in cells and cell["device"] == device
            assert len(cell["seconds"]) == 6 and all(t > 0 for t in cell["seconds"])
            cells[key] = cell
        expected = set(itertools.product([1, 64, 256], [1, 8], ["get_state", "public_step_states_all"], ["legacy", "column"]))
        assert set(cells) == expected, (path, set(cells) ^ expected)
        for batch, horizon, operation in itertools.product([1, 64, 256], [1, 8], ["get_state", "public_step_states_all"]):
            count = batch * (horizon if operation == "public_step_states_all" else 1)
            rates = {mode: [count / t for t in cells[batch, horizon, operation, mode]["seconds"]] for mode in ("legacy", "column")}
            rows.append(dict(env=NAMES[domain], device=device, batch=batch, horizon=horizon, operation=operation,
                             legacy_rate_mean=statistics.mean(rates["legacy"]), legacy_rate_sd=statistics.stdev(rates["legacy"]),
                             column_rate_mean=statistics.mean(rates["column"]), column_rate_sd=statistics.stdev(rates["column"]),
                             speedup=statistics.mean(rates["column"]) / statistics.mean(rates["legacy"]),
                             frame_mse_vs_legacy=0, host=report["host"], gpu=report["gpu"], job=report["job"], report=str(path)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(len(rows), "paired comparisons; all 24 registration/device gates PASS")


if __name__ == "__main__":
    main()
