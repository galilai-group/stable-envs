"""Validate all DMC export gates and summarize their paired public-API timings."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from dmc_profile import DOMAINS
from dmc_state_export import NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    p.add_argument("--source", type=Path, required=True, help="Frozen candidate checkout")
    p.add_argument("--reference-adapters", type=Path, required=True, help="Frozen baseline adapter directory")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    assert len(set(args.domains)) == len(args.domains)
    candidate_hashes = reference_hashes = script_hash = None
    for domain, device in itertools.product(args.domains, ("cpu", "cuda")):
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
            script_hash = report["script_sha256"]
            for name, digest in candidate_hashes.items():
                assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest, name
            for name, digest in reference_hashes.items():
                assert hashlib.sha256((args.reference_adapters / Path(name).name).read_bytes()).hexdigest() == digest, name
            assert hashlib.sha256((args.source / "scripts/benchmarks/online_stepping/dmc_state_export.py").read_bytes()).hexdigest() == script_hash
        assert (candidate_hashes, reference_hashes, script_hash) == (report["candidate_hashes"], report["reference_hashes"], report["script_sha256"])
        cells = {}
        for cell in report["timings"]:
            key = (cell["batch"], cell["horizon"], cell["operation"], cell["mode"])
            assert key not in cells and cell["device"] == device
            assert len(cell["seconds"]) == 6 and all(math.isfinite(t) and t > 0 for t in cell["seconds"])
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
    print(len(rows), f"paired comparisons; {2 * len(args.domains)} explicitly selected registration/device gates PASS")


if __name__ == "__main__":
    main()
