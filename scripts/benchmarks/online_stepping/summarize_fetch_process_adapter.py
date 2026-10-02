"""Aggregate independent timing blocks without mixing devices or API scopes."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for path in args.inputs:
        report = json.loads(path.read_text())
        assert report["status"] == "PASS", path
        groups = defaultdict(list)
        for row in report["timings"]:
            assert row["exact_parity"] == "PASS"
            groups[row["batch"], row["horizon"], row["stage"], row["layout"]].extend(row["seconds"])
        rates = {key: [key[0] * key[1] / t for t in times] for key, times in groups.items()}
        for (batch, horizon, stage, layout), seconds in groups.items():
            samples = rates[batch, horizon, stage, layout]
            baseline = statistics.mean(rates[batch, horizon, stage, "threads_1x4"])
            rows.append(
                dict(
                    env="swm/FetchPush-v3",
                    output_device=report["device"],
                    simulation_device="cpu",
                    host=report["host"],
                    job=report["job"],
                    batch=batch,
                    horizon=horizon,
                    scope=stage,
                    layout=layout,
                    samples=len(samples),
                    steps_per_second_mean=statistics.mean(samples),
                    steps_per_second_sd=statistics.stdev(samples),
                    chunk_seconds_mean=statistics.mean(seconds),
                    chunk_seconds_sd=statistics.stdev(seconds),
                    speedup_vs_threads_1x4=statistics.mean(samples) / baseline,
                    rendering_included=False,
                    full_state_included=True,
                    parity="EXACT",
                    evidence=path.name,
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
