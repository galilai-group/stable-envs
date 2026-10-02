"""Validate and aggregate Fetch family measurements, keeping hosts/scopes separate."""

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
    output = []
    seen = set()
    for path in args.inputs:
        report = json.loads(path.read_text())
        assert report["status"] == "PASS", path
        config = report["configuration"]
        for gate in report["gates"]:
            assert gate["frame_mse_vs_current"] == (0.0 if gate["device"] == "cuda" else None), gate
        gates = {
            (g["env"], g["device"], g["layout"])
            for g in report["gates"]
            if g["state_task_rng_replay_lifecycle_variation"] == "PASS"
        }
        groups = defaultdict(list)
        for row in report["timings"]:
            assert row["exact_parity"] == "PASS", row
            assert row["frame_mse_vs_current"] == (0.0 if row["frames"] else None), row
            assert all(t > 0 for t in row["seconds"]), row
            key = row["env"], row["device"], row["batch"], row["horizon"], row["stage"], row["layout"]
            groups[key].append(row)
        # Require every requested cell; a PASS marker alone is not sufficient.
        expected_keys = set()
        for suffix in config["suffixes"]:
            env = "Fetch" + report["task"] + ("" if suffix == "plain" else suffix)
            assert env not in seen, env
            seen.add(env)
            for device in config["devices"]:
                for layout in ("process4", "process8"):
                    assert (env, device, layout) in gates, (env, device, layout)
                for batch in config["batches"]:
                    modes = ["step_state"]
                    if device == "cuda" and batch in config["pipeline_batches"]:
                        modes += ["step_render_all", "render_only"]
                    layouts = ["current"] + [f"process{p}" for p in (4, 8) if batch >= p]
                    for horizon in config["horizons"]:
                        for mode in modes:
                            for layout in layouts:
                                expected_keys.add((env, device, batch, horizon, mode, layout))
        assert set(groups) == expected_keys, (set(groups) - expected_keys, expected_keys - set(groups))
        rates = {}
        for key, rows in groups.items():
            assert sorted(r["block"] for r in rows) == list(range(config["blocks"])), key
            assert all(len(r["seconds"]) == config["repeats"] for r in rows), key
            seconds = [t for row in rows for t in row["seconds"]]
            count = rows[0]["frames"] if key[4] == "render_only" else rows[0]["transitions"]
            rates[key] = statistics.mean(count / t for t in seconds)
        for key, rows in groups.items():
            env, device, batch, horizon, mode, layout = key
            seconds = [t for row in rows for t in row["seconds"]]
            row = dict(
                registration="swm/" + env + "-v3",
                output_device=device,
                simulation_device="cpu",
                host=report["host"],
                gpu=report["gpu"],
                job=report["job"],
                partition=report["partition"],
                batch=batch,
                horizon=horizon,
                stage=mode,
                layout=layout,
                samples=len(seconds),
                seconds_mean=statistics.mean(seconds),
                seconds_sd=statistics.stdev(seconds),
                speedup_vs_current=rates[key] / rates[(*key[:-1], "current")],
                exact_state_task_parity="PASS",
                frame_mse_vs_current=rows[0]["frame_mse_vs_current"],
                image_size=config["size"],
                evidence=path.name,
            )
            for label, count in (("steps_s", rows[0]["transitions"]), ("frames_s", rows[0]["frames"])):
                values = [count / t for t in seconds]
                row[label + "_mean"] = statistics.mean(values)
                row[label + "_sd"] = statistics.stdev(values)
            output.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Validated {len(seen)} registrations; wrote {len(output)} comparison rows.")


if __name__ == "__main__":
    main()
