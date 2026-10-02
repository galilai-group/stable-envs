"""Validate completed pipeline sweeps and report paired, same-host throughput."""

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
    output, seen = [], set()
    for path in args.inputs:
        d = json.loads(path.read_text())
        assert d["status"] == "PASS", path
        c = d["configuration"]
        assert d["task"] not in seen
        seen.add(d["task"])
        names = {"Fetch" + d["task"] + ("" if s == "plain" else s) for s in c["suffixes"]}
        devices = {c["device"]} | ({"cpu"} if c.get("production") else set())
        expected_gates = {(name, device) for name in names for device in devices}
        assert len(d["gates"]) == len(expected_gates)
        assert {(g["env"], g["device"]) for g in d["gates"]} == expected_gates
        for gate in d["gates"]:
            assert gate["state_task_replay_ownership"] == "PASS"
            assert gate["cases"] == 13
            assert gate["frame_mse_vs_serial"] == (0.0 if gate["device"] == "cuda" else None)
        groups = defaultdict(list)
        for r in d["timings"]:
            assert r["exact_parity"] == "PASS"
            assert r["frame_mse"] == (0.0 if r["images"] else None)
            assert r["steps"] == r["batch"] * r["horizon"]
            assert r["images"] == r["batch"] * (r["horizon"] if r["frames"] == "all" else int(r["frames"] == "final"))
            assert all(s > 0 for s in r["seconds"])
            groups[r["batch"], r["horizon"], r["frames"], r["mode"]].append(r)
        expected = (
            {
                (batch, horizon, frames, mode)
                for batch in c["batches"]
                for horizon in c["horizons"]
                for frames in (["none", "final", "all"] if c["device"] == "cuda" else ["none"])
                for mode in ("serial_legacy", "serial_fused", "pipeline")
            }
            if "plain" in c["suffixes"]
            else set()
        )
        assert set(groups) == expected
        means = {}
        for key, rows in groups.items():
            assert sorted(r["block"] for r in rows) == list(range(c["blocks"]))
            assert all(len(r["seconds"]) == c["repeats"] for r in rows)
            means[key] = statistics.mean(rows[0]["steps"] / s for r in rows for s in r["seconds"])
        for key, rows in groups.items():
            batch, horizon, frames, mode = key
            seconds = [s for r in rows for s in r["seconds"]]
            row = dict(
                registration="swm/Fetch" + d["task"] + "-v3",
                host=d["host"],
                gpu=d["gpu"],
                job=d["job"],
                output_device=c["device"],
                simulation_device="cpu",
                processes=8,
                threads_per_process=1,
                cpu_cores=len(d["affinity"]) if d["affinity"] else None,
                batch=batch,
                horizon=horizon,
                frames=frames,
                mode=mode,
                samples=len(seconds),
                speedup_vs_serial_fused=means[key] / means[batch, horizon, frames, "serial_fused"],
                speedup_vs_serial_legacy=means[key] / means[batch, horizon, frames, "serial_legacy"],
                exact_state_task_parity="PASS",
                frame_mse=rows[0]["frame_mse"],
                image_size=c["size"],
                evidence=path.name,
            )
            for label, count in (("steps_s", rows[0]["steps"]), ("frames_s", rows[0]["images"])):
                rates = [count / s for s in seconds]
                row[label + "_mean"] = statistics.mean(rates)
                row[label + "_sd"] = statistics.stdev(rates)
            output.append(row)
    assert output, "No timing rows"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Validated {len(seen)} tasks; wrote {len(output)} comparison rows.")


if __name__ == "__main__":
    main()
