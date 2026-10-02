"""Validate independent process gates and summarize public stepping throughput.

Run against frozen result directories. CPU physics is used for both output
devices; timings are not renderer FPS or upstream-source speedups.
"""

import argparse
import csv
import hashlib
import itertools
import json
import statistics
from pathlib import Path

DOMAINS = (
    "acrobot", "pendulum", "cartpole", "reacher", "cheetah", "hopper",
    "walker", "ball_in_cup", "finger", "manipulator", "quadruped", "humanoid",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--domains", choices=DOMAINS, nargs="+", default=DOMAINS)
    parser.add_argument("--timings", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert len(set(args.domains)) == len(args.domains)
    rows, hashes = [], None
    for domain, device in itertools.product(args.domains, ("cpu", "cuda")):
        path = args.results / domain / f"{device}.json"
        report = json.loads(path.read_text())
        assert report["status"] == "PASS" and report["domain"] == domain, path
        assert len(report["gates"]) == 1, path
        gate = report["gates"][0]
        assert gate["domain"] == domain and gate["device"] == device, path
        assert gate["cases"] == 12 and gate["overlaps"] > 0, path
        assert gate["state_task_rng_lifecycle_ownership"] == "PASS", path
        assert gate["physics_and_visual_variations"] == "PASS", path
        assert gate["frame_mse_vs_local"] == 0, path
        gl = report["opengl"]
        assert gl["software"] == (device == "cpu"), path
        assert ("mesa" if device == "cpu" else "nvidia") in gl["vendor"].lower(), path
        if hashes is None:
            hashes = report["source_sha256"]
            assert hashes
            for name, digest in hashes.items():
                assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == digest, name
        assert report["source_sha256"] == hashes, path
        common = dict(domain=domain, device=device, frame_mse_vs_local=0,
                      host=report["host"], gpu=report["gpu"], job=report["job"],
                      report=str(path))
        if not args.timings:
            rows.append(dict(**common, cases=gate["cases"], overlaps=gate["overlaps"],
                             status="PASS", gl_renderer=gl["renderer"]))
            continue
        cells = {}
        for cell in report["timings"]:
            key = (cell["batch"], cell["horizon"], cell["processes"])
            assert key not in cells and cell["device"] == device, path
            assert len(cell["seconds"]) == 6 and all(t > 0 for t in cell["seconds"]), path
            cells[key] = [cell["batch"] * cell["horizon"] / t for t in cell["seconds"]]
        assert set(cells) == set(itertools.product((64, 256), (1, 8), (0, 4, 8))), path
        for (batch, horizon, processes), rates in cells.items():
            rows.append(dict(**common, batch=batch, horizon=horizon, processes=processes,
                             steps_per_s_mean=statistics.mean(rates),
                             steps_per_s_sd=statistics.stdev(rates),
                             speedup_vs_local=statistics.mean(rates) / statistics.mean(cells[batch, horizon, 0])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {len(args.domains) * 2} independent gates; wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
