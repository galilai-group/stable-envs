"""Reject incomplete/mixed Fetch campaigns before writing same-device FPS rows."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

from benchmarks.online.fetch_source_pipeline import MODES, REGISTRATIONS, require_image_gate
from benchmarks.online.summarize_dmc_source import checked_seconds, require_gl


def summarize(results, candidate, source, source_manifest, batches):
    manifest = json.loads(source_manifest.read_text())
    assert manifest["commit"] == "176d54532b3c0a251a4f509b74f2fdacc143a63d" and len(manifest["files"]) == 5
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((source / rel).read_bytes()).hexdigest() == digest, rel
    reports, cells, components, provenance, pids = {}, {}, [], None, set()
    for mode, device in itertools.product(MODES, ("cpu", "cuda")):
        report = json.loads((results / mode / f"{device}.json").read_text())
        assert report["status"] == "PASS" and report["gate_only"] is False
        assert report["registration"] in REGISTRATIONS and report["mode"] == mode and report["device"] == device
        assert report["job"] and len(report["affinity"]) == 8
        assert report["parent_torch_threads"] == (8 if device == "cpu" else 1)
        assert (
            report["source_threads_per_worker"] == 1 and report["output_size"] == 64 and report["backend"] == "opengl"
        )
        assert report["gpu"] is None if device == "cpu" else "3090" in report["gpu"]
        assert report["source_gpu_renderer_available"] and report["source_physics_device"] == "cpu"
        assert report["source_state_export_in_online_timing"] is False and report["ours_state_history"] is False
        for flag in (
            "source_readback_and_output_transfer_in_timing",
            "ours_mandatory_final_state",
            "source_observations_computed",
            "fresh_strategy_process",
            "oracle_closed_before_population",
        ):
            assert report[flag] is True
        assert report["pid"] not in pids
        pids.add(report["pid"])
        rendered = "PickAndPlace" not in report["registration"]
        assert report["rendered_protocol"] == rendered and report["source_internal_size"] == (
            480 if rendered else None
        )
        assert report["source_pin"] == manifest["commit"] and report["source_files"] == manifest["files"]
        assert set(report["versions"]) == {"numpy", "mujoco", "gymnasium", "gymnasium-robotics", "torch", "tensordict"}
        assert report["versions"]["mujoco"] == "3.11.0"
        required = {
            "benchmarks/online/fetch_source_pipeline.py",
            "benchmarks/online/fetch_parallel_source.py",
            "stable_envs/_stepping.py",
            "stable_envs/_adapters/fetch/process.py",
            "stable_envs/envs/fetch_opengl.py",
        }
        assert required <= report["candidate_files"].keys()
        for rel, digest in report["candidate_files"].items():
            assert hashlib.sha256((candidate / rel).read_bytes()).hexdigest() == digest, rel
        rel = "stable_worldmodel/envs/gymnasium_robotics/fetch.py"
        assert [h for f, h in report["source_runtime_files"].items() if f.endswith("/" + rel)] == [
            manifest["files"][rel]
        ]
        identity = {
            k: report[k]
            for k in (
                "registration",
                "host",
                "job",
                "affinity",
                "versions",
                "candidate_files",
                "action_policy",
                "initial_state_policy",
            )
        }
        if provenance is None:
            provenance = identity
        assert identity == provenance
        if rendered:
            gate = results / "image-gates" / f"{device}.json"
            digest = require_image_gate(
                gate, report["registration"], device, manifest, candidate_root=candidate, versions=report["versions"]
            )
            assert digest == report["image_gate_sha256"]
        else:
            assert report["image_gate_sha256"] is None
        selections = ("none", "final", "all") if rendered else ("none",)
        expected = set(itertools.product(batches, (1, 8), selections))
        current = {}
        for cell in report["cells"]:
            key = tuple(cell[k] for k in ("batch", "horizon", "selection"))
            assert key not in current and key in expected
            assert cell["physics_workers"] == (1 if mode == "source_serial" else 8)
            assert cell["exact_state_task"] == cell["output_ownership"] == "PASS"
            assert cell["valid_steps"] == cell["batch"] * cell["horizon"]
            frames = cell["batch"] * (
                cell["horizon"] if cell["selection"] == "all" else int(cell["selection"] == "final")
            )
            assert cell["frames_checked"] == frames
            checked_seconds(cell["seconds"])
            for field in ("initial_state_sha256", "endpoint_state_sha256", "source_tasks_sha256"):
                assert len(cell[field]) == 64 and all(c in "0123456789abcdef" for c in cell[field])
            assert len(cell["source_gl"]) == int(rendered)
            assert len(cell["actor_gl"]) == (
                0 if mode == "ours" or not rendered else 1 if mode == "source_serial" else 8
            )
            for gl in cell["source_gl"] + cell["actor_gl"]:
                require_gl(gl, device)
            if mode == "ours" and frames:
                assert 0 <= cell["mse_mean"] <= cell["mse_max"] <= 0.001
                assert math.isfinite(cell["mse_sd"]) and cell["mse_sd"] >= 0
            else:
                assert all(cell[k] is None for k in ("mse_mean", "mse_sd", "mse_max"))
            current[key] = cell
        assert set(current) == expected
        cells[mode, device] = current
        operations = {"task_no_state_with_ipc_transfer", "get_state", "state_cpu_to_device"}
        if rendered:
            operations.add("render_saved_state")
        seen = set()
        for component in report["components"]:
            key = component["batch"], component["operation"]
            assert mode == "ours" and key not in seen
            seen.add(key)
            seconds = checked_seconds(component["seconds"])
            rates = [component["batch"] / t for t in seconds]
            components.append(
                dict(
                    env=report["registration"],
                    device=device,
                    batch=component["batch"],
                    operation=component["operation"],
                    rate_mean=statistics.mean(rates),
                    rate_sd=statistics.stdev(rates),
                    seconds_mean=statistics.mean(seconds),
                    seconds_sd=statistics.stdev(seconds),
                    rate_unit="frames/s"
                    if component["operation"] == "render_saved_state"
                    else "steps/s"
                    if component["operation"] == "task_no_state_with_ipc_transfer"
                    else "states/s",
                    host=report["host"],
                    job=report["job"],
                )
            )
        assert seen == (set(itertools.product(batches, operations)) if mode == "ours" else set())
        reports[mode, device] = report
    rows = []
    for device in ("cpu", "cuda"):
        for key, ours in sorted(cells["ours", device].items()):
            group = {mode: cells[mode, device][key] for mode in MODES}
            for field in ("initial_state_sha256", "endpoint_state_sha256", "source_tasks_sha256"):
                assert len({cell[field] for cell in group.values()}) == 1
            row = dict(
                env=provenance["registration"],
                device=device,
                batch=key[0],
                horizon=key[1],
                frames=key[2],
                mse_mean=ours["mse_mean"],
                mse_sd=ours["mse_sd"],
                mse_max=ours["mse_max"],
                source_internal_size=reports["ours", device]["source_internal_size"],
                output_size=64,
                host=provenance["host"],
                job=provenance["job"],
            )
            for mode, cell in group.items():
                steps = [cell["valid_steps"] / t for t in cell["seconds"]]
                frames = [cell["frames_checked"] / t for t in cell["seconds"]]
                row[mode + "_steps_s_mean"], row[mode + "_steps_s_sd"] = (
                    statistics.mean(steps),
                    statistics.stdev(steps),
                )
                row[mode + "_fps_mean"] = statistics.mean(frames) if cell["frames_checked"] else None
                row[mode + "_fps_sd"] = statistics.stdev(frames) if cell["frames_checked"] else None
            fastest = max(row[m + "_steps_s_mean"] for m in ("source_serial", "source_parallel"))
            row["speedup_vs_best_same_device_source"] = row["ours_steps_s_mean"] / fastest
            rows.append(row)
    return rows, components


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[64])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows, components = summarize(args.results, args.candidate, args.source, args.source_manifest, args.batches)
    args.output.mkdir(parents=True, exist_ok=False)
    for filename, data in [("online.csv", rows), ("components.csv", components)]:
        with (args.output / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)


if __name__ == "__main__":
    main()
