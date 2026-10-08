"""Join only complete, same-host, same-device isolated DMC source comparisons."""

import argparse
import csv
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path

MODES = ("source_serial", "source_parallel", "ours")
OPERATIONS = ("task_no_state_with_ipc_transfer", "get_state", "render_saved_state")


def require_gl(identity, device):
    software = any(x in identity["renderer"].lower() for x in ("llvmpipe", "softpipe", "swrast"))
    assert identity["software"] == software
    assert software if device == "cpu" else "nvidia" in identity["vendor"].lower() and not software


def checked_seconds(values):
    assert len(values) == 6 and all(math.isfinite(t) and t > 0 for t in values)
    return values


def summarize(results, source, source_root, source_manifest, batches):
    manifest = json.loads(source_manifest.read_text())
    assert manifest["commit"] == "176d54532b3c0a251a4f509b74f2fdacc143a63d" and len(manifest["files"]) == 20
    for f, h in manifest["files"].items():
        assert hashlib.sha256((source_root / f).read_bytes()).hexdigest() == h, f
    reports, seen_pids, all_cells, component_rows = {}, set(), {}, []
    expected = set(itertools.product(batches, (1, 8), ("none", "final", "all")))
    provenance = None
    for mode, device in itertools.product(MODES, ("cpu", "cuda")):
        report = json.loads((results / mode / f"{device}.json").read_text())
        assert report["status"] == "PASS" and report["gate_only"] is False
        assert report["mode"] == mode and report["device"] == device
        assert len(report["affinity"]) == report["source_workers"] == report["parent_torch_threads"] == 8
        assert report["source_threads_per_worker"] == 1 and report["size"] == 64
        assert report["gpu"] is None if device == "cpu" else "3090" in report["gpu"]
        assert report["source_physics_device"] == "cpu" and report["source_render_device"] == device
        assert report["source_gpu_supported"] is True and report["source_readback_in_timing"] is True
        assert report["source_state_export_in_timing"] is False and report["ours_state_history"] is False
        assert report["ours_mandatory_final_state"] is True and report["oracle_closed_before_population"] is True
        assert report["fresh_strategy_process"] is True
        assert report["pid"] not in seen_pids
        seen_pids.add(report["pid"])
        assert report["source_pin"] == manifest["commit"] and report["source_files"] == manifest["files"]
        assert set(report["versions"]) == {
            "torch",
            "numpy",
            "gymnasium",
            "mujoco",
            "dm-control",
            "tensordict",
            "mujoco-warp",
            "warp-lang",
        }
        required = {
            "benchmarks/online/dmc_source_pipeline.py",
            "benchmarks/online/dmc_parallel_source.py",
            "benchmarks/online/classic_parallel_source.py",
            "benchmarks/online/common.py",
            "stable_envs/_stepping.py",
            "stable_envs/env.py",
        }
        assert required <= set(report["candidate_files"])
        for rel in (
            "stable_worldmodel/envs/dmcontrol/dmcontrol.py",
            f"stable_worldmodel/envs/dmcontrol/{report['domain']}.py",
        ):
            loaded = [h for f, h in report["source_runtime_files"].items() if f.endswith("/" + rel)]
            assert loaded == [manifest["files"][rel]], rel
        key = {
            f: report[f]
            for f in (
                "domain",
                "host",
                "job",
                "affinity",
                "versions",
                "candidate_files",
                "initial_state_policy",
                "action_policy",
            )
        }
        assert report["job"] is not None
        if provenance is None:
            provenance = key
        else:
            assert key == provenance
        for f, h in report["candidate_files"].items():
            assert hashlib.sha256((source / f).read_bytes()).hexdigest() == h, f
        require_gl(report["opengl"], device)
        cells = {}
        for cell in report["cells"]:
            key = tuple(cell[f] for f in ("batch", "horizon", "selection"))
            assert key not in cells
            assert cell["exact_state_task"] == cell["output_ownership"] == "PASS"
            require_gl(cell["source_gl"], device)
            actors = 0 if mode == "ours" else (1 if mode == "source_serial" else 8)
            assert len(cell["actor_gl"]) == actors
            for identity in cell["actor_gl"]:
                require_gl(identity, device)
            workers = (0 if report["domain"] == "quadruped" else 8) if mode == "ours" else actors
            assert cell["physics_workers"] == workers
            assert cell["valid_steps"] == cell["batch"] * cell["horizon"]
            frames = cell["batch"] * (
                cell["horizon"] if cell["selection"] == "all" else int(cell["selection"] == "final")
            )
            assert cell["frames_checked"] == frames
            if mode == "ours" and frames:
                assert 0 <= cell["mse_mean"] <= cell["mse_max"] <= 0.001
                assert math.isfinite(cell["mse_sd"]) and cell["mse_sd"] >= 0
            else:
                assert all(cell[k] is None for k in ("mse_mean", "mse_sd", "mse_max"))
            checked_seconds(cell["seconds"])
            for name in ("initial_state_sha256", "endpoint_state_sha256", "source_tasks_sha256"):
                assert len(cell[name]) == 64 and all(c in "0123456789abcdef" for c in cell[name])
            cells[key] = cell
        assert set(cells) == expected
        all_cells[mode, device] = cells
        component_keys = set()
        for cell in report["components"]:
            key = cell["batch"], cell["operation"]
            assert key not in component_keys and mode == "ours"
            seconds = checked_seconds(cell["seconds"])
            rates = [cell["batch"] / t for t in seconds]
            component_rows.append(
                dict(
                    domain=report["domain"],
                    device=device,
                    batch=cell["batch"],
                    operation=cell["operation"],
                    rate_mean=statistics.mean(rates),
                    rate_sd=statistics.stdev(rates),
                    rate_unit="frames/s"
                    if cell["operation"] == "render_saved_state"
                    else ("states/s" if cell["operation"] == "get_state" else "steps/s"),
                    seconds_mean=statistics.mean(seconds),
                    seconds_sd=statistics.stdev(seconds),
                    host=report["host"],
                    job=report["job"],
                )
            )
            component_keys.add(key)
        assert component_keys == (set(itertools.product(batches, OPERATIONS)) if mode == "ours" else set())
        reports[mode, device] = report
    rows = []
    for device in ("cpu", "cuda"):
        for key in sorted(expected):
            cells = {mode: all_cells[mode, device][key] for mode in MODES}
            for field in ("initial_state_sha256", "endpoint_state_sha256", "source_tasks_sha256"):
                assert len({cell[field] for cell in cells.values()}) == 1, (device, key, field)
            ours = cells["ours"]
            row = dict(
                domain=provenance["domain"],
                device=device,
                batch=key[0],
                horizon=key[1],
                selection=key[2],
                source_render_device=device,
                source_physics_device="cpu",
                source_gpu_supported=True,
                source_parallel_workers=8,
                ours_physics_workers=ours["physics_workers"],
                mse_mean=ours["mse_mean"],
                mse_sd=ours["mse_sd"],
                mse_max=ours["mse_max"],
                frames_checked=ours["frames_checked"],
                valid_steps=ours["valid_steps"],
                host=provenance["host"],
                job=provenance["job"],
            )
            for mode, cell in cells.items():
                rates = [cell["valid_steps"] / t for t in cell["seconds"]]
                row[f"{mode}_steps_per_s_mean"] = statistics.mean(rates)
                row[f"{mode}_steps_per_s_sd"] = statistics.stdev(rates)
                frames = [cell["frames_checked"] / t for t in cell["seconds"]] if cell["frames_checked"] else []
                row[f"{mode}_fps_mean"] = statistics.mean(frames) if frames else None
                row[f"{mode}_fps_sd"] = statistics.stdev(frames) if frames else None
            row["ours_vs_serial"] = row["ours_steps_per_s_mean"] / row["source_serial_steps_per_s_mean"]
            row["ours_vs_parallel"] = row["ours_steps_per_s_mean"] / row["source_parallel_steps_per_s_mean"]
            row["ours_vs_best_source"] = min(row["ours_vs_serial"], row["ours_vs_parallel"])
            rows.append(row)
    return rows, component_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, components = summarize(args.results, args.source, args.source_root, args.source_manifest, args.batches)
    args.output.mkdir(parents=True, exist_ok=False)
    for name, records in (("comparison.csv", rows), ("components.csv", components)):
        with (args.output / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)
    print("Validated", len(rows), "source-comparison rows and", len(components), "component rows")


if __name__ == "__main__":
    main()
