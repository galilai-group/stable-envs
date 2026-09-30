"""Export actual paired measurements; never infer source timings or fill failures."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("root", type=Path)
    a = p.parse_args()
    plan = json.loads((a.root / "plan.json").read_text())
    hashes, data, long_rows = {}, {}, []
    for task in plan["tasks"]:
        path = a.root / ("cpufull" if task["device"] == "cpu" else "raw") / str(task["index"]) / "result.json"
        retry = a.root / "cpuretry2" / str(task["index"]) / "result.json"
        if task["device"] == "cpu" and retry.exists() and json.loads(retry.read_text())["status"] == "COMPLETE":
            path = retry
        r = json.loads(path.read_text()) if path.exists() else {}
        data[task["registration"], task["device"]] = r
        if not r:
            continue
        hashes[str(path.relative_to(a.root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for m in r["measurements"]:
            record = dict(
                env=r["registration"],
                candidate_device=r["device"],
                source_render_device=r.get("source_render_device"),
                output_device=m["output_device"],
                batch=m["batch"],
                status=r["status"],
                backend=r["backend"],
                cpu=r["profile"]["cpu"],
                gpu=r["profile"]["gpu"],
                host=r["profile"]["host"],
                independent_states=r["independent_states"],
                source_batch_method=r["source_batch_method"],
                speedup_source_over_ours=m["speedup"],
                paired_mse_mean=m.get("paired_mse", r["paired_mse"])["mean"],
                paired_mse_sd=m.get("paired_mse", r["paired_mse"])["sd"],
                evidence=str(path.relative_to(a.root)),
            )
            for side in ["ours", "source"]:
                for key in ["mean_ms_per_frame", "sd_ms_per_frame", "frames_per_second", "count"]:
                    record[f"{side}_{key}"] = m[side][key]
            long_rows.append(record)
    rows = []
    for task in plan["tasks"][:42]:
        env = task["registration"]
        row = dict(env=env, scope=task["scope"], batch=256)
        for device, label in [("cpu", "cpu"), ("cuda", "gpu")]:
            r = data.get((env, device), {})
            row[f"{label}_status"] = r.get("status", "PENDING")
            row[f"{label}_host"] = r.get("profile", {}).get("host", "")
            row[f"{label}_hardware"] = r.get("profile", {}).get("gpu" if device == "cuda" else "cpu", "")
            row[f"{label}_backend"] = r.get("backend", "")
            row[f"{label}_source_pixel_variation_note"] = (
                r.get("source_pixel_variation", "") if r.get("source_warm_repeat_max_uint8", 0) > 1 else ""
            )
            row[f"{label}_source_render_device"] = r.get("source_render_device", "")
            row[f"{label}_paired_mse_mean"] = r.get("paired_mse", {}).get("mean", "")
            row[f"{label}_paired_mse_sd"] = r.get("paired_mse", {}).get("sd", "")
            for destination in ["cpu"] if device == "cpu" else ["cpu", "cuda"]:
                matches = [
                    m for m in r.get("measurements", []) if m["batch"] == 256 and m["output_device"] == destination
                ]
                m = matches[0] if matches else {}
                prefix = f"{label}_render_to_{destination}"
                for side in ["ours", "source"]:
                    for key in ["mean_ms_per_frame", "sd_ms_per_frame", "frames_per_second", "count"]:
                        row[f"{prefix}_{side}_{key}"] = m.get(side, {}).get(key, "")
                row[f"{prefix}_speedup_source_over_ours"] = m.get("speedup", "")
                row[f"{prefix}_paired_mse_mean"] = m.get("paired_mse", {}).get("mean", "")
                row[f"{prefix}_paired_mse_sd"] = m.get("paired_mse", {}).get("sd", "")
        rows.append(row)
    for filename, records in [("paired_rendering_42_b256.csv", rows), ("paired_rendering_all_batches.csv", long_rows)]:
        if not records:
            continue
        with (a.root / filename).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    summary = []
    for row in rows:
        item = dict(env=row["env"], batch=256, output_location="cpu", scope=row["scope"])
        for label in ["cpu", "gpu"]:
            item[f"{label}_status"] = row[f"{label}_status"]
            item[f"{label}_hardware"] = row[f"{label}_hardware"]
            item[f"{label}_host"] = row[f"{label}_host"]
            item[f"{label}_backend"] = row[f"{label}_backend"]
            item[f"{label}_source_pixel_variation_note"] = row[f"{label}_source_pixel_variation_note"]
            item[f"reference_device_for_{label}_comparison"] = row[f"{label}_source_render_device"]
            prefix = f"{label}_render_to_cpu"
            for side, title in [("ours", "ours"), ("source", "ground_truth")]:
                for statistic in ["mean", "sd"]:
                    item[
                        f"{title}_{label}_ms_per_image_{statistic}"
                        if side == "ours"
                        else f"ground_truth_for_{label}_comparison_ms_per_image_{statistic}"
                    ] = row[f"{prefix}_{side}_{statistic}_ms_per_frame"]
            item[f"ours_{label}_fps"] = row[f"{prefix}_ours_frames_per_second"]
            source_device = row[f"{label}_source_render_device"]
            expected_device = "cuda" if label == "gpu" else "cpu"
            matched_device = source_device == expected_device
            item[f"ground_truth_{label}_fps"] = row[f"{prefix}_source_frames_per_second"] if matched_device else ""
            item[f"ours_{label}_vs_ground_truth_{label}_speedup"] = (
                row[f"{prefix}_speedup_source_over_ours"] if matched_device else ""
            )
            if label == "gpu":
                item["ground_truth_cpu_fps_on_gpu_host"] = (
                    row[f"{prefix}_source_frames_per_second"] if source_device == "cpu" else ""
                )
                item["ours_gpu_vs_cpu_only_ground_truth_speedup"] = (
                    row[f"{prefix}_speedup_source_over_ours"] if source_device == "cpu" else ""
                )
            for statistic in ["mean", "sd"]:
                item[f"{label}_mse_{statistic}"] = row[f"{prefix}_paired_mse_{statistic}"]
        summary.append(item)
    with (a.root / "paired_rendering_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    (a.root / "result-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(dict(Counter(r.get("status", "PENDING") for r in data.values())))


if __name__ == "__main__":
    main()
