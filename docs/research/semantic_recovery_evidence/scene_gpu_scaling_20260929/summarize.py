"""Export the completed Scene scaling diagnostic without rerunning rendering."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    path = ROOT / "results/30/result.json"
    result = json.loads(path.read_text())
    assert result["status"] == "COMPLETE"
    assert result["state_nonmutation_passed"]
    rows = []
    for measurement in result["measurements"]:
        ours, source = measurement["ours"], measurement["source"]
        paired = [s - o for s, o in zip(source["samples_ms"], ours["samples_ms"], strict=True)]
        n = len(paired)
        # Seven samples in this experiment. The interval is descriptive: thermal
        # drift and serial correlation can violate independent-sample assumptions.
        critical = {7: 2.446912, 8: 2.364624, 9: 2.306004, 10: 2.262157,
                    11: 2.228139, 12: 2.200985, 13: 2.178813,
                    14: 2.160369, 15: 2.144787}[n]
        difference = statistics.mean(paired)
        half_width = critical * statistics.stdev(paired) / math.sqrt(n)
        row = {
            "env": "swm/OGBScene-v0", "device": "cuda",
            "batch": measurement["batch"], "output_device": measurement["output_device"],
            "mse_mean": measurement["paired_mse"]["mean"],
            "mse_sd": measurement["paired_mse"]["sd"],
            "ours_fps": ours["frames_per_second"],
            "ours_fps_sample_sd": statistics.stdev(1000 * measurement["batch"] / t for t in ours["samples_ms"]),
            "source_fps": source["frames_per_second"],
            "source_fps_sample_sd": statistics.stdev(1000 * measurement["batch"] / t for t in source["samples_ms"]),
            "speedup": measurement["speedup"],
            "ours_ms_per_frame": ours["mean_ms_per_frame"],
            "ours_ms_per_frame_sd": ours["sd_ms_per_frame"],
            "source_ms_per_frame": source["mean_ms_per_frame"],
            "source_ms_per_frame_sd": source["sd_ms_per_frame"],
            "paired_samples": n,
            "paired_saved_ms_per_batch": difference,
            "paired_saved_ms_ci95_low": difference - half_width,
            "paired_saved_ms_ci95_high": difference + half_width,
            "source_state_count": result["independent_states"],
        }
        for label in ("ours", "source"):
            phase = measurement.get("telemetry", {}).get(label, {})
            row[f"{label}_telemetry_status"] = phase.get("status", "NOT_SAMPLED")
            row[f"{label}_process_cpu_duty_percent"] = phase.get("process_cpu_duty_percent")
            row[f"{label}_other_gpu_pids"] = json.dumps(phase.get("other_gpu_pids"))
            for metric in ("gpu_busy_percent", "memory_busy_percent", "vram_used_mib", "power_watts"):
                stats = phase.get("metrics", {}).get(metric, {})
                row[f"{label}_{metric}_mean"] = stats.get("mean")
                row[f"{label}_{metric}_sd"] = stats.get("sd")
        rows.append(row)
    with (ROOT / "scene_scaling.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(ROOT.rglob("*")) if p.is_file() and p.name != "sha256.json"}
    (ROOT / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
