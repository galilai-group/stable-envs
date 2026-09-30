"""Readable mean ± sample-SD view of the independently exported numeric CSV."""

import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    with a.input.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 42

    def mean_sd(row, prefix, precision):
        mean, sd = row[prefix + "_mean"], row[prefix + "_sd"]
        return f"{float(mean):.{precision}g} ± {float(sd):.{precision}g}" if mean and sd else ""

    output = []
    for row in rows:
        gpu = "ours_gpu_vs_source_gpu" if row["source_gpu_status"] == "COMPLETE" else "ours_gpu_vs_source_cpu"
        item = {
            "env": row["env"],
            "mse CPU vs source CPU (mean ± SD)": mean_sd(row, "ours_cpu_vs_source_cpu_mse", 7),
            "mse GPU vs source (mean ± SD)": mean_sd(row, gpu + "_mse", 7),
            "GPU comparison source": "GPU" if gpu.endswith("source_gpu") else "CPU",
            "mse source CPU vs GPU (mean ± SD)": mean_sd(row, "source_cpu_vs_source_gpu_mse", 7),
            "time on CPU ms (mean ± SD)": mean_sd(row, "ours_cpu_time_ms", 6),
            "time on GPU ms (mean ± SD)": mean_sd(row, "ours_gpu_time_ms", 6),
            "original time CPU ms (mean ± SD)": mean_sd(row, "source_cpu_time_ms", 6),
            "original time GPU ms (mean ± SD)": mean_sd(row, "source_gpu_time_ms", 6),
            "max MSE CPU": row["ours_cpu_vs_source_cpu_mse_max"],
            "max MSE GPU": row[gpu + "_mse_max"],
            "CPU backend": row["ours_cpu_backend"],
            "GPU backend": row["ours_gpu_backend"],
            "both devices all 1000 frames ≤ 0.001": row["cpu_and_gpu_max_mse_le_001"],
            "both devices all 1000 frames ≤ 0.0005": row["cpu_and_gpu_max_mse_le_0005"],
            "scope": row["scope"],
            "source CPU vs GPU status": row["source_cpu_vs_source_gpu_status"],
        }
        output.append(item)
    with a.output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
