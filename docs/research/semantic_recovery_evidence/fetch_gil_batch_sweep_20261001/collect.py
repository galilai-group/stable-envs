"""Summarize immutable direct GIL and batch sweep evidence."""

import csv
import json
import statistics as st
from pathlib import Path

root = Path(__file__).resolve().parent
raw = json.loads((root / "result.json").read_text())
assert raw["status"] == "PASS"
summary = []
keys = sorted({(r["batch"], r["processes"], r["threads"], r["shared"]) for r in raw["rows"]})
for batch, processes, threads, shared in keys:
    rows = [
        r
        for r in raw["rows"]
        if (r["batch"], r["processes"], r["threads"], r["shared"]) == (batch, processes, threads, shared)
    ]
    baseline = [s for r in rows if not r["sampled"] for s in r["samples"]]
    sampled = [s for r in rows if r["sampled"] for s in r["samples"]]
    occupancy = [s["gil_occupancy"] * 100 for s in sampled]
    total = sum(sum(s["gil_samples"]) for s in sampled)
    held = sum(sum(s["gil_held"]) for s in sampled)
    worker_rates = [
        sum(s["gil_held"][i] for s in sampled) / sum(s["gil_samples"][i] for s in sampled) * 100
        for i in range(processes)
    ]
    row = dict(
        batch=batch,
        processes=processes,
        threads_per_process=threads,
        shared_memory=shared,
        repeats=len(baseline),
        sampled_repeats=len(sampled),
        steps_s_mean=st.mean(s["steps_s"] for s in baseline),
        steps_s_sd=st.stdev(s["steps_s"] for s in baseline),
        batch_step_ms_mean=st.mean(s["seconds"] / 8 * 1000 for s in baseline),
        batch_step_ms_sd=st.stdev(s["seconds"] / 8 * 1000 for s in baseline),
        worker_cpu_cores_mean=st.mean(s["worker_cpu_cores"] for s in baseline),
        voluntary_switches_per_transition=st.mean(sum(s["voluntary_switches"]) / (batch * 8) for s in baseline),
        sampled_steps_s_mean=st.mean(s["steps_s"] for s in sampled),
        gil_held_percent_mean=st.mean(occupancy),
        gil_held_percent_sd=st.stdev(occupancy),
        gil_held_percent_pooled=held / total * 100,
        gil_samples=total,
        gil_held_samples=held,
        gil_worker_percent_min=min(worker_rates),
        gil_worker_percent_max=max(worker_rates),
        exact_trajectory="PASS",
    )
    row["sampled_throughput_change_percent"] = 100 * (row["sampled_steps_s_mean"] / row["steps_s_mean"] - 1)
    summary.append(row)
with (root / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for shared in (True, False):
    print("SHARED" if shared else "PIPES")
    for batch in sorted({r["batch"] for r in summary}):
        p, h = [
            next(r for r in summary if r["batch"] == batch and r["processes"] == n and r["shared_memory"] == shared)
            for n in (8, 4)
        ]
        print(
            f"{batch}: {p['steps_s_mean']:.0f} ± {p['steps_s_sd']:.0f} | {h['steps_s_mean']:.0f} ± {h['steps_s_sd']:.0f} | "
            f"hybrid {100 * (h['steps_s_mean'] / p['steps_s_mean'] - 1):+.1f}% | "
            f"GIL {p['gil_held_percent_mean']:.1f} ± {p['gil_held_percent_sd']:.1f}% / "
            f"{h['gil_held_percent_mean']:.1f} ± {h['gil_held_percent_sd']:.1f}% | "
            f"CPU {p['worker_cpu_cores_mean']:.2f} / {h['worker_cpu_cores_mean']:.2f}"
        )
