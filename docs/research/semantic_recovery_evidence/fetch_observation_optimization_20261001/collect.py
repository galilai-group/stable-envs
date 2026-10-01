"""Derive before/after tables from preserved terminal results."""

import csv
import json
import statistics as st
from pathlib import Path

root = Path(__file__).resolve().parent
bench = json.loads((root / "benchmark/result.json").read_text())
adapter = json.loads((root / "acceptance/adapter-timing.json").read_text())
assert bench["status"] == adapter["status"] == "PASS"
rows = []
for batch, processes, threads in sorted({(r["batch"], r["processes"], r["threads"]) for r in bench["rows"]}):
    pair = {}
    for optimized in (False, True):
        samples = [
            s
            for r in bench["rows"]
            if (r["batch"], r["processes"], r["threads"], r["optimized"]) == (batch, processes, threads, optimized)
            for s in r["samples"]
        ]
        pair[optimized] = dict(mean=st.mean(s["steps_s"] for s in samples), sd=st.stdev(s["steps_s"] for s in samples))
    rows.append(
        dict(
            scope="source_task_and_transport",
            batch=batch,
            processes=processes,
            threads=threads,
            baseline_mean=pair[False]["mean"],
            baseline_sd=pair[False]["sd"],
            optimized_mean=pair[True]["mean"],
            optimized_sd=pair[True]["sd"],
            speedup=pair[True]["mean"] / pair[False]["mean"],
        )
    )
for r in adapter["rows"]:
    rows.append(
        dict(
            scope="production_adapter_step_including_state",
            batch=r["batch"],
            processes=1,
            threads=r["workers"],
            baseline_mean=r["baseline"]["mean"],
            baseline_sd=r["baseline"]["sd"],
            optimized_mean=r["optimized"]["mean"],
            optimized_sd=r["optimized"]["sd"],
            speedup=r["optimized"]["mean"] / r["baseline"]["mean"],
        )
    )
with (root / "comparison.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
for r in rows:
    print(
        f"{r['scope']} B{r['batch']} {r['processes']}p{r['threads']}t: "
        f"{r['baseline_mean']:.0f} ± {r['baseline_sd']:.0f} → {r['optimized_mean']:.0f} ± {r['optimized_sd']:.0f}, {r['speedup']:.2f}×"
    )
