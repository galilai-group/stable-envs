"""Summarize preserved raw profiling records. Run only in this evidence folder."""

import csv
import json
import statistics as st
from pathlib import Path

root = Path(__file__).resolve().parent
raw = json.loads((root / "result.json").read_text())
assert raw["status"] == "PASS"
transitions = raw["batch"] * raw["horizon"]
summary = []
for config in dict.fromkeys(r["config"] for r in raw["rows"]):
    rows = [r for r in raw["rows"] if r["config"] == config]
    plain = [s for r in rows if not r["instrumented"] for s in r["samples"]]
    profiled = [s for r in rows if r["instrumented"] for s in r["samples"]]
    row = dict(
        config=config,
        steps_s_mean=st.mean(s["steps_s"] for s in plain),
        steps_s_sd=st.stdev(s["steps_s"] for s in plain),
        profiled_steps_s_mean=st.mean(s["steps_s"] for s in profiled),
    )
    phases = []
    for sample in profiled:
        totals = {
            k: 0.0
            for k in (
                "dispatch_ms",
                "critical_worker_start_delay_ms",
                "critical_worker_task_ms",
                "join_tail_ms",
                "scatter_ms",
                "pack_ms",
                "return_ms",
                "copy_ms",
            )
        }
        workers = []
        for t, call in enumerate(sample["calls"]):
            children = [c[t] for c in sample["children"]]
            # The last child to finish defines this step's observed critical path.
            last = max(children, key=lambda c: c["end"])
            worker = max(last["workers"], key=lambda w: w["end"])
            values = (
                last["start"] - call["start"],
                worker["start"] - last["start"],
                worker["end"] - worker["start"],
                last["joined"] - worker["end"],
                last["pack_start"] - last["joined"],
                last["end"] - last["pack_start"],
                call["returned"] - last["end"],
                call["copied"] - call["returned"],
            )
            for key, value in zip(totals, values):
                totals[key] += value * 1000
            for child in children:
                workers.extend(child["workers"])
        for key in workers[0]:
            if key not in ("start", "end"):
                totals[key + ("_us_per_transition" if key.endswith(("wall", "cpu")) else "_per_chunk")] = sum(
                    w[key] for w in workers
                ) * (1e6 / transitions if key.endswith(("wall", "cpu")) else 1)
        phases.append(totals)
    for key in phases[0]:
        row[key] = st.mean(p[key] for p in phases)
    summary.append(row)
(root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
with (root / "summary.csv").open("w") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
print(json.dumps(summary, indent=2))
