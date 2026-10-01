"""Summarize all hybrid runs, retaining the noisy initial attempt."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
rows = []
for stage in ("initial", "controlled", "confirmation"):
    report = json.loads((ROOT / stage / "result.json").read_text())
    assert report["status"] == "PASS"
    assert len(report["rows"]) == len(report["gates"]) == (4 if stage == "confirmation" else 12)
    assert all(g["exact_task_qpos_qvel_and_ownership"] == "PASS" for g in report["gates"])
    for name, expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / stage / name).read_bytes()).hexdigest() == expected, name
    for r in report["rows"]:
        samples = r["samples"]
        if stage != "initial":
            assert report["pin_workers"]
            cpus = [c for worker in r["worker_info"] for c in worker["affinity"]]
            assert len(cpus) == len(set(cpus)) == 8
            assert all(len(worker["affinity"]) == r["threads_per_process"] for worker in r["worker_info"])
        rows.append(
            dict(
                run=stage,
                env="FetchPush",
                host=report["host"],
                batch=r["batch"],
                horizon=8,
                processes=r["processes"],
                threads_per_process=r["threads_per_process"],
                shared_memory=r["shared_memory"],
                pin_workers=report.get("pin_workers", False),
                repeats=r["repeats"],
                steps_s_mean=r["steps_s_mean"],
                steps_s_sd=r["steps_s_sd"],
                worker_cpu_cores_mean=statistics.mean(s["worker_cpu_cores"] for s in samples),
                parent_cpu_cores_mean=statistics.mean(s["parent_cpu_cores"] for s in samples),
                call_seconds_mean=statistics.mean(s["seconds"] for s in samples),
                voluntary_switches_mean=statistics.mean(s["voluntary_switches"] for s in samples),
                construction_seconds=r["construction_seconds"],
            )
        )
rows.sort(key=lambda r: (r["run"], r["batch"], r["processes"], not r["shared_memory"]))
with (ROOT / "hybrid.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
comparisons = []
for batch in (64, 256, 1024):
    selected = [
        r for r in rows if r["batch"] == batch and r["run"] == ("confirmation" if batch == 256 else "controlled")
    ]
    lookup = {(r["processes"], r["shared_memory"]): r for r in selected}
    comparisons.append(
        dict(
            batch=batch,
            hybrid_shared_vs_8_process_shared=lookup[4, True]["steps_s_mean"] / lookup[8, True]["steps_s_mean"],
            hybrid_pipe_vs_8_process_pipe=lookup[4, False]["steps_s_mean"] / lookup[8, False]["steps_s_mean"],
            hybrid_shared_vs_hybrid_pipe=lookup[4, True]["steps_s_mean"] / lookup[4, False]["steps_s_mean"],
        )
    )
(ROOT / "summary.json").write_text(
    json.dumps(
        dict(
            status="PASS",
            jobs={"initial": 6922309, "controlled": 6922504, "confirmation": 6923133},
            comparisons=comparisons,
            production_changes=False,
            scope=report["scope"],
        ),
        indent=2,
    )
    + "\n"
)
print(json.dumps(comparisons, indent=2))
