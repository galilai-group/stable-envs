"""Rebuild this checkpoint's CSV and verify its final production source hashes."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
rows = []
for stage in ("profile-v2", "profile-v3"):
    report = json.loads((ROOT / stage / "result.json").read_text())
    assert report["status"] == "PASS"
    assert all(g["exact_task_qpos_qvel"] == "PASS" for g in report["gates"])
    for r in report["rows"]:
        samples = r["samples"]
        rows.append(
            dict(
                stage=stage,
                host=report["host"],
                env="FetchPush",
                simulator_device="cpu",
                executor=r["kind"],
                batch=r["batch"],
                workers=r["workers"],
                horizon=8,
                repeats=len(samples),
                steps_s_mean=r["steps_s_mean"],
                steps_s_sd=r["steps_s_sd"],
                worker_cpu_cores_mean=statistics.mean(s["worker_cpu_cores"] for s in samples),
                worker_cpu_cores_sd=statistics.stdev(s["worker_cpu_cores"] for s in samples),
                call_seconds_mean=statistics.mean(s["seconds"] for s in samples),
                worker_voluntary_switches_mean=statistics.mean(s["voluntary_switches"] for s in samples),
                worker_involuntary_switches_mean=statistics.mean(s["involuntary_switches"] for s in samples),
                disk_blocks_in=sum(s["disk_blocks_in"] for s in samples),
                disk_blocks_out=sum(s["disk_blocks_out"] for s in samples),
                workload="engine_only_diagnostic"
                if r["kind"] == "engine_threads"
                else "source_task_plus_observation_qpos_qvel_transport",
            )
        )
with (ROOT / "scaling.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
manifest = json.loads((ROOT / "slices-v3/manifest.json").read_text())
for path, sha in manifest.items():
    if path.startswith("stable_envs/"):
        assert hashlib.sha256((REPO / path).read_bytes()).hexdigest() == sha, path
for stage, batch in [("slices-v2", 64), ("slices-v3", 3)]:
    for device in ["cpu", "gpu"]:
        data = json.loads((ROOT / stage / f"fetch-{device}.json").read_text())
        assert data["status"] == "PASS"
        gates = [g for g in data["gates"] if "state_task_thread_and_endpoint" in g]
        assert len(gates) == 12 and all(g["batch"] == batch for g in gates)
summary = dict(
    baseline_commit="3f8a30f7e372391c38f791c72df0d55301523372",
    production_matches_final_gpu_stage=True,
    jobs=dict(
        profile_initial_failed=6916221,
        profile_completed=6916606,
        batch_sweep=6916898,
        slices_pending_cancelled=6916672,
        slices_batch64=6916954,
        slices_final=6917357,
    ),
    local_tests="278 passed, 4 CUDA-only skips; final selector suite rechecked: 84 passed, 4 CUDA skips",
    local_oracle="20 passed, includes DMC/Reacher frozen oracle and semantic harness tests",
    gpu_tests="114 passed, 1 optional Matplotlib skip on each GPU stage; includes four native GPU selector cases",
    fetch="12 registrations independently pass CPU and CUDA-output gates at B64; final source revalidated at B3",
    gpu_scope="Correctness only: batch64 NVIDIA RTX A5500; final source NVIDIA A40. No new GPU performance comparison.",
    production_executor="Unchanged: bounded Fetch thread pool. Process execution is a measured candidate.",
    matrix="39 VERIFIED / 3 INCOMPLETE; unchanged",
    limitations=[
        "CPU profiling uses fixed eight-action chunks and a benchmark-only shard facade.",
        "Process checks cover observations, reward/flags/success, qpos and qvel; not exhaustive State restoration or lifecycle acceptance.",
        "No new renderer throughput/MSE table or simulation/render overlap is claimed.",
        "Concurrent-thread cProfile output is retained but not used for percentage attribution; use serial/process profiles.",
        "Engine-only rows bypass task/controller logic and are not equivalent task simulations.",
    ],
)
(ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print("CSV rows:", len(rows), "Final production hashes: PASS")
