# Report reproduction

The [report](report.md) and `summary.json` combine the retained raw measurements.
Every accepted registration has its own CPU and GPU record. GPU records cover
rendering for 37 registrations and dynamics/State operations for all four Craftax
registrations; other families' dynamics are CPU-backed. Symbolic Craftax has no
renderer. Raw JSON retains every operation's status and refusal reason.

From `/private/tmp/stable-envs-semantic-recovery`, the exact report command was:

```sh
PYTHONPATH=.:/private/tmp/native-nav-py312-deps .venv/bin/python scripts/benchmarks/report_state_matrix.py \
  --cpu docs/research/semantic_recovery_evidence/state_performance_cpu \
  --cpu docs/research/semantic_recovery_evidence/state_performance_cpu_retry \
  --gpu docs/research/semantic_recovery_evidence/state_performance_gpu_6693308/results \
  --gpu docs/research/semantic_recovery_evidence/state_performance_gpu_6694195/results \
  --cold docs/research/semantic_recovery_evidence/state_performance_cold_cpu \
  --cold docs/research/semantic_recovery_evidence/state_performance_cold_gpu_6694865/results \
  --dmc-native docs/research/semantic_recovery_evidence/state_performance_dmc_native/result.json \
  --output docs/research/semantic_recovery_evidence/state_performance_report
```

Use a **fresh output directory** when reproducing. No accepted evidence collector
is invoked. The report verifies registration/batch coverage and recomputes timing
statistics from retained samples before writing. The final checkpoint independently
checks source artifact hashes and evidence seals. The PNG and SVG are standalone
exports of the same CPU B1 plot.

Individual measurement commands, Python environments and package versions are in
each raw result and the CPU README files. GPU submission scripts, source bundle
hashes, corrected-profile retry and terminal logs are in
`../state_performance_gpu_submission/`. The initial failed CPU/GPU records are
preserved and only their corrected, complete successors enter these tables.

The main harness recorded first measured calls after checkpoint preparation;
the separate cold supplement uses sixteen fresh processes, covering public and
oracle APIs for each Craftax registration on CPU and GPU. These individual cold
observations are not latency distributions. Warm measurements retain at least
10 warmups and 50 samples. B16 is measured except B4 for Cube/Scene; B64 is deferred
for runtime, not declared unsupported.
