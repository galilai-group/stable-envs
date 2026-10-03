# Five-registration direct-source online checkpoint

All five array tasks in 6968542 independently completed with exit 0:0. Each
registration passes 36 complete-State/task gates and 72 timing cells across
CPU/CUDA output, B1/64/256, T1/8 and no/final/all frames. Each also has 18 component
cells. Every measured source image MSE is zero. These are six-sample means and
sample standard deviations, with startup/restore/validation outside timing.

At B256/T8/all frames (FPS, mean +/- SD):

| Registration | Source CPU | Ours CPU | Ours GPU |
|---|---:|---:|---:|
| swm/CartPoleControl-v1 | 373 +/- 2.5 | 506 +/- 0.3 | 10,129 +/- 15.7 |
| swm/PendulumControl-v1 | 276 +/- 2.2 | 539 +/- 1.7 | 12,211 +/- 34.5 |
| swm/AcrobotControl-v1 | 216 +/- 3.9 | 338 +/- 6.2 | 11,569 +/- 491.1 |
| swm/MountainCarControl-v0 | 451 +/- 1.5 | 558 +/- 1.2 | 12,466 +/- 23.9 |
| swm/MountainCarContinuousControl-v0 | 451 +/- 1.3 | 553 +/- 0.5 | 14,111 +/- 38.7 |

Source physics/rendering are serial CPU; source GPU rendering is unsupported.
The main CSV separately includes source CPU images delivered to GPU memory.
GPU/native-CPU ratios use the explicitly permitted fallback; they are not
GPU-versus-GPU claims. Acrobot uses eight candidate physics owners at large
batches; the other four use local scalar physics. All have the same allocated
eight-CPU budget, but this is not a parallel-source performance comparison.

The workload requests no intermediate State history. Ours still returns complete
final State, which source does not supply. Plain stepping remains slower than
source in some registrations; source rendering dominates its combined pipeline.
Do not reuse these ratios as raw physics speedups, all-State-history results, or
claims that the 35-registration goal is complete.

`pipeline.csv` has 180 rows, `components.csv` 90, and `batch256_allframes.csv` five.
The reports and raw logs remain in ../completed. Earlier two-registration outputs
are preserved unchanged. Reproduce all tables in a fresh directory:

```sh
python scripts/benchmarks/online_stepping/collect_classic_source_pipeline.py \
  docs/research/semantic_recovery_evidence/classic_source_pipeline_20261003/completed/results \
  NEW_OUTPUT_DIRECTORY --source FROZEN_CANDIDATE_CODE
```

Remaining work includes a resource-matched parallel-source baseline, larger
memory/performance sweeps, and a gated classic `_advance_without_snapshot` path
to avoid exporting unrequested intermediate State. The existing shared API and
renderer fidelity need no redesign for that candidate.
