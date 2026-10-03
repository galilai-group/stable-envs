# Online source-performance audit — five measured, thirty still open

This is a reconciliation of existing evidence at `fd58472e3ced25677f0f82e3ec24b1d04ac7d7a6`,
not a fresh benchmark run or an acceptance promotion. It keeps online pipeline
throughput separate from renderer-only results and library-versus-library controls.
The [coverage CSV](coverage.csv) has all 42 registrations; the
[measured CSV](measured_online.csv) retains all 120 completed source-comparison
rows rather than selecting only fast batches or favorable comparisons.

## What is actually complete?

Of the **35 accepted non-Craftax registrations**:

| Evidence status | Count | Registrations |
|---|---:|---|
| Completed, qualified source comparison | 5 | Classic control |
| Source timing campaign prepared, not submitted | 6 | Acrobot/Cartpole/Pendulum DMC; Navigation three |
| Source image gate prepared, not submitted | 3 | OGBMaze/Cube/Scene |
| No completed online source comparison in this checkpoint | 21 | Fetch twelve; other DMC nine, including Quadruped |

The four Craftax registrations remain deferred; PushT two and Rocket remain
correctness-blocked. The semantic matrix is 39 VERIFIED / 3 INCOMPLETE. The routing
tally for the accepted non-Craftax tasks is 27 eligible process routes, three
native Torch routes and five local routes. None of these counts is a source-speedup
count. Fetch's completed pipeline/adapter sweeps compare library implementations;
DMC's completed pipeline sweep also uses a library control. Those measurements
are useful but cannot fill the source columns in this table.

## Completed source results at a common workload

B256, eight actions per lane, every frame requested, no intermediate State
history; the library still returns complete final State. Each number below is
online FPS, not isolated rendering or raw physics throughput. Six samples per
strategy/device; the CSVs include sample SD. Every measured frame has normalized
source MSE zero in this bounded seed-17 workload.

| Registration | Source CPU serial | Source CPU P8 | Ours CPU | CPU / best source | Source CPU P8 + CUDA delivery | Ours GPU | GPU / best CPU + delivery |
|---|---:|---:|---:|---:|---:|---:|---:|
| AcrobotControl | 338 | 1,772 | 368 | 0.21x | 781 | 11,394 | 14.58x |
| CartPoleControl | 229 | 1,254 | 562 | 0.45x | 941 | 13,443 | 14.29x |
| MountainCarContinuousControl | 451 | 2,076 | 563 | 0.27x | 1,860 | 14,069 | 7.56x |
| MountainCarControl | 450 | 2,036 | 533 | 0.26x | 1,877 | 12,656 | 6.74x |
| PendulumControl | 273 | 1,500 | 524 | 0.35x | 1,402 | 12,190 | 8.69x |

Source pygame has **no native GPU renderer**. Therefore source-GPU FPS cells stay
blank; GPU ratios are explicitly labeled permitted CPU-source fallbacks with CUDA
delivery. They are never described as GPU-versus-GPU wins. Source serial and
parallel rates remain available in the detailed CSV. Each comparison stays within
its own allocation; task rows come from different hosts and are not pooled.

The completed campaign limited the library's parent Torch renderer to one CPU
thread while allocating eight cores. All five CPU results lose to the parallel
source in this configuration. This is **not the best demonstrated CPU renderer
configuration**: a separate thread sweep shows substantial gains with eight
threads, and source comparison array 6971892 was submitted with that setting.
Its results remain uncollected while Oscar authentication is unavailable. We do
not replace these measured numbers with projections or infer that the job ended.

## Validation and provenance

The generator checks all frozen closure hashes, requires five terminal jobs with
COMPLETED/0:0 and matching report hosts/IDs, then reruns the existing strict classic
summarizer against the original frozen candidate. It reproduces each committed
per-registration CSV byte-for-byte. That validates 120 correctness cells and 360
six-sample timing cells. The recorded production file hashes also match the current
checkout; this is a check of the files named by those reports, not a new full-code
performance test. Full proof remains in the original reports and campaign manifests.

`audit.json` records input hashes and explicit incomplete status. No missing FPS,
MSE or source-GPU result is filled with zero. No-frame rows retain blank FPS and
MSE; their steps/s still include the requested public call and final State. They
must not be relabeled bare simulator throughput. Offline renderer-only tables,
small local smoke timings and internal-control speedups are deliberately separate
measurement categories and have not been used to manufacture source results.

Reproduce into a new directory:

```bash
python3 scripts/benchmarks/online_stepping/audit_online_source_results.py \
  --frozen-classic-code /path/to/classic-parallel-source-v1-20261003/code \
  --output /tmp/new-online-source-audit
```

Next: collect the submitted eight-thread classic/source and OGBench validator
jobs; run the prepared DMC/Navigation source campaigns and OGBench image gate;
add direct same-device Fetch source comparisons, then cover remaining DMC and
larger/sustained workloads. Preserve per-registration acceptance and source
correctness instead of extrapolating a sibling's timing or MSE.
