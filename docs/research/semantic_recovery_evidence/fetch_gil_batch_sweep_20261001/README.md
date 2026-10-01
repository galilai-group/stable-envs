# Direct GIL occupancy and Fetch batch scaling

Continuation of the [hybrid attribution experiment](../fetch_hybrid_attribution_20261001/README.md).
Base integration SHA: `1f1952c396d706bafff9ed0724dadaecd4e56a60`.
Benchmark-only changes; no production executor, semantic promotion or renderer change.

## What is measured

The two layouts are eight processes × one thread and four processes × two
threads. Both use **eight simulation cores**. The job reserves a ninth physical
core solely for native observation threads; it is excluded from simulator and
parent affinity in both sampled and unprofiled runs. The benchmark runs both
layouts with shared observations and pipe transport, batches 64–1,024, eight
actions per lane, seed 19 actions and seed 17 + lane for source resets.

Throughput is measured with the GIL sampler **stopped and joined**. Each cell has
ten repeats in two blocks with reversed layout order. It includes source task
stepping, observation/reward/flag/success/qpos/qvel packing, transport, and owned
parent copies. It excludes construction, reset, warmup and correctness checks.
These are aggregate **environment transitions/s**, not renderer FPS or batch/s.
Batch/s for a single action per lane is transitions/s divided by batch size.

Each measured chunk, including sampled chunks, is compared exactly against an
independent serial source rollout for every collected field. The trajectory
does not reach the 50-step TimeLimit. This is not a full State/lifecycle gate.
Larger batches add differently seeded lanes, so increasing batch also changes
the mixture of contact workloads; it is not a pure fixed-work scheduler test.

## Direct GIL observation

Oscar sets `kernel.yama.ptrace_scope=2`, preventing ordinary external profiler
attachment. No permissions or machine security settings were changed. Instead,
`gil_probe.c` builds a diagnostic extension against the exact installed CPython
3.12.9 headers. A native pthread reads the actual atomic
`interp->ceval.gil->locked` field without acquiring the GIL or calling Python.
CPython documents this field as the lock's taken/released state in its
[3.12.9 source](https://github.com/python/cpython/blob/v3.12.9/Include/internal/pycore_gil.h).
The extension checks the runtime version and requires the lock to be held
during initialization. Header paths/hashes and the compile command are recorded.

Checks occur after randomized 0.5–1.5 ms sleeps; OS scheduling adds delay. The
randomization avoids synchronizing checks with a repeating task phase. Sampling
is enabled only inside each child shard's `step`, including its thread waits
and packing. Reset, construction, IPC waiting outside `step`, and parent work
are excluded. Boundary counters are synchronized with a native mutex; that
mutex is separate from the GIL. Raw per-worker, per-chunk held/total/invalid
counts are preserved. Every sampled configuration runs at least five chunks
and three seconds of measured work in each of two order blocks.

**Occupancy is the fraction of checks at which the process's GIL was held.**
It is a sampled estimate of lock occupancy during the measured windows, not an
exact time trace, CPU utilization, or the fraction of time a particular thread
waited for the GIL. A holder can itself be descheduled. Threads in a hybrid
process share one lock, so 100% is the maximum for that process, not 200%.
Different processes have independent locks. Percentages are not summed across
processes. Waiting for CPU scheduling, memory or a future is not automatically
GIL waiting.

Separate calibration workloads validate the same read against known lock states:

| Control | GIL-held observations | Occupancy |
|---|---:|---:|
| Native CPU loop explicitly releasing GIL | 0 / 1,413 | 0% |
| Native CPU loop explicitly retaining GIL | 1,430 / 1,430 | 100% |
| Pure Python busy loop | 1,407 / 1,407 | 100% |
| Alternating equal held/released intervals | 721 / 1,433 | 50.3% |
| Sleeping | 0 / 1,419 | 0% |

No invalid lock values occurred in calibration. The release control performs
CPU work, demonstrating that this measures the GIL rather than CPU busyness.

## Reproducibility

Use a new directory for new runs; the harness refuses to overwrite its result.
`run.sbatch` requests the actual CPU resources. `manifest.json` preserves the
submitted script hashes. `result.json` records CPU topology, source hashes,
versions, controls and every measured sample. `summary.csv` and `summary.json`
are derived by `collect.py`. The terminal log and Slurm status must both show
success before this checkpoint is accepted. The extension is intentionally
CPython-version-specific and is not imported by production code.

Throughput and occupancy SD describe variation across repeated chunks; they
are not confidence intervals. Occupancy SD includes finite sampling noise.
The CSV also preserves the pooled held/total ratio, worker ranges, CPU usage,
context switches, sample counts and sampled-run throughput changes. Sampled
throughput is diagnostic only: comparison with baseline combines sampler
effects and ordinary temporal variation and is not an isolated overhead proof.

## Results and interpretation

**Accepted diagnostic:** job `6925154`, node1955, completed in 12:40 with exit 0
and `FETCH_GIL_SWEEP_PASS`. All **621 measured chunks** match the serial source
exactly. The observer collected **985,513 valid lock-state samples**, with zero
invalid values. Peak job RSS was approximately 26.1 GiB.

Shared-buffer results (mean ± one sample SD; GIL SD in percentage points):

| Batch | 8×1 steps/s | 4×2 steps/s | Hybrid change | 8×1 GIL held | 4×2 GIL held | Worker CPU cores, 8×1 / 4×2 |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 4,216 ± 281 | 3,580 ± 399 | -15.1% | 39.0 ± 1.8% | 66.3 ± 3.1% | 6.84 / 5.87 |
| 128 | 4,309 ± 212 | 3,937 ± 338 | -8.6% | 39.6 ± 0.9% | 67.9 ± 3.2% | 7.03 / 6.28 |
| 256 | 4,772 ± 170 | 4,104 ± 203 | -14.0% | 39.1 ± 0.8% | 70.2 ± 1.5% | 7.53 / 6.71 |
| 512 | 4,578 ± 186 | 4,235 ± 166 | -7.5% | 38.4 ± 0.7% | 69.9 ± 1.2% | 7.58 / 6.97 |
| 1,024 | 4,726 ± 108 | 4,249 ± 145 | -10.1% | 38.3 ± 0.5% | 71.0 ± 0.9% | 7.73 / 7.22 |

Pipe transport gives the same ranking; retain all rows rather than selecting a
transport per batch:

| Batch | 8×1 steps/s | 4×2 steps/s | Hybrid change | 8×1 GIL held | 4×2 GIL held |
|---|---:|---:|---:|---:|---:|
| 64 | 4,406 ± 358 | 3,473 ± 354 | -21.2% | 39.1 ± 1.5% | 66.0 ± 2.9% |
| 128 | 4,870 ± 13 | 3,466 ± 153 | -28.8% | 38.8 ± 1.0% | 68.0 ± 2.2% |
| 256 | 4,708 ± 208 | 3,968 ± 169 | -15.7% | 38.8 ± 1.0% | 67.3 ± 4.1% |
| 512 | 4,680 ± 191 | 4,011 ± 101 | -14.3% | 38.3 ± 0.7% | 70.6 ± 0.7% |
| 1,024 | 4,752 ± 95 | 4,213 ± 159 | -11.4% | 38.4 ± 0.4% | 71.4 ± 0.9% |

Eight independent processes hold their individual GILs about 38–40% of the
observed task windows. Hybrid processes hold their shared GILs about 66–71%.
This directly establishes higher lock occupancy, consistent with the earlier
thread-switch and off-CPU evidence for contention. It does **not** establish
100% GIL saturation, an exact GIL-wait percentage, or that the entire throughput
penalty is caused by the GIL. Bursts of competing Python work can incur waits
while average occupancy remains below 100%; worker scheduling and packing also
contribute, as the earlier phase profile showed.

Increasing batch improves hybrid CPU use from about 5.87 to 7.22 cores with
shared buffers, but does not make it outperform eight independent processes.
At batch 1,024, eight processes use 7.73 of eight allocated simulation cores
and reach ~4.7k transitions/s. This leaves little unused CPU capacity for more
batching alone. Throughput is broadly plateauing rather than growing in
proportion to batch. A batch of 1,024 is not a demonstrated universal optimum:
shared 8×1 peaks at batch 256 within this sweep, and there is ordinary variation.

The hybrid penalty varies by batch, transport, and node: shared-buffer losses
here range 7.5–15.1%, compared with 3.1% in the prior node1828 attribution run.
All configurations here share one job, node and core allocation, but the node
is not exclusive. Reversed order blocks and reported SD expose variability;
they do not remove all effects of frequency changes and other tenants.

Sampling-run throughput changes range from −4.8% to +5.7% relative to each
configuration's unprofiled mean. Positive changes illustrate temporal noise;
these are not claims that sampling accelerates execution. Main throughput
numbers always use the sampler-off runs. Occupancy is directly observed in
the sampled runs, whose behavior can still be affected by observation.

[Complete CSV](summary.csv) includes both transports, per-batch mean latency,
CPU usage, switches, GIL sample totals and sampled-run rates. Raw records are
in [result.json](result.json). These results support retaining independent
process owners as the stronger measured option for this source Fetch workload;
they do not implement or accept a production process adapter.

