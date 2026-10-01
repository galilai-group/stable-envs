# Fetch hybrid process/thread experiment

Starting checkpoint: `c8506c2d4a491ea2810d9cf3d258dc5e60752113` on
`repair/render-config-fixes`. This is a CPU-only diagnostic; production code and
semantic acceptance are unchanged.

The requested comparison is **four processes with two persistent simulation
threads per process**, using Gymnasium AsyncVectorEnv with either shared output
buffers or pipe transport. A fresh **eight-process, one-thread** control runs in
the same allocation. Both layouts have eight simulation workers and eight
allocated CPU cores. The coordinator shares the allocation; there is no hidden
extra CPU allocation for it.

## Results

**The hybrid did not beat eight single-threaded processes in the pinned runs.**
The longer batch-256 confirmation (15 repetitions, mean ± sample SD) measured:

| Layout | Shared output buffers | Pipe output transport | Worker CPU equivalents, shared / pipe |
| --- | ---: | ---: | ---: |
| 4 processes × 2 threads | 3574 ± 144 steps/s | 3621 ± 75 steps/s | 6.20 / 6.25 |
| 8 processes × 1 thread | 3876 ± 160 steps/s | 3914 ± 300 steps/s | 6.44 / 6.48 |

The hybrid is about 7.8% slower with shared buffers and 7.5% slower with pipes
based on those means. Shared versus pipe transport differs by only about 1.3%
for the hybrid here, smaller than the per-repeat variability; no reliable
transport-mode winner is established at this batch size.

The five-repeat pinned sweep also covered smaller and larger batches:

| Batch | 4×2, shared | 4×2, pipe | 8×1, shared | 8×1, pipe |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 3384 ± 44 | 3413 ± 249 | 4172 ± 25 | 3669 ± 307 |
| 256 | 3410 ± 714 | 3960 ± 202 | 4253 ± 75 | 3948 ± 102 |
| 1024 | 4009 ± 140 | 3584 ± 252 | 4216 ± 96 | 3966 ± 110 |

The five-repeat batch-256 pipe layouts are effectively tied; the longer repeat
favors eight single-threaded processes. Absolute rates vary between runs, so
these observations do not establish a universal layout or transport ranking.
The initial unpinned run was substantially slower and highly variable. It also
favored the hybrid at batch 1024: 923 versus 735 steps/s with shared buffers,
and 813 versus 631 with pipes. Its batch-256 eight-process rows, however,
differed dramatically between transports (546 ± 30 shared versus 3286 ± 1342
pipe). These conflicting observations are retained in the CSV with their
run/host labels; they reinforce that the pinned result is not a universal
hybrid-versus-process ranking. The appropriate
conclusion is that **4×2 offers no demonstrated speed advantage on the tested
pinned allocation**, rather than that hybrid executors cannot help elsewhere.

All completed configuration and per-repeat trajectory checks match the serial
source exactly for the tested task fields, and output ownership checks pass.
The CPU shared buffers here are Gymnasium's cross-process observation buffers,
not CUDA pinned memory, VRAM, or a separate transfer pipeline.

## Method

- FetchPush, batches 64/256/1024, eight known float64 actions per lane, identical
  lane seeds and action arrays for all configurations. Five timing repeats in
  the sweep; fifteen in the batch-256 confirmation.
- Each process owns disjoint scalar environments; its two threads advance
  disjoint subshards. The pool persists across calls and all futures finish
  before the process packs output arrays. No simulator is stepped concurrently
  by two threads.
- `shared_memory=True` selects Gymnasium's preallocated cross-process observation
  buffers. `False` sends the same arrays through pipes. Both retain `copy=True`
  and collect the same observations, reward/termination/truncation/success,
  qpos and qvel. Threads do not write directly into a custom ring buffer.
- Timings include task execution, packing, IPC, and coordinator collection.
  Construction, reset, warmup, and correctness assertions are excluded.
- Every measured repeat is checked exactly against a separate serial source
  trajectory. Returned-array ownership is also checked across further stepping
  and reset. This is not exhaustive State/RNG/restore or terminal-lifecycle
  acceptance for a production process adapter.
- Worker CPU time uses process CPU time, including both threads and per-process
  packing. Parent CPU time is recorded separately. These are active CPU
  equivalents, not sampled core-utilization traces.
- NumPy/BLAS/OpenMP threading is limited to one; no rendering or CUDA transfer
  occurs. No renderer FPS/MSE or full online pipeline speedup is claimed.

## Why there are three runs

The initial run (6922309, node2410) showed very large timing variance, including
in its eight-process controls. Its raw results remain in `initial/` and in the
CSV; they are not silently discarded.

The controlled sweep (6922504, node2303) uses the Cascade Lake CPU family and
pins each process to its own core or pair of cores. Actual worker affinities are
recorded and verified disjoint. Both host/CPU family and affinity changed from
the initial attempt, so improved stability cannot be attributed to pinning
alone. Compare layouts within each allocation, rather than mixing absolute
rates across hosts.

The controlled sweep still had a noisy hybrid shared-buffer cell at batch 256,
so confirmation job 6923133 repeats that batch fifteen times per configuration
on node2303 with the same pinning protocol. All attempts and individual samples
are retained. Configuration order reverses at batch 256; configurations are
measured sequentially to keep resident simulator memory bounded.

The generated [CSV](hybrid.csv) labels every run, host, layout, buffer mode,
repeat count, mean, standard deviation, CPU use, call latency and construction
time. Values are aggregate task transitions per second, not batch calls or rendered
frames.

## Reproduction

The runnable benchmark is
[`scripts/benchmarks/online_stepping/fetch_hybrid.py`](../../../../scripts/benchmarks/online_stepping/fetch_hybrid.py).
Its companion `fetch_scaling.py` defines the benchmark-only shard/task payload.
Run it on a Linux CPU node, for example:

```bash
python scripts/benchmarks/online_stepping/fetch_hybrid.py \
  --output fresh-results.json --batches 64 256 1024 --repeats 5 --pin-workers
```

Both exact source files, package versions, hashes, submission scripts, results
and terminal logs are retained per run. `collect.py` validates completion,
trajectory checks, hashes and CPU ownership before rebuilding the CSV/summary.
`sha256.json` inventories this evidence directory.

Shared arrays speed output transport; they do not remove Python/GIL contention
inside each process. No production executor change follows solely from these
measurements: complete State restoration, RNG, Variation and episode lifecycle
still require their existing acceptance gates.


All three jobs completed with exit code 0 and a `FETCH_HYBRID_PASS` terminal
marker. Twenty-eight configuration gates passed across the three runs, with
exact trajectory checks on every timed repeat. `job_status.txt` preserves the
scheduler record. Ruff and Python compilation checks passed for the new harness.
