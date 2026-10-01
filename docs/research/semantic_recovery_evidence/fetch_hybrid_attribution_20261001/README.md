# Fetch hybrid slowdown attribution

Fresh CPU diagnostic, job **6924165**, node1828, eight distinct Cascade Lake
cores, October 1 2026. Completed with exit 0 and `FETCH_HYBRID_PROFILE_PASS`.
Base integration commit: `722fa870b80453934b7d9803305556e3c3729954`.
No production executor or semantic status changes. Historical evidence preserved.

## Answer

Four processes with two threads each have the same eight-core budget as eight
single-threaded processes, but only four independent Python interpreters. Fetch
does substantial controller, observation, and task work around native physics.
The hybrid adds contention, worker start delays, and larger serial packing per
process. Its small transport savings do not offset these costs.

Evidence strongly supports Python/GIL contention as the leading explanation,
not disk I/O or slow shared buffers. This is an inference from scoped CPU/wall
timings and context switches, not direct GIL occupancy sampling. The earlier
[scaling experiment](../fetch_scaling_slices_20261001/README.md) additionally
showed native-engine-only thread scaling, while full-task threading scaled much
less. Engine-only trajectories are not equivalent task benchmarks.

## Uninstrumented throughput

Batch 256, eight known actions/lane, 2,048 transitions per sample; source
observations, reward, flags, success, qpos and qvel collected through shared
memory. Two configuration-order blocks (forward/reverse), seven repeats each.
Mean ± sample SD across 14 samples, not a confidence interval:

| Configuration | Source transitions/s |
|---|---:|
| 8 processes × 1 thread | 4,262 ± 13 |
| 4 processes × 2 threads | 4,129 ± 38 |
| Hybrid, each thread pinned to one core | 3,911 ± 107 |
| Hybrid, Python switch interval 100 µs instead of 5 ms | 4,103 ± 64 |

The ordinary hybrid penalty is **3.1%** here. The previous controlled node2303
confirmation measured ~7.8% with shared buffers and ~7.5% with pipes. The penalty
is not a universal constant. Neither tested intervention improved throughput on
this node. Thread pinning is applied when each thread task begins; it is not a
separate persistent per-core work queue.

## Where time goes

Separate instrumented passes (three repeats per block) wrap six source methods
with thread CPU and wall timers. No cProfile hooks are installed. Throughput
with instrumentation was 4,212 vs 4,051 transitions/s (8×1 vs 4×2), respectively
1.2% and 1.9% below their uninstrumented rates. These timings explain the measured
profiled run; they are not an exact decomposition of the uninstrumented gap.

| Worker time per transition | 8×1 | 4×2 |
|---|---:|---:|
| Entire task, wall | 1,639 µs | 1,764 µs |
| Entire task, thread CPU | 1,630 µs | 1,657 µs |
| Difference (wall minus CPU) | 9 µs | 107 µs |
| `_mujoco_step`, wall / CPU | 937 / 930 µs | 1,004 / 941 µs |
| `_get_obs`, wall / CPU | 488 / 483 µs | 521 / 491 µs |
| Voluntary switches inside task work per 2,048 transitions | 0 | 10,075 |

The roughly 97 µs increase in off-CPU time is most of the 124 µs increase in
average worker task duration. Native-step CPU work barely increases, while its
wall time increases much more. A timed Python call around native physics also
includes scheduling and reacquiring Python execution after native work returns;
its wall duration must not be read as native engine compute time. Observation
and callback regions also accumulate waiting. Wall-minus-CPU includes any
descheduling, not exclusively the GIL. Clock wrapper overhead prevents exact
subtraction of summed method timings from the outer task measurement.

The observed critical path below selects the last-finishing child for each
batch step and that child's last-finishing worker. Columns sum across eight
steps (milliseconds). Parallel workers' durations are **not** summed into
elapsed time. Dispatch includes delivery/start latency; return includes shared
observation writes, pipe notification, parent collection and scheduling.

| Critical path component | 8×1 | 4×2 | Change |
|---|---:|---:|---:|
| Parent dispatch → child start | 1.18 | 1.14 | −0.05 |
| Child start → critical worker start | 0.09 | 3.49 | +3.40 |
| Critical worker executing source tasks | 478.75 | 492.81 | +14.06 |
| Worker finish → coordinator join | 0.09 | 0.31 | +0.22 |
| Scatter results | 0.04 | 0.06 | +0.02 |
| Pack arrays | 2.43 | 4.45 | +2.02 |
| Child finish → parent return | 3.29 | 3.10 | −0.19 |
| Parent-owned output copies | 0.34 | 0.35 | +0.01 |

Of the ~19.5 ms extra profiled critical-path time, about 72% is inside the task,
17% precedes the critical thread starting, and 10% is larger packing. Worker
start delay includes thread-pool wakeup, scheduling, and Python contention; it
is not a pure queue-cost measure. Some load imbalance remains in either layout.
No claim is made that each of these percentages generalizes to other nodes or
batch sizes. The previous shared-versus-pipe experiment, not this run, measures
the transport choice directly.

## Implication and limits

Eight independent process owners remain the stronger measured Fetch option
under this eight-core budget. A hybrid becomes more attractive if task-side
Python work is reduced or moved into native/vectorized operations; adding more
threads alone does not provide more cores or independent Python execution.
Any production process adapter still needs its lifecycle and full State gates.

All 80 measured chunks, including instrumented and intervention runs, exactly
match an independent serial source trajectory for every collected field.
Reset, construction, warmup and correctness checks are outside timing. These
are source stepping/transport diagnostics: no renderer, GPU transfer, exhaustive
State export, terminal/autoreset recovery, or online policy-speed claim.

`result.json` retains all raw timestamps and samples; `summary.csv` and
`summary.json` are derived by `collect.py`. `manifest.json` records submitted
source hashes; raw results additionally record installed package versions,
upstream source paths/hashes and CPU topology. `run.sbatch` and `terminal.log`
preserve the execution. Run the script from its directory with the benchmark
dependencies installed and exactly eight allocated CPU cores:

```bash
python fetch_hybrid_profile.py --output new-result.json
```

The script refuses to overwrite its output. Preserve this accepted evidence;
future runs must use a new output directory.
