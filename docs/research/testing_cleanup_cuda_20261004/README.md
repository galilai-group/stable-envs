# Testing cleanup: fresh RTX 3090 closure

The five checks skipped in the local cleanup run now pass on Oscar. Job
**7004892** completed with exit **0:0** on **gpu2110**, using an **NVIDIA GeForce
RTX 3090** in `3090-gcondo`. All **9 required cases passed**, with **0 skipped,
failed, errored or crashed** cases.

The tested code is `sami-dev` commit
`73b58b1308e169345ecf18069f237679928d3c97`. Before testing, the job verified all
1,611 source/test/bundle file hashes against `submission/manifest.json`.

| Check | Cases | Result |
|---|---:|---|
| CUDA sequence frames and final State versus repeated public steps: CartPole, Pendulum, TwoRoom, Piecewise | 4 | PASS |
| CPU SimplePointMaze sequence versus scalar calls, with Matplotlib installed | 1 | PASS |
| Renamed legacy render-projection round-trip: CartPole, Pendulum, TwoRoom, Piecewise | 4 | PASS |

CUDA cases cover full, final, sparse, reversed and empty frame/State selections.
They check actual CUDA frame placement and exact equality. Render-projection
round-trips remain distinct from full simulator-State restoration.

These are fresh correctness checks, **not** new source-image parity measurements,
an all-registration GPU acceptance run or a performance benchmark. The existing
38/38 scoped non-Craftax acceptance count is unchanged. The earlier cleanup's
56/56 CPU source-profile results and two independently reproduced legacy
provenance failures also remain unchanged.

The supervisor ran one segment. Pytest reported 9 passes in 3.94 seconds; total
SLURM elapsed time was 42 seconds including setup. These durations are diagnostic
job timings and are not renderer/online throughput measurements.

Evidence includes scheduler terminal status, hardware/runtime identity, JUnit,
supervisor events, exact-case verification and the executed launch/validation
scripts. Local verification independently checked the downloaded JUnit, required
node IDs, zero-skip counts, candidate identity and terminal success marker.
`SHA256SUMS` seals this new directory. The [earlier checkpoint](../testing_cleanup_20261004/README.md)
and its seal were left unchanged.
