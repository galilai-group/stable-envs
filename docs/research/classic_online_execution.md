# Classic-control State export checkpoint

Parent: `6ce4e82931a629b68218bbe5331296bdc0a946d2`, branch
`repair/render-config-fixes`. This covers the five accepted historical classic
registrations mapped to `CartPole`, `Pendulum`, `Acrobot`, `MountainCar`, and
`MountainCarContinuous`. Semantic acceptance remains unchanged.

## Profile and change

These registrations delegate scalar CPU physics to Gymnasium. At local B256,
adapter step/export/task-only diagnostic times in milliseconds were:

| Domain | Full adapter step | Separate State export | Task-only diagnostic |
| --- | ---: | ---: | ---: |
| CartPole | 11.65 | 11.01 | 0.72 |
| Pendulum | 10.72 | 9.54 | 1.19 |
| Acrobot | 12.53 | 8.92 | 3.68 |
| MountainCar | 8.54 | 7.54 | 1.01 |
| MountainCarContinuous | 9.27 | 8.99 | 0.41 |

These are separate measurements, not additive timing partitions. The task-only
cell substitutes a pre-owned State solely to diagnose export cost; that result
is never used as a valid transitioned State. Source stepping is unchanged.

The candidate reuses OGBench's small `owned_columns` helper: collect the accepted
field values into fresh batch columns, then construct/transfer each tensor once.
It removes per-lane TensorDicts, clones and device transfers. Full PCG64 limbs
and cached words, elapsed TimeLimit steps, CartPole's post-terminal counter,
Pendulum's last action, and continuous MountainCar's float32 continuation flag
remain in the same schema. Returned snapshots retain independent ownership.
No process pool, numerical approximation, public flag or renderer change is
introduced. Process overhead must now be compared against this cheaper path.

## Paired local public throughput

At B256/T8, every State retained and frames disabled, six samples in two
reversed strategy-order blocks gave:

| Registration | Previous steps/s ± SD | Column export steps/s ± SD | Speedup |
| --- | ---: | ---: | ---: |
| CartPole | 20,796 ± 2,833 | 101,230 ± 949 | 4.87× |
| Pendulum | 23,856 ± 123 | 93,839 ± 782 | 3.93× |
| Acrobot | 19,706 ± 2,093 | 48,916 ± 86 | 2.48× |
| MountainCar | 30,350 ± 112 | 114,649 ± 584 | 3.78× |
| MountainCarContinuous | 28,088 ± 57 | 136,771 ± 742 | 4.87× |

The [60-row local CSV](semantic_recovery_evidence/classic_online_20261002/local/comparison.csv)
contains B1/64/256, T1/8, public step and export comparisons. Restore, reset,
construction and warmup are excluded. All timing workloads were checked to
perform B×T real transitions, so padded terminal slots do not inflate the rates.
These are local CPU results, not renderer FPS, RTX 3090 results or source speedups.

## Correctness and queued device gates

Local tests: **17 passed, five CUDA cases skipped**. The new tests compare all
fields against the frozen `6ce4e829` exporter, including PCG cached words,
continuation/replay, mutation isolation, closed adapters, terminal counters and
the continuous MountainCar dtype transition. Existing classic semantic tests
also passed.

All five local public gates independently passed exact frames (MSE 0 against
the old exporter), one/eight actions, slice edge cases, time-limit/terminal
boundaries, unseeded resets, State ownership, physical variations and discrete
scalar actions. Frames use each renderer's native resolution. The initial
CartPole harness incorrectly requested size 64 and was corrected after it
failed locally; its report/source are preserved.

Oscar array **`6964284`** is queued on `3090-gcondo` / RTX 3090. Each registration
runs CPU/CUDA exact-export tests and separate device public gates, then paired
throughput on the same host. Source manifests and launchers are preserved in
the [evidence directory](semantic_recovery_evidence/classic_online_20261002/).
No CUDA gate is claimed complete from CPU results or a sibling registration.

The first array `6964063` was canceled while pending after the OGBench cohort
revealed a missing reference-helper fixture in the shared bundle recipe. V2
adds `tests/fixtures/regression/support_matrix.json`; production and benchmark
code are unchanged. All ten tests collect successfully from the isolated V2
archive, independently of the developer checkout.

After those gates, compare bounded processes/native opportunities against the
optimized baseline, particularly Acrobot. Then complete renderer-only and
combined pipeline timings with same-device source baselines. The broader
35-registration goal remains active.

## CPU/CUDA closure (October 3)

All five registrations independently pass both device gates in RTX 3090 array
6964284. [Terminal reports and the 120-row paired CSV](semantic_recovery_evidence/classic_export_closure_20261003/)
validate source hashes, exact State/task/frame checks and every timing cell.
B256/T8 State-retaining public stepping improves 2.80–6.40× for CPU output and
5.48–13.39× for CUDA output against the old exporter, with CPU physics in both
cases. These are library-exporter comparisons, not upstream or render speedups.
Process/default/full pipeline measurements remain outstanding.
