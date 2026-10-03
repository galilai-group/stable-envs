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

## Post-export profiling (October 3)

The [fresh RTX 3090-host component profiles](semantic_recovery_evidence/classic_postexport_profile_20261003/)
cover all five registrations through B1024 with exact replay, six samples and
CPU/CUDA output. Acrobot's RK4 dominates task execution; State conversion still
dominates the cheapest tasks. The next bounded candidates are direct writes into
owned State columns and a process comparison for Acrobot. Neither is enabled
from profile evidence alone. These diagnostics do not measure render throughput.

## Direct-column export accepted (October 3)

The [next paired checkpoint](semantic_recovery_evidence/classic_direct_columns_20261003/)
removes the temporary NumPy array and repeated field checks for every lane/field.
The fixed schema allocates each owned batch column once; source values are copied
directly into it. RNG, dtype/lifecycle fields and snapshot ownership are unchanged.
All five registrations independently pass exact CPU/CUDA State/task/frame gates.

Against the preceding accepted column exporter, B1024/T8 public steps retaining
all States improve 1.15–1.64x. The 160-row sweep preserves every B1/64/256/1024,
T1/8, export/step and CPU/CUDA-output cell. Three noisy B1/T1 public-step means
are slightly lower, so this is not a universal speedup claim. Processes remain
a candidate for Acrobot and must be compared with this updated local baseline.

## Acrobot process default accepted (October 3)

The [Acrobot process sweep](semantic_recovery_evidence/acrobot_process_20261003/)
compares local, four and eight persistent owners on the same RTX 3090 host.
Across B64/256/1024 and T1/8, eight-owner stepping improves 2.25–6.71x over the
accepted direct-column local baseline. At B1024/T8, CPU output improves
9,197 +/- 145 to 61,675 +/- 644 steps/s; CUDA output improves 9,352 +/- 30 to
57,960 +/- 1,481. All States are retained and frames disabled; these are neither
upstream speedups nor renderer FPS.

The [independent default gate](semantic_recovery_evidence/acrobot_default_20261003/)
passes 132 tests without skips and 25 public-constructor cases per CPU/CUDA
mode, including exact frames, State/RNG/lifecycle, ownership, variation and
failed-restore isolation. The tested constructor now selects up to eight owners
for Acrobot on Linux at B>=64 with >=4 affinity CPUs outside daemon processes.
Other classic domains and smaller/resource-limited cases retain the local path.
The slice-only API and renderer are unchanged. Full pipeline and same-device
upstream comparisons remain required.

## Direct-source pipeline sweep staged (October 3)

[Array 6968542](semantic_recovery_evidence/classic_source_pipeline_20261003/)
measures the five registrations against direct pinned upstream calls, with
source-State validation outside timing. B1/64/256, T1/8, no/final/all frames,
CPU/CUDA output and six samples cover task stepping, saved-State rendering,
State export and the online pipeline. This workload disables intermediate
State retention; the mandatory final State remains on our side. Source does
not pay an invented per-step full-State export cost.

Local B2 CPU checks pass independently for all five, with exact State/task
outputs and zero image MSE. No cluster measurement is accepted from those local
checks. Source is serial CPU; Pygame has no GPU renderer. Reports distinguish
native source CPU output from delivery to GPU memory and explicitly retain
source worker count versus our process count. A parallel-source comparison
remains open, so the resulting ratios must not be labeled universal source
speedups. The sweep is queued, not yet a completed source/pipeline table.

CartPole and Pendulum have now independently completed the full staged sweep;
[their two-device CSVs and component tables](semantic_recovery_evidence/classic_source_pipeline_20261003/completed/)
contain zero source image MSE across all checked frames. At B256/T8/all frames,
ours CPU is 1.36x and 1.95x the serial source respectively; ours GPU produces
10,129 and 12,211 FPS. Source GPU rendering is unavailable. Plain stepping is
still slower than direct source; repeated unrequested State export is a concrete
next optimization candidate using the existing family hook. No parallel-source
or universal speedup is claimed, and the other three registrations remain pending.

The [complete five-registration source checkpoint](semantic_recovery_evidence/classic_source_pipeline_20261003/family-closure/)
is now closed: 180 pipeline rows, 90 component rows, and zero measured source
image MSE on both output devices for every registration. At B256/T8/all, GPU
online rates span 10,129–14,111 FPS, CPU online 338–558 FPS, and serial source CPU
216–451 FPS. The CSV retains means/SDs, every measured cell and source GPU
unavailability. This is a serial-source comparison; Acrobot uses eight candidate
physics owners, and a parallel-source baseline remains open. No broader goal
completion or universal stepping speedup follows from these results.

## Snapshot-selection candidate (October 3)

The [five-registration correctness closure](semantic_recovery_evidence/classic_snapshot_20261003/closure/)
passes 147 CPU/CUDA tests per job without skips, 28 public cases per registration
and 320 paired rows through B1024. The isolated candidate uses the existing
advance-without-snapshot hook and avoids worker exports for advance requests.
Complete owned final State, explicit intermediate selections and exact frames
are preserved. Production remains unchanged pending performance review.

At B1024/T8 without intermediate States or frames, ratios versus the previous
adapter are 1.93x CartPole, 1.53–1.55x Pendulum, 1.62–1.63x MountainCar,
2.55–2.62x continuous MountainCar, and 1.01–1.05x Acrobot. Some Pendulum and
Acrobot cells regress or have large timing spikes; unchanged-code complete
repeats investigate those results. Original samples are retained. All-State
requests often change little, and no upstream or universal speedup is claimed.

Both [complete repeats](semantic_recovery_evidence/classic_snapshot_20261003/repeat-review/)
have now passed correctness. Pendulum reproduces the same slow fourth/fifth
CUDA samples, so these cannot be dismissed as generic noise. Acrobot has
variable slower cells and small gains. The candidate remains isolated while
GC-event diagnostic 6970565 correlates collection durations with timing windows;
no collection is disabled or forced, and original measurements remain intact.

The [completed GC trace](semantic_recovery_evidence/classic_snapshot_20261003/gc-diagnostic/completed/)
records a 155-ms generation-2 pause inside one CPU sample. Instrumentation shifts
collection placement, and the earlier slow CUDA samples do not recur at the same
timed locations. This supports investigating allocation/collection costs but
does not prove the earlier CUDA spikes' cause. Sustained independent-strategy
measurements must retain naturally occurring GC costs before activation.
