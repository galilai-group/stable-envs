# Fetch process pipeline checkpoint

The public `step` API accepts both `[B,A]` and `[B,T,A]`. Fetch now overlaps
process-owned CPU simulation with the parent's rendering of owned snapshots.
No new public constructor flag, capture mode or sequence method is introduced.
The shared scheduler is reusable; Fetch is the only family opted into it here.
Other family execution strategies and the semantic support matrix are unchanged.

## Execution and ownership

Large Linux Fetch batches (B≥64, at least four allocated CPUs, non-daemon
caller) automatically select up to eight persistent processes, one simulation
thread per process, pinned within the available CPU affinity. Other cases keep
the local adapter. Scripts use the standard `if __name__ == "__main__":` guard.

At most one transition is in flight. Workers export selected full State and task
outputs in one request. The parent owns copies before sending the next action;
workers may then reuse shared storage while rendering consumes the prior State.
The OpenGL/CUDA renderer stays on the caller's thread. Exceptions with a step
in flight abort/reap the owners, and concurrent adapter requests are refused
before overwriting buffers. There is no work left running after a successful
`step` call, no implicit reset, and no guessed action across calls.

Both action shapes use the same dispatcher. A single action has no later step
to overlap. The default renders only the final endpoint, so intermediate-render
overlap requires `frames=slice(None)` or a sparse slice selecting intermediate
frames. Final-frame-only calls are measured separately and do not claim the
all-frame throughput gain. State/frame selections, early termination, reverse
and empty slices, replay, and owned outputs retain their previous semantics.

## Paired measurement

The [final-source RTX 3090 CSV](semantic_recovery_evidence/fetch_pipeline_20261002/comparison-rtx3090.csv)
contains 216 aggregate cells. Each is six samples across two reversed-order
blocks, with means and sample standard deviations. B=64/256/1024 and T=1/8;
all States retained; frames disabled, final-only, or all. Restoration, startup,
warmup, correctness checks and policy inference are outside the timed regions;
CUDA is synchronized around each sample. Each comparison uses the same CPU
allocation, GPU, environment and renderer. This measures **CPU stepping plus
GPU rendering**, not renderer-only FPS or a comparison to upstream FPS.

Controls are `serial_legacy` (separate advance/export requests), `serial_fused`
(the same fused export as the pipeline, but deferred dispatch prevents overlap),
and `pipeline`. Comparing against the fused control isolates overlap from RPC
savings. FPS counts emitted images; steps/s counts transitions. With T=8 and
final-only capture these counts differ by eight.

At B=1024, T=8, all frames, 64×64 OpenGL, on RTX 3090 allocations:

| Task | Fused serial FPS ± SD | Pipeline FPS ± SD | Overlap speedup | Host |
| --- | ---: | ---: | ---: | --- |
| FetchReach | 2,440 ± 20 | 2,769 ± 32 | 1.135× | gpu2109 |
| FetchPush | 2,389 ± 10 | 2,711 ± 93 | 1.134× | gpu2113 |
| FetchSlide | 2,279 ± 17 | 2,585 ± 66 | 1.134× | gpu2113 |
| FetchPickAndPlace | 2,465 ± 24 | 2,833 ± 30 | 1.149× | gpu2109 |

Host CPU performance differs, so do not interpret absolute cross-row differences
as isolated task effects. All four final-source RTX 3090 jobs in array 6960169
completed and passed. The [A5500 sweep](semantic_recovery_evidence/fetch_pipeline_20261002/comparison-v3.csv)
is retained separately; its measurements are never pooled with the RTX 3090 CSV.
Across the twelve task/batch cells at T=8 with all frames, overlap adds 13–42%
over the fused serial control on the 3090 allocations. At B=1024 it adds 13–15%.
Single-action and final-frame-only results remain separate in the CSV; no blanket
speedup is claimed for those modes. The six-sample one-action/all-frame cells
range from 0.953× to 1.020× the fused control, with the longer B64 follow-up below.

## Acceptance and the single-action check

All 12 registrations independently pass CPU/CUDA full-State/task, checkpoint/RNG
replay, distinct time limits, variations, and output-ownership checks through the
unpatched public constructor. GPU frame comparisons are exact against the serial
library renderer (MSE 0), including one/eight actions and output slices. This
preserves existing fidelity; it is not a new upstream renderer-parity survey.
The final GPU unit suite passes 188 tests, with one optional Matplotlib skip.
The local full suite before the last GPU-sync-only change passed 185 with four
CUDA skips; the affected local API/selection suite then passed 154 with four
CUDA skips. Those CUDA cases pass on Oscar.

An initial six-sample B64/T1 Slide cell was 13.6% below the legacy control.
A [40-sample before/after follow-up](semantic_recovery_evidence/fetch_pipeline_20261002/production_v3/single_action/)
on the original host did not reproduce that drop in either version. The final
pipeline measured 1814.5±81.0 FPS, versus 1783.9±91.3 legacy and 1838.0±42.9 fused.
These measurements do not establish a single-action speedup or a causal timing
benefit from the small fix. The fix removes a logically unnecessary GPU scalar
read of the known-all-true initial active mask; long-chunk overlap is the main
measured gain. Both original and follow-up evidence are retained.

## Reproduction and provenance

Implementation files are `_stepping.py`, `_adapters/fetch_process.py`,
`_adapters/fetch_execution.py`, and `envs/_fetch_semantics.py`. The source hashes
in [the final manifest](semantic_recovery_evidence/fetch_pipeline_20261002/production_v3/source_manifest.json)
identify the tested code. Implementation commit: `080b525c0d2852e0e6d8ff597505ea6dda513e03`; every
archived production source hash matches that commit. The upload was staged from
checkout `5c23f8ee` plus those changes before the commit was made.
`fetch_pipeline.py` runs the public API gates and comparisons;
`summarize_fetch_pipeline.py` refuses partial reports or missing cells and writes
the aggregate CSV. `pack_fetch_family.py` makes the reproducible archive and
launcher. See [evidence](semantic_recovery_evidence/fetch_pipeline_20261002/) for
raw sample times, logs, hardware, manifests, launchers and the staged iterations.

The earlier all-registration four/eight-process family sweep is separate: job
6950614 completed Reach but hit time/memory limits on three other tasks. The
four unfinished registrations completed and passed individually in 6957513 using
the unchanged original source, with larger RAM allocations. The [completed family
CSV](semantic_recovery_evidence/fetch_family_3090_20261002/completed/comparison.csv)
contains 1,488 aggregate rows for all 12 registrations, CPU/CUDA output,
B=1/8/32/64/128/256/512/1024, T=1/8, and four/eight-process versus current-local
execution. This earlier sweep does not include the new overlap implementation.
Historical outputs are preserved; `collect_fetch_family.py` recovers only complete
per-registration sweeps, validates every cell and never joins measurements across
hosts for one row. Pipeline throughput is measured on the four plain tasks;
Dense/Dict registrations also pass their own final pipeline correctness gates.

## Direct-source online comparison prerequisite (October 3)

The [direct-source CPU audit](semantic_recovery_evidence/fetch_source_prerequisite_20261003/)
passes 48 independent full-State/task/restore/RNG cases across all 12 registrations.
It also traces a source-render lifecycle constraint: in 576 callback/continuation
cases, Reach/Push/Slide remain unchanged, while all three PickAndPlace registrations
change controller caches and subsequent trajectories in every tested post-step case.
An ordinary source step/render loop therefore cannot be used as a matched-trajectory
oracle for PickAndPlace without explicitly accounting for this effect. These are
CPU callback checks, not GL/MSE or throughput acceptance; no production semantics
or acceptance status changed.

The same audit records internal versus output dimensions: the default source
renders at 480x480 before resizing to 224x224; `resolution=64` still renders at
480x480. Future online source rows must record both dimensions and separately gate
any alternative internal resolution. Existing pipeline numbers above remain
library-control comparisons, not upstream speedups.

The [shared-memory source prerequisite](semantic_recovery_evidence/fetch_source_transport_20261003/)
now passes all 12 registrations independently: 96 CPU B4/P2 cases cover T1/T8,
float32/64, mixed episode limits, exact source/adapter State and task outputs,
owned shared results, restored replay and unseeded-reset RNG. All 48 workers
exit cleanly. This benchmark executor reuses Gym's process/shared-memory transport;
source State RPCs are untimed and pickled. No rendering, source FPS or CUDA output
result is claimed. The next rendered source gates must retain the PickAndPlace
lifecycle and raster-size qualifications above.

The [selected-frame source checkpoint](semantic_recovery_evidence/fetch_source_frames_20261003/)
adds shared image columns while preserving source raster/resize behavior. Final
CPU no-frame and empty-frame regressions each pass 96 cases across all 12
registrations; 36 synthetic checks also pass from the frozen upload bundle.
The prepared actual-image gate compares independent source P1/P2 owners with the
public library API for each of the nine Reach/Push/Slide registrations on CPU and
CUDA. It requires exact source task/State/parallel-image agreement, render
nonmutation, and per-frame library MSE <= 0.001. It has **not been uploaded or
submitted** because Oscar SSH authentication remains unavailable. It is a B4
correctness prerequisite, not throughput or large-batch process acceptance.
PickAndPlace nonempty source capture is rejected before execution; its no-frame
path remains valid, and its rendered comparison protocol remains unresolved.

The subsequent [source throughput campaign](semantic_recovery_evidence/fetch_source_pipeline_20261003/)
is prepared for all 12 registrations at B64/T1/T8 on eight-core RTX 3090
allocations. It runs independent source P1/P8 and public-library strategies in
fresh interpreters, with six samples and separate stepping/export/copy/render
components. The nine admissible rendered registrations must pass their own
CPU/CUDA image gates first; PickAndPlace produces no-frame rows only. A strict
collector reports mean/SD and speedup against the best same-device source.
Local source-State projection/transport prerequisites and synthetic collector
checks pass, but this integrated campaign is not uploaded/submitted and has
no measured source FPS. Broader batch/memory sweeps remain pending.
