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

The [final-source A5500 CSV](semantic_recovery_evidence/fetch_pipeline_20261002/comparison-v3.csv)
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

At B=1024, T=8, all frames, 64×64 OpenGL, on RTX A5500 allocations:

| Task | Fused serial FPS ± SD | Pipeline FPS ± SD | Overlap speedup | Host |
| --- | ---: | ---: | ---: | --- |
| FetchReach | 2,590 ± 4 | 2,888 ± 54 | 1.115× | gpu2501 |
| FetchPush | 2,331 ± 6 | 2,602 ± 44 | 1.116× | gpu2501 |
| FetchSlide | 1,509 ± 1 | 1,784 ± 2 | 1.182× | gpu2503 |
| FetchPickAndPlace | 2,445 ± 4 | 2,791 ± 10 | 1.141× | gpu2504 |

Host CPU performance differs, so do not interpret absolute cross-row differences
as isolated task effects. The separate final-source RTX 3090 sweep is job array
6960169; its results are collected separately and never pooled with this CSV.

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
identify the tested code, based on checkout `5c23f8ee` plus the recorded changes.
`fetch_pipeline.py` runs the public API gates and comparisons;
`summarize_fetch_pipeline.py` refuses partial reports or missing cells and writes
the aggregate CSV. `pack_fetch_family.py` makes the reproducible archive and
launcher. See [evidence](semantic_recovery_evidence/fetch_pipeline_20261002/) for
raw sample times, logs, hardware, manifests, launchers and the staged iterations.

The earlier all-registration four/eight-process family sweep is separate: job
6950614 completed Reach but hit time/memory limits on three other tasks. The
four unfinished registrations run individually in 6957513 using the unchanged
original source, with larger RAM allocations. Historical outputs are preserved;
`collect_fetch_family.py` can recover only complete per-registration sweeps,
validates every cell and never joins measurements across hosts for one row.
