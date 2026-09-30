# Renderer profiling: all 42 registrations

Fresh Oscar measurements on 2026-09-28. **Complete: 97 renderer cases and four
NO_RGB device entries, covering all 42 registrations.** Every rendered case
passed full-State/live-environment nonmutation and batch-repeat consistency
checks at B=1,8,64,256. Production code was unchanged.

- [Interactive sortable table](dashboard.html)
- [42-row performance CSV](per_environment_42.csv): accepted fidelity backends,
  CPU/GPU public-call latency, preparation, prepared rendering and B256
  throughput, with sample SD, medians, sample counts and hardware.
- [All backend/stage measurements](timings.csv), [case coverage](coverage.csv),
  [Python call costs](functions.csv), [Torch/CUDA operator costs](torch_operators.csv).
- [Validation](validation.json), [terminal job evidence](slurm-terminal.txt),
  [case plan and production hashes](plan.json).
- [Raw measurements, cProfile files and logs](delivery-6760082.tar.gz),
  [291 Chrome-format Torch traces](delivery-6760082-traces.tar.gz).
  Per-case JSON is also directly available under `raw/results/`.

## Where performance is weakest

These are measured costs, not promised optimization gains. CPU measurements
use Xeon Platinum 8268 hosts. GPU measurements use A40/EPYC 7413 or
L40S/EPYC 9554 hosts; cross-environment rankings are prioritization signals,
not controlled hardware-normalized comparisons. Backend siblings share the
same allocation.

| Priority | Observation | Concrete next target |
|---|---|---|
| Craftax pixel registrations | Full Craftax GPU public call **43.01 ± 1.01 ms**; Classic **27.94 ± 0.17 ms** (L40S). B256 prepared rendering still costs **14.244 ± 0.717** and **7.483 ± 0.036 ms/frame**, respectively. | Per-call package metadata reads, full-State validation, repeated Torch→NumPy→JAX reconstruction, and scalar per-lane rendering. |
| Cube / Scene / Rocket State preparation | Cube GPU public **32.86 ± 0.28 ms**, Rocket **29.35 ± 0.83 ms** (A40), Scene **13.51 ± 0.08 ms** (L40S). Separate preparation probes cost **21.41**, **29.26**, and **10.22 ms**. | Consolidate per-leaf validation/synchronization and conversions while retaining rejection and nonmutation guarantees. Cache stable dependency checks rather than reparsing package metadata each render. |
| Walker GPU fidelity throughput | OpenGL B256 **0.678 ± 0.005 ms/frame**, versus fast GPU **0.02461 ± 0.00022** on the same A40 allocation: **27.54×** observed mean cost. | Preserve batched GPU execution while adding the missing fidelity features. Serial OpenGL is cheap at B1 but scales poorly relative to the fast batch renderer. |
| BallInCup CPU | Public call **226.53 ± 9.43 ms**; preparation only **0.230 ± 0.005 ms**. | Capsule ray intersections, dot/cross products and shadow-occlusion traversal. The bottleneck is rendering, not State preparation. |
| Classic-control GPU small batches | Acrobot and CartPole public calls **11.87 ± 0.14** and **10.90 ± 0.11 ms** (A40). Acrobot's prepared-render trace records **970 CUDA kernel launches** for one frame. | Reduce geometry/clipping launch overhead; the current fused pixel kernel does not eliminate all the tiny Torch geometry operations. |
| OGBMaze CPU batching | Prepared-render B1 **40.76 ± 0.47 ms/frame**, B256 **75.01 ± 0.77 ms/frame**. | Investigate larger-batch ray traversal and memory behavior. The profile locates reductions, cross products and primitive intersections; it does not yet establish the cause of this scaling reversal. |

The largest absolute CPU public-call costs are **Scene 334.02 ± 16.01 ms**,
**Cube 297.18 ± 20.22 ms**, and **BallInCup 226.53 ± 9.43 ms**. Scene/Cube
use software OpenGL; their source renderers were also expensive in the prior
campaign. No new source-wide CPU/GPU speedup ratios are claimed here.

Craftax illustrates why the wrapper cost matters: full Craftax's actual
upstream jitted pixel computation takes **0.886 ± 0.005 ms** on the L40S,
while rendering prepared library inputs takes **15.905 ± 0.068 ms**. The
standalone JAX pixels→Torch conversion probe takes **0.082 ± 0.005 ms**;
most bridge cost is elsewhere. The call stacks show package metadata parsing
and repeated state decoding, not merely the final image copy. This upstream
probe starts from decoded State and excludes bridge work; it is not an
end-to-end source-versus-library speedup claim.

## Fidelity throughput tradeoffs

Same allocation, same saved inputs, prepared batch 256. All timing samples
are retained. Mean ratios and median ratios are both supplied because Cube
and Scene's fast-path samples contain isolated long stalls.

| GPU path | Fast mean ms/frame | Fidelity mean ms/frame | Mean cost ratio | Median cost ratio |
|---|---:|---:|---:|---:|
| Cube: auto → mjwarp_fidelity, A40 | 0.3789 | 0.6150 | 1.62× | 1.85× |
| Scene: auto → OpenGL, L40S | 0.2365 | 0.7688 | 3.25× | 4.26× |
| Walker: auto → OpenGL, A40 | 0.02461 | 0.67770 | 27.54× | 27.68× |

Cube's fast B256 samples are mostly about 84–85 ms/batch, with one
455.70 ms sample; Scene's are mostly 45–47 ms, with one 475.75 ms sample.
The current traces cover B1, so the causes of these B256 stalls remain
unresolved. Neither outlier was discarded.

At B1 the tradeoff differs: Scene OpenGL public rendering is 13.51 ms versus
16.36 ms for auto; Walker OpenGL is 2.12 ms versus 2.50 ms for auto. Walker's
CPU OpenGL path also improves the measured public call from 77.48 to
46.55 ms. Optimizations should target the intended batch size, not just a
single-frame number.

## Method and limits

1. The full-State public call, validation/projection and prepared render are
   independently timed. Their means **must not be added or subtracted as an
   exact cost decomposition**: they are separate probes with different cache,
   clock and allocation histories. Profiler self/cumulative times include
   instrumentation overhead; cumulative entries overlap.
2. Each RGB source-backed case selects up to six actual saved States spread
   through one compatible simulator configuration from the accepted parity
   corpus, carrying each State's appearance inputs. Batch throughput repeats
   these poses to fill B=1,8,64,256. Higher batches exclude full-State
   preparation and simulator allocation; no batch-256 end-to-end claim is made.
3. The timer warms each selected pose, synchronizes CUDA, and records 3–30
   samples per probe, with a three-second accumulated sample budget and
   explicit minimum counts. SD uses ddof=1. At B256, SD is variation between
   measured *batch-average per-frame costs*, not within-batch frame latency.
   `first_call_ms` is the first call within that probe, not process startup.
4. cProfile covers three separate instrumented calls per stage; the Torch
   trace covers one. Neither is part of the latency samples. Torch allocator
   memory excludes external allocations, and the Torch trace can miss native
   OpenGL, Warp or JAX GPU kernels. Zero Torch-observed launches for an OpenGL
   or JAX path does not mean no GPU work occurred.
5. This is a **new performance profile**, not another 1,000-frame MSE run.
   Previous pixel-parity evidence remains unchanged. Batch-repeat tolerance
   is at most one uint8 level; that diagnostic is not a new source-MSE gate.
6. The two symbolic Craftax registrations have no RGB renderer. Both PushT
   entries profile the available shared candidate renderer diagnostically;
   there is no new source-matching or registration-specific semantic claim.
   FetchPickAndPlace profiles use reset-only source States. Craftax GPU uses
   its compatible GPU State corpus, not incompatible CPU State signatures.
7. The concise CSV uses CPU OpenGL for Fetch/Cube/Scene/Walker, GPU
   mjwarp_fidelity for Cube, and GPU OpenGL for Scene/Walker. These match the
   previously accepted fidelity paths. The full CSV also records alternatives.

## Provenance and execution

Base integration SHA `df80c9c54590afa9fc0482a4dcb241d21f6b63a3`, plus the
unchanged accepted renderer overlay. The exact production Python hashes are
in `plan.json` and every case result. All jobs read the immutable Oscar
`render-parity-df80c9c5-20260928/source-v12` tree. The profiling harness was
loaded separately. Package/source profiles, host hardware, thread settings,
input hashes, corpus hashes and SLURM job IDs are retained per case.

Campaign root:
`/oscar/scratch/sboughan/se-semantic-recovery/renderer-profiles-df80c9c5-20260928`.
Pilots: CPU 6759929 / GPU 6759930. Initial CPU bulk 6759954. CPU retries
6760005 / norm-GPU sweep 6760006. Validated package 6760082 completed with
exit code 0. A pending condo array 6759955 was cancelled because of its
allocation-wide GPU cap. A diagnostic-helper filename collision caused
29 CPU jobs to fail before measurements; those jobs were retried after the
helper was renamed. Original failure logs and successful retry evidence are
both preserved. No completed measurement was overwritten.

All 291 stage profiles and all four batch checks per rendered case were
verified, along with production hashes and sibling-backend host/device
consistency. Both downloaded archives match the remote SHA-256 values in
`archive-sha256.json`. Five local harness tests and lint/compilation checks
passed. No production optimization, semantic promotion, or push occurred.
