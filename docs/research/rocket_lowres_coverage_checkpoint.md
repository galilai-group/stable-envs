# Rocket low-resolution GPU coverage investigation (2026-10-04)

**37/38 non-Craftax registrations are accepted. Rocket remains INCOMPLETE.**
Both PushT registrations were independently accepted and pushed on `sami-dev`
in `d543e669`. This checkpoint integrates a Rocket-local GPU raster candidate and preserves the
investigation evidence. State validation and the acceptance threshold are unchanged.
Fresh production acceptance is pending Oscar job **6996062**.

## The actual failing case

The prior CUDA online gate passed the one-action flight case. Its failure is in
an eight-action flight sequence, at 96px: lane 0, time index 2 has normalized
MSE 0.001889559927; lane 1, time index 3 has 0.001889557146. The threshold remains
0.001 **per frame**, not an average that conceals failing frames.

Independent CPU source pixels are exact. Numeric source State matches each
transition. Online rendering, restored rendering, separate singleton rendering,
chunk-size-one rendering, and reversed batching all produce identical GPU pixels.
This rules out stale captured State and batch-lane contamination in this witness.
The major error is 24 omitted fin-edge pixels: source dark gray becomes white.

The projected thin top faces straddle the coarse rasterizer's coverage boundary.
Offset changes and supersampling do not fix the mismatch; the recorded controls
are worse. The installed `RasterizeGLContext` is an alias of the CUDA context,
so the identical GL control is not an independent hardware rasterizer result.
A correction using only triangle IDs in adjacent pixels also fails: entire thin
faces can be absent from the initial raster.

An exhaustive floating-point triangle-coverage reference fixes the saved sequence.
A separate triangle-bounding-box Triton prototype produces byte-identical images
to that reference: worst normalized MSE **3.8936235351e-9** over 16 frames.
This is a bounded diagnostic result, not registration acceptance. Both initial
successful prototypes ran on gpu3001; a later job on that node reported an
uncorrectable ECC error. The candidate must therefore also pass on another node.

## Replay portability is a separate issue

The saved A5000-node sequence rejected exact replay on a newer AMD CPU host.
Seven or more derived velocity/cache fields differed by up to 3.55e-15; physical
body State did not account for that first rejection. The source performs NumPy
matrix multiplication in these paths.

A controlled fresh-process experiment on the same destination host established:

| Destination NumPy BLAS dispatch | Foreign checkpoints restored exactly |
|---|---:|
| Automatic SkylakeX | 0/8 |
| Explicit Haswell | 8/8 |

This establishes a math-profile portability limit, not a reason to weaken the
State guard. Production does not alter global BLAS settings or accept approximate
reconstruction. Cross-profile replay remains unsupported by this evidence.
Matching package versions alone does not establish bitwise replay portability.

## Evidence and reproduction

[Raw diagnostic bundle](semantic_recovery_evidence/final_three_20261004/rocket/diagnostics-first/)
contains the immutable executed script archives and SHA256 manifests, raw pixel
and numeric NPZ witnesses, result JSON, hardware information, and logs for both
successful and rejected controls. Failed prototypes remain distinguishable from
accepted evidence. The earlier B16 performance job's partial output is preserved
separately; it is not a completed benchmark.

Recompute the saved-frame comparisons locally:

```sh
python scripts/research_rocket_production/verify_coverage_diagnostics.py \
  docs/research/semantic_recovery_evidence/final_three_20261004/rocket/diagnostics-first
```

The verifier recomputes squared pixel differences using integer accumulation,
checks executed-script hashes and input provenance, and compares the two
independent coverage implementations. Its tiny relative tolerance applies only
to floating-point serialization of reported MSE, not to State or acceptance.

The production candidate still needs clipping/occlusion coverage beyond the
saved flight witness, the complete CPU/CUDA online source gate, existing public
State/renderer regression gates, and bounded render-overhead measurements. The
research scripts use scoped monkeypatches solely to investigate a candidate;
these are not a proposed public API or production integration.

## Broader candidate results and integration

Two clipped prototypes passed the complete 64-case / 224-frame online source gate:

| Prototype | Mean MSE | Worst-frame MSE |
|---|---:|---:|
| Float64 coverage, job 6995956 | 0.0000208509 | 0.0002093924 |
| Float32 coverage, job 6995996 | 0.0000645899 | 0.0003739141 |

The pixel threshold remains 0.001 per frame. Both were recomputed from raw pixels.
The float32 candidate was selected because it avoids a large-batch slowdown:
[short saved-state render timings](semantic_recovery_evidence/final_three_20261004/rocket/prototype-render-overhead.csv)
cover batches 1/16/64 at 96px and 480px on an L40S. At 480px, candidate time is
1.022× / 0.962× / 0.953× baseline, respectively. These ten-repeat measurements
use saved flight states; they are neither a 1,000-frame representative benchmark
nor a comparison to source throughput, and are not RTX 3090 numbers.

The integrated kernel is `stable_envs/envs/rocket_raster.py`. Rocket overrides a
small private coverage-stage method in the existing renderer. All other renderer
families retain the same nvdiffrast call, offsets, shading and material path.
The frozen legacy Rocket renderer, physics, State format, exact restore guard,
and action-chunk API are unchanged. CUDA Rocket rendering additionally requires
Triton; the CPU installation does not import it.

The earlier unclipped prototype failed contact frames (MSE about 0.0336).
The first production bundle failed before test collection due to absent test
fixtures. Inspection also found its reused base predates the shared PushT
`_stack_states` hook. The corrected bundle overlays **all current tracked
production sources and tests**, includes the new files, and checks every bundled
file hash before execution. No old-base prototype is counted as current-branch
production acceptance.

Production job 6996062 initially passed **67 CPU tests and 2 CUDA regression
tests**, with zero skips. Local Mac checks passed 31 shared-render tests; the two
CUDA tests were skipped locally and then actually executed on Oscar. Public
480px CPU/CUDA gates, current-branch online gates, independent branch suffixes,
and persistent Variation checks remain pending; the matrix is not promoted.
