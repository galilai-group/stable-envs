# CPU primitive-bound component layout, 2026-09-29

The targeted optimization changes only the CPU `_rays_near` implementation in
`stable_envs/render/raycast3d.py`. It keeps each x/y/z component contiguous through
cross products and dot products, avoiding packed `(B,N,3)` temporaries and their
strided component reads. There is no compiler, shader, fidelity, State, or CUDA
algorithm change. CPU source and candidate both render actual saved poses; no
rollout, loading, State preparation, or setup is timed.

The prior profile identified this bound predicate as 73% of BallInCup and 52% of
Finger B64 render wall time. Smaller work groups were not a general solution, and
an earlier Inductor experiment added compilation costs without enough improvement.
This change addresses the array work directly.

## Numerical gate and rejected first attempt

The first component expression used separate multiplication/subtraction. It
passed the real B64 image corpus and improved throughput, but adversarial rays
within 32 ulps of tangency exposed different bound decisions: 12 of 16 dtype,
scale and stride cases failed. CPU `torch.cross` contracts one multiply with the
subtraction, so equivalent algebra was not equivalent floating-point execution.
The pilot is diagnostic only. V1 CPU array 6810370 and queued CUDA 6810373 were
cancelled; retained partial output is not acceptance evidence.

V2 uses `torch.addcmul` to preserve that contraction. It retains the legacy
arithmetic order and tolerances. All 77 tests passed on both Apple M5 and Oscar
x86 CPU: the new 16 exact legacy-mask tests plus 61 existing primitive-bound
checks. These include float32/float64, contiguous/strided rays, four spatial
scales, tangent rays, zero and reversed directions, and primitive combinations.
Oscar test/staging job 6810676 passed; its XML is retained. Ruff passed.

## Measurement protocol

CPU array 6810700 tests BallInCup, Finger and Manipulator at B64/128/256, using
16 distinct saved States repeated into batches. Each baseline/candidate pair
runs in a fresh process on the same allocated host. Each leg warms source and
ours, then takes three alternating paired timings. Full baseline/candidate
output SHA-256 values must match at every batch. Full saved source State, input
render State and live candidate State must remain unchanged. The timings are
single-thread CPU analytic rendering against source software OpenGL, including
batch assembly/output destination identically on each side.

Manipulator uses the existing accepted source-repeat policy: repeated source
MSE <= 0.0001, raw maximum-byte difference retained, and actual paired MSE sampled
outside timing. Ours still has strict repeatability and exact baseline/candidate
output hashes. This is source GL variation at unchanged State, not permission
for candidate pixels to change.

The source checkpoint is the immutable accepted Humanoid chunk4 snapshot,
`dmc-humanoid-chunk-gate-20260929/source`. V2 copies it into
`dmc-components-v2-gate-20260929/source` and overlays only `raycast3d.py`.
The candidate file SHA-256 is
`d0a55325ed934695bbb9003b7735b0b391c9d35f8f5dc7d5fcc7662eb9ddcdc5`.
Every driver verifies the production source hashes before execution.

## Terminal CPU results

All three CPU tasks completed with unchanged State and exact whole-batch pixels.
Each row compares source and candidate on the same host; machines differ between
registrations, so these FPS should not be mixed with the faster pilot hardware.

| Registration | Batch | Baseline FPS | Candidate FPS | Source CPU FPS | Gain over baseline | Candidate/source |
|---|---:|---:|---:|---:|---:|---:|
| swm/BallInCupDMControl-v0 | 64 | 10.767 | 15.639 | 14.902 | 1.453x | 1.049x |
| swm/BallInCupDMControl-v0 | 128 | 11.003 | 14.349 | 15.182 | 1.304x | 0.945x |
| swm/BallInCupDMControl-v0 | 256 | 10.059 | 15.005 | 14.533 | 1.492x | 1.032x |
| swm/FingerDMControl-v0 | 64 | 13.271 | 17.257 | 21.245 | 1.300x | 0.812x |
| swm/FingerDMControl-v0 | 128 | 12.982 | 16.909 | 21.378 | 1.303x | 0.791x |
| swm/FingerDMControl-v0 | 256 | 11.512 | 12.716 | 21.530 | 1.105x | 0.591x |
| swm/ManipulatorDMControl-v0 | 64 | 16.565 | 20.982 | 22.891 | 1.267x | 0.917x |
| swm/ManipulatorDMControl-v0 | 128 | 15.087 | 19.564 | 23.116 | 1.297x | 0.846x |
| swm/ManipulatorDMControl-v0 | 256 | 13.635 | 16.524 | 22.803 | 1.212x | 0.725x |

BallInCup reaches a near tie with source at B256; its 3.3% advantage should not
be presented as a robust large win. Finger and Manipulator remain slower than
source CPU despite meaningful improvements over the prior implementation.
Large batches do not improve these CPU renderers consistently. Per-frame timing
SD and individual samples are retained in the raw results. The complete CPU jobs
peaked at 4.36–4.45 million KiB RSS (whole-job footprint, including source contexts).

## Cross-consumer and CUDA coverage

`_rays_near` also serves CPU3d (including Humanoid), the analytic MuJoCo renderer,
and composites used by maze renderers. PointMaze's sphere-only scene bypasses
this bounded primitive path. The primitive tests cover the actual capsule,
cylinder and box bound callers.

Cross-consumer job 6810781 passed Humanoid B64 exact image equality and eight
MuJoCoPusher exact images. Both Humanoid legs retain the accepted chunk4 setting.
Humanoid MSE remains 0.0004328633; sequential-leg timing was 10.081 -> 10.690 FPS
(about 1.06x), a secondary diagnostic rather than a strong small-effect claim.
This B64 check does not update its B256 benchmark row.

CUDA job 6810719 passed BallInCup and Finger exact baseline/candidate images at
B64 for both CPU and CUDA output destinations, plus 262144 exact primitive
predicates per leg on RTX 3090. Its Manipulator baseline stopped at the obsolete
one-byte source-repeat assertion. This attempt is preserved. A fresh retry,
6811231, passed with the already established source-repeat MSE policy: ours
remains byte-exact for both output destinations, and all 262144 predicates match.
The three CUDA candidates measured 0.98–0.99x their baselines in these short
sequential legs, within the scale of sampled timing variation; no GPU speedup is
claimed. This checks shared dispatch/compilation, although the CUDA expression
is unchanged. It is not a new 42-registration benchmark. All own jobs and
collectors are terminal. `collection_status.json` records the full PASS and
`comparison.csv` contains all measured comparisons.

Raw harnesses, source hashes, sampled timings, terminal records and pixel hashes
are under `semantic_recovery_evidence/dmc_cpu_bounds_components_20260929/`.
The bounded 16-state corpus is performance and regression evidence, not 1000
independent new visual observations. No accepted prior evidence was overwritten.
