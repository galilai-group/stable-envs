# Measured bottlenecks and next changes

The ratios below are the previous October 8 batch-256 render measurements,
using matching output destinations. Ratios below 1 mean we are slower.
Profiles are new diagnostics against candidate `65ec1f38`; their instrumented
wall times explain where work goes, not replacement benchmark results.

| Target | Ours/source FPS | Evidence | Next change and acceptance check |
| --- | --- | --- | --- |
| PushT, continuous and discrete, CPU | 0.266 / 0.276 | Actual saved corpus at B64 and B256 rebuilds 1,024 scenes per 1,024-frame pass. B1024 rebuilds zero after warmup. Local continuous diagnostic: 2.70 s at B256 vs 0.88 s at B1024. | Separate immutable drawing geometry from pose/reset/goal values. Reuse geometry without replaying reset. Compare all saved pixels, varied goals/geometries, contact marks, and input/live-State nonmutation. Benchmark alternating configurations, not only one cached scene. |
| OGBScene, NVIDIA OpenGL | 0.074 | B256 diagnostic: 2.39 of 2.63 s in NumPy supersample averaging; GL rendering/readback about 0.20 s. | Replace the strided generic mean with an exact box reduction, preserving rounding. Then test an equivalent GPU reduction to remove the host step. Require pixel identity to the current implementation and source MSE gates at every batch. |
| OGBCube, MJWarp fidelity | 0.511 | Six raster calls across three configuration groups; 0.273 s of the 0.407 s diagnostic waits in Warp synchronization. Six render megakernels consume 350.67 of 388.98 ms of profiled CUDA time; each uses 181 registers/thread, with a register limit of four resident 64-thread blocks per SM (a 16.7% theoretical thread-occupancy ceiling). CUDA kernel compute throughput 21.3%, DRAM throughput 10.0% during that replay. | Reduce register pressure in the dominant render megakernel and measure the effect on residency/spills; reduce duplicated geometry/fidelity work, and remove unnecessary pass synchronization only where ownership permits. Retain transparent objects, shadows, reflections, and the full varied-state parity corpus. |
| Fetch, all 12, CPU | 0.567–0.839 | Reach representatives spend nearly all their time inside the OpenGL frame path. In one B256 diagnostic, 5.13 s is inside per-frame resolve/blit code and 1.85 s in `mjr_render`; State restoration is only 6 ms. | Profile the software-GL resolve/resize separately. Compare a CPU-native resize/readback path with the current batched framebuffer path. Keep the source's 480px sampling/context behavior; do not regain speed by dropping fidelity. |
| Finger, CPU | 0.834 | B256 diagnostic: 4.08/6.64 s primary ray intersections; 1.66 s shadow occlusion. | Share the CPU ray-kernel optimization with Reacher, using fused native loops/BVH pruning or a parity-tested software-raster path. Preserve shadows and exact camera/material configuration. |
| Reacher, CPU | 0.895 | B256 diagnostic: 3.01/6.84 s primary rays; reflection path 2.42 s (nested shadow work overlaps). | Same CPU kernel work; do not sum overlapping profile times. Re-run small and large batches because Python overhead matters more at B1. |
| Humanoid, CPU | 0.663 | B256 diagnostic: 27.36 s in CPU rendering; shading 11.15 s, nearest intersections 8.08 s, texture filtering 4.48 s (nested totals overlap). | Fuse shading/intersection/texture work to avoid repeated Torch temporaries and dispatch. Compare with a source-compatible Mesa route before adding a new custom kernel implementation. |
| Quadruped, CPU | 0.527 | B256 diagnostic: 29.00 s CPU rendering; nearest intersections 10.49 s, shading 9.07 s, reflections 5.49 s (nested totals overlap). | Same shared CPU-renderer change as Humanoid; maintain filtered floor textures and reflected geometry. |

Cube CPU is essentially tied with source (~1.000×); this is not evidence of a
material regression. PushT's CUDA configuration still uses CPU drawing plus
upload, so its ~0.26× result is not a CUDA rasterization result. There is no
source GPU renderer for that comparison.

Priority: Scene reduction, PushT geometry ownership, shared CPU rendering,
then Cube's fidelity kernels. Fetch's CPU GL resolve needs a focused A/B probe.
Do not remove fidelity passes or change the comparison protocol to claim wins.

## New online findings

The new forward-trace workload is broader than the saved render table.
FetchSlide's three registrations currently exceed the per-frame 0.001 gate
at H8 in a B1 GPU pilot (first failure: 0.00100050036). Keep those cells failed;
a small exceedance is not a reason to move the acceptance threshold.

Rocket's first online pilot exposed an invalid ignition value in the benchmark
input generator; that generator has been corrected to the maintained valid
powered-flight action. The failed attempt is retained and the retry is separate.
The older saved Rocket GPU corpus also still fails replay verification; that
is a different failure and is not repaired by correcting benchmark actions.

## Evidence

Oscar diagnostic jobs: 7123087 (Fetch GPU), 7123089 (Scene), 7123090 (Cube),
7123114/7123115 (Fetch CPU), 7123091 (Finger), 7123102 (Humanoid),
7123103 (Quadruped), and 7123084 (Reacher). The archive hashes are in
[provenance.json](provenance.json). PushT local diagnostic scripts and JSON are
preserved in `branch-backups/presentation-20261008/`.
