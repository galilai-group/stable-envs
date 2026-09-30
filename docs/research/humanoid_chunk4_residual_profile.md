# Humanoid CPU: residual work after the chunk-four fix

Subsequent bounded change: the CPU component-layout optimization documented in
`dmc_cpu_bounds_components.md` passed exact Humanoid B64 image equality and
improved its secondary sequential-leg timing by about 6%. The detailed cost
breakdown below profiles the preceding accepted chunk4 snapshot. It was not
rerun after that bounds change, and the main B256 table row remains unchanged.

This is a fresh profile of the accepted `CPU_RENDER_CHUNK=4` implementation,
not the earlier chunk-sixteen profile. Oscar job 6810114 uses the immutable
`dmc-humanoid-chunk-gate-20260929/source` snapshot, checks its source hashes,
and renders the same 16 saved States at batches 64 and 256 on a CPU-only node.
The benchmark uses one Torch thread and one llvmpipe worker; four CPUs are
allocated to accommodate the process and graphics runtime.

The renderer is busy doing image work. At batch 64, internal pose preparation
costs 1.18 ms out of a 3,895 ms instrumented render (0.03%). It already uses
MuJoCo kinematics, CoM, and camera/light updates rather than full dynamics.
Source installation and our RenderState projection happen before the timer;
this is prepared-input rendering, not rollout or disk-loading throughput.

The fresh non-overlapping cost buckets are approximately:

| Batch-64 work | Time | Share |
|---|---:|---:|
| Primitive bounds | 801 ms | 20.6% |
| Primitive intersections | 1,113 ms | 28.6% |
| Texture bilinear sampling | 257 ms | 6.6% |
| Vector normalization | 108 ms | 2.8% |
| Primary/shadow reduction and shadow packing | 453 ms | 11.6% |
| Remaining chunk work | 1,150 ms | 29.5% |

The last bucket includes shading, plane calculations, gather/scatter work,
output conversion, and tensor bookkeeping; the profile does not justify
calling all of it pure shading. Capsules account for 979 ms of the intersection
bucket. Primary casting totals 1,241 ms and shadow preparation/casting totals
1,125 ms, but those totals overlap bounds/intersections above and must not be
added to them.

A separate Torch operator profile exposes the mechanism: 2,928 advanced-index
operations, 42,018 multiplies, 28,769 adds, and 512 cross-products. Indexing is
the largest individual operator by measured self CPU time. Positive operator
memory accounting totals 12.55 GB for one batch-64 render: substantial temporary
tensor allocation activity, not 12.55 GB of simultaneous memory use and not a
DRAM-bandwidth measurement. The combined benchmark worker holds about 3.15 GB
RSS before the Torch profiler; profiler instrumentation itself subsequently
raises it above 8 GB, so that later value is not the render working set.

Unprofiled process CPU time is 99.1% of one core, with zero major page faults.
The source uses about 104.5% of one core. Relative to the four-CPU allocation,
these are about 25% and 26%; that reflects the deliberately single-worker
comparison, not evidence that our one rendering thread is idle. More CPU
threads are a separate experiment and must give the source comparable resources.

Source profiling is dominated by native software OpenGL: 3,213 ms in drawing
and 951 ms in pixel readback out of a 4,392 ms profiled call. Our implementation
instead pays for many eager tensor passes over ray bundles and hit-point arrays.
This evidence points toward reducing intermediate arrays and dispatch/gather
passes in the ray and shading kernels. It does not support optimizing State
serialization, replacing the already-minimal kinematics, or increasing batch
size as a substitute for fixing per-frame work.

No production code changed during this investigation. A possible next bounded
experiment is a compiled CPU kernel for a measured expensive operation, with
primitive tie order, numerical slack, and pixel output checked against the
current implementation. Thread-count scaling should be measured separately.
The existing source/candidate comparison and quality settings must remain
visible while evaluating either change.

All raw traces, operator rows, State/corpus provenance, harness hashes, and
scheduler evidence belong under
`semantic_recovery_evidence/humanoid_chunk4_residual_profile_20260929/`.
Job 6810114 completed with exit 0 in 8m46s on node2355 (AMD EPYC 9684X).
The final alternating unprofiled timings are:

| Batch | Ours ms/frame ± SD | Source CPU ms/frame ± SD | Ours FPS | Source FPS | Speedup |
|---|---:|---:|---:|---:|---:|
| 64 | 60.289 ± 0.249 | 69.938 ± 2.060 | 16.587 | 14.298 | 1.160× |
| 256 | 60.265 ± 0.051 | 68.376 ± 0.693 | 16.593 | 14.625 | 1.135× |

Each row uses three alternating timing samples. Increasing the outer batch
from 64 to 256 leaves our per-frame throughput essentially unchanged, because
both execute the same four-world internal chunks on one thread. Batch 256
also repeats the cost shares: bounds 20.6%, intersections 28.5%, texture 6.65%,
remaining chunk work 29.5%, and pose preparation 0.026%.

Normalized MSE across the 16 independent States is 0.000432863 ± 0.000073295
(SD), maximum 0.000542634, at both batches. These fresh diagnostic timings use
a different host from earlier acceptance runs and do not silently replace the
accepted comparison table. Source and ours within each row share the same host,
State corpus, batch, and CPU output destination.

The CPU-only Torch profiler emitted a GPU device-count discovery warning on a
node without CUDA hardware; it still completed its requested CPU activity
profile. The result files and scheduler exit status both confirm completion.
