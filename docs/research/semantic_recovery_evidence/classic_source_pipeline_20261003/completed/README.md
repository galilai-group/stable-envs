# First two independent source/pipeline closures

CartPole and Pendulum independently completed both output devices, every
B1/64/256, T1/8 and no/final/all-frame cell. Each report passes 36 complete
State/task gates, 72 timing cells and 18 component cells. Every measured source
image error is zero, including 2,048 B256/T8/all images per output device.
Scheduler completion and raw logs are retained separately from application PASS.

At B256/T8/all, mean +/- sample SD FPS:

| Environment | Source CPU, native output | Ours CPU | Ours GPU | Source CPU delivered to GPU |
|---|---:|---:|---:|---:|
| CartPole | 372.55 +/- 2.55 | 505.68 +/- 0.34 | 10128.82 +/- 15.69 | 363.41 +/- 0.70 |
| Pendulum | 275.82 +/- 2.17 | 538.83 +/- 1.66 | 12210.58 +/- 34.52 | 267.15 +/- 4.55 |

Source has no GPU renderer. GPU comparisons here use the explicitly supported
CPU-source fallback, with both native CPU and GPU-delivery measurements shown.
Source is serial; the candidates for these two domains also use local scalar
physics, not process pools. Both run under the same eight-CPU allocation and
one Torch/BLAS thread; this is not a parallel-source scalability result.

Plain no-frame stepping remains slower than source. For CUDA output at B256/T8,
CartPole is 26,268 +/- 792 steps/s versus direct-source 52,305 +/- 158; Pendulum
is 28,840 +/- 83 versus 45,539 +/- 122. Ours additionally returns mandatory final
State. Inspection shows the local classic path still exports State inside every
step, even when no intermediate snapshot is requested. The shared dispatcher
already supports `_advance_without_snapshot`; a family-local implementation is
a concrete next candidate, requiring separate correctness/performance gates.
No production change is made from this observation alone.

The optional component-CSV export was added after the cluster run. It consumes
the same immutable reports; regenerated main CSVs are byte-identical to the
original summary. The measured source and benchmark driver are unchanged.
Reproduce using `summarize_classic_source_pipeline.py REPORT NEW_CSV --source
FROZEN_CODE --components-output NEW_COMPONENT_CSV`. The other three registrations
remain independently pending; these two cannot satisfy their gates.
