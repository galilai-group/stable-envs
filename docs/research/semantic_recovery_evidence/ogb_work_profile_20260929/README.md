# Cube/Scene exact-workload diagnosis — 2026-09-29

Immutable production source-v12, RTX 3090, saved-State batch256, same 16 selected poses and CPU uint8 output as the paired campaign. No production files changed. Array 6800850. New diagnostic scripts and raw profiles are retained here. Accepted parity/benchmark outputs were not overwritten.

## Cube

Three instrumented calls averaged 2230.628 ms/batch. Context misses accounted for 2045.475 ms/batch (91.70%). Independent uninstrumented paired timing after profiling was 2224.171 ms/batch, so the wrapper instrumentation did not introduce the two-second behavior.

The batch contains three static configuration groups. Each needs two MJWarp contexts (normal and opaque pass), at world capacities 256,32,16 respectively. The existing context cache has capacity four. Each of three warmed calls produced six misses, six hits (the fidelity reader retrieves the context a second time), and six evictions. Six different contexts are rebuilt each render. Context construction deep-copies the model, constructs Warp model/data/render buffers, executes warmup, and captures a CUDA graph.

Nested timing categories (per call; do not sum overlapping inclusive times):

| Section | ms | Calls |
|---|---:|---:|
| renderer_boundary | 2215.071 | 1 |
| pass.draw | 2154.857 | 6 |
| pass.render | 2154.125 | 6 |
| context.miss | 2045.475 | 6 |
| build_render_context | 856.085 | 6 |
| context.warmup_capture_frame | 651.948 | 12 |
| warp_wait | 150.117 | 12 |
| configuration_hash | 9.514 | 259 |
| mj_forward | 1.630 | 12 |
| validate_render_state | 1.137 | 1 |
| warp_graph_launch | 0.886 | 6 |
| torch_wait | 0.430 | 8 |
| configured_bundle | 0.253 | 3 |
| context.hit | 0.054 | 6 |

CProfile's self/cumulative times include its own overhead and GPU wait attribution; they are not clean GPU kernel durations. The lightweight wrappers do not add CUDA synchronizations. Their wall-clock intervals include the existing synchronization in the renderer. Output repeat was bit-exact for Cube. CPU validation and configuration hashing are small compared with rebuilding contexts.

## Scene and causal follow-up

Initial Scene diagnostic required bit-exact output and stopped when that stronger-than-benchmark repeat assertion failed. The accepted paired harness permits max 1/255 channel variation; no failed output is promoted. Separate retry array 6800940 records the actual repeat difference and uses that existing tolerance. It also runs an isolated Cube cache-capacity experiment sized to the observed working set, requires bit-exact Cube pixels, and restores the original capacity afterward. That experiment is not a production fix or acceptance promotion.
