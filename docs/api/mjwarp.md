# MJWarp rendering

This CUDA backend renders batches of MuJoCo configurations as `(B, H, W, 3)`
uint8 frames. Fetch and OGBench Cube/Scene use it, alongside model preparation,
static background composition, and family-specific material handling.

## Rendering contract

Model preparation selects visible geom groups, removes invisible geometry,
sets shadow behavior, and records unsupported texture mappings. Packed BGRA
output is converted to RGB. GL-compatible rounding must be active while the
render graph is captured; changing the pack function afterward cannot change
an already captured graph. Tracking cameras also require center-of-mass data.

Context construction temporarily patches upstream texture upload and byte
packing helpers. Their names are checked when the optional backend is loaded.
Construction must be exclusive with other users of those upstream helpers;
these patches are not thread-safe. Importing this module defaults
`WARP_CACHE_PATH` to `~/.cache/warp` only if the caller has not supplied a path.

The light specialization is enabled only for its supported upstream version
and model; other configurations use stock MJWarp. Exact regression checks
protect internal refactors. They do not imply exact source OpenGL pixels:
texture filtering, coverage, and shading can retain registration-specific MSE.

## Recorded performance

An archived saved-state run measured Cube with `mjwarp_fidelity`, batch 256,
on one RTX 3090. With CUDA output, ours took **0.59445 ± 0.00598 ms/frame**
(1,682 FPS), versus NVIDIA OpenGL's **0.70748 ± 0.03504 ms/frame** (1,413 FPS):
1.19×. Normalized RGB MSE was **0.00046346 ± 0.00010413** across the corpus.
Timing statistics used six warmed measurements.

Both paths used the same saved states and output destination. State loading,
rollout, restoration, and render-state preparation were excluded. These are
archived renderer-only results, not a fresh benchmark of this cleanup or a
claim about online stepping.
[Full table and evidence paths](../research/semantic_recovery_evidence/paired_saved_rendering_20260929/paired_rendering_42_b256.csv).

### Cube default checkpoint

Cube's CUDA default now selects `mjwarp_fidelity` and 2× supersampling when
sampling is unspecified. Explicit backends and sampling in supplied RenderStates
keep their previous meaning. CPU rendering remains software OpenGL.

A paired RTX 3090 run at 64×64 measured normalized MSE **0.0006046 ± 0.0000364**,
with maximum **0.000691** across 32 independently reached States:

| Batch | Ours GPU FPS | Source GPU FPS | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 144 | 1,948 | 0.074× |
| 64 | 4,105 | 1,791 | 2.29× |
| 256 | 5,812 | 1,787 | 3.25× |

Public rendering and synchronization are timed after three warmups, with six
samples. Loading, rollout, projection, initial transfer and batch indexing are
excluded. Source OpenGL loops over independently replayed ready instances and
returns NumPy images including readback; ours returns CUDA images. Both use the
same GPU allocation. Larger batches repeat the 32 States. This table measures
renderer throughput, not online stepping or a matched CUDA-output pipeline.

At B256, removing unused composition and reducing configuration overhead makes
this 1.60× faster than the previous 2× fidelity implementation. It is still 22%
slower than the old default, whose mean MSE was 0.002059. B1 remains slower than
source. This Cube checkpoint does not certify Scene; its separate gate follows
in the [testing record](../testing.md).
[CSV with latency and FPS dispersion](https://github.com/galilai-group/stable-envs/blob/sami-dev/tests/fixtures/testing/cube-default-cuda-performance.csv)
and [acceptance scope](../testing.md#cube-default-cuda-renderer-acceptance).

::: stable_envs.render.mjwarp
