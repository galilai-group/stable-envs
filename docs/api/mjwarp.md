# MJWarp rendering

This CUDA backend renders batches of MuJoCo configurations as `(B, H, W, 3)` uint8 frames. Fetch and
OGBench Cube/Scene use it, with model preparation, static background composition and family-specific
material handling.

## Contract

- Model preparation selects visible geom groups, removes invisible geometry, sets shadow behaviour and
  records unsupported texture mappings.
- Packed BGRA output is converted to RGB.
- GL-compatible rounding must be active when the render graph is captured. Changing the pack function
  later does not change a captured graph.
- Tracking cameras need centre-of-mass data.

## Limits

- Context construction temporarily patches upstream texture-upload and byte-packing helpers. The
  backend checks their names when it loads. These patches are not thread-safe: do not construct
  contexts concurrently with other users of those helpers.
- Importing the module sets `WARP_CACHE_PATH` to `~/.cache/warp` if the caller has not set it.
- The light specialization applies only to its supported upstream version and model. Other
  configurations use stock MJWarp.
- Exact regression tests protect internal refactors. They do not mean the output equals source
  OpenGL. Texture filtering, coverage and shading can leave a per-class MSE.

## Cube and Scene defaults on CUDA

With `backend="auto"` on CUDA:

- Cube uses the two-pass fidelity renderer with 2× supersampling.
- Scene uses four samples per pixel aligned with NVIDIA OpenGL, and two passes for transparency and
  shadows. Its static background is baked once at the matching sample positions.

Explicit backends and sampling settings in a supplied RenderState keep their meaning. CPU rendering
uses software OpenGL.

Measured on one RTX 3090 at 64×64, against source NVIDIA OpenGL, on 32 independently reached saved
States. Larger batches repeat those States. Three warmups and six timed samples. Timing covers public
rendering and synchronization. It excludes rollout, loading, projection, initial transfer and batch
indexing. The source returns NumPy images including readback; this renderer returns CUDA images.

| class | batch | ours FPS | source FPS | speedup |
| --- | ---: | ---: | ---: | ---: |
| Cube | 1 | 144 | 1,948 | 0.074× |
| Cube | 64 | 4,105 | 1,791 | 2.29× |
| Cube | 256 | 5,812 | 1,787 | 3.25× |
| Scene | 1 | 117 | 1,641 | 0.071× |
| Scene | 64 | 3,297 | 1,480 | 2.23× |
| Scene | 256 | 4,769 | 1,476 | 3.23× |

Normalized MSE against the source: Cube 0.0006046 ± 0.0000364 (worst 0.000691); Scene
0.000768 ± 0.000106 (worst 0.000950). Both are below the 0.001 per-frame limit. Batch 1 is slower than
the source.

::: stable_envs.render.mjwarp
