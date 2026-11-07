# MJWarp rendering

This CUDA backend renders batches of MuJoCo configurations as `(B, H, W, 3)` uint8 frames. Fetch and
OGBench Cube/Scene use it, with model preparation, static background composition and family-specific
material handling.

## Contract

- Model preparation selects visible geom groups, removes invisible geometry, sets shadow behaviour and
  records unsupported texture mappings.
- The backend converts packed BGRA output to RGB.
- GL-compatible rounding must be active when the backend captures the render graph. Changing the pack function
  later does not change a captured graph.
- Tracking cameras need centre-of-mass data.

## Limits

- Context construction temporarily patches upstream texture-upload and byte-packing helpers. The
  backend checks their names when it loads. These patches are not thread-safe. Do not construct
  contexts concurrently with other users of those helpers.
- Importing the module sets `WARP_CACHE_PATH` to `~/.cache/warp` if the caller has not set it.
- The light specialization applies only to its supported upstream version and model. Other
  configurations use stock MJWarp.
- Exact regression tests protect internal refactors. They do not mean the output equals source
  OpenGL. Texture filtering, coverage and shading can leave a per-class MSE.

## Cube and Scene dispatch

Cube and Scene default to `supersample=1`. With `backend="auto"`:

- CPU rendering uses software OpenGL.
- CUDA rendering at 1× sampling uses NVIDIA OpenGL with the public class's compiled appearance.
- CUDA rendering at 2× sampling or more uses the two-pass MJWarp fidelity renderer. At 2×, Scene's
  four samples per output pixel align with NVIDIA OpenGL.

Explicit backends and sampling settings, including settings stored in a supplied RenderState, keep
their meaning.

## H200 Cube and Scene throughput

The candidate measurements below use CUDA MJWarp fidelity at `supersample=2`, not the default
`supersample=1` `auto` path (NVIDIA OpenGL). They compare MuJoCo/MuJoCo Warp 3.17 on one H200 at
64×64 against the registered stable-envs NVIDIA OpenGL path with the same compiled appearance. The
MJWarp candidate returns CUDA images; the OpenGL baseline returns NumPy images, including GPU readback.

| class | batch | candidate FPS | OpenGL baseline FPS | observed ratio |
| --- | ---: | ---: | ---: | ---: |
| Cube | 1 | 237 | 261 | 0.907× |
| Cube | 64 | 8,147 | 264 | 30.8× |
| Cube | 256 | 13,506 | 264 | 51.1× |
| Scene | 1 | 184 | 230 | 0.803× |
| Scene | 64 | 6,160 | 230 | 26.7× |
| Scene | 256 | 9,971 | 230 | 43.3× |

The run uses 32 independently reached saved states, repeated for larger batches, with three warmups
and six timed samples. Timing includes public rendering and synchronization and excludes rollout,
loading, projection, initial transfer and batch indexing. FPS values are rounded for display; ratios
are derived from the unrounded timings.

These are H200-specific observed throughput ratios, not a general speedup attributable to a MuJoCo
version or hardware class. The ratios are amplified because the stable-envs OpenGL baseline was about
6–7× slower on this H200 than in the prior RTX 3090 measurements. The GPU, driver, host and MuJoCo
version all changed, so the comparison does not isolate a hardware effect.

::: stable_envs.render.mjwarp
