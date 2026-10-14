# Analytic raycast and CPU3D

`raycast3d` intersects spheres, capsules, cylinders, ellipsoids, boxes, and finite
planes. It supports Torch on CPU and CUDA, plus a fused Triton cast on CUDA.
OGBench mazes, fixed-camera DM Control and other analytic MuJoCo render paths
use these operations. `cpu3d` uses them to render a complete primitive scene on
CPU when a moving camera prevents reuse of a static background.

## Rendering contract

Fixed-camera rendering bakes a static RGB/depth plate, casts moving primitives
inside conservative screen bounds, and shades only covered samples. CPU3D adds
sky and floor textures, directional shadows, and specular lighting; unsupported
geometry and shadow-floor configurations are rejected by its scene constructor.

MuJoCo's mono camera averages its two stereo eyes. Its depth buffer measures
perpendicular distance, whereas the composite compares radial ray parameters.
Specular is additive before texture modulation. These conventions, arithmetic
order, and final byte conversion affect silhouette and highlight pixels.

Regression tests require exact agreement between the optimized compositor and
its full-window reference, and between Torch and Triton casts on float32 inputs.
Bounded casts must match the unbounded Torch cast. Agreement with source OpenGL
is measured per class. Silhouettes, antialiasing, textures and lighting can leave
a nonzero MSE.

Use `backend="auto"` for device-appropriate dispatch. Explicit `"triton"` requires
Triton, CUDA float32 inputs on one device, and packed trailing axes. Unsupported
explicit requests raise `ValueError`; `"auto"` falls back to Torch. The Torch cast
accepts `bounds=False` for diagnostics. Regression overrides belong in scoped test
fixtures, not process-wide production switches.

## Performance

Batch-64 `DMCBallInCupCatch` renders from saved states. Both renderers used the same states, output
destination and device class. The source GPU renderer is NVIDIA OpenGL. Timing excludes rollout,
state loading, restoration and render-state preparation.

| Device and output | Ours FPS | Source FPS | Ratio | Normalized MSE, mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| One Xeon Platinum 8268 thread, CPU output | 15.64 | 14.90 | 1.05× | 0.00004790 ± 0.00001818 |
| One RTX 3090, CPU output | 2,192.28 | 240.24 | 9.13× | 0.00005607 ± 0.00002248 |

Sixteen states were repeated to fill each batch. Each timing used three warmed
samples. Other configurations will differ. See the
[testing guide](../testing.md) for contributor checks.

::: stable_envs.render.raycast3d
