# Analytic raycast and CPU3D

`raycast3d` intersects spheres, capsules, cylinders, ellipsoids, boxes, and finite
planes. It supports Torch on CPU and CUDA, plus a fused Triton cast on CUDA.
OGBench mazes, Maze2D, fixed-camera DM Control, and analytic MuJoCo render paths
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
its private full-window reference, and between Torch and Triton casts on supported
float32 inputs. Bounds must also match the unbounded Torch cast. That is an
internal implementation guarantee: agreement with source OpenGL is a separate,
registration-specific measurement. Analytic silhouettes, antialiasing, textures,
and lighting can leave a nonzero source-image MSE.

Use `backend="auto"` for device-appropriate dispatch. Explicit `"triton"` requires
Triton, CUDA float32 inputs on one device, and packed trailing axes. Unsupported
explicit requests raise `ValueError`; `"auto"` falls back to Torch. The Torch cast
accepts `bounds=False` for diagnostics. Regression overrides belong in scoped test
fixtures, not process-wide production switches.

## Recorded performance

The September 29 saved-state investigation measured these batch-64 BallInCup
renders. Both renderers used the same saved states, output destination, and device
class; source GPU means NVIDIA OpenGL. Timers excluded rollout, state loading,
restoration, and render-state preparation.

| Device and output | Ours FPS | Source FPS | Ratio | Normalized MSE, mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| One Xeon Platinum 8268 thread, CPU output | 15.64 | 14.90 | 1.05× | 0.00004790 ± 0.00001818 |
| One RTX 3090, CPU output | 2,192.28 | 240.24 | 9.13× | 0.00005607 ± 0.00002248 |

These are archived results, not a fresh benchmark of this cleanup. Sixteen
independent states were repeated into batches; each timing used three warmed
samples. They do not establish speed or fidelity for other configurations. See the
[fidelity methodology](../concepts/fidelity-methodology.md) for source comparison
and [testing guide](../testing.md) for contributor checks.

::: stable_envs.render.raycast3d
