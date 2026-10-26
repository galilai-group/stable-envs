# Constant-alpha surface replay

This extension supports fixed-camera rigid recordings with constant, straight-ray
alpha surfaces. G90/G36 use perspective cameras; G54/G61 use orthographic cameras.
G01 acrylic additionally enables transmission weight 0.25 and IOR 1.45. It uses
the separate [sampled dielectric policy](object-permanence-optics.md), preserving
this straight-ray alpha policy rather than silently changing its meaning.

The new recording schema is `stable-envs/object-permanence-rigid-v4`, with material
policy `stable-envs/constant-alpha-surfaces-v1` and visibility contract
`stable-envs/transparent-pixel-rays-v1`. These versions describe different things.
Old data, opaque visibility defaults, G34 State and acceptance budgets remain
unchanged. The current audit certifies 150/150 first nominal members: 7 legacy,
94 opaque-v1, 44 opaque-v2, 4 alpha cases and 1 sampled optical case. G98/G132 now use [hash-pinned native realizations](object-permanence-native-scenes.md). Nine additional members (two optical and seven opaque variants) are
supported; 38 remain unimported, including two explicit G05 capability deferrals.

## Material and event contract

Each object's existing `rgba` now carries its effective constant surface alpha
when explicitly using v4. Alpha 0 omits that surface; alpha 1 is opaque. One
unlinked Principled BSDF connected directly to Material Output is supported,
without volume, displacement, image textures or nonzero transmission. Changing
alpha, base color or local geometry is still rejected by all-frame extraction.
Constant alpha does not need another trajectory array or repeated mesh files.
Poses remain float32 `[T,D,7]`, with optional asset-relative scales `[T,D,3]`.

The same fixed perspective/orthographic pixel-center rays and clipping rules are
used. Surface events are ordered by exact binary64 camera-axis depth and then
semantic object ID. A shared triangle edge at the same object/depth is one event;
the lowest original face index supplies interpolation. Distinct coplanar objects
remain distinct events. A solid glass mesh's front and back surfaces contribute
separately. This explicit surface-alpha model does not implement volumetric
absorption, refraction, Fresnel reflection, source shadows or original shaders.

For front-to-back event `k`, contribution is `T * alpha[k]`; then
`T *= 1-alpha[k]`. RGB accumulates simplified linear-color Tiny shading and ends
with `T * background`, followed by the existing uint8 quantization. The first
alpha-one event terminates visibility. There is no small-transmittance cutoff.
Very small float32 contribution weights can underflow; geometric masks do not
depend on those rounded weights or on a perceptual visibility threshold.

`max_layers` is a runtime memory bound, default 16, not an approximation knob.
The renderer checks an additional event and raises on overflow, including after
static/dynamic merging. It never silently drops visible layers. The four
references use at most 5/7/9/9 layers respectively (G90/G36/G54/G61).

## Output and lifetime

`RigidSceneRenderer` requires the explicit alpha contract for v4 data. Online
replay imports neither Blender nor MuJoCo. The existing `render` and `render_many`
entry points and default RGB/depth/ID shapes are retained. New modes are opt-in:

| Mode | CUDA dtype and shape | Meaning |
|---|---|---|
| `rgb` | uint8 `[B,H,W,3]` | Alpha-composited simplified colors |
| `depth` | float32 `[B,H,W]` | Nearest nonzero-alpha surface, including glass |
| `ids` | int16 `[B,H,W]` | That surface's semantic object ID |
| `layer_depth` | float32 `[B,H,W,L]` | Ordered axial depths; unused entries zero |
| `layer_ids` | int16 `[B,H,W,L]` | Ordered semantic IDs; unused entries zero |
| `masks` | bool `[B,G,H,W]` | Overlapping object support before the first opaque surface |
| `weights` | float32 `[B,G,H,W]` | Sum of alpha contributions for each object |

`G` follows manifest object order, while IDs are the manifest's semantic values.
Depth/ID output alone cannot describe a ball behind glass: the ball appears in
the overlapping mask and later depth layers. Empty rays have zero depth/IDs.
The recording remains caller-owned and must outlive its renderer. Caller output
buffers are validated and never alias retained cache state.

```python
with RigidTrajectory(recordings, "g90-nominal") as clip:
    with RigidSceneRenderer(
        clip, visibility_contract="stable-envs/transparent-pixel-rays-v1"
    ) as renderer:
        fields = renderer.render_many([0, 44, 119], modes=("rgb", "masks"))
```

Dense diagnostic weights/masks cost memory proportional to object count; ordinary
RGB training does not allocate them. Select only the modes the caller needs.

## Correctness baseline and reusable optimization

`AlphaRenderer` reuses the existing fixed-order transforms, canonical planes,
pixel rays, exact ray/triangle intersections and nvdiff attribute interpolation.
The full baseline repeatedly selects the next unique surface event. It retains
exact depth/semantic-ID ordering instead of relying on native raster depth alone
for distinct coplanar surfaces. `cache_static=False` keeps this full comparison.

Two generic optimizations follow validation of that baseline:

1. Shade only layers that contain events, retaining the public padded output
   layouts. A diagnostic full-capacity control reproduces the initial baseline.
2. Cache static ordered event depths and face/semantic keys once. Draw moving
   geometry, then perform an exact two-list CUDA merge. An all-opaque dynamic
   partition needs only its nearest event. There are no task names, prescribed
   motions or single-mover assumptions in this logic.

The static seed is not a flattened background RGB plate. A moving ball can lie
between static glass and the floor, so their separate depths must survive.
Geometry/projection/rays, alpha, labels, clipping and capacity are bound by exact
cache snapshots. Material changes rebuild the seed; static geometry varying
within a batch uses full drawing without poisoning the retained seed. Immutable
topology/partition are required. Cross-batch/chunk tests require one build and no
fallback for the fixed-source recordings.

## Independent validation

`freeze_object_permanence_alpha.py` runs audited source preparation independently
of the candidate exporter. `object_permanence_alpha_reference.py` uses projected
triangle edge tests and NumPy sorting, importing no runtime intersection code,
candidate arrays, Torch or renderer. All source meshes retain their identities;
source world poses and effective alphas are recorded alongside layer depths/IDs.
Lossless XOR depth/ID parts remove repeated static data without changing values.
The new oracles coexist with existing fixtures and refuse to overwrite output.

The checker binds source hashes, recipes, camera, alpha, object/frame order and
policy. Every frame checks each layer's depth MSE and each present object's mask,
plus overlapping masks for every object. Budgets remain MSE <= 1e-8 and IoU >= .99.
Fresh-process world checks retain the 2e-6 vertex and 3e-6 corner-normal budgets.
Cached/full RGB, depth, IDs, masks, weights and layers must be bit-identical at
160x90 and 320x180 across chunks 1/3/4/7 and repeated/shuffled requests.

Source-reference masks validate this straight-ray material contract, not Cycles
refraction or perceptual recognizability. Native-source/shared-replay GIFs provide
an additional visual check. Downstream task accuracy has not been measured.

Use the pinned Blender interpreter for export/freezing and the CUDA interpreter
for checking, as described in the [recording guide](object-permanence-recordings.md):

```bash
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --tasks G90,G36,G54,G61 --report <export.json> \
  --visibility-contract stable-envs/transparent-pixel-rays-v1
python -m scripts.testing.freeze_object_permanence_alpha --source-root <source> \
  --output <new-oracle> --task G90
python -m scripts.testing.check_object_permanence_alpha --recordings <recordings> \
  --oracle <new-oracle> --task G90 --report <check.json>
python -m benchmarks.object_permanence_alpha --recordings <recordings> \
  --output <benchmark.json> --batches 1,32 --profile
```

The benchmark alternates synchronized baseline/active-layer/cached measurements,
includes CPU pose selection/upload and internal temporary allocation, and checks
pixel equality. Construction, JIT, first seed build and caller output allocation
are outside steady-state timing. It records visibility-stage CUDA events, seed
memory and PyTorch-allocator peak temporary bytes (not total device/context memory).
After visibility caching, interpolation,
shading and composition account for a larger fraction; a future shared fused
compositor or static shading cache must pass the same geometry and parity gates.

Measured on an RTX 3070 Ti Laptop GPU, 320x180 RGB, batch 32, chunk 4, 7 alternating
samples after warmup, including CPU pose upload:

| Scene | Full-capacity FPS | Cached FPS | Speedup | Full/cached peak temporary MiB |
|---|---:|---:|---:|---:|
| G90 | 46.3 | 246.0 | 5.31x | 160.4 / 79.4 |
| G36 | 23.0 | 277.1 | 12.06x | 164.2 / 97.6 |
| G54 | 27.5 | 248.9 | 9.04x | 160.4 / 111.0 |
| G61 | 28.0 | 265.7 | 9.49x | 160.5 / 112.0 |

The persistent static event buffers add 4.39/6.15/7.91/7.91 MiB respectively,
plus validation snapshots. Visibility-stage event spans fall from about 556–1281 ms
to 38–67 ms per 32-frame call; those spans include host submission gaps. After
caching, interpolation, shading, composition and submission account for roughly
half to three quarters of the profiled call. G90's two dynamic balls retain a
larger visibility share. This is renderer throughput, not an end-to-end model
training speedup or a comparison with MP4 decoding.

Larger chunks trade temporary memory for throughput. With the same batch 32,
G90/G61 reach 371/408 FPS at chunk 16, using about 308/439 MiB temporary memory;
chunk 32 reaches 387/440 FPS at about 616/875 MiB. These outputs also match smaller
chunks bitwise and reuse one static seed. The default remains 4 to keep bounded
memory; callers with available GPU memory can choose `chunk_size=16`. Reproduce
with the benchmark's `--chunk-size` option. No device-specific default or scene
branch is introduced.
