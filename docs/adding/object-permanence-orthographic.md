# Orthographic replay contract (v2)

`stable-envs/opaque-pixel-rays-v2` adds fixed orthographic cameras to opaque rigid
replay. V1 remains perspective-only. Its defaults, references, mask IoU ≥ 0.99
and depth MSE ≤ 1e-8 are unchanged. RGB remains simplified; source shading is a
visual reference, not a pixel-equality gate.

The G09/G11/G75 pilots cover a tall car and beam, a moving container, and a
moving ball/panel. The same implementation then accepts 41 first nominal members:

| Batch | Accepted | Worst all-frame depth MSE | Lowest object mask IoU |
|---|---:|---:|---:|
| G08–G26 | 17 | 4.024e-12 | 1.0 |
| G43–G78 | 15 | 3.305e-15 | 1.0 |
| G91–G128 | 9 | 7.736e-14 | 1.0 |

Together with 7 legacy and 88 V1 scenes, 136 of 150 first nominal members are
certified. The 14 remaining deferrals are 6 Curve/FONT scenes, 5 transparent
scenes, 2 independent source-normal failures and 1 animated-geometry scene.
G54's `guardwall_a_back` and G61's `spiral_guard_00` use active glass with alpha
0.24; supporting their camera does not make their transmission opaque. G46's
camera is supported but its FONT identity labels remain outside mesh export.
The 47 additional member scripts remain unaudited and unregistered.

## Projection and depth

Use the recorded Blender camera world matrix `W` and OpenGL projection `P`.
For a top-left image with width `w`, height `h`, the pixel-center coordinates are
`nx = 2*(x+0.5)/w-1`, `ny = 1-2*(y+0.5)/h`.

| Camera | Local origin | Local direction |
|---|---|---|
| Perspective | `(0,0,0)` | `((nx+P02)/P00, (ny+P12)/P11, -1)` |
| Orthographic | `((nx-P03)/P00, (ny-P13)/P11, 0)` | `(0,0,-1)` |

Transform the origin as a point and the direction as a vector by `W`. Do not
normalize the direction or move the origin to the near plane. In
`point = origin + t*direction`, `t` is positive camera-axis depth, not Euclidean
range from the camera center. Thus the public depth unit stays the source scene
unit, and background stays zero. Orthographic apparent size does not shrink with
distance. Shifted cameras are represented by `P03/P13`.

The CUDA implementation uses a `float64[H,W,3]` origin grid and `float64[3]`
direction for orthographic projection. Perspective retains the opposite layout.
Compile-time layout flags select addressing inside the same triangle intersection,
nearest-depth, semantic-ID and interpolation kernels. No scene ID is consulted.
The origin grid and mesh/camera resources stay on the GPU across calls; CPU
`float32[B,D,7]` poses are uploaded through the existing renderer interface.

Each orthographic hit must satisfy `near <= t <= far`; triangles may cross these
depth planes because orthographic projection has no perspective division
singularity. Perspective retains V1's whole-triangle depth-plane restriction.
Both use closed, two-sided triangles. Exactly equal canonical plane depth chooses
the lowest positive semantic object ID, then the original triangle index.
No silhouette epsilon, mask dilation or omitted boundary pixels are added.

## Versioned storage and independent reference

Orthographic recordings declare schema
`stable-envs/object-permanence-rigid-v2`, explicit camera
`projection_type: orthographic`, and `visibility_contract: stable-envs/opaque-pixel-rays-v2`.
Mesh arrays, `frames.npy`, `poses.npy` (`float32[T,D,7]`, xyz + wxyz), object IDs
and content-hashed asset sharing retain their existing formats. The V2 reader
accepts V1 recordings too; an older reader rejects the new schema instead of
silently constructing a perspective camera.

Pass `visibility_contract=CONTRACT_V2` explicitly to `RigidSceneRenderer`. The
old render/render_many signatures and defaults remain available. V2 also accepts
standard perspective cameras and produces the same pixels as V1 on those inputs.
This does not automatically recertify existing data under a different version.

New independent references live beside existing fixtures at
`tests/fixtures/semantics/object_permanence/<task>/opaque_pixel_rays_v2/nominal/`.
They retain the lossless oracle-v2 array encoding. The visibility contract version
is separate from that encoding version. The freezer evaluates the pinned Blender
source itself; it never reads candidate recordings or imports runtime hit code.
Its NumPy projected-edge coverage is independent of CUDA ray/triangle coverage.
Both evaluate the explicit camera contract.

Candidate and reference metadata bind source commit, script hash, preparation
recipe, camera, object identities, frame numbers and optional official surface
`DIVERSITY` context. V2 references also hash the reference preparation algorithm.
A builder that already creates animation is called once with an empty external
animation entry; source rendering/output functions are never used for extraction.

`source_semantic_targets` lists audited source task objects. `targets` lists the
subset seen at least once in the independent source ID images, for visibility
count probes. Never-visible targets remain in every frame's poses and in the
**all-object** depth/mask checks. This distinction avoids inventing visibility
for an object the source camera never sees; it does not exclude any object or pixel.

## Reproduce and verify

Use the pinned source and isolated Blender interpreter from the
[recording guide](object-permanence-recordings.md). Run export and freeze in
separate processes, without overwriting an existing directory:

```bash
python -m scripts.export.batch_object_permanence --source-root <source> \
  --output <recordings> --tasks G09,G11,G75 --report <export-report.json> \
  --visibility-contract stable-envs/opaque-pixel-rays-v2
python -m scripts.testing.freeze_object_permanence_rays --source-root <source> \
  --output <new-oracle> --task G09 \
  --visibility-contract stable-envs/opaque-pixel-rays-v2
python -m scripts.testing.check_object_permanence_rendering --recordings <recordings> \
  --oracle <new-oracle> --task G09 --report <pixel-report.json> \
  --visibility-contract stable-envs/opaque-pixel-rays-v2
```

The source runtime verifies evaluated world vertices/normals at every frame,
including hidden objects. CPU regressions verify all poses, provenance and lossless
reference reconstruction. GPU gates compare all 120 frames against source depth
and every object mask, both cached and full. At 90×160 and 180×320, RGB/depth/IDs
must be bit-identical with caching, shuffled/repeated requests, output mutation
and chunk sizes 1/3/4/7. Static seeds must build once across those requests.
Synthetic geometry additionally tests camera shifts/rotations, axial depth,
near/far crossing, exact ties and one-ULP origin-grid invalidation.

Source/replay GIFs show all 120 aligned frames. Native source previews retain
Cycles CUDA, 64 samples, denoising disabled; replay retains simplified shading.
Temporary PNGs are removed after decoding the finished GIF. GIFs remain local
under `.cache/object_permanence/evaluation/`; the machine-readable
[source audit](object-permanence-audit.json) records accepted and deferred entries.
