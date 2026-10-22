# Limitations

This page lists the known limits of this release.

## Optional dependencies

`pip install stable-envs` installs the core. Each family needs its extra. See the
[installation page](../install.md).

```python
snv.import_errors()     # families this interpreter could not import
```

## Devices

| classes | CPU | GPU |
|---|---|---|
| PushT, PushTDiscrete | source physics and pygame rendering | output storage only (rendering stays on CPU) |
| classic control, navigation, OGBench mazes | yes | yes |
| Fetch (12 classes) | software OpenGL | MJWarp or explicit NVIDIA OpenGL |
| OGBCube, OGBScene | software OpenGL | MJWarp or explicit NVIDIA OpenGL |
| PyFlytRocketLanding | TinyRenderer | CUDA rendering |
| DM Control body-tracking | CPU3D | MJWarp |

`device=` is a constructor argument. A class that cannot run on a device refuses at construction.

For Cube and Scene, physics runs on CPU whatever the render backend. CPU rendering needs Mesa software
OpenGL, configured before the first MuJoCo import. See [System setup](../system-setup.md).

## Agreement with the source

- The tests measure the per-frame MSE between our frames and the source frames, and compare it with a
  recorded value for each class. See [Frame regression checks](../testing.md#frame-regression-checks).
- The DM Control body-tracking classes, and OGBCube, OGBScene and PyFlytRocketLanding on CUDA, draw
  with a different rasterizer from the source. `env.describe()["visual_fidelity"]` gives their
  measured MSE. `env.describe()["visual_note"]` tells where the frames differ.

## Rendering details

- Fetch `sky.color` at its default draws the registered gradient skybox. The official wrapper's
  `background.color` cannot express it, because `background.color` fills the sky with one colour.
- Fetch does not draw the translucent mocap markers (the 1 cm cube and the three crosshair bars).
  MJWarp has no alpha blending and would draw them opaque.
- DM Control body-tracking cells draw `floor.color` from a per-texel palette texture.
  `DMCWalkerWalk` `floor.rotation_y` tilts the floor per row.
- OGBench `light.intensity`, `camera.angle_delta`, `cube.size` and Scene `lock_color` are per-world
  inputs of one MJWarp context per family. They do not add contexts or compiled models.

## Versions and reproducibility

- The DM Control reference profile (`pin-dmc`) is `mujoco==3.12.0` and `dm-control==1.0.45`. The
  `VISUAL_FIDELITY` declarations of the body-tracking cells record the versions they were measured with.
- nvdiffrast on x86 CPU has a one-ULP batch dependence in a sparse stage. No registered class uses that
  path on CPU.
- `HumanoidMazeGiantNavigate` on CUDA is not reproducible across processes on at least one GPU.

## Sibling libraries

There is no runtime interop with other stable-* packages. See [Sibling projects](worldmodel.md).
