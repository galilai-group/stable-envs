# Backend and device limitations

Known limits of this release.

## Optional dependencies

`pip install stable-envs` installs the core. Each family needs its extra; see the
[installation page](../install.md).

```python
snv.import_errors()     # families this interpreter could not import
```

## Devices

| classes | CPU | GPU |
|---|---|---|
| PushT, PushTDiscrete | source physics and pygame rendering | output storage only; rendering stays on CPU |
| classic control, navigation, OGBench mazes | yes | yes |
| Fetch (12 classes) | software OpenGL (`backend="opengl"`) | MJWarp or explicit NVIDIA OpenGL |
| OGBCube, OGBScene | software OpenGL | MJWarp or explicit NVIDIA OpenGL |
| PyFlytRocketLanding | TinyRenderer | CUDA rendering |
| DM Control body-tracking (visual tier) | CPU3D | MJWarp |

`device=` is a constructor argument. A class that cannot run on a device refuses at construction.

On CPU, Fetch with `backend="auto"` does not fall back to OpenGL. Pass `backend="opengl"`.

For Cube and Scene, physics runs on CPU whatever the render backend. CPU rendering needs Mesa software
OpenGL, configured before MuJoCo is imported. See [System setup](../system-setup.md).

## Fidelity

- **Parity tier**: interior pixels match the official renderer. The floor is 1.0 unless the class
  declares a `parity_note` that names the cause.
- **Visual tier**: a different rasterizer of the same scene. Use it for training frames. It is not
  interchangeable with official pixels at evaluation. `info(name)["fidelity_tier"]` reports the tier.

See [Fidelity tiers](fidelity-tiers.md).

## Rendering details

- Fetch `sky.color` at its default draws the registered gradient skybox. The official wrapper's
  `background.color` cannot express it, because it fills the sky with one colour.
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
