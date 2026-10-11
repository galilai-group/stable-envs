# Backend and device limitations

Accepted facts about this release. They are not hidden, and they are not treated as regressions
against an unpublished predecessor.

## Optional dependencies

`pip install stable-envs` gives the core. Each family extra installs what that family's
**reference** needs (parity, CPU-vs-GPU benches, gallery). Most extras are not required to
*render* — the exceptions are called out on the [installation page](../install.md).

```python
snv.import_errors()     # families this interpreter could not import
```

Renderer-only ports of other families live in the repository's `contrib/` directory and are not
part of this release; see `contrib/README.md`.

## Device honesty

| cells | CPU | GPU |
|---|---|---|
| PushT / PushTDiscrete | source physics and Pygame rendering | tensor/output storage only; no GPU renderer on this public path |
| classic control, navigation, OGBench mazes, most 2-D | yes | yes |
| Fetch (12 ids) | software OpenGL (`backend="opengl"`) | MJWarp or explicit NVIDIA OpenGL |
| OGBCube, OGBScene | software OpenGL | MJWarp or explicit NVIDIA OpenGL |
| PyFlytRocketLanding | TinyRenderer | public CUDA rendering path |
| DM Control trackcom / visual-tier body-tracking | CPU3D path where declared | MJWarp |
| nvdiff locomotion (visual tier) | not a supported consumer of the x86 sparse-stage path | nvdiffrast |

`device=` is part of the constructor. A cell that cannot run on CPU says so at construction, not halfway through a sweep.

## Fidelity

- **Parity tier:** interior pixels match the official renderer (floor 1.0 unless a named `parity_note` mechanism is declared).
- **Visual tier:** a different rasterizer of the same scene. Suitable for live training frames; **not** interchangeable with official pixels at eval. `info(name)["fidelity_tier"]` reports it.

Fetch `sky.color` at its default draws the registered gradient skybox, which the official wrapper's `background.color` cannot express: it always fills the sky flat.

DMC trackcom `floor.color` has no baked plate (the camera rides the body).

OGBench `light.intensity` / `camera.angle_delta` / `cube.size` and Scene `lock_color` are not wired onto per-world model arrays.

## Certification / provenance

- DM Control certification interpreter is `mujoco==3.11.0` + `dm_control==1.0.44`. `mujoco >= 3.12` re-anchors finite-plane textures; the 3.12 arm is recorded as reference drift, not as our renderer moving.
- nvdiff on x86 CPU has a documented sparse-stage one-ULP batch dependence. There is no CPU consumer of that path in the historical 42-id set.
- `HumanoidMazeGiantNavigate` on CUDA is not process-reproducible across runs on at least one measured card (`gpu1503`).
- Craftax's live official wrapper JITs `reset`/`step`. Certified pixel identity is against the **un-jitted** official renderer. Historical jitted-wrapper measurements live in `tests.helpers.reference.craftax_jit`.

## Sibling libraries

There is no runtime interop contract with other stable-* packages. See [Sibling projects](worldmodel.md).
