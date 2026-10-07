# Backend and device limitations

Accepted facts about this release. They are not hidden, and they are not treated as regressions
against an unpublished predecessor.

## Optional dependencies

`pip install stable-envs` gives the core. Each family extra installs what that family's
**reference** needs (parity, CPU-vs-GPU benches, gallery). Most extras are not required to
*render* — the exceptions are called out on the [installation page](../install.md)
(nvdiffrast for visual-tier MuJoCo locomotion; jaxmarl for Overcooked).

```python
snv.import_errors()     # families this interpreter could not import
```

Overcooked is the one extra that needs its **own interpreter** (`jaxmarl` pins). HumanoidBench
registers only when `STABLE_ENVS_HUMANOIDBENCH=1` is set: `mujoco_warp`'s batched render is not
bit-reproducible across a long conformance session, which violates "the same state gives the
same frame".

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

Fetch `background.color` is **partial**: the floor half is mapped; the skybox half is not. The official wrapper flat-fills a 16-colour gradient. Supporting the skybox would move the family's frozen default frame.

Fetch `rendering.transparent_arm` has no alpha path in `mujoco_warp`.

DMC trackcom `floor.color` has no baked plate (the camera rides the body).

OGBench `light.intensity` / `camera.angle_delta` / `cube.size` and Scene `lock_color` are not wired onto per-world model arrays.

## Certification / provenance

- DM Control certification interpreter is `mujoco==3.11.0` + `dm_control==1.0.44`. `mujoco >= 3.12` re-anchors finite-plane textures; the 3.12 arm is recorded as reference drift, not as our renderer moving.
- nvdiff on x86 CPU has a documented sparse-stage one-ULP batch dependence. There is no CPU consumer of that path in the historical 42-id set.
- `HumanoidMazeGiantNavigate` on CUDA is not process-reproducible across runs on at least one measured card (`gpu1503`).
- Craftax's live official wrapper JITs `reset`/`step`. Certified pixel identity is against the **un-jitted** official renderer. Historical jitted-wrapper measurements live in `tests.helpers.reference.craftax_jit`.

The measurement record is in the [research archive](../research/index.md).

## Sibling libraries

There is no runtime interop contract with other stable-* packages. See [Sibling projects](worldmodel.md).
