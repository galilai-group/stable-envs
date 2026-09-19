# Backend and device limitations

These are accepted Step-8 facts. Step 9 does not re-freeze them, hide them, or treat them as regressions.

## Optional dependencies

`pip install -e .` gives the core. Each family extra installs what that family's **reference** needs (parity, CPU-vs-GPU benches, gallery). Most extras are not required to *render* — the exception is called out on the extra in `pyproject.toml` (nvdiffrast for visual-tier MuJoCo locomotion; jaxmarl for Overcooked).

```python
snv.import_errors()     # families this interpreter could not import
```

Overcooked is the one extra that needs its **own interpreter** (`jaxmarl` pins). HumanoidBench registers only when `STABLE_ENVS_HUMANOIDBENCH=1` is set: `mujoco_warp`'s batched render is not bit-reproducible across a long conformance session, which violates "the same state gives the same frame".

## Device honesty

| cells | CPU | GPU |
|---|---|---|
| PushT, classic control, navigation, OGBench mazes, most 2-D | yes | yes |
| Fetch (12 ids) | **no backend** | yes |
| OGBCube, OGBScene, PFRocketLanding | **refuse** (`device="cpu"` raises; no silent CUDA fallback) | required |
| DM Control trackcom / visual-tier body-tracking | CPU3D path where declared | MJWarp |
| nvdiff locomotion (visual tier) | not a supported consumer of the x86 sparse-stage path | nvdiffrast |

`device=` is part of the constructor. A cell that cannot run on CPU says so at construction, not halfway through a sweep.

## Fidelity

- **Parity tier:** interior pixels match the official renderer (floor 1.0 unless a named `parity_note` mechanism is declared).
- **Visual tier:** a different rasterizer of the same scene. Suitable for live training frames; **not** interchangeable with official pixels at eval. `spec(name)["fidelity_tier"]` reports it.

Fetch `background.color` is **partial**: the floor half is mapped, the skybox residual is measured (swm flat-fills a 16-colour gradient). Supporting the skybox would move the family's frozen default frame; that is a product decision, not a silent gap.

Fetch `rendering.transparent_arm` has no alpha path in `mujoco_warp`.

DMC trackcom `floor.color` has no baked plate (the camera rides the body).

OGBench `light.intensity` / `camera.angle_delta` / `cube.size` and Scene `lock_color` wiring onto per-world model arrays was not done in Step 8.

## Certification / provenance

- DM Control certification interpreter is `mujoco==3.11.0` + `dm_control==1.0.44`. `mujoco >= 3.12` re-anchors finite-plane textures; the 3.12 arm is recorded as reference drift, not as our renderer moving.
- nvdiff on x86 CPU has a documented sparse-stage one-ULP batch dependence. There is no CPU consumer of that path in the 42-id set.
- `HumanoidmazeGiantNavigate` on CUDA is not process-reproducible across runs on at least one measured card (`gpu1503`); classified `DOCUMENTED_LIMITATION`, not a Step-9 regression.
- Craftax's live SWM wrapper JITs `reset`/`step`. Certified pixel identity is against the **un-jitted** official renderer. The class attribute `SWM_WRAPPER_GATE_F` declares that.

The full lists: [Step 8 closure](../research/step8_closure.md) §12, [support matrix](../research/step8_support_matrix.md), [upstream findings](../research/step8_upstream_findings.md).

## Interop

Native `swm.World(env=snv.PushT(...))` is **not** a public API: installed `stable-worldmodel` takes `env_name: str` and always `gym.make`s it. Use `stable_envs.interop` as documented in [worldmodel](worldmodel.md).
