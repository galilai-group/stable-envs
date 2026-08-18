# Choosing a 3-D backend for stable-envs

**Verdict: adopt MJWarp's built-in batch raytracer (`mujoco_warp` 3.11).**
It is the only candidate that installs with `pip install`, runs on this cluster, scales to
thousands of worlds, needs no geometry glue, and — measured, not assumed — agrees with
MuJoCo's own renderer to **98.9% exactly-equal interior pixels** on flat-material scenes.

One caveat carried up front: that parity holds for flat-material and lightly-textured scenes
(OGBench, DM Control, classic control), and degrades as texture coverage rises. §2 has the numbers.

> **Correction (2026-08-10).** An earlier version of this document reported that MJWarp renders
> RoboCasa's textured surfaces as flat green (7.6% exact pixels) and told users not to ship
> RoboCasa-class scenes on it. That was a model-side problem, not a texture-path failure: **MJWarp
> never reads alpha**, and RoboCasa leaves 154 fully transparent marker geoms in the rendered geom
> group, which MJWarp paints opaque green over ~40% of the frame. Excluding them takes that scene
> from mean 79.3/255 to **interior mean 4.96 / 89.4% of interior pixels within 8**, i.e. into the
> "lightly-textured" band below. See `robocasa.md` §7 for the diagnosis and the one-line pre-pass.
>
> **Second correction (2026-08-10, same day).** Do not read the 89.4% as "close to parity". That
> figure excludes a 1-pixel band around silhouettes; scored the way `stable_envs.testing.conformance`
> defines *interior* — a pixel whose 3×3 neighbourhood in the reference is one flat colour — the same
> pipeline reaches **5.9–23.1% of interior pixels exact** across four RoboCasa kitchens, with a 5×
> spread between them (`robocasa.md` §11). The DM Control agent measured the same shape independently
> (19–31% exact on side-camera domains, 0.5% top-down) and shipped a plate + ray-cast renderer
> instead. **The rule that survives both: MJWarp is parity on flat-material scenes and is not parity
> on textured ones, and no flag combination closes it.**
>
> The truncate-vs-round correction does not close it either, and that is measured rather than
> assumed. A truncation-only residual is `reference - mjwarp` in `{0, +1}` on every channel and never
> negative; on four RoboCasa kitchens only **14.2-64.7%** of interior pixels have that signature, and
> the signed histograms are two-sided (16-57% of pixels 4+ levels below the reference, 6-34% 4+
> above) where Fetch's are `{0: 86.8%, +-1: 13.1%}`. The fix is worth what it says on flat materials
> and caps a RoboCasa kitchen at 0.65 interior-exact. `robocasa.md` Section 11.

> **Third correction (2026-08-10): the variable is MINIFICATION, not texture coverage, and there is a
> frame-loop bug for tracking cameras.** The rule above — "parity on flat-material scenes, not parity
> on textured ones" — is right about the outcome and wrong about the cause, and Gymnasium's six
> locomotion domains separate the two. Their scene is *lightly* textured by any coverage measure: one
> checker plane, one gradient skybox, primitives. With every correction applied (including
> `gl_rounding` and one material per geom carrying the resolved colour) MJWarp reaches **25-34% of all
> pixels and 0.22-0.56 of interior pixels** there; take the textures off both sides and the same
> pipeline reaches **0.86-1.00 interior exact**. The whole gap is one plane seen to the horizon:
> `create_warp_texture` builds every `wp.Texture2D` at `num_mip_levels=1` and the render kernel calls
> `wp.texture_sample` with no LOD, so where MuJoCo's mipmap has averaged the checker into one grey
> (luminance spread 1.5-3.4) MJWarp paints full contrast (spread 92), and **0.0017-0.0069% of those
> pixels are bit-exact**. Supersampling to 4×4 reaches 0.6-1.4% there and makes the whole frame worse.
> So antmaze's 78.2% and humanoidmaze's 38.3% are not "textured scenes are ~40-80%" — they are
> checker floors at two viewing angles, and the angle is the variable.
>
> Independently: **`kinematics` does not compute `subtree_com`**, so the four-call frame loop
> recommended in §3 leaves a `TRACKCOM`/`TARGETBODYCOM` camera or light frozen at its `put_data` pose
> — silently, for every state. `stable_envs.render.mjwarp.needs_com_pos` detects it and
> `BatchRenderer` inserts `com_pos` when the model needs it. `mujoco_locomotion.md` §6.
>
> **Setup cost (2026-08-10): `create_render_context` has a one-line performance bug, and it is most
> of the per-scene setup.** `render_util.create_warp_texture` builds one texture but does
> `wp.array(mjm.tex_data, dtype=wp.uint8)` *inside itself*, and `mjm.tex_data` is MuJoCo's single
> concatenated blob for every texture in the model — so a context uploads `ntex` copies of the whole
> blob to use one slice of each. On a RoboCasa kitchen (109 textures, 356 MB) that is 38.8 GB of
> host-to-device traffic. Hoisting it out: **context creation 8.98 s → 1.34 s, transient device
> memory 2,338 MB → 168 MB, frames bit-identical (max abs diff 0)**. Skipping mesh BVHs for meshes no
> enabled geom can reach takes it to 1.07 s. `stable_envs.render.mjwarp.create_render_context` wraps
> the fix for every caller here; it is worth reporting upstream.
>
> Two neighbouring terms behave the same way and are worth knowing before amortizing anything:
> `put_model` is 3.70 s on a full RoboCasa kitchen, **0.048 s** on the same kitchen stripped to the
> geometry it draws, and 0.012 s on a two-geom model — so it is dominated by the *collision* half of
> the model, which the renderer never touches. `make_data` replaces `put_data` at 0.98 s against
> 5.68 s. Full decomposition in `robocasa.md` §12.

All numbers below were measured on this cluster: one NVIDIA H200 (sm_90, 143 GB, driver
590.48.01), CUDA 12.9/13.0 toolkits available, via `srun` on `open_research`. Scripts and raw
JSON live in `scratch/backend3d/`.

---

## 1. Headline numbers

Frames per second at **224×224**, one camera, batched, textures on, shadows off. Higher is better.

| Backend | antmaze-medium | humanoidmaze-medium | cube-single (28 meshes) | Max batch | Installs? |
|---|---:|---:|---:|---:|---|
| **MJWarp raytracer** | **53,549** | **60,132** | **20,343** | 4096+ | `pip install mujoco-warp` |
| nvdiffrast 0.4.0 + own glue | 52,481 | 51,457 | 11,488 | 1024 (OOM at 4096) | builds, needs CUDA headers fixup |
| Warp `OpenGLRenderer` tiled | 8,402 | — | — | **324** (hard cap) | `pip install pyglet` |
| MuJoCo EGL (baseline) | 185 | 178 | 203 | 1 per context | already there |
| madrona_mjx | — | — | — | — | **builds, does not run** |

Same table at **64×64**, B=1024 (MJWarp): antmaze **488,587** fps, humanoidmaze **619,484** fps,
cube-single **163,533** fps.

**MJWarp is 290× MuJoCo's EGL renderer** at 224px on antmaze (337× on humanoidmaze, 100× on the
mesh-heavy cube-single), and unlike EGL it saturates rather than degrading with batch size
(53.0k fps at B=256 → 53.5k at B=4096).

### The EGL baseline gets *worse* with threads

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
| fps (antmaze, 224px) | 184.9 | 160.2 | 98.9 | 96.4 | 81.9 |

Multiple GL contexts do not help; they contend. There is no "just parallelise MuJoCo's renderer"
option, which is the whole reason this repo exists.

---

## 2. Fidelity: does it match MuJoCo?

This matters more than throughput. A backend that renders a *different-looking* scene at a million
fps is not a substitute for a stored dataset.

Measured as agreement against `mujoco.Renderer` (EGL) on the identical model, camera, and qpos, with
geom groups and render flags matched explicitly on both sides. Interior pixels only
(`edge_analysis.py`, shadows + skybox on):

| Scene | interior px exactly equal | interior MAE (/255) | interior within 8 |
|---|---:|---:|---:|
| antmaze-medium, **textures removed** | **98.9%** | **0.30** | 99.8% |
| humanoidmaze-medium, **textures removed** | **92.5%** | **0.08** | 100% |
| antmaze-medium, textured | 78.2% | 1.75 | 86.1% |
| cube-single, textured | 20.8% | 5.93 | 46.6% |
| humanoidmaze-medium, textured | 38.3% | 12.09 | 53.8% |

The last two rows are pessimistic: they use one fixed config, whereas the 32-config sweep
(`compare_agreement.py`, `results/agreement_224.json`) finds materially better settings for both —
cube-single reaches MAE 2.80 / 90.7% within 8 with `shadows=False, specular=False`, and
humanoidmaze reaches MAE 4.64 / 76.0% within 8 with `specular=False`. Both scenes want specular
highlights off; the maze scenes want shadows on. There is no single config that is best everywhere,
which is itself worth knowing.

"Interior" excludes a 1-pixel band around silhouettes. That exclusion is the honest one to make:
MuJoCo's GL path multisamples, MJWarp casts one ray per pixel, so the two *must* disagree on
partially-covered edge pixels, and that disagreement disappears under any downsampling. Edge pixels
are 7.6–32.1% of the frame and carry most of the residual error (edge MAE 10–17 vs interior 0.1–1.8
on the flat and antmaze cases).

**Conclusion, split by scene type:**

- **Flat-material scenes (OGBench mazes, most classic control): effectively a match.** 98.9% / 92.5%
  of interior pixels are *bit-identical*, mean error ≤0.3/255. This is parity, not "close enough".
- **Lightly-textured scenes (OGBench checker floors, cube-single): close but not identical.**
  Texture sampling and filtering differ (MuJoCo does mipmapped GL texturing). Perceptually
  near-identical — see `images/antmaze_medium_STRIP.png`, where the MuJoCo and MJWarp panels are
  hard to tell apart — but not pixel-exact, and the gap is scene-dependent (antmaze interior MAE
  1.75, humanoidmaze 4.64 at its best config).
- **Heavily-textured asset sets (RoboCasa kitchens): closer than recorded, and still not parity.**
  The 60.8/255 · 7.6%-exact figure was MJWarp drawing zero-alpha marker geoms opaque (see the
  correction at the top). With those geoms excluded the same kitchen scores **interior mean 4.96,
  89.4% within 8** at 128px on the silhouette-band definition — comparable to `cube-single` and
  `humanoidmaze` textured — but only **5.9-23.1% of interior pixels exact** on this repo's own
  flat-neighbourhood definition, which is the one the conformance suite applies. Two model-side caveats
  remain: MJWarp derives texture UVs for `PLANE` and `MESH` only, so a textured `BOX`/`CYLINDER`/
  `SPHERE` samples one corner texel and comes out flat (RoboCasa mostly gets away with it because its
  box materials use near-uniform textures); and transparency is not rendered at all. Full numbers in
  `robocasa.md` §7.

### ⚠ The BGRA trap — real, worth 4–6× in error, and *not* the RoboCasa story

MJWarp packs pixels as `(a<<24)|(r<<16)|(g<<8)|b` into a `uint32`
(`mujoco_warp/_src/render_util.py:pack_rgba_to_uint32`). On little-endian hardware a **byte view of
`rc.rgb_data` is B, G, R, A — not R, G, B, A.**

Reading it without the swap leaves geometry, camera, pose, and shadows pixel-aligned while making
every colour wrong. It looks exactly like "the renderer is not applying the model's materials".
Measured cost of getting this wrong:

| Scene | MAE with swap | MAE without | exact px with swap | without |
|---|---:|---:|---:|---:|
| cube-single | 2.80 | 14.14 | 53.4% | **1.3%** |
| humanoidmaze-medium | 4.64 | 27.01 | 31.8% | 30.1% |
| antmaze-medium | 3.14 | 11.96 | 67.7% | 66.4% |

4–6× in mean error, and up to 40× in exact-pixel agreement.

**It is invisible on grey scenes.** On achromatic pixels `r == g == b`, so a swap changes nothing —
the two readings score *identically* on our colour-stripped variants. Any fidelity test run on a
grey or untextured scene will pass while the bug is present.

Use `mjw.get_rgb()` (which unpacks correctly) or swap explicitly: `bgra[..., [2, 1, 0]]`.

**But it does not explain the RoboCasa result.** `scratch/robocasa/probe_mjwarp_calib.py` does read
the buffer as BGR (line 103), so the hypothesis was worth testing. It was tested:
`robocasa_channel_check.py` re-runs that probe's own pinned scene, qpos, camera and group-1
convention and scores both readings.

| RoboCasa kitchen (1251 geoms, 76 textures, 128px) | mean err | exact | within 8 |
|---|---:|---:|---:|
| as the probe reads it (BGR) | 62.14 | 7.57% | 29.0% |
| channel-corrected (RGB) | 60.76 | 7.60% | 42.0% |

Correcting the channels moves the mean error by 2%, so the channel bug is not what was wrong with
RoboCasa. Re-scored independently through `mjw.get_rgb` on the same pinned scene the numbers
reproduce (81.67 → 79.34 mean, 7.57% → 7.60% exact).

What *was* wrong is that green is the colour MJWarp paints RoboCasa's 154 invisible
`rgba="0 1 0 0"` marker geoms, because it never reads alpha — and green is also the one channel a
red/blue swap leaves alone, which is why the swap looked irrelevant. Hiding those geoms takes the
same scene to mean 7.47 / 82.8% within 8 (`robocasa.md` §7).

So the two bugs are independent and both real: **the channel bug is expensive on OGBench-class
scenes; the alpha bug is what made RoboCasa look like a texture-path failure.** Fixing one does not
fix the other, and neither is a texture-path defect.

### Two more measurement traps, confirmed

- **Geom groups.** `mujoco.Renderer` shows groups 0–2 by default. A model whose collision geoms sit
  in a visible group will have them painted over the visual mesh on one side only. Set
  `MjvOption.geomgroup` and MJWarp's `enabled_geom_groups` to the same explicit list.
- **Render flags.** `mjRND_SHADOW` / `mjRND_REFLECTION` / `mjRND_SKYBOX` live on **`MjvScene.flags`**
  (`mjtRndFlag`), not on `MjvOption.flags` (`mjtVisFlag`), and must be set *after* `update_scene`.
  Setting them on the wrong object silently does nothing and yields a perfect 0.0 difference between
  two configs, which reads as "this feature is unused".

Enabling shadows *improves* agreement (antmaze PSNR 27.8 → 30.4 dB): MuJoCo renders shadows by
default, so MJWarp's `use_shadows=False` default is a mismatch, not a saving.

---

## 3. Integration surface

Exactly the shape stable-envs wants — model + a batch of qpos in, `(B, H, W, 3)` uint8 on device out,
no host round-trip. Reference implementation: `scratch/backend3d/mjwarp_surface.py`.

```python
rc = mjw.create_render_context(mjm, nworld=B, cam_res=(224, 224), render_rgb=True,
                               cam_active=[...], use_textures=True,
                               use_shadows=True, render_skybox=True)
bgra = wp.to_torch(rc.rgb_data).view(torch.uint8).reshape(B, 224, 224, 4)  # zero copy

def render(qpos):                       # (B, nq) float32 on device
    wp.to_torch(d.qpos).copy_(qpos)
    wp.capture_launch(graph)            # kinematics [+ com_pos] + camlight + refit_bvh + render
    return bgra[..., [2, 1, 0]]         # (B, 224, 224, 3) uint8, contiguous, on device
```

Per-stage cost, B=1024 at 224×224, antmaze-medium:

| stage | ms | share |
|---|---:|---:|
| `kinematics` | 0.117 | 0.6% |
| `camlight` | 0.069 | 0.3% |
| `refit_bvh` | 0.042 | 0.2% |
| `render` | 27.86 | 99% |
| full frame, eager | 30.95 | — |
| full frame, **CUDA graph** | 21.37 | **1.45× faster** |
| qpos → BGRA view (zero copy) | 20.05 | 51,076 fps |
| qpos → RGB uint8 (contiguous) | 20.70 | **49,466 fps** |

Three things worth carrying into the design:

1. **Kinematics is free.** Producing world transforms from a state vector costs 0.6% of a frame. The
   "pure function of a compact state vector" premise holds with room to spare — you could re-derive
   state every frame and never notice.
2. **CUDA graph capture is worth 45%.** The whole pipeline captures cleanly; do it once at
   construction.
3. **The RGB unpack costs 3%.** Not worth training on BGRA to avoid.

Also supported by the same context: depth, segmentation (object ID/type per pixel), per-world camera
selection, heightfields, flex bodies, and multiple cameras per world.

---

## 4. Why not the others

### nvdiffrast 0.4.0 — fast, but you write the renderer

Installs and runs (see build note below). Throughput on primitive-only scenes is competitive:
52,481 fps vs MJWarp's 53,549 on antmaze. But:

- **It knows nothing about MuJoCo.** `scratch/backend3d/mjgeom.py` (≈240 lines) tessellates MuJoCo
  primitives, and `bench_nvdiffrast.py` does the transform, camera, interpolation, and shading. To
  *match* MuJoCo you would additionally have to reimplement its material, texture, skybox and
  lighting model on top of `dr.texture` — which is precisely the work MJWarp has already done and
  validated. Our version renders correct geometry with wrong colours
  (`images/antmaze_medium_STRIP.png`, rightmost panel).
- **It does not scale on mesh scenes.** Instanced mode needs the full `(B, V, 4)` clip-space vertex
  buffer, so memory grows with batch × scene vertex count. cube-single (394k verts) manages 11,488
  fps and **OOMs at B=4096**; MJWarp does 20,343 fps at B=4096 because it shares geometry across
  worlds and stores only per-geom transforms.
- Differentiable rendering is its actual selling point, and stable-envs does not need gradients
  through pixels.

Worth keeping in mind if a differentiable-render requirement ever appears.

### Warp's own `OpenGLRenderer` — structurally capped

Warp 1.16 ships one rasterizer: a pyglet/OpenGL scene-graph renderer. Headless mode *does* work on
this cluster, and `setup_tiled_rendering` + `get_pixels(..., use_uint8=True)` does land pixels in a
`wp.array` on device. But:

- **Batch is capped by framebuffer area.** Every batch element is a tile in one window: 324 tiles at
  224px in a 4096×4096 framebuffer, and that is the ceiling. 8,402 fps there.
- Instance transforms are updated through a **per-instance Python call**
  (`update_shape_instance(name, pos=...)`), so a real per-world update means ~17k Python calls per
  frame at 324 worlds × 53 geoms. Our number is a *static* scene, i.e. an upper bound.

Note MJWarp's raytracer is itself built on Warp, so choosing MJWarp is not choosing against Warp.

### madrona_mjx — deprecated upstream, and it does not run here

Upstream added a deprecation notice (commit `1505699`, 2026-02-17) telling users to migrate to the
MJX/MJWarp renderer. We built it anyway, to check.

**It builds.** Not easily, but it builds:

| obstacle | fix |
|---|---|
| requires CUDA ≤ 12.5.1; cluster has 12.9/13.0 | `conda create -p envs/cuda125 -c nvidia/label/cuda-12.5.1 cuda-toolkit` |
| `CUDA_NVJITLINK_LIBRARY`/`CUDA::nvrtc` NOTFOUND | the conda CUDA 13 env has no `libnvrtc.so`; use the real 12.5.1 toolkit |
| `no member named 'computeMode' in 'cudaDeviceProp'` | **`CFLAGS`/`CXXFLAGS` inherited from the activated conda env** injected CUDA 13 headers as `-I`, which outrank the correct toolkit's `-isystem`. `unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS` |
| `pip install -e .` fails | its build backend needs `tomli` and `pyproject_metadata` pre-installed |
| requires `jax<0.6.0` | `jax[cuda12]==0.5.3` in a clean venv |
| `mjx.Model has no attribute light_type` | needs mujoco-mjx ≥ 3.3.3; 3.3.7 works with jax 0.5.3 |

**It does not run.** With everything above resolved, on H200 / driver 590.48.01:

- raytracer path: initialises, compiles its GPU engine, then
  `MWCudaExecutor::run → an illegal memory access was encountered`. Reproduces at **B=8, 64×64**, so
  it is not memory pressure.
- rasterizer path: segfault (it needs Vulkan, which the README's own `-DLOAD_VULKAN=OFF` build
  disables).

Also structurally awkward regardless: one renderer per process (a second `create` dies in
`setCudaHeapSize`), and train/eval batch sizes must match. **Do not adopt.**

### PyBatchRender — surveyed, not adopted

2026 arXiv entry (2601.01288), Panda3D-based, MIT, reports 1.6M fps at 64×64 and claims ~4× over
Madrona MJX. Not adopted because it is **physics-agnostic with no MuJoCo/MJCF path** — it would need
the same asset-translation layer as nvdiffrast, plus a Panda3D scene graph — and its headline is
CartPole (3 geoms) at 64px. For calibration, MJWarp does 619k fps at 64px on humanoidmaze, a far
heavier scene, on our hardware. Worth revisiting only if MJWarp becomes a bottleneck.

### Also considered

- **Isaac Lab tiled rendering** — requires closed-source Isaac Sim; wrong dependency for this repo.
- **ManiSkill3 / SAPIEN** — genuinely fast (30k+ fps) but renders SAPIEN scenes, not MuJoCo models.

### Re-surveyed 2026-08-10 — the recommendation is unchanged

The whole field was re-checked against current releases, because three env families
([Adroit](adroit.md), [Franka Kitchen](kitchen.md), [Meta-World](metaworld.md)) had been rejected on
this backend's limits rather than on their own merits. **Nothing displaces `mujoco-warp`.** The
disqualifier is the same in every case and it is not speed: every alternative renders a scene
reconstructed from MJCF rather than MuJoCo's own scene, and most of them drop the camera and the
lights on the way in — so the picture differs before any shading law is compared.

| candidate | current | why not |
|---|---|---|
| **Isaac Lab / Isaac Sim** | Sim 6.0.1 (2026-06-22), Lab 3.0.0-beta2 | **Unusable on this cluster**: Isaac Sim requires RT cores and states "GPUs without RT Cores (A100, H100) are not supported" — H200 is the same architecture family. Its MJCF path delegates to `mujoco-usd-converter` (alpha), which converts **neither cameras nor lights** and supports "only standard 2D file textures", explicitly not checkerboards or cube maps. Default AA is **DLSS**, a temporally-accumulated neural denoiser. No published pixel comparison against `mujoco.Renderer` exists. |
| **Newton** (NVIDIA/DeepMind/Disney) | 1.4.0 GA (2026-07-16) | Its batched `SensorTiledCamera` is a Warp ray tracer with `enable_textures=False` and `enable_shadows=False` by default, **pure Lambertian** shading, no specular/emission/skybox/alpha, hardcoded attenuation and a shadow floor. Its MuJoCo solver docs state outright that "camera/light primitives in the source asset are ignored". NVIDIA positions it as "basic visualization for debugging purposes". Newton depends on `mujoco_warp`; it does not replace it. |
| **Genesis** | 1.3.2 (2026-08-07) | Its MJCF loader handles `mjTEXTURE_2D` only (no cube, no skybox), does not extract reflectance or emission, and does not implement sites. It remaps MuJoCo materials onto a PBR/glossiness model and renders through pyrender / LuisaRender / Nyx, none of which implement MuJoCo's fixed-function shading. A maintainer states there is no parity commitment of any kind. Batch rendering supports "only basic materials, lighting, and shadows". |
| **ManiSkill3 / SAPIEN 3** | MS 3.0.1 (2026-04-21), SAPIEN 3.0.3 | An MJCF loader **does** now exist (`mani_skill/utils/building/_mjcf_loader.py`) — the earlier "URDF-only" note is out of date — but it remaps materials onto SAPIEN PBR (`shininess → metallic`), does not handle sites or procedural textures, and documents its geom-group handling as showing groups 0 and 2 only, which is not MuJoCo's default. SAPIEN issue #250, "headless rendering on an H100", is **still open** since 2025-06-24. |
| **madrona-mjx** | deprecated | Its README now directs users to "the official MJX renderer via MJWarp… for better long-term support, performance, and feature parity". Pinned to CUDA ≤ 12.5.1 and jax < 0.6.0. |

Two facts from the survey that bear on how this repo states its results:

1. **`mujoco_warp`'s renderer is written against `render_gl3.c` on purpose**, which is why it is the
   only candidate with any parity evidence. PR #1364 (merged 2026-06-02, shipped in 3.11.0) reports
   canonical-fixture SSIM against `mujoco.Renderer` rising from ~0.3–0.5 to **0.917–0.994**. Those
   SSIM tests did **not** land in `main`: `render_test.py` asserts segmentation and backface culling
   **exactly** and depth to 1 %, and has no RGB regression test at all. This repo's per-env interior
   numbers are, as far as the survey found, the only ongoing RGB parity measurement anywhere.
2. **"MuJoCo's own renderer" is becoming ambiguous.** MuJoCo now ships a second, **Filament**-based
   PBR renderer selectable at build time, which is not pixel-identical to the classic OpenGL `mjr_*`
   path; and 3.11.0 added `light/softness` with the migration note "set softness to 1 to reproduce
   the previous appearance of existing models". Parity claims here should keep naming the exact
   `mujoco` version and the classic-OpenGL build, which they do.

A **batched Meta-World** was searched for specifically, since a GPU-native port with a seedable draw
would have closed that family's state gate directly. Exactly one exists — `AlgoForce-I/MTCWorldMJX`
(PyPI 0.2.0, 2026-08-07, single author, no test CI) — and its own documentation says parity with
Meta-World is "tolerance-based, not bit-exact" (`OBS_ATOL=5e-3`, `REWARD_ATOL=2e-2`). It is not a
reference. It did not matter in the end: [metaworld.md](metaworld.md) closes that gate against the
official env directly. **No MJX/JAX/Warp port of Adroit exists at all** — `mujoco_menagerie` does not
even carry an Adroit model.

---

## 5. Recommendation

Adopt **`mujoco-warp` 3.11's `RenderContext`** as the 3-D backend, mirroring the role the fused
Triton kernel plays in 2-D.

Concretely:

1. Wrap `create_render_context` + `kinematics/camlight/refit_bvh/render` behind a
   `render(qpos) -> (B, H, W, 3) uint8` surface, CUDA-graph captured at construction.
   `scratch/backend3d/mjwarp_surface.py` is a working reference.
2. **Swap channels** — the buffer is BGRA. Costs 3%. Getting this wrong is the single most expensive
   mistake available here, and it is silent on grey scenes.
3. Default to `use_shadows=True, render_skybox=True, use_textures=True` — that maximises agreement
   with MuJoCo (30.4 dB vs 27.8 dB with shadows off). Shadows cost ~1.7× throughput; keep the flag
   exposed.
4. Set `enabled_geom_groups` explicitly rather than relying on the default.
5. Budget ~50k fps at 224px and ~500k fps at 64px for primitive scenes; ~20k / ~160k for
   mesh-heavy manipulation scenes. Mesh complexity, not batch size, is the scaling variable.

Honest limits to document for users, in order of severity:

- **Alpha is ignored.** MJWarp's shader reads `color[0..2]` and packs a constant `255`, so any geom
  MuJoCo draws transparent is drawn opaque. Scenes that use invisible marker geometry — RoboCasa
  puts 154 of them in the rendered geom group — come out unrecognisable. Hide them before rendering:
  compute the effective alpha (material's if `geom_matid >= 0`, else `geom_rgba`) and move
  `alpha <= 0` geoms into a group `enabled_geom_groups` excludes. Partially transparent geoms have
  no correct handling at all.
- **Texture UVs exist for `PLANE` and `MESH` only.** A textured `BOX`, `CYLINDER`, `SPHERE`, `CAPSULE`
  or `ELLIPSOID` samples texel (0, 0) and renders one flat colour (checkered box: MuJoCo 167 distinct
  colours, MJWarp 2; box-pixel mean error 50.3). Still true on `mujoco_warp` `main` at 3.11.0
  (2026-07-28). Re-expressing such geoms as meshes with texcoords fixes it (box-pixel mean 50.3 →
  6.43), but matching MuJoCo's own box texgen exactly is unsolved here.
- Lightly-textured scenes match closely but not exactly; best flags are scene-dependent (mazes want
  shadows on, mesh scenes want specular off, RoboCasa's wrist camera wants specular off).
- Full Phong: specular, shininess, emission, headlight and per-light attenuation/spot/ambient
  landed in mujoco_warp PR #1364 (merged 2026-06-02, shipped in 3.11.0). An earlier version of this
  line said "Lambertian diffuse only", which is stale. Still no global illumination and no
  physically-based materials.
- Silhouettes differ by ~1px because MuJoCo multisamples and MJWarp does not.

For flat-material scenes, which is most of what stable-envs targets in 3-D, it is pixel-parity.

---

## Reproducing

```
venv:    /mnt/vast/home/randall.balestriero/venvs/se3d          (mujoco 3.11, mujoco-warp 3.11, warp 1.16, torch 2.11+cu130, nvdiffrast 0.4.0)
venv:    /mnt/vast/home/randall.balestriero/venvs/madrona       (jax 0.5.3, mujoco-mjx 3.3.7, madrona_mjx)
toolkit: /mnt/vast/home/randall.balestriero/envs/cuda125        (CUDA 12.5.1, for madrona only)
scenes:  scratch/backend3d/scenes/*.xml                          (dumped from ogbench via make_scenes.py)
```

| script | what it does |
|---|---|
| `make_scenes.py` | dumps OGBench envs to standalone MJCF + assets |
| `bench_mjwarp.py` | MJWarp throughput sweep |
| `bench_nvdiffrast.py`, `mjgeom.py` | nvdiffrast path incl. MuJoCo→triangle-soup glue |
| `bench_warp_gl.py` | Warp OpenGL tiled renderer |
| `bench_egl.py` | MuJoCo EGL baseline, 1–16 threads |
| `bench_madrona.py` | madrona_mjx (fails; kept for the error trail) |
| `compare_agreement.py` | 32-config agreement sweep vs MuJoCo, both channel readings |
| `edge_analysis.py` | splits disagreement into interior vs silhouette |
| `mjwarp_surface.py` | the proposed integration surface + per-stage cost |
| `robocasa_channel_check.py` | re-scores the RoboCasa probe's scene under both channel readings |

nvdiffrast build needs the CUDA headers on `CPATH` (torch's `cusparse.h` include is not satisfied by
the conda toolkit alone):

```bash
INC=$VENV/lib/python3.11/site-packages/nvidia/cu13/include
CUDA_HOME=<conda env with nvcc> CPATH=$INC TORCH_CUDA_ARCH_LIST=9.0 \
  pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
```
