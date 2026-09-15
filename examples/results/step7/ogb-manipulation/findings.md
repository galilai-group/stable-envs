# Step 7 / W5 `ogb-manipulation` — phase 0 findings

**Classification: `FINDING`.** Diagnosis only; no renderer written, no shared file touched.
Two sub-verdicts, both measured:

* **GPU: feasible on the `fetch.py` pattern, and the Fetch blocker probably does not transfer.**
  Every precondition the Fetch cell needed is satisfied here and two of them are satisfied MORE
  cleanly (a model camera instead of a recovered free camera; zero texture gaps). The one Step-6
  `fetch-gpu` blocker — `BATCH_SCALES` 7.6–11.7x against a required 20x — was recorded as a
  **480 px** failure that the same scene PASSES at ≤240 px (26–99x). These two envs render at
  **224 px and 200 px**, inside that passing regime. The measurement is queued (job 6394992).
* **CPU: `BLOCKER`, and roughly an order of magnitude harder than the Fetch blocker.** The binding
  CPU bar leaves **0.75 ns per triangle** for 193,113 triangles. The Step-6 `CPU3D-FETCH-1` span
  rasteriser's *setup-only* floor — transform, bbox, area, cull, no fill, no shade — is 24 ns per
  triangle, i.e. **32–35x over the binding bar** and **3.7x over the weaker `S_cpu ≥ 5` conjunct on
  its own**. At the full measured 135 ns/triangle it is 180–198x over, and its `S_cpu` against this
  env's own official renderer is **0.24** — four times SLOWER than the reference, so not
  `CPU_FAST`, not `CPU_FASTER`, not `CPU_PARITY`. Recorded as **`CPU3D-OGB-1`**.

Baseline `76dc5116f432e61dd47fba8f9e879cf9212c8e46`; branch `step7/ogb-manipulation`.
Hardware/venvs: Apple M5 (`arm64`, macOS 26.6.2), new local venv `/Users/samib/se7-ogb-venv`
(python 3.12.11, ogbench 1.2.1, mujoco 3.12.0, torch 2.14.0, numpy 2.5.3, gymnasium 1.3.0,
dm_control 1.0.45, scipy 1.18.1, pillow 12.3.0, plus the worktree installed `-e`; full freeze in
`examples/results/step7/ogb-manipulation/venv_se7-ogb-venv_m5.txt`), with
`stable_worldmodel` taken from the source checkout `/Users/samib/stable-worldmodel` via
`PYTHONPATH`. Local numbers are NOT certification numbers; the Oscar legs are queued (below).

**`MUJOCO_GL=glfw` works on macOS for the official renderer.** Step 1 recorded these two envs as
"needs EGL" because they raise `X11: The DISPLAY environment variable is missing` on a headless
Linux node. On macOS GLFW opens a CGL context against the Metal-backed GL and both envs render;
every local number in this file was taken that way. The headless-EGL leg is still required and is
queued, because the bars have to come from the certification host.

---

## 1. Exact registration

`stable_worldmodel/envs/__init__.py` registers both ids **with no kwargs at all**, pointing at
swm's own forked subclasses of `ogbench.manipspace.envs.manipspace_env.ManipSpaceEnv` — **not** at
an ogbench gym id. The upstream ogbench counterparts exist (`cube-single-v0`, `scene-v0`, and the
`visual-*` variants at 64 px) and are **not** what swm instantiates.

| | `swm/OGBCube-v0` | `swm/OGBScene-v0` |
|---|---|---|
| entry point | `stable_worldmodel.envs.ogbench.cube_env:CubeEnv` | `stable_worldmodel.envs.ogbench.scene_env:SceneEnv` |
| upstream counterpart | `cube-single-v0` (`env_type='single'`) | `scene-v0` (`env_type='scene'`) |
| registered default resolution | **224 x 224** (`CubeEnv.__init__` sets `height=width=224`) | **200 x 200** — `SceneEnv` does NOT override, so it inherits `ogbench...envs.env`'s `width=height=200` |
| `ob_type` | `'states'` | `'states'` |
| observation | `Box(-inf, inf, (28,), float64)` | `Box(-inf, inf, (40,), float64)` |
| default render camera | `'front_pixels'` (both swm subclasses override `render(camera='front_pixels')`; the ogbench base would have used `'front'` for `ob_type='states'`) | `'front_pixels'` |
| render backend | `mujoco.Renderer(model, height, width)` — whatever `MUJOCO_GL` selects; `visual.global_.offwidth/offheight` are set to the render size at compile | same |
| `mode` / `visualize_info` / `pixel_transparent_arm` | `'task'` / `True` / `True` | `'task'` / `True` / `True` |
| `nq, nv, nmocap` | 21, 20, 1 | 25, 24, 1 |
| state vector (swm's own `reset(options={'state': ...})` contract) | `concat(qpos, qvel)`, **41 floats** | `concat(qpos, qvel)`, **49 floats** |

Two registration facts that matter downstream:

* **The observation is not the state.** The 28-D / 40-D observation is a *scaled, re-encoded*
  vector (joint pos/vel, effector position centred on `(0.425, 0, 0)` and x10, effector yaw as
  `(cos, sin)`, gripper opening x3, gripper contact, then per-cube position/quat/yaw). It is not
  invertible to `qpos`, and it is **not** the renderer's input. The renderer's input is `qpos` plus
  the extras in §3.
* **The goal image is in neither the observation nor the info dict.** `_render_goal` is `False` at
  the registered default, and swm's `get_reset_info` / `get_step_info` overrides drop ogbench's
  `goal_rendered` key entirely and rename `goal` to **`target`** — the 28-D/40-D goal *observation*,
  not an image. Measured: the reset-info keys are exactly `{qpos, qvel, target, success, time,
  control, env_name, prev_qpos, prev_qvel, proprio/*, privileged/*}` (+ `button_states`,
  `prev_button_states` and the drawer/window/button privileged keys for Scene). The goal reaches the
  *frame* only through the translucent `target_object_0` marker and its mocap pose.
* **The info dict already exposes almost all of the render-relevant state.** `qpos`, `qvel` and (for
  Scene) `button_states` are all there. The one thing it does NOT expose is `mocap_pos` /
  `mocap_quat`, which position the goal marker and are the remaining 7 floats of §3's state vector.
  A cell would have to read them off `data` rather than off `info`.
* **`pixel_transparent_arm` is inert at the registered default.** It only fires under
  `ob_type == 'pixels'`, where ogbench recolours the Robotiq materials purple and drops the arm's
  materials to alpha 0.1. At `ob_type='states'` the arm is opaque and wears its own UR5e materials.
  A later decision to register `ob_type='pixels'` would change the picture substantially and would
  add eight partially-transparent materials to §5's alpha problem.
* **`SceneEnv.reset(options={'state': ...})` is broken upstream.** It calls
  `self.set_state(qpos, qvel)` with no button kwargs, and `SceneEnv.set_state` requires
  `button_state_0`/`button_state_1`; the documented reset-from-state path raises `KeyError: 0`.
  Measured, not inferred (`probe/purity_fps_OGBScene.json`, `G1f_reset_options_state_error`).

---

## 2. Render census

Counts are from `examples/results/step7/ogb-manipulation/census/census_*.json`. "Visible" means
`geom_group ∈ {0,1,2}` (the env's own `_scene_option.geomgroup = [1,1,1,0,0,0]`) and effective
alpha > 0 — groups 3 (collision) and, for Scene, 4 (workspace bounding-box sites) are not drawn.

| | `OGBCube` | `OGBScene` |
|---|---:|---:|
| `nbody / ngeom / nmesh / nmat / ntex / nlight / ncam / nsite` | 26 / 64 / 28 / 10 / 2 / 2 / 4 / 5 | 38 / 137 / 48 / 23 / 3 / 2 / 3 / 11 |
| geoms by type (all) | MESH 44, BOX 10, CAPSULE 8, CYLINDER 1, PLANE 1 | MESH 63, BOX 45, CAPSULE 17, CYLINDER 11, PLANE 1 |
| **visible** mesh geoms / primitive geoms | **34** / **3** (1 PLANE floor, 2 BOX: the cube and its target) | **53** / **8** (1 PLANE, 2 BOX, 2 CYLINDER, 3 CAPSULE — the window handle) |
| **visible triangles (instanced)** | **193,113** | **196,633** |
| of which the movers own | 188,275 (34 geoms) | 190,175 (47 geoms) |
| distinct visible meshes | 28 | 42 |
| meshes that are their own convex hull | **4 of 28** (`robotiq/pad`, `robotiq/silicone_pad`, x2 each) | 8 of 42 |
| largest meshes | `robotiq/base` 34,248 tri, `robotiq/base_mount` 21,834, `upperarm_3` 20,756, `upperarm_2` 14,489, `shoulder_0` 11,435 | identical (same assets) |
| textures | `grid` 512x512 2D (on the floor PLANE only), `skybox` 1024x6144 SKYBOX | + `T_btn` 512x512 **CUBE** (on 2 button MESH geoms) |
| materials with `reflectance > 0` | 0 | **10** (0.7–1.0) — **but measured contribution is 0–1 pixel at 1/255**; MuJoCo does not reflect these mesh surfaces |
| lights | 2: `global` DIRECTIONAL, `castshadow=1`, diffuse 0.7; `spotlight` SPOT, `castshadow=0`, diffuse 0.3, cutoff 100, exponent 10 | same, `global` diffuse 0.6 |
| headlight | active, diffuse 0.6, specular 0, ambient 0.1 — **98.7 % / 100 % of pixels, max 163/255** | same |
| camera | `front_pixels`, **mode FIXED** (0), fovy 45, `pos (1.053, -0.014, 0.639)` | `front_pixels`, FIXED, fovy 45, `pos (0.905, 0, 0.762)` |
| skybox / floor | yes / `grid` textured PLANE, `texuniform=1`, `texrepeat (1,1)` | same |
| MSAA | `offsamples = 4` | `offsamples = 4` |
| shadow map | `shadowsize = 4096`, `shadowclip 1.0`, `shadowscale 0.6` | same |
| haze | 0.3 declared — **measured contribution 0 pixels** | 0.3 — 1 pixel at 1/255 |
| per-episode geometry change | **none**: `ngeom/nmesh/nq` constant across resets; object count fixed at 1 cube | none; object count fixed (1 cube + drawer + window + 2 buttons) |
| per-episode appearance change | **none under the registered default variation set** (`cube.start_position/start_yaw/goal_position/goal_yaw` only) — every `MjModel` appearance array is bit-identical across resets | same, plus the 2 button bits (§3) |

**The render-distinct set is one asset library, not two.** All **28** of OGBCube's meshes appear in
OGBScene under the same names with **bit-identical vertex and face arrays** (sha256 over
`mesh_vert ‖ mesh_face`). That is **185,489 of OGBScene's 188,485 model triangles — 98.4 %** shared.
OGBScene adds exactly 20 meshes (drawer, drawer case, drawer handle, window frames/glass/base,
stop-button parts) totalling 2,996 triangles. The cameras are **not** shared: same names, different
poses, and OGBCube has a fourth (`side`) camera OGBScene lacks. So a cell pair would share one UR5e
+ Robotiq asset set and one shading law, and differ in camera pose, scene furniture and state
layout.

**No state-independent part dominates.** Hiding every mover leaves a plate that is **bit-constant**
across 6 states (0 pixels move, max diff 0) — so the Fetch plate-bake precondition *is* satisfied —
but the plate is only the floor plane and the skybox, and it is **85.4 % (Cube) / 72.1 % (Scene)**
of the frame. The remaining pixels split, measured against MuJoCo's own segmentation buffer, into
mover silhouette **10.2 % / 19.3 %** and mover shadow on the floor **≈ 4.4 % / 8.6 %**. The arm
casts a real shadow; a mover pass must draw it, exactly as `cpu3d`'s M1 does for the DM Control
trackcom cells.

---

## 3. Purity and state size

Probes in `scratch/step7/ogb-manipulation/probe_purity_fps.py`, `probe_hidden_bits.py`,
`probe_determinism_bars.py`; results under `examples/results/step7/ogb-manipulation/probe/`.

**A methodological trap, recorded because it silently inverts the answer.**
`ogbench...envs.env.compile_model_and_data` **rebinds** `_model` and `_data` on every dirty reset,
and these envs are dirty on every reset. A probe that caches `m, d = u.model, u.data` before a
reset compares a struct with itself and reports "qpos identical, frames differ, no hidden bit
found" — which is what the first pass of `probe_purity_fps.py` produced. Both probes re-fetch after
every reset now; the first-pass artefact is the reason `probe_hidden_bits.py` exists.

**OGBCube: the frame is a pure function of `(qpos, qvel, mocap_pos, mocap_quat)`.** Measured three
ways:

* **G1a** — 32 renders of one unchanged `MjData`: **1 distinct frame**, 0 pixels move. Bit-stable.
* **G1b** — drive the sim 25 random steps away and restore `(qpos, qvel)` through the public
  `set_state`: **bit-identical**, max abs diff 0.
* **G1d/G1e** — take a state from episode A and impose `(qpos, qvel, mocap_pos, mocap_quat)` on a
  **different episode** (different seed, recompiled model): **bit-identical, 0 pixels differ**.
  Across two `reset(seed=7)` calls the only arrays that differ are `qpos`, `mocap_pos` and their FK
  derivatives; **zero** `MjModel` arrays differ.

There is one genuine non-determinism, and it is in the *reset*, not the renderer: `CubeEnv.reset`
calls `reset_variation_space(..., seed=None, ...)`, so the default variations are resampled from an
RNG the gym seed does not control and two `reset(seed=7)` calls give different pictures (7.6 % of
pixels). That is a reproducibility property of the reference env, not a purity failure — the frame
is still a function of the exposed state.

**OGBScene: pure in `(qpos, qvel, mocap_pos, mocap_quat, button_state_0, button_state_1)`.** The two
button bits are **not** in `qpos`. They are required keyword arguments of
`SceneEnv.set_state(qpos, qvel, **kwargs)` — so swm's own API does expose them — and
`_apply_button_states` writes them into `geom_rgba` of `btngeom_0/1` and `mat_rgba` of
`drawer_handle` / `window_handle`. Measured at fixed `qpos`: flipping them moves **1.03 %, 1.18 %
and 2.21 %** of pixels for bit patterns `10`, `01`, `11`. Two discrete bits, fully enumerable.

OGBScene has one further wrinkle: **the official renderer is not quite bit-stable on this host.**
32 renders of one unchanged state gave **2 distinct frames**, 0.01 % of pixels moving by **1/255**
— four pixels. Almost certainly the two coincident translucent window-glass geoms under 4x MSAA.
It bounds what an "exact" claim can mean and must be re-measured under EGL on Linux (queued).

**Provenance note for the cluster leg.** The local numbers use the `stable_worldmodel` source
checkout at `/Users/samib/stable-worldmodel` (`sha256` prefixes: `cube_env.py` `143e5e81f424ff7a`,
`scene_env.py` `d0eebdce0131f86e`, `envs/__init__.py` `118a262677a647c6`); the Oscar venv installs
`stable-worldmodel==0.0.6` from PyPI, and job 6394784 prints the same three digests so the two can
be compared rather than assumed equal.

**Bytes.**

| | render-relevant state | bytes (f64) | frame bytes | frame / state |
|---|---|---:|---:|---:|
| `OGBCube` | `qpos(21) + qvel(20) + mocap(7)` = 48 floats | 384 | 224·224·3 = 150,528 | **392x** |
| `OGBScene` | `qpos(25) + qvel(24) + mocap(7)` = 56 floats + 2 bits | 464 | 200·200·3 = 120,000 | **259x** |

JPEG-q90 of the official frame is 6,115 B (Cube) and 10,150 B (Scene), i.e. **16x and 22x** the
state — the G4 economics are comfortable, as they were for the mazes.

---

## 4. The official renderer and the two frozen bars

`probe_determinism_bars.py`, median of 7 blocks of 100 renders, warm, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, one torch thread, at the **registered** resolution and camera. Decode is the
Step-6 `profile_note` protocol: JPEG-q90 the env's own frame, then one-core `PIL` decode, median of
7 blocks of 300.

| env | res | official `render()` | JPEG q90 | 1-core decode | **`S_cpu ≥ 5` bar** | **`cores_cpu ≥ 1` bar** | binding |
|---|---|---:|---:|---:|---:|---:|---|
| `OGBCube` | 224 | **159.9 f/s** (max 175.9) | 6,115 B | **6,893 f/s** | **800 f/s** | **6,893 f/s** | decode |
| `OGBScene` | 200 | **157.6 f/s** (max 162.0) | 10,150 B | **7,452 f/s** | **788 f/s** | decode |

Both bars are absolute fps targets for a stable-envs renderer of these scenes on this host. The
binding conjunct is `cores_cpu`, as it was for every Step-6 CPU cell: **145 µs/frame (Cube)** and
**134 µs/frame (Scene)**, i.e. **2.89 and 3.35 ns per output pixel**.

Two caveats the coordinator should hold: (a) these are M5 laptop numbers under GLFW/Metal, not
certification numbers — the Oscar `batch`/`normal` EGL leg is queued as job **6394784**; (b) this
official renderer is unusually *fast* relative to Step-6's CPU references (160 f/s here against
llvmpipe's 39.6–53.3 f/s for Fetch at 480 px), because the frame is small. That makes `S_cpu ≥ 5`
harder here than the frame size suggests.

---

## 5. Fetch-pattern feasibility on GPU

Checked precondition by precondition against what `stable_envs/envs/fetch.py` documents and
`stable_envs/render/mjwarp.py` implements. Measured with the repo's own helpers
(`MJ.texture_gaps`, `MJ.effective_alpha`) on the real registered models.

| Fetch precondition | Fetch | `OGBCube` / `OGBScene` | verdict |
|---|---|---|---|
| mover pass needs a mesh ray-caster | 18 meshes, 31,720 tri | 28 / 42 meshes, **193,113 / 196,633** visible tri | **same shape, 6.1x bigger** |
| static plate provably static | yes (reflectance 0, `castshadow=false`) | **yes** — movers hidden, 6 states, **0 pixels move** | ✅ satisfied |
| plate share of the frame | 91 % | **85.4 % / 72.1 %** | ✅ comparable |
| camera | free camera, pose **recovered** from `mjvScene`'s two `mjvGLCamera` eyes | **model camera `front_pixels`, mode FIXED, fovy 45** | ✅ **easier than Fetch** |
| MJWarp colour-from-material conflict | 11 of 24 geoms; needed per-geom material cloning | **0 geoms (Cube)**, **2 geoms (Scene)** — `btngeom_0/1`, which are exactly the geoms carrying the hidden button bits | ✅ / ⚠️ small, but it is the button state |
| MJWarp draws no sites | `target0` had to be re-expressed as a geom | **0 visible sites** in either env (all sites are group 4, not rendered) | ✅ nothing to do |
| MJWarp `sample_texture` UV gap | — | **`texture_gaps` is EMPTY**: the only 2D texture is `grid` on the floor **PLANE**, which MJWarp UVs correctly | ✅ (see mipmap caveat) |
| MJWarp never reads alpha | 4 marker geoms hidden, declared as a 1.2–1.5 % `parity_note` | **1 geom (Cube)**: `target_object_0` at alpha 0.2, owning **0.61 %** of pixels; **3 geoms (Scene)**: + two window-glass meshes at 0.1, **0.15 %** total | ⚠️ same shape, **smaller**, but the Cube marker IS the goal |
| MJWarp truncates where GL rounds | `gl_rounding` moved interior 0.791 → 0.9988 | same correction applies unchanged | ✅ reuse |
| shadows | Fetch's one light is `castshadow=false` | **`global` is `castshadow=1`** and the mover shadow is **4.4 % / 8.6 %** of the frame | ⚠️ needed, but MJWarp has `use_shadows` and `cpu3d` M1 already transcribed the law |
| `BATCH_SCALES` | **7.6–11.7x, FAIL** at 480 px; **26–99x, PASS** at ≤240 px | renders at **224 px / 200 px** — inside the passing regime | **the Fetch blocker probably does not transfer; queued** |

**Quantified geometry.** Median projected triangle edge at the registered resolution through the
registered camera: **0.713 px (Cube)**, **0.723 px (Scene)**; 61.8 % / 61.7 % of triangles are under
one pixel across; p90 edge 2.28 / 2.45 px; summed triangle area / frame area = **0.90 (Cube)** and
**3.10 (Scene)**. Fetch measured 0.50 px median at 480 px with 31,720 triangles. **Same regime —
sub-pixel, setup-bound — at 6.1x the triangle count into a 4.6x smaller frame.**

**The MSAA question, and the good news in it.** The official renderer runs `offsamples = 4`; a ray
caster does one sample per pixel. Measured by rebuilding the renderer at `offsamples = 0`: 1x versus
4x differs on **5.4 % (Cube)** and **13.0 % (Scene)** of pixels, mean 0.52 and 2.24/255 — but
`interior_exact_frac` between them is **1.0** and **0.99983**. **MSAA lives entirely on edge pixels
and the interior metric is blind to it.** A 1-sample ray caster is therefore not excluded from an
interior-parity claim by antialiasing alone. What it *does* mean is that the whole-frame exact
fraction has a hard ceiling around 0.95 / 0.87 unless the backend supersamples.

**How much of the frame is even measurable as parity.** `conformance.interior_mask` keeps
**41.6 % (Cube)** and **27.6 % (Scene)** of pixels — low by this repo's standards (the 2-D families
sit near 1.0) and a direct consequence of 62 % sub-pixel triangles, but not degenerate. An
`interior_exact_frac` here is a real claim about two fifths / one quarter of the picture, and the
declaration should say so.

**The minification risk is measured, and it is small.** MJWarp builds textures at
`num_mip_levels=1`; `nvdiff.py` records that this one fact carries 99.45-100.00 % of the interior
error on the Gymnasium locomotion floors. These scenes have exactly one 2-D texture — `grid`
512x512 on the floor — so the same failure mode was the open risk. Priced analytically
(`probe_floor_footprint.py`: cast the registered camera's rays at the plane, convert the hits to
texel coordinates under the PLANE texgen law with `texuniform=1`, finite-difference between
neighbouring pixels):

| | floor share of rays | texel/pixel median | p90 | p99 | max | > 2 texels/px |
|---|---:|---:|---:|---:|---:|---:|
| `OGBCube` | 97.8 % | 0.44 | 1.47 | 2.82 | **3.72** | 4.7 % |
| `OGBScene` | 100 % | 0.34 | 0.84 | 1.34 | **1.63** | 0 % |

The floor is a **bounded 2 m x 2 m plane** (`geom_size (1, 1, 0.01)`), the camera is 0.6-0.8 m above
it looking down, and the horizon is not in frame — so the far-field that ruins the locomotion floor
does not exist here. Most of the floor is **magnified** (median under half a texel per pixel) and the
worst pixel sees under 4 texels. Point sampling and a mipmap can differ over at most a few percent
of floor pixels, and the fallback if even that matters is free: the floor is part of the bit-static
plate, which is baked through MuJoCo itself.

**Appearance terms, measured so none is assumed.** Headlight **98.7 % / 100 %** of pixels, max
163/255 — the dominant light. Specular **25.5 % / 33.3 %**, max 47/43. Reflectance **0 pixels
(Cube)** and **1 pixel at 1/255 (Scene)** despite 10 materials declaring 0.7–1.0 — MuJoCo does not
mirror these mesh surfaces, so the Fetch "reflectance must be 0" precondition is satisfied *in
effect* even though it is violated *in declaration*. Haze 0 pixels. All four are things MJWarp's
megakernel already implements or can ignore.

---

## 6. Alternative shared paths

**`raycast3d` primitives: no.** The arm is not boxes and cylinders. Only **4 of 28** visible meshes
are their own convex hull, and those are the 300-face Robotiq pads; the load-bearing geometry is
`robotiq/base` (34,248 faces from 17,120 vertices, hull 1,636 faces), `upperarm_3` (20,756 faces
from 61,912 vertices, hull 3,366) and friends — organic CAD surfaces whose hulls are 5–21x smaller
than the mesh. Expressing them as primitives is not a re-encoding, it is a different object.
`raycast3d` is not the smallest capability here.

**`mjwarp`: yes, and it is the smallest.** It already has mesh BVHs, shadows, the `grid` PLANE
texture, spot and directional lights and the headlight, and this repo already carries the four
corrections (`gl_rounding`, `prepare_model`, material cloning, `needs_com_pos`) that Fetch needed.
The only capability MJWarp lacks that these scenes use is **alpha**, and it costs 0.61 % / 0.15 % of
pixels — a smaller `parity_note` than Fetch's 1.2–1.5 %, already accepted.

**`nvdiff`: not needed, and expensive here.** Its two reasons to exist are texgen for primitives
(`texture_gaps` is empty here) and minification (one 512x512 floor texture — power-of-two, so
mippable, and it *is* seen at grazing incidence, which is the one open fidelity risk). Against that,
`nvdiff.py`'s own docstring records 3–5x slower than MJWarp at 224 px and, decisively, **geometry is
per world**: the `(B, V, 4)` clip-space buffer at **390,445 vertices** costs ~25 MB per world, so
~1.6 GB at B=64 and ~6.4 GB at B=256 before attributes. MJWarp shares one BVH across worlds. Do not
route this family through nvdiff on the census as it stands. (W6's MuJoCo-free `Scene.from_triangles`
is not needed by this workstream and nothing here was touched.)

**CPU: a demonstrated blocker — `CPU3D-OGB-1`.** Using the Step-6 `CPU3D-FETCH-1` constants,
measured on the real Fetch mesh set with a scalar C span rasteriser (`-O3 -march=native`,
backface-culled, edge-function bbox scan, z-test, Gouraud, Blinn-Phong) on an AMD EPYC 9684X:

| | budget | at 24 ns/tri (setup only) | at 135 ns/tri (full, culled) |
|---|---:|---|---|
| `OGBCube` — 193,113 tri, binding bar 145 µs/frame | **0.75 ns/tri** | 4.63 ms = 216 f/s — **32x over** the binding bar, **3.7x over** `S_cpu ≥ 5` alone | 26.07 ms = **38 f/s** — **180x over**, `S_cpu = 0.24` |
| `OGBScene` — 196,633 tri, binding bar 134 µs/frame | **0.68 ns/tri** | 4.72 ms = 212 f/s — **35x over**, **3.7x over** `S_cpu ≥ 5` | 26.55 ms = **38 f/s** — **198x over**, `S_cpu = 0.24` |

For comparison, Fetch was 3.3–3.8x short at the full rate and **at** the bar at the setup floor.
These scenes are **~10x harder than Fetch at the setup floor and ~47x harder at the full rate**, and
unlike Fetch they fail `CPU_PARITY` too: 38 f/s against an official renderer running at 160 f/s is
`S_cpu = 0.24`, four times *slower* than the reference. The mechanism is not the fill — summed
triangle area is 0.90 / 3.10 frames, comparable to Fetch — it is **193 thousand sub-pixel triangles
into a 50 thousand pixel frame**. No rasteriser architecture transforms, culls and bins a triangle
in 0.75 ns. **Recommendation: do not fund a CPU backend for this family in Step 7.** If
`device="cpu"` must be honoured, the honest options are (a) raise `ValueError` at construction as
`cpu3d` already does for `DMCSwimmerSwimmer6`, or (b) route to the official `llvmpipe` renderer,
which is `S_cpu ≈ 1` by construction and is not a stable-envs backend.

---

## 7. Classification and proposal

**`FINDING`.** Smallest missing capability, named: **an `mjwarp`-backed env cell on the `fetch.py`
pattern (static plate + mesh mover pass), with a model camera instead of a recovered free camera,
and `(qpos, qvel, mocap_pos, mocap_quat[, 2 button bits])` as the state.** No new shared renderer
primitive is required — unlike W4's ellipsoid/heightfield, every capability these scenes use already
exists in `stable_envs/render/mjwarp.py`.

**Proposed fidelity tier: `parity` on GPU with a declared alpha `parity_note`; no CPU backend.**
The alpha note is 0.61 % (Cube) / 0.15 % (Scene) of pixels, strictly smaller than Fetch's accepted
1.2–1.5 %, and it falls almost entirely on edge pixels of a translucent marker — the same reason
Fetch's markers reached 0.000002 of interior pixels. The declaration must also state the interior
fraction (0.416 / 0.276), because an `interior_exact_frac` here covers much less of the frame than
in the 2-D families.

**Shared vs per-env.** Shared: the MJWarp backend and all four Fetch corrections; the UR5e +
Robotiq asset set (28 meshes, 185,489 triangles, bit-identical between the two envs); the plate bake;
the shading law. Per-env: camera pose, state layout, the scene furniture meshes (2,996 triangles),
and — Scene only — the two button bits and the `btngeom_0/1` material-vs-rgba resolution.

**Open questions for the coordinator** are in §9.

---

## 8. Cluster legs

Oscar SSH was denied at spawn time and is now restored; the jobs below were **submitted**, not only
prepared, per the coordinator's mid-task update. Scripts are versioned under
`scratch/step7/ogb-manipulation/`.

| job | script | partition | what it settles |
|---|---|---|---|
| **6394991** | `00_build_venv_and_bars.sbatch` | `batch` / `normal`, 4 cpu, 48G, 3 h | builds `$SCR/venvs/se7-ogb` (the only interpreter with ogbench **and** mujoco_warp **and** stable_worldmodel — `se-step5` lacks ogbench, `se-step6` lacks the other two, and AGENT_COMMON.md forbids installing into either); then proves both envs render under **EGL** headless and retakes the census, the bars, the purity probes and the interior/MSAA probe at 1 thread on a certification node |
| **6394992** | `01_mjwarp_trial.sbatch` | `gpu` / `norm-gpu`, 1x geforce3090, `--exclude=gpu2108,gpu2113,gpu2110,gpu2116`, 2 h, `afterok:6394991` | puts the registered `MjModel` through the **existing** `MJ.BatchRenderer` at the registered resolution and `front_pixels` camera: does it build at 193k triangles, VRAM, first fidelity numbers against MuJoCo's own EGL frames with `gl_rounding`, and **`BATCH_SCALES` = fps(1024)/fps(1)** |

Two earlier attempts failed and are recorded rather than hidden, because both failure modes are
traps a later phase would hit again:

* **6394700 / 6394701** — a bare `python3` on a `batch` node is **3.9**, so every `pip install -q`
  for a 3.12-only wheel failed *silently* and the venv imported nothing. The script now loads
  `python/3.12.8-zr3c`, asserts the version, drops `-q`, and refuses to exit 0 unless the named
  output files exist — so a dependent GPU job cannot be chained onto an empty result. `6394701` was
  cancelled by the `afterok` dependency, as intended.
* **6394784** — `torch==2.14.0` came from the cu126 index but `torchvision` was resolved later from
  PyPI, which picks the default-variant wheel; `import torchvision` (pulled in by
  `stable_worldmodel`) then dies with `RuntimeError: operator torchvision::nms does not exist`. Both
  now come from the cu126 index in one resolve, `stable-worldmodel` installs `--no-deps`, and the
  import assertion covers `torchvision` explicitly.

One GPU job in flight. No `simclr` job was touched.

---

## 9. Open questions for the coordinator

1. **Which resolution is the frozen one?** `OGBCube` registers at 224 and `OGBScene` at **200** —
   the latter only because `SceneEnv` forgets to override the ogbench default while swm's own
   collection scripts pass `width=height=224`. Freeze the registered defaults (224/200), or
   normalise both to 224 and record the deviation?
2. **`visualize_info=True` or `False`?** The registered default draws the translucent goal marker
   (0.61 % of Cube's pixels, the one thing MJWarp cannot draw). swm's own `collect_cube.py`,
   `collect_scene.py` and `cube.yaml` all pass `visualize_info=False`, which removes it entirely and
   would make the alpha `parity_note` unnecessary. Certify the registration or the usage?
3. **Are the 2 OGBScene button bits part of the state contract?** They are required kwargs of
   `SceneEnv.set_state`, they move 1.0–2.2 % of pixels, and swm's own `reset(options={'state': ...})`
   path cannot supply them (it raises `KeyError`). Proposal: carry them as 2 extra state dimensions
   and report the upstream `reset` defect to swm.
4. **How much of the variation space is in scope?** Phase 0 measured the registered default set
   (`cube.start_position/start_yaw/goal_position/goal_yaw`), under which every `MjModel` appearance
   array is constant and the plate is bit-static. `floor.color`, `light.intensity`,
   `camera.angle_delta`, `cube.size`, `agent.color` and `lock_color` each make the plate a function
   of the variation vector — tractable as per-env tables (the DMC cells already do this) but it
   changes the bake's cache key and should be decided before implementation.
5. **`CPU3D-OGB-1`: confirm "no CPU backend".** The recommendation is to raise `ValueError` on
   `device="cpu"` rather than build one, per the Sami policy addendum's honest-device-selection
   rule. That leaves these two ids GPU-only. Acceptable?
6. **The mipmap risk is now priced and is NOT the blocker it was for locomotion** (§5): the floor is
   a bounded 2 m x 2 m plane with no horizon in frame, worst-case 3.72 texels per pixel and 4.7 % of
   floor pixels above 2. The largest *open* fidelity risk is instead the whole-frame ceiling imposed
   by 4x MSAA (≈ 0.95 Cube / 0.87 Scene) against a 1-sample ray caster, which is an edge-pixel
   effect the interior metric does not see. Is an `interior`-only parity claim, with the whole-frame
   number declared and explained, acceptable for these two ids?
7. **OGBScene's official renderer is not bit-stable on macOS** (2 distinct frames in 32, 4 pixels at
   1/255). If that reproduces under EGL on Linux, the frozen reference corpus needs a stated
   tolerance rather than a digest.
