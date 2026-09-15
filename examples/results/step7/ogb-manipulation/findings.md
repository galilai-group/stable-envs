# Step 7 / W5 `ogb-manipulation` — findings

Phase 0 (census, purity, bars, feasibility) is §1-§9. **Phase 1** is §10 onward: M1
attribution, M2 the `texture_gaps` fix, and the Q11 numbers prepared for Sami.

**Classification: `FINDING`.** Diagnosis only; no renderer written, no shared file touched.
Two sub-verdicts, both measured:

* **GPU: `OGBCube` goes through the existing MJWarp backend unmodified but fails `BATCH_SCALES`;
  `OGBScene` does not build at all.** Measured on an RTX 3090 (jobs 6395461, 6395736, one env per
  process). `OGBCube` builds at 193,113 triangles and runs **2,232 f/s at B=1 → 6,531 f/s at
  B=4096**, so **`BATCH_SCALES` = 2.93x** against a required 20x. **This refutes the hypothesis this
  file carried before the GPU leg** — that Fetch's 7.6–11.7x failure was a 480 px effect these
  224/200 px scenes would escape. It is the reverse: at 224 px the per-frame device cost is small
  enough that `fps(1)` is *already* 2,232 f/s and throughput is within 3 % of its ceiling by **B=64**,
  so there is nothing left to amortise. `OGBScene` **crashes the megakernel** with
  `Warp CUDA error 700: an illegal memory access`, and the cause is bisected to **one CUBE texture**
  (`T_btn`, 512x512, on two button meshes): detaching that single texture and nothing else makes the
  scene render. `stable_envs/render/mjwarp.py::texture_gaps` gives this model a **false all-clear**
  because it classifies by geom type and never reads `tex_type` — a shared-file finding, reported
  not fixed.
  And `OGBCube`'s raw fidelity, with only `gl_rounding` and no plate bake or per-env correction, is
  **0.0101 interior-exact, mean |Δ| 10.5/255** against Fetch's corrected 0.9988. The smallest
  capability is still MJWarp, but it does not work out of the box on either env.

* **CPU: `BLOCKER` for `CPU_FAST` on the `cores_cpu` conjunct; `CPU_FASTER` is feasible.**
  Measured on the certification host (`node2351`, `batch`/`normal`, one thread, Mesa llvmpipe
  through EGL): the official renderer runs at **6.31 / 5.17 f/s** and one-core JPEG-q90 decode at
  **2,774 / 2,106 f/s**. The Step-6 `CPU3D-FETCH-1` span rasteriser's measured 135 ns/triangle puts
  193,113 triangles at **38.4 f/s**, which is `S_cpu` **6.07 / 7.29 — comfortably over the ≥ 5
  conjunct** but `cores_cpu` **0.014 / 0.018, short by 72x / 56x**. Even the architecture-independent
  *setup-only* floor (24 ns/triangle: transform, bbox, area, cull; no fill, no shade) is **12.9x /
  9.9x** short of `cores_cpu ≥ 1`. So the verdict is the same shape as Fetch's — do not fund a
  `CPU_FAST` backend — but the conjunct that fails is different, and a `CPU_FASTER` backend at
  ~6-7x the official renderer IS reachable if `device="cpu"` is wanted for its own sake. Recorded as
  **`CPU3D-OGB-1`**.

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

Both hosts, same probe (`probe_determinism_bars.py`), median of 7 blocks of 100 warm renders,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, one torch thread, at the **registered** resolution and
camera. Decode is the Step-6 `profile_note` protocol: JPEG-q90 the env's own frame, one-core `PIL`
decode, median of 7 blocks of 300.

**Certification numbers — Oscar `node2351`, `batch`/`normal`, `MUJOCO_GL=egl` resolving to Mesa
llvmpipe (job 6395175):**

| env | res | official `render()` | JPEG q90 | 1-core decode | **`S_cpu ≥ 5` bar** | **`cores_cpu ≥ 1` bar** | binding |
|---|---|---:|---:|---:|---:|---:|---|
| `OGBCube` | 224 | **6.31 f/s** | 6,125 B | **2,774 f/s** | **31.6 f/s** | **2,774 f/s** | decode |
| `OGBScene` | 200 | **5.17 f/s** | 10,180 B | **2,106 f/s** | **25.8 f/s** | **2,106 f/s** | decode |

The binding conjunct is `cores_cpu`, as it was for every Step-6 CPU cell: **361 µs/frame (Cube)** and
**475 µs/frame (Scene)**, i.e. 7.19 and 11.87 ns per output pixel.

**Local M5 numbers, for contrast — and the contrast is the point:**

| env | official `render()` | 1-core decode | `S_cpu ≥ 5` bar | `cores_cpu ≥ 1` bar |
|---|---:|---:|---:|---:|
| `OGBCube` | 159.9 f/s | 6,893 f/s | 800 f/s | 6,893 f/s |
| `OGBScene` | 157.6 f/s | 7,452 f/s | 788 f/s | 7,452 f/s |

**The M5's official renderer is 25-30x faster than the certification host's, and a phase-0 CPU
verdict computed from it would have been wrong in both directions.** macOS GLFW gets a
Metal-accelerated GL; the `batch` node gets Mesa llvmpipe in software, and llvmpipe is what
`S_cpu` is defined against. Against the laptop, a 38 f/s rasteriser looks like `S_cpu = 0.24` and
fails every tier; against the certification host it is `S_cpu = 6.1` and clears `CPU_FASTER`
outright. The decode bar moved far less (6,893 -> 2,774), because decode is a real CPU workload on
both. This is precisely the "local CPU timings are NOT certification numbers" rule in
`AGENT_COMMON.md`, and it changed a verdict, not a decimal.

**The official renderer IS bit-stable on the certification host.** 32 renders of one unchanged
`MjData`: **1 distinct frame for BOTH envs**, 0 pixels moving. The 2-distinct-frames result for
`OGBScene` on macOS (4 pixels at 1/255) was a Metal/MSAA artefact and does not reproduce under
EGL/llvmpipe — so the frozen corpus can be a digest rather than a tolerance. Open question 7 is
answered.

**Everything else reproduced across the two hosts**, which is the reason to trust the local half:
the census is identical (`ngeom` 64/137, `nmesh` 28/48, model triangles 185,489/188,485, instanced
263,243/266,763, `nq` 21/25, `nlight` 2, `ntex` 2/3, obs 28/40); `pure(qpos,qvel) = True` for
`OGBCube` on both; `plate_static = True` with 0 pixels moving on both; mover share 15.3 % / 31.3 %
(local 14.6 / 27.9); interior fraction 41.86 % / 27.15 % (local 41.61 / 27.64); MSAA 1x-vs-4x
`interior_exact` 0.99995 / 0.99974 (local 1.0 / 0.99983); median projected triangle edge 0.698 /
0.723 px (local 0.713 / 0.723); translucent share 0.62 % for `OGBCube`.

**Provenance.** `oscar/provenance_6395175_readback.txt`: python 3.12.8, torch 2.14.0+cu126,
torchvision 0.29.0+cu126, mujoco 3.12.0, mujoco-warp 3.12.0, warp-lang 1.17.0, ogbench 1.2.1,
dm-control 1.0.45, numpy 2.5.2, scipy 1.18.1, pillow 12.3.0, gymnasium 1.3.0. The three
`stable_worldmodel` file digests match the local checkout **exactly** (`cube_env.py`
`143e5e81f424ff7a`, `scene_env.py` `d0eebdce0131f86e`, `envs/__init__.py` `118a262677a647c6`), so
both halves measured the same registration. That file is a **login-node read-back**, not the job's
own output, and says so in its first line: an edit to section 1 of the sbatch had deleted the
provenance block, so job 6395175 shipped a *stale* `versions.txt` from the failed 6394700 (python
3.9, everything ABSENT) beside numbers taken under python 3.12. The stale file is deleted, the block
is restored, and the script now refuses to exit 0 without a job-tagged provenance file. Found by
reading the whole log rather than its tail.

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
| `BATCH_SCALES` | **7.6–11.7x, FAIL** at 480 px; **26–99x, PASS** at ≤240 px | **2.93x MEASURED** on a 3090, B=1→4096 (job 6395736) | ❌ **FAIL, and worse than Fetch** — see below |

**`BATCH_SCALES`, measured, and why the resolution argument was wrong.** `OGBCube` through the
existing `MJ.BatchRenderer` on an RTX 3090, registered 224 px, registered `front_pixels` camera,
CUDA graph captured, median of 7 (job 6395736; job 6395461 agrees within 3 % on every row):

| B | ms/render | f/s | peak VRAM |
|---:|---:|---:|---:|
| 1 | 0.45 | **2,232** | 0 MiB |
| 16 | 2.78 | 5,749 | 2 MiB |
| 64 | 10.05 | 6,369 | 9 MiB |
| 256 | 39.20 | 6,531 | 36 MiB |
| 1024 | 155.96 | **6,566** | 148 MiB |
| 4096 | 627.13 | 6,531 | 588 MiB |

**`BATCH_SCALES` = 6,531 / 2,232 = 2.93x** against a required 20x. The reasoning this file carried
before the job landed — that Fetch's failure was a 480 px effect and these smaller frames sit in the
regime where the same scene scored 26–99x — is **refuted by the measurement**, and the mechanism is
the opposite of the one assumed: `BATCH_SCALES` is large only when `fps(1)` is *bad*, and here it is
already 2,232 f/s because 224 px of a 6.1x-denser BVH is still a cheap frame. Throughput is within
3 % of its ceiling by **B=64** and flat from B=256 to B=4096, so no larger batch rescues the ratio,
and VRAM is not the constraint (588 MiB at B=4096).

**`OGBScene` does not build through MJWarp, and the cause is one CUBE texture.** The crash is
`Warp CUDA error 700: an illegal memory access` from `wp_cuda_graph_begin_capture`, reproduced in
two jobs, the second with one env per process — and in that run the error surfaces first at
`wp_cuda_context_synchronize` **during the warm-up frame, before the capture**, so it is a fault in
`kinematics`/`camlight`/`refit_bvh`/`render` on this model, not contamination from `OGBCube`'s
teardown (which is what job 6395461's `OGBScene` row was, and why that row is void).

Bisected rather than guessed at (job **6396447**, each configuration in its own subprocess because
an illegal access poisons the context for everything after it):

| configuration | builds and renders? |
|---|---|
| baseline | ❌ CUDA 700 |
| `capture=False` (no CUDA graph) | ❌ CUDA 700 — not a graph-capture bug |
| `skybox=False` | ❌ CUDA 700 |
| `shadows=False` | ❌ CUDA 700 |
| `textures=False` | ✅ renders, frame std 88.0 |
| **detach only the CUBE textures, keep 2-D + skybox** | ✅ **renders, frame std 60.9, 1 texture detached** |

**One texture.** `OGBScene`'s `btn_metal` material wears `T_btn`, a 512x512 **cube map**
(`mjTEXTURE_CUBE`), on two button MESH geoms; `OGBCube` has no cube map and builds. Detaching that
single texture — leaving every 2-D texture and the skybox in place — makes the scene render. So
`mujoco_warp`'s texture path reads a cube map through the 2-D sampler and walks off the allocation.

**This also exposes a false all-clear in this repo's own helper, which is a shared-file concern and
is reported rather than fixed.** `stable_envs/render/mjwarp.py::texture_gaps` returns **empty** for
`OGBScene`, because its rule is "which geom TYPES can MJWarp derive UVs for" and `MESH` is in that
set. It never looks at the texture's `tex_type`. A cube map on a mesh therefore passes the gap check
and then crashes the renderer. The narrowest fix is one clause in `texture_gaps` — report
`mjTEXTURE_CUBE` on any geom as a gap — but `mjwarp.py` is shared and the coordinator assigns
ownership; W5 has not touched it. Until then, an empty `texture_gaps` should not be read as "this
scene's textures are safe for MJWarp".

**Raw fidelity is far from parity and is the real phase-1 question.** With only `gl_rounding`
applied — no plate bake, no per-env correction, the whole scene through the megakernel — against
MuJoCo's own EGL frames at 16 matched states: **`interior_exact` 0.0101**, `all_exact` 0.035,
`all_within8` 0.547, mean |Δ| **10.5/255**, `interior_frac` 0.415. Fetch's uncorrected number was
0.791 interior and its corrected one 0.9988, so this is two orders of magnitude further away and is
not explained by any of the four known Fetch corrections (`texture_gaps` is empty, there is no
material/rgba conflict on `OGBCube`, there are no visible sites, and the one translucent geom owns
0.62 % of pixels). The likely candidates — the `castshadow` directional light, the spot light's
cutoff 100 / exponent 10, and the headlight that owns 98.7 % of pixels — are exactly the terms
`cpu3d`'s M1 had to transcribe for the DM Control trackcom cells, and none of them is in the Fetch
scene. **Phase 1 must start here, not at performance.**

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

**CPU: `CPU3D-OGB-1` — a demonstrated blocker for `CPU_FAST`, on `cores_cpu` alone.** Using the
Step-6 `CPU3D-FETCH-1` constants (a scalar C span rasteriser, `-O3 -march=native`, backface-culled,
edge-function bbox scan, z-test, Gouraud, Blinn-Phong, measured on the real Fetch mesh set on an
AMD EPYC 9684X) against the **certification-host** bars of §4:

| | official / decode (node2351) | at 24 ns/tri (setup only) | at 135 ns/tri (full, culled) |
|---|---|---|---|
| `OGBCube` — 193,113 tri | 6.31 / 2,774 f/s | 4.63 ms = **216 f/s**: `S_cpu` 34.2 ✅, `cores_cpu` **0.078 — 12.9x short** | 26.07 ms = **38.4 f/s**: `S_cpu` **6.07 ✅**, `cores_cpu` **0.014 — 72x short** |
| `OGBScene` — 196,633 tri | 5.17 / 2,106 f/s | 4.72 ms = **212 f/s**: `S_cpu` 41.0 ✅, `cores_cpu` **0.101 — 9.9x short** | 26.55 ms = **37.7 f/s**: `S_cpu` **7.29 ✅**, `cores_cpu` **0.018 — 56x short** |

The `S_cpu ≥ 5` conjunct **passes** — llvmpipe pushing 193 k triangles is itself slow, at 6.3 f/s —
and `cores_cpu ≥ 1` fails by 56-72x. The floor row is the architecture-independent part: even a
rasteriser that wasted no bbox test, wrote no pixel and shaded nothing would still pay ~24 ns per
triangle to transform, bound, area and cull it, and that alone is **9.9-12.9x** over the decode bar.
No rasteriser architecture — scalar, SIMD-binned, or tiled — closes a 10x gap on pure vertex setup.
The mechanism is not the fill: summed triangle area is 0.90 / 3.10 frames, comparable to Fetch's.
It is **193 thousand sub-pixel triangles into a 50 thousand pixel frame**.

For comparison, Fetch was 3.3-3.8x short at the full rate and *at* the bar at the setup floor, so
these scenes are **~4-6x harder than Fetch on the binding conjunct**. Unlike the laptop-based
estimate this file carried before the cluster leg landed, they are **not** slower than their own
reference: `CPU_FASTER` at ~6-7x is comfortably reachable.

**Recommendation: do not fund a `CPU_FAST` backend for this family in Step 7.** If `device="cpu"`
must be honoured, the two honest options are (a) raise `ValueError` at construction, as `cpu3d`
already does for `DMCSwimmerSwimmer6`, or (b) build the shared span-rasteriser mover pass and
declare `CPU_FASTER` with `cores_cpu` reported and failing — which is what the Step-6 policy
addendum's "measured and reported honestly" clause allows, and what Step 2 §12.1
`PERFORMANCE_MEASURED` asks for.

---

## 7. Classification and proposal

**`FINDING`.** Smallest missing capability, named: **an `mjwarp`-backed env cell on the `fetch.py`
pattern (static plate + mesh mover pass), with a model camera instead of a recovered free camera,
and `(qpos, qvel, mocap_pos, mocap_quat[, 2 button bits])` as the state.** No new shared renderer
primitive is required — unlike W4's ellipsoid/heightfield, every capability these scenes use already
exists in `stable_envs/render/mjwarp.py`.

**Proposed fidelity tier: NOT YET PROPOSABLE — phase 1 must first explain a 0.0092 interior.**
Before the GPU leg landed this file proposed `parity` on GPU with an alpha `parity_note`. The
measurement does not support that yet: raw MJWarp is **0.0092 interior-exact** against the official
EGL frames, two orders of magnitude from Fetch's corrected 0.9988, and none of the four known Fetch
corrections applies here (§5). The alpha note is still small and still right (0.62 % / 0.15 %, below
Fetch's accepted 1.2–1.5 %), and the interior fraction (0.414 / 0.272) must be declared alongside
any interior claim — but the tier itself is a phase-1 output, not a phase-0 proposal.

**Performance tiers, measured and reported honestly (Step 2 §12.1 `PERFORMANCE_MEASURED`).** On GPU:
`GPU_FAST`-class throughput (6,352 f/s at B=1024 on a 3090) with **`BATCH_SCALES` = 2.89x, FAIL** —
the same qualifier Step 6 closed `fetch-gpu` with, and Step 7's own rule says a new cell is not
required to pass it to ship. On CPU: `CPU_FAST` blocked on `cores_cpu` by 56–72x, `CPU_FASTER`
reachable at `S_cpu` ≈ 6–7 (§6). Neither number blocks coverage; both must be declared.

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

| job | script | partition | outcome |
|---|---|---|---|
| **6395175** | `00_build_venv_and_bars.sbatch` | `batch`/`normal`, node2351 | **COMPLETED.** Built `$SCR/venvs/se7-ogb` (the only interpreter with ogbench **and** mujoco_warp **and** stable_worldmodel — `se-step5` lacks ogbench, `se-step6` lacks the other two, and AGENT_COMMON.md forbids installing into either); proved both envs render headless; reproduced the census exactly; took the certification bars, the purity probes and the interior/MSAA probe at 1 thread. §4. |
| **6395461** | `01_mjwarp_trial.sbatch` | `gpu`/`norm-gpu`, gpu2101, RTX 3090 | **COMPLETED.** `OGBCube` builds through the existing `MJ.BatchRenderer` at 193 k triangles; `BATCH_SCALES` 2.89x; raw fidelity 0.0092 interior. §5. The `OGBScene` row is **void** — CUDA context poisoned by `OGBCube`'s teardown in the same interpreter (an error-700 storm from `wp_free_device_async`, then `wp_cuda_graph_begin_capture` failed). |
| **6395736** | `01_mjwarp_trial.sbatch`, one env per process | `gpu`/`norm-gpu`, RTX 3090 | **COMPLETED.** `OGBCube` reproduces job 6395461 within 3 % on every row and extends the sweep to B=4096 (`BATCH_SCALES` 2.93x, 588 MiB). `OGBScene` fails again in a clean process, with the error before the graph capture — so it is that model's own fault. §5. |
| **6396447** | `02_mjwarp_bisect.sbatch` | `gpu`/`norm-gpu`, RTX 3090 | **COMPLETED.** Localises the `OGBScene` fault to a single CUBE texture: `capture`/`skybox`/`shadows` off all still crash, `textures=False` renders, and detaching only the cube map renders. §5. |

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
  now come from the cu126 index in one resolve, and the import assertion covers `torchvision`
  explicitly.
* **6394991** — and this one is a *finding*, not just a build error: **PyPI
  `stable-worldmodel==0.0.6` is not the same code as the source checkout.** The PyPI build's
  `data/dataset.py` imports `h5py`; the checkout's `data/__init__.py` imports its own
  `formats.lance` and never mentions `h5py`. Same version string, different code — so a cluster
  number taken against the PyPI wheel would not have been measuring the registration the local
  census measured. The checkout is now rsynced to `$SCR/se7/swm-src` and put on `PYTHONPATH`, the
  PyPI build is uninstalled so it cannot shadow it, and the job prints the same three file digests
  the local run recorded (`cube_env.py` `143e5e81f424ff7a`, `scene_env.py` `d0eebdce0131f86e`,
  `envs/__init__.py` `118a262677a647c6`) so the two are compared rather than assumed equal.
  `6395006` was cancelled with it.
* **6395139** — `MUJOCO_GL=egl` on a `batch` node: see the backend note below. Not a renderer
  result, and the probe that now catches it is the reason it is recorded here rather than sitting
  in an artifact as a "these envs do not render headless" conclusion.

**The GL backend is probed before any measurement** (`gl_probe.py`, run first in both scripts),
because two different things wear the same traceback and must not be confused:

* **A node fault.** W4 found `gpu2104` with healthy torch/CUDA and broken EGL context creation, and
  a fault of that shape reaches the artifacts looking like a *renderer* finding. On the GPU job the
  probe therefore runs `--require egl`: EGL or exit 9, never a software fallback that would silently
  change the reference. `gpu2104` is excluded by name as well.
* **The wrong backend for the partition.** Job **6395139** died on `node2326` with
  `Cannot initialize a EGL device display ... PLATFORM_DEVICE`. That is not a fault — a `batch` node
  has no GPU, so EGL's device path cannot initialise, and Mesa surfaceless EGL or `osmesa`
  (llvmpipe) is what renders there, exactly as the Step-6 CPU scripts probe for. The CPU job now
  tries `egl`, `osmesa`, `glfw` in order and records which one won in `oscar/gl_backend.txt` — which
  also names the reference the `S_cpu` number is computed against.

The probe renders a real lit scene in a **child process** (a failed GL init leaves the OpenGL
bindings in a state the next attempt inherits, so probing three backends in one interpreter makes
the second answer depend on the first) and rejects a context that creates but returns a **constant
frame**. `sbatch` snapshots the script at submit time, so every script change costs a
cancel-and-resubmit; the live pair is **6395175** (CPU) and **6395176** (GPU, `afterok`).

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
5. **`CPU3D-OGB-1`: `ValueError` on `device="cpu"`, or a `CPU_FASTER` backend?** `CPU_FAST` is out
   (`cores_cpu` short by 56-72x, and by 9.9-12.9x even at the vertex-setup floor), but the cluster
   leg overturned this file's first, laptop-based reading: `S_cpu ≈ 6-7` against the certification
   host's llvmpipe, so a shared span-rasteriser mover pass would honestly certify `CPU_FASTER`.
   Refuse `device="cpu"` and keep these two ids GPU-only, or build the CPU path at `CPU_FASTER` with
   `cores_cpu` reported and failing?
6. **The mipmap risk is now priced and is NOT the blocker it was for locomotion** (§5): the floor is
   a bounded 2 m x 2 m plane with no horizon in frame, worst-case 3.72 texels per pixel and 4.7 % of
   floor pixels above 2. The largest *open* fidelity risk is instead the whole-frame ceiling imposed
   by 4x MSAA (≈ 0.95 Cube / 0.87 Scene) against a 1-sample ray caster, which is an edge-pixel
   effect the interior metric does not see. Is an `interior`-only parity claim, with the whole-frame
   number declared and explained, acceptable for these two ids?
7. **Answered, no action needed.** `OGBScene`'s official renderer is not bit-stable on macOS
   (2 distinct frames in 32, 4 pixels at 1/255) but **is** bit-stable on the certification host
   (1 distinct frame in 32, both envs, job 6395175). The macOS result was a Metal/MSAA artefact; the
   frozen corpus can be a digest.
8. **The 0.0092 interior is the gating unknown, and it is a phase-1 question.** Raw MJWarp on
   `OGBCube` is `interior_exact` 0.0092, mean |Δ| 10.4/255 against the official EGL frames, and none
   of the four Fetch corrections applies. The terms the Fetch scene does not have and these do — a
   `castshadow` directional light, a spot light with cutoff 100 / exponent 10, and a headlight that
   owns 98.7 % of pixels — are exactly the ones `cpu3d`'s M1 had to transcribe from the megakernel
   for the DM Control trackcom cells. Is W5 authorised to spend phase 1 attributing that gap term by
   term (the M1 recipe, on the GPU side) before any cell is written?
9. **`BATCH_SCALES` = 2.89x: ship with the qualifier, as Fetch did?** Step 7's rules say a new cell
   need not be `GPU_FAST` to ship provided the tiers are measured and declared. These cells saturate
   by B=64 at 6,352 f/s on a 3090 with 148 MiB at B=1024, so the ratio is a property of a cheap frame
   and a dense BVH, not of a fixable bottleneck. Confirm that `GPU_*`+`BATCH_SCALES_FAIL` is
   acceptable for coverage, so phase 1 spends its budget on fidelity rather than on a ratio that
   Step 6 already escalated once and closed.
10. **Who owns the `texture_gaps` fix?** `stable_envs/render/mjwarp.py::texture_gaps` classifies by
    geom type and never reads `tex_type`, so a cube map on a mesh passes the check and then crashes
    the megakernel with an illegal memory access (job 6396447). The narrowest fix is one clause, but
    `mjwarp.py` is shared with the Fetch, DM Control and Meta-World families and AGENT_COMMON.md says
    the coordinator assigns ownership of shared files — so W5 has recorded it and changed nothing.
    Any workstream reading an empty `texture_gaps` today is reading a false all-clear.
11. **Is `OGBScene` in scope at all if MJWarp cannot draw its buttons?** The cube map is the button
    metal. Options: (a) upstream/patch `mujoco_warp`'s cube sampling; (b) bake `T_btn` to its mean
    colour via the existing `bake_primitive_textures` recipe, which is a declared approximation on
    two small geoms; (c) drop `OGBScene` from Step 7 and ship `OGBCube` alone. (b) is cheap and in
    keeping with the RoboCasa precedent, but it is a fidelity decision, not an engineering one.

---

# Phase 1

Authorised by the coordinator after phase 0, with rulings on all eleven open questions. Milestones
**M1 attribution -> M2 `texture_gaps` -> M3 OGBCube cell -> M4 tiers -> M5 OGBScene (pending Sami)**.

## 10. M1 — the interior gap is one term, and it is upstream

Full report and data: `m1/attribution.md`, `m1/m1_attribution.json`, `m1/m1_mujoco_spot_cutoff.json`.
Job **6396646**, `gpu2109`, RTX 3090, 8 real states, 224 px, `gl_rounding` throughout. Each term is
neutralised **in the `MjModel`** and BOTH renderers are then driven from that same model, so the
measurement isolates the term instead of changing one side's picture.

| term neutralised on both sides | interior exact | Δ | interior mean abs |
|---|---:|---:|---:|
| *baseline* | 0.0082 | — | 8.78 |
| **spot light** | **0.4568** | **+0.4486** | **1.54** |
| headlight | 0.0454 | +0.0372 | 8.25 |
| MSAA 4x -> 1x | 0.0111 | +0.0029 | 8.89 |
| specular | 0.0078 | −0.0004 | 9.56 |
| shadows | 0.0040 | −0.0042 | 13.06 |
| skybox | 0.0039 | −0.0043 | 8.90 |

The signed histogram confirms it: spread over **−3 to −8** at baseline — MJWarp uniformly darker,
which is not a quantisation signature — collapsing to **{0: 61.8 %, +1: 21.3 %}** once the spot is
removed, which is exactly what one shading law quantised twice looks like.

**Mechanism, read from both sources.** `mujoco_warp` attenuates a spot as
`cos_theta < cos(cutoff) -> unlit`, then `pow(cos_theta, exponent)`. This scene's spot is
`cutoff = 100°`, `exponent = 10`. MuJoCo never applies that, because it **discards** the cutoff:
measured through MuJoCo alone, `cutoff = 100°` renders **bit-identically to 180°** (max abs **0**)
while a valid `90°` changes **95.2 %** of pixels and darkens the frame by 7.4 levels of mean. That is
`GL_SPOT_CUTOFF`'s rule — `[0, 90]` or the sentinel `180`, anything else `GL_INVALID_VALUE` and left
at 180 = not a spotlight — reproduced exactly at its boundary. **`mujoco_warp` honours an
out-of-range spot cutoff that MuJoCo silently discards.** Not per-env, and not predicted by any Fetch
correction, because the Fetch scene has no spot light.

**Two sharp caveats on reading the interior number here.** The interior mask on this scene is
**99.2 % floor** (static 0.9917, sky 0.0043, mover 0.0040), so the whole-frame interior figure is
almost entirely a statement about the plate the shipped cell bakes through MuJoCo exactly.
`interior_exact_mover` is 0.667 at baseline and 0.852 with the headlight neutralised, but over 0.4 %
of interior pixels — it is reported with that sample size, not quoted as a mover-pass parity number.

## 11. M2 — `texture_gaps` was a false all-clear (shared-file fix, assigned)

`stable_envs/render/mjwarp.py`. The helper classified by geom **type** alone and never looked at the
texture or the mesh, so `swm/OGBScene-v0` returned an **empty** gap list and then killed the renderer.

**Root cause, read from `mujoco_warp` and confirmed on the model** — sharper than phase 0's
"cube maps are a different sampler". `sample_texture`'s MESH branch reads
`mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][i]]` with no guard, and
OGBScene's two button geoms wear mesh `stopbuttonrim` with **`texcoordadr = -1, texcoordnum = 0`** —
no UVs at all — so the index is computed from a negative offset. A cube-mapped mesh normally carries
no UVs, because MuJoCo generates cube texgen from position and normal; that is why the two symptoms
travel together. `OGBCube` has **zero** textured mesh geoms and does not crash.

Two conditions added, both checked regardless of geom type because the case that found them is a
MESH — the type the old rule declared safe: a texture whose `tex_type` is not 2-D, and a textured
mesh with `mesh_texcoordnum == 0`. They are independent (a 2-D texture on a UV-less mesh crashes with
no cube map in sight). Every texture role is examined rather than `np.max` over `mat_texid`.

Proof it is additive: **21 synthetic unit cases** (`tests/unit/test_mjwarp_texture_gaps.py`);
a registry sweep running BOTH implementations, the old one reproduced verbatim — **40 families build
locally (25 OGBench maze, 11 DM Control, 4 Mujoco) and all 40 give identical answers**, including
models containing cube and skybox textures no rendered geom wears; a **positive control** so the
sweep can tell a correct fix from a no-op (`swm/OGBScene-v0` goes 0 -> 2, naming the two button
geoms); and `tests/unit` at **830 passed, 2 failed**, both reproducing identically at the stashed
baseline. Data: `m2/texture_gaps_sweep.json`. Still owed on the cluster: the sweep where the
cuda-only families build (Fetch above all) and a Fetch/DMC byte-identity replay — job 6396748,
pending behind a saturated GPU partition (346 queued).

## 12. Q11 — the three OGBScene options, priced for Sami (nothing chosen)

Full memo: `m5/q11_for_sami.md`, data `m5/q11_button_bake.json`. Option (b), the bake, measured
MuJoCo-against-MuJoCo so no MJWarp term is mixed in, 8 states, 200 px:

| quantity | value |
|---|---:|
| the two button geoms | 36 px/frame = **0.090 %** of the frame |
| pixels changed by the bake | 55.5 px/frame = **0.139 %** |
| max abs / mean abs on the buttons | 37/255 · 21.6/255 |
| mean abs over the whole frame | **0.023/255** |
| **`interior_exact` (baked vs official)** | **1.00000** |

The residual touches **no interior pixel**: every changed pixel fails the 3x3-flat test. Fetch's
accepted alpha `parity_note` is 1.2–1.5 % of pixels; this is a tenth of that. Recorded alongside, so
the decision is not made on the residual alone: **(b) is not a call to `bake_primitive_textures`** —
the shipped helper refuses this material twice (non-2-D texture; a material worn by MESH geoms is
classified `mixed`) — so it is a change to a shared helper, ~a dozen lines plus tests; **(a)** is ~3
lines in a third-party `wp.func` whose real cost is the delivery route, and it is the only option
that costs no pixels and fixes the general case; **(c)** is 39/40 coverage and reversible. The
escalation stays open.

## 13. Two lessons the coordinator asked to be recorded

* **PyPI `stable-worldmodel==0.0.6` is not the source checkout.** Same version string, different
  code: the PyPI build's `data/dataset.py` imports `h5py`, the checkout's `data/__init__.py` imports
  its own `formats.lance` and never does. A cluster number taken against the wheel would not have
  measured the registration the local census measured. The checkout is rsynced to `$SCR/se7/swm-src`
  and put on `PYTHONPATH`, the wheel uninstalled, and the three file digests confirmed identical
  (`cube_env.py` `143e5e81f424ff7a`, `scene_env.py` `d0eebdce0131f86e`, `envs/__init__.py`
  `118a262677a647c6`).
* **llvmpipe versus Metal changed a verdict, not a decimal.** macOS GLFW gets a Metal-accelerated GL
  running the official renderer at 159.9 f/s; the certification `batch` node gets Mesa llvmpipe at
  **6.31 f/s**, and `S_cpu` is defined against the latter. Against the laptop a 38 f/s span
  rasteriser scores `S_cpu` 0.24 and fails every tier; against the certification host it scores 6.07
  and clears `CPU_FASTER`. The phase-0 CPU verdict written from laptop numbers was wrong, and only
  the Oscar leg caught it.
