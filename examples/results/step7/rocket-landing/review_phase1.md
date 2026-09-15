# Step 7 · W6 `rocket-landing` — independent review (Phase 1)

**Reviewer:** independent (did not write this code).
**Branch:** `step7/rocket-landing`, worktree `/Users/samib/stable-envs-step7-rocket-landing`,
**HEAD `92d9b28`** (reviewed at `6e50f34` and re-checked at `92d9b28`; the extra commit touches only
`scratch/step7/rocket-landing/measure_tier.py`), baseline `76dc5116f432e61dd47fba8f9e879cf9212c8e46`.
**Machine:** Apple M5, no NVIDIA GPU. venvs `/Users/samib/se6-venv` (repo tests) and
`/Users/samib/se7-rocket-venv` (PyFlyt 0.29.0, pybullet 3.2.7, swm 0.1.1, gymnasium 1.3.0).
No file in the worktree was edited; nothing committed; nothing pushed; no SLURM job submitted.

---

## VERDICT: **APPROVE-WITH-CHANGES**

The engineering is right and, where I could check it without a GPU, it is right for the reasons the
author gives — I reproduced the reference recipe, the lag law, the camera, the link frames, the colour
law, the hull silhouette and the TinyRenderer shading law independently, and I proved the default
nvdiff path is bit-identical to the baseline for all six locomotion models. Two defects are
**demonstrated, not suspected**, and both must be fixed before this is merged or before the M4 GPU job
is trusted:

* **R1 (blocking the certification run).** `PFRocketLanding.reference_render` raises
  `AssertionError` on **36 of the 64 states `env.sample_states(64)` produces at the very first draw
  seed `measure_tier.py` uses**. `check_visual_fidelity` — the function `measure_tier.py` calls — is
  therefore guaranteed to abort, and because `json.dump(R, ...)` is the *last* statement of
  `measure_tier.py`, the job writes **no output file at all**. Jobs 6395008/6395009 cannot produce the
  numbers that are supposed to replace the UNMEASURED declaration.
* **R2 (unit-suite regression + a false statement in `findings.md`).** The branch takes
  `tests/unit` from **2 failures to 3** (from **1 to 2** with `test_examples.py` deselected, as
  briefed). `findings.md` states the opposite: *"After M3 the suite is 820 passed, 37 skipped, 2 failed
  — the same two pre-existing failures and nothing of mine"*. Both new failures are caused by
  `findings.md` lines 68–69, i.e. by the sentence that claims the defects are fixed.

Everything else in the checklist passes, several items with independent reproduction.

---

## Required changes

### RC-1 (blocking) — `stable_envs/envs/rocket_landing.py`, `_reference_frame` (≈ line 451)

The pose assertion compares a **float32** state's quaternion against Bullet's **renormalised** one.
`sample_states` returns float32 and `state_dim` is float32 by contract, so the quaternion the caller
hands in has `‖q‖ = 1 ± ~3e-8`, Bullet returns a unit quaternion, and `min(|q_got−q|, |q_got+q|)` is
~1e-8–3e-8 — larger than the hard-coded `1e-8`. This is **not** a physics failure: the velocity fixed
point converges to 5e-14 and the *position* lands to ~1e-16 in every case I logged.

Demonstrated (`se7-rocket-venv`):

```
env = object.__new__(PFRocketLanding); env.rng = np.random.default_rng(1000)   # measure_tier draw 0
st  = env.sample_states(64)            # dtype float32, max ‖q‖ error 2.96e-08
-> per-state: 28 ok, 36 AssertionError out of 64
-> the batch call raises: "the reference did not reach the requested pose (max 1.053e-08)"
-> the SAME 64 states with q renormalised in float64: batch call OK
```

and over the frozen corpus (100 of the 200 states, both precisions):

| states fed to the recipe | pose-assertion failures | frames byte-identical | max differing px |
|---|---|---|---|
| `obs30[:, :13]` (float64) | **0 / 100** | **99 / 100** | 1 |
| `state13` (float32, what the env holds) | **56 / 100** | 3 / 100 | 79 |

Minimum fix: normalise the quaternion in float64 inside `lagged_pose` (or at the top of
`_reference_frame`) and make the tolerance float32-aware — compare the *rotation*
(`1 − |q·q_got|`) with a bound derived from float32 eps, not an absolute 1e-8 on raw components.
`quat_to_mat` already normalises internally, so the render side is unaffected and this is not a
pixel change for the cell; it is a change to the reference bridge only.

Please also wrap the `check_visual_fidelity` loop in `measure_tier.py` the way the mandatory checks
are wrapped, and write `R` to disk before the fidelity stage, so one raising state cannot cost a whole
GPU job.

**Second-order item to record, not necessarily to fix:** the 199/200 byte-identity headline was
measured on float64 `obs30` states, and `findings.md` says *"This is `PFRocketLanding.reference_render`"*.
With the env's own float32 states the same recipe reproduces 3/100 exactly (up to 79 differing px of
230,400). That is inherent float32 state quantisation, not a recipe defect, but the claim as written
overstates what the shipped function does with the states the suite actually feeds it, and the corpus
replay in `measure_tier.py` (which uses `state13`) will carry that term inside its "renderer" residual.

### RC-2 (blocking) — `examples/results/step7/rocket-landing/findings.md` lines 68–69

The sentence describing the fix re-cites the two bad paths verbatim, so the citation extractor picks
them up again:

```
68: files (a `scratch/step7/rocket-landing/*` glob that matched gitignored `.log` output, and a
69: `scratch/.../oscar_fidelity.sbatch` that was never written) are fixed.
```

Measured, same interpreter (`/Users/samib/se6-venv`), `-p no:randomly`:

| run | baseline 76dc511 | branch |
|---|---|---|
| `tests/unit --deselect tests/unit/test_examples.py` | **1 failed**, 713 passed, 6 skipped | **2 failed**, 741 passed, 7 skipped |
| `tests/unit` (full) | **2 failed**, 823 passed, 6 skipped | **3 failed**, 851 passed, 7 skipped |

* `test_every_cited_scratch_path_resolves`: baseline lists **9** unresolved paths; branch lists **10**
  — the extra one is `scratch/.../oscar_fidelity.sbatch`, *cited at findings.md:69*.
* `test_no_cited_probe_is_excluded_by_gitignore`: **passes on baseline, fails on branch**, listing six
  gitignored `.log` files matched by the `scratch/step7/rocket-landing/*` glob *at findings.md:68*.

Fix: name the defects without writing paths the extractor will match (e.g. "a `*` glob under the
workstream's scratch directory", "a sbatch file name that was never written"), and correct the
suite totals in the same paragraph — the recorded "820 passed, 37 skipped, 2 failed … nothing of
mine" does not match any run I could reproduce.

### RC-3 (non-blocking, before certification) — `scratch/step7/rocket-landing/measure_tier.py`

Against the frozen §6 definitions in `docs/research/certification.md`:

| frozen rule | implemented? |
|---|---|
| `R = stable-envs GPU fps / official CPU renderer fps`, same host, same job | **yes** — `official` is `u.render()` on the live swm env, timed warm in the same process (§6.3) |
| `cores = GPU fps / one-core JPEG-q90 decode fps` of this env's own frame, from memory | **yes** — PIL q90 encode of the corpus frames, warm, decode timed; `jpeg_bytes_mean` recorded |
| class S ladder `R<1 / 1≤R<2 / 2≤R<4 / R≥4 ∧ cores≥1 / R≥40 ∧ cores≥10` | **yes**, transcribed exactly, including the `BELOW_G4` cap and qualifier |
| `BATCH_SCALES: fps(4096) ≥ 20·fps(1)` | **yes** |
| `NO_LARGE_BATCH_REGRESSION: fps(4096) ≥ 0.9·fps(256)` | **yes** |
| mandatory batch sweep includes 4096 | **yes** — default `[1, 16, 64, 256, 1024, 4096]` |
| **"No env is ever certified on a B = 1 number" (§6.2)** | **NO** — `best_b = max(a.batches, key=thr)` includes `1` in the candidate set. Exclude `1` from the argmax. |
| **`BATCH_SCALES` is "required for any tier above `GPU_PARITY`"** (§6.7) | **NO** — computed and reported in `batch_checks`, but `tier` is not capped at `GPU_PARITY` when it fails. Cap it. |
| **`HOST_BOUND` qualifier (`cpu_frac ≥ 1.5` at the certified batch)** (§6.7) | **NO** — `cpu_frac` is never measured. This cell does per-row float64 kinematics on the host inside `render()` and chunks in a Python loop, so it is exactly the shape that earns `HOST_BOUND`; it must be measured, not assumed absent. |

Step-6 `profile_note.py` conventions: the script is host-tagged (`<host>.json`) and records versions,
GPU name and python — the important part. It does **not** reuse `scratch/step6/profile_note.py` and
does not record `threads`, `load_average`, `SLURM_JOB_ID` or the operator attribution that
AGENT_COMMON asks the Step-6 harness to produce ("reuse it and extend it rather than writing a
parallel harness"). It also does not pin `OMP_NUM_THREADS=1` for the one-core decode comparator.
Recommend: add `threads`, `load_average`, `slurm_job` and `cpu_frac` to `R`, and pin threads around
the decode timing. The `throughput_vs` block carrying the **local M5** numbers (34.25 / 1190.9 f/s)
next to the same-node ones is a cross-host comparison waiting to be quoted — label it explicitly as
"local, not certification" or drop it.

### RC-4 (non-blocking) — declaration honesty

`visual_fidelity` carries real-looking numbers (`mean_abs 3.0`, `within8_frac 0.95`, `ssim 0.95`,
`mean_abs_max 6.0`, `within8_min 0.85`, `ssim_min 0.90`). The framework leaves no alternative:
`check_fidelity_declaration` requires all seven keys to be *present* for a visual-tier class to
register. The author put the disclosure in `measured_at` ("UNMEASURED — provisional Phase-0 estimate,
not a measurement…") and it is surfaced by `describe()`. That is about as honest as the contract
allows, and `reference_versions {PyFlyt 0.29.0, pybullet 3.2.7}` is complete and well-formed
(validator run directly: `check_fidelity_declaration` and `check_decoder_declaration` both pass).
However the validator only checks *presence*, so `"mean_abs": None` would also register — which would
make it impossible for a consumer reading `describe()["visual_fidelity"]["mean_abs"]` to mistake a
guess for a measurement. Suggest either that, or coordinator sign-off that the numeric placeholders
are acceptable until the 3090 run. Note that no local test exercises this cell's declaration at all:
`test_declarations.py` iterates `se.REGISTRY`, and the cell is not registered without nvdiffrast.

### RC-5 (non-blocking) — `missing_backend()` is incomplete

Registration is gated on nvdiffrast only, but `_nvd()` also needs `scipy` (`ConvexHull` in
`tiny_cylinder`), `pybullet_data` and `PIL` (`checker_texture` reads `checker_blue.png` from the
installed `pybullet_data`). On a machine with nvdiffrast but without those, the env registers and then
dies at construction with a bare `ImportError` instead of reporting a missing capability. The ground
texture is also an *installed-package asset*, not a committed one, so its provenance rests entirely on
the `pybullet 3.2.7` declaration — worth one sentence in the docstring.

---

## What I verified, and how

### 1 · nvdiff generic extensions — defaults unchanged (PASS, with a strong positive result)

**Line-by-line diff of the default path** (`git diff 76dc5116..HEAD -- stable_envs/render/nvdiff.py`):

* `BatchRenderer.__init__`: `m` becomes `m=None`; every new option is **keyword-only** and appended
  after `aniso`. No positional reordering, so every existing call site is unaffected.
* `render()`: `pos = offset_clip(pos, R, self.sample_offset)` — `offset_clip` returns **the input
  object** (`is pos`, pinned by a test) when the offset is `(0,0)`; no arithmetic is performed.
* `base = gather_material(S.mat_rgb if mat_rgb is None else mat_rgb, gid)` — for a 2-D table this is
  literally `mat[gid]`, the previous expression.
* `_texture()`: the `nearest` branch is inserted **before** the existing `if self.mipmap and
  mipmap_ok: … else: …` chain; with `tex_filter="linear"` the chain is byte-for-byte the old one.
* `_shade` is untouched; `shade_tiny` is reached only under `shading="tiny"`.
* `prepare_texture` was split into `prepare_texture_array`. The two added guards are inert on the
  MuJoCo path: `texture_array()` returns float32-scaled data (never uint8 at that point) and always
  `a[..., :3]` or `np.repeat(a,3,-1)` (never 4 channels). `np.ascontiguousarray` is a layout change,
  not a numeric one.
* `Scene.__init__` → `_assemble` is a pure extraction: same tensors, same dtypes (`face` int32,
  `vgeom`/`fgeom` int64), same order (`_assemble` → `_appearance` → `_sparse`).

**Empirical bit-identity (my own test, not the author's).** I built `nvdiff.Scene` on the CPU from the
six locomotion `MjModel`s under both checkouts in the same interpreter and hashed every member:

```
scene_identity.py {branch|baseline}  ·  HalfCheetah/Hopper/Walker2d/Ant/Humanoid/Swimmer-v5
hashed: vert norm uv vgeom face fgeom mat_rgb alpha spec shininess emission texid mjgeom
        S4 S3 S3p (densified) geoms textures{tex,cube,resampled,mipmap_ok,shape} skybox
-> "SCENE/TEXTURE STATE IDENTICAL across baseline and branch"   (4414 bytes of digests, diff clean)
```

So everything the rasterizer consumes — geometry, normals, UVs, index buffers and dtypes, the three
sparse transforms, the appearance table, and every texture pyramid including the skybox and the
pow2-resample notes — is unchanged. Only `dr.rasterize`/`dr.texture` themselves are untestable here.

**`from_triangles` / `_assemble` correctness.** Winding is the caller's (the rocket cell winds its hull
outward and asserts it in `test_tiny_cylinder_is_bullets_support_hull`); no double-sided handling —
nvdiffrast rasterizes both faces and the depth test decides, same as the MjModel path. Per-vertex
normals/UVs, per-face geom id derived from `vgeom[face[:,0]]` **with a check that all three corners
agree** (`ValueError("every triangle must lie in one geom")`, tested). `face` → int32. Validation of
`vert/norm/vert_geom` lengths and of `vgeom` range is present. `clip()` is checked against a dense
transform in `test_from_triangles_clip_matches_a_dense_transform`.
Two gaps worth a comment (not defects here): `from_triangles` does **not** drop zero-alpha geoms (the
MjModel path does, `skip_alpha_zero`), and it cannot carry a cube texture (`texcube_np` is forced
False and `prepare_texture_array` is called without `cube=`). Neither is exercised by the rocket cell
— TinyRenderer has no transparency and the cell's only texture is 2-D.

**`gather_material` shapes.** `(G,3)` → `mat[gid]`; `(B,G,3)` → `mat[arange(B).view(-1,1,1), gid]`,
which broadcasts `(B,1,1)` against `(B,H,W)` correctly. Pinned by `test_gather_material_shared_and_per_row`
including the row-swap case.

**TinyShading vs the measured TinyRenderer.** The law is
`out = trunc(255 · base · (0.6 + 0.35·max(0, n·l) + 0.05·max(0, r_z)²))` with
`r = 2(n·l)n − l` built from the **unclamped** `n·l`, `l = normalize(−50, 30, 100)` (world-fixed),
`r_z` the **world** z. I recomputed both back-face cases by hand and then through the code:

```
n = (0,0,−1):      n·l = −0.86387  r_z = 0.86387  term = 0.637313  -> 162 ; clamped law would give 153
n = R_x(150°)·ẑ:   n·l = −0.87771  r_z = 0.65637  term = 0.621541  -> 158 ; clamped law would give 153
shade_tiny(...)                                                    -> 162 and 158
```

`phase0/tiny_law4_backface.json` records the **measured** TinyRenderer pixels 162 and 158 alongside
`pred_ambient_only: 153` and `pred_unclamped_r: 162/158`. So the implemented law **is** the measured
one, the discrepancy is explained by the mechanism (the reflection uses the unclamped `n·l`), and the
explanation is recorded in the JSON, in the unit-test comment, in the `TinyShading` docstring and in
the cell's module docstring. `tiny_law2.json:frame` discriminates world-fixed from head-lamp across
three eye positions (measured 255 / 52 / 219 matching `pred_if_world`, refuting `pred_if_view`), and
`specular.tilts` pins exponent 2 (`190 = int(255·0.8639²)`). Bullet normalises `r`; with unit `n`
and `l` the un-normalised form is already unit, so omitting it is exact. The `.clamp_min(0)` on `r_z`
is Bullet's `b3Max(r.z, 0)`. `quant="trunc"` matches Bullet's `int(...)`; clamping after the texture
multiply matches Bullet's single `b3Min(..., 255)` on the product.
*Not verifiable here:* that these constants are what `TinyRendererVisualShapeConverter` literally
contains — I judged the law only against the author's measurements on real PyBullet frames, which is
the stronger evidence anyway.

**Nearest sampling + corner offset.** `dr.texture(..., filter_mode="nearest", boundary_mode="wrap")`;
`wrap` is required for the ground's 0–100 UV range and is present. `offset_clip` scales by `w` so the
shift is exactly a screen-space translation after the divide, and is pinned at 0.5 px
(`test_offset_clip_moves_by_exactly_the_requested_pixels`). The 0.5 px convention is supported by
`tiny_law2.json:row_rule` (13/13 rows under the bottom-corner rule, 3/13 under the centre rule).

**Tests run.** The two files are `tests/unit/test_nvdiff_triangles.py` and `tests/unit/test_rocket_landing.py`
(there is no `tests/unit/test_nvdiff.py`; `test_nvdiff_texgen.py` is pre-existing and untouched):

```
/Users/samib/se6-venv/bin/python -m pytest tests/unit/test_rocket_landing.py tests/unit/test_nvdiff_triangles.py -q -p no:randomly
-> 29 passed, 1 skipped   (skip: test_checker_texture_is_flipped_for_nvdiffrast, no pybullet_data in se6-venv)
```

`se7-rocket-venv` has no pytest, so the rocket test file was run under `se6-venv`; the one test it
skips there is the `pybullet_data` one, which I covered separately by reading the real asset through
the rocket venv.

### 2 · Reference corpus and recipe (M1) — PASS

* **Stepped, not teleported.** `build_reference_corpus.py` runs `env.step(a)` in a loop over 24
  episodes (seeds 1000–1023), records the frame `u.render()` produced by that step, and only then does
  the trick validation on a `saveState`/`restoreState` copy. 200 states, 24 episodes, heights 1.9–489 m.
* **Byte-identical regeneration.** Re-running the committed recipe against the committed frames from
  a **fresh process and a fresh `reference_env`**: 100 corpus states at float64 → **99/100
  byte-identical, 1 frame off by 1 px, 0 pose-assertion failures**. This independently reproduces the
  committed `reference_trick_validation.json` (`exact 199/200, max_diff_px 1`, the single non-exact
  frame being index 124, which is in my sample). The one off-by-1-px frame **is** recorded, in the
  JSON and in `findings.md`.
* **Lag law on a fresh rollout** (seeds 4242/4243, not in the corpus; 16 sampled steps):

```
pos_prev vs PyBullet's cached link-0 position:   max 3.5e-18
q_prev   vs PyBullet's cached link-0 quaternion: max 2.2e-16
obs[3:7] vs getBasePositionAndOrientation:       max 1.1e-16
R(q)·obs[0:3], R(q)·obs[7:10] vs getBaseVelocity: max 3.6e-15
```

  So `pos − v·dt` and `exp(−ω·dt)·q` with **world** velocities recovered from the body-frame obs is
  exact, on data the author never saw.
* **Corpus size is modest:** `reference/` 1.4 MB, `phase0/` 1.7 MB, whole branch +7,194 lines.
* Note for the record: `meta.json:validation.trick_*` (max 9,054 differing px, only 42/200 exact) is
  the **naive** teleport measured during the M1 build, superseded by the fixed-point recipe in
  `reference_trick_validation.json`. The two files are easy to mistake for each other; a one-line
  note in `meta.json` would help.

### 3 · The cell (M3) — PASS on contract, geometry, camera and colour; see RC-1/RC-5

* **State contract.** `swm/PFRocketLanding-v0` observation is `Box(..., (30,), float64)`; the first 13
  entries are `ang_vel_body(3), quat_xyzw(4), lin_vel_body(3), lin_pos(3)` — confirmed against
  PyBullet directly (numbers above). `obs[:13]` is exactly what swm exposes, and the frame is a
  function of those 13 plus the style leaves: `geom_poses` uses only the 13, `camera_pose` uses only
  the lagged pose, `material_table` uses only the style.
* **Style leaves.** 5 RGBs + flame bit, 16 columns, `gymnasium.spaces.Dict` alphabetical layout,
  addressed by path everywhere (`style_layout()`, `corpus_style_to_env`, `_reference_apply_style`) —
  I checked no code assumes the corpus column order.
* **`sky_color` is genuinely dead.** A grep over the whole installed `stable_worldmodel` package finds
  `sky_color` at exactly one line — its own declaration in the variation space
  (`pyflyt_rocketlanding.py:127`). Never read. Correctly not offered.
* **The colour law is right, and the way it is right is subtle.** swm's `_modify_rocket_urdf` writes
  3-decimal strings **and comments out `booster_color`** — but `reset()` then overrides the rocket with
  `changeVisualShape` at full `value/255` precision, applying `booster_color` to the `booster` link by
  link-name match. The pad is *not* overridden and keeps the URDF's `f"{v:.3f}"` rounding. The cell's
  `material_table` reproduces exactly that (`/255` for the rocket, 3-decimal for the pad, white fuel
  tank, white ground material), which is why `meta.json:rgba_err` is 3e-8 and not 5e-4.
* **Default style = the official default appearance.** A plain `reset(seed=0)` with **no** variation
  request, `getVisualShapeData` on all 11 rocket links + the pad vs `material_table(_default_style_row())`:
  **max RGB error 1.19e-08** (float32). URDF yellow (255,204,0) / grey (51,51,51) / black (0,0,0),
  white pad, unignited flame — all confirmed.
* **`device="cpu"`.** Raises `NotImplementedError` naming the missing capability ("needs a CPU
  textured-triangle rasteriser, which this repo does not have (cpu3d/raycast3d cast MuJoCo
  primitives)"), **before** `super().__init__`, and there is no fallback anywhere. The measured
  alternative is cited (`cpu_raster_probe.json`: 1.92 f/s at 480 px, 1 thread, vs the 171.25 f/s the
  `S_cpu ≥ 5` bar needs and 1190.9 f/s one-core decode — `S_cpu 0.056`, `cores_cpu 0.0016`). The test
  that asserts the raise is skipped on this machine (it is gated on nvdiffrast being importable), so
  the raise is verified by reading, not by execution.
* **Gate-F shape.** `render_resolution` default is `(480, 480)`; `u.render()` returns `(480, 480, 3)`
  uint8. `default_size = 480`, `default_supersample = 1`, `background = white` — corner pixels of the
  frozen frames are `(255,255,255)` in 94.6 % of corners, consistent with TinyRenderer's white clear
  and the cell's `skybox=False` flat background.
* **Geometry, independently reproduced.** I re-ran `validate_env_math.py` with its output redirected
  out of the worktree; **every committed number reproduces bit-for-bit**:

```
euler_from_quat_max_err              4.6629367034256575e-15   (vs p.getEulerFromQuaternion, 500 quats)
quat_from_euler_max_err              2.220446049250313e-16
quat_to_mat_max_err                  6.661338147750939e-16
camera_view_and_proj_max_err         0.0004438460610387551    (200 attitudes, vs p.computeViewMatrix
                                                               and p.computeProjectionMatrixFOV)
link_frames_max_err                  1.8978016314008528e-06   (vs getLinkState on the real rocket.urdf)
visual_local_frames_identity         true
hull_silhouette_iou_vs_tinyrenderer  0.9948576454932977       (tiny 132480 px, ours 132862, xor 684)
hull_vertices 42   hull_triangles 80
```

  The silhouette check is a real one: it renders a PyBullet cylinder through TinyRenderer at 512 px
  and compares the segmentation mask with a polygon fill of the author's hull. 684 xor pixels on a
  ~205 px-radius disc is a sub-pixel band. Caveat: it is a **top-down** view, so it tests the cap
  polygon and the margin, not the side silhouette; the side is covered only by the unit test's bounds
  assertions (`r ∈ [R, R+margin]`, `|z| ∈ [L/2, L/2+margin]`, normals = normalised positions).
* **The `btShapeHull` claim.** The construction matches Bullet's: the 2×32 ring interleaved
  `(top_i, bottom_i)` as `btConvexHullShape` stores it, support point per direction with
  first-maximum tie-breaking, `+ margin·d` (`localGetSupportingVertex` adds the margin along the
  normalised direction), the 42 `sUnitSpherePoints` at `highres=0`, normals = normalised position (as
  `TinyRendererVisualShapeConverter` sets them). All 42 directions land on distinct points → 42
  vertices, 80 triangles, which is the Euler characteristic of a closed convex triangulation
  (2·42−4 = 80). **What I cannot check locally:** that Bullet's `HullLibrary::CreateConvexHull` does
  not itself move the vertices (it normalises and merges within `mNormalEpsilon`). The author declares
  only the *triangulation* of the planar caps as a residual, not possible vertex motion. The 0.9949
  IoU against real TinyRenderer is good evidence that any motion is sub-pixel, but the declaration
  should say "vertex positions and cap triangulation are scipy's/Bullet's respectively, verified to
  IoU 0.995 on one pose" rather than implying the vertex set is exact.
* **Camera.** `camera_pose` reproduces PyFlyt's recipe (link-0 position + `R·(−1,0,3)`, Euler with
  +30° pitch, up = that rotation's z, target = the link origin, `gluLookAt`) to 4.4e-4 against
  PyBullet's own `computeViewMatrix` over 200 random attitudes, and the projection matches
  `computeProjectionMatrixFOV(90, 1, 0.1, 255)` to 6e-8. Against PyFlyt's **live** `camera.view_mat`
  on my fresh rollouts the max element error was 1.009e-3, all of it in the 350 m translation column
  (relative 2.4e-6, i.e. float32 — PyFlyt stores the matrix as float32). At 480 px / 90° one pixel is
  3.3 mrad, and this is ~0.3 mrad, so it is sub-pixel; but note the committed
  `camera_view_and_proj_max_err < 1e-3` assertion in `test_camera_is_pyflyts_rocket_camera` is a check
  on a stored JSON value, and my live measurement (1.009e-3) sits just the wrong side of that bound.
  Worth converting the residual into a stated float32 tolerance rather than a hair-thin constant.
* **No per-env code in `stable_envs/render/`.** Confirmed: the only "rocket" strings in `nvdiff.py`
  are two citations of the W6 probe paths in docstrings. No PyFlyt/PyBullet/cylinder/hull logic. The
  cell owns `tiny_cylinder`, `build_scene_arrays`, `checker_texture`, `camera_pose`, `material_table`.
* **`_NVD_CACHE`** is the same module-level, key-bounded, clear-at-max pattern
  `mujoco_locomotion.py` already uses (`_NVD_CACHE`/`_NVD_CACHE_MAX`), keyed by `(size, ss, device)`
  and holding only static geometry — not a hidden state cache, no batch/order dependence.

### 4 · Tests — see RC-2

Commands and counts are in RC-2. The two pre-existing failures on baseline are
`test_examples.py::…[robocasa_visual/render_mjwarp.py]` and
`test_repo_contracts.py::test_every_cited_scratch_path_resolves` (9 Step-6 Oscar-scratch citations).
The branch keeps both and adds `test_no_cited_probe_is_excluded_by_gitignore` plus a 10th unresolved
citation.

### 5 · Hygiene — PASS

```
git log --format=%B 76dc5116..HEAD | grep -ci -E "co-authored-by|generated with"   ->  0
git log --format='%an <%ae>' 76dc5116..HEAD | sort -u   ->  sami-bg <samibghanem@gmail.com>
git diff --name-only 76dc5116..HEAD -- examples/results/step4 step5 step6           ->  0 files
non-step7 files touched: stable_envs/envs/rocket_landing.py, stable_envs/render/nvdiff.py,
                         tests/unit/test_nvdiff_triangles.py, tests/unit/test_rocket_landing.py
reference/ 1.4 MB · phase0/ 1.7 MB · 67 files, +7,194 / −19
```

### 6 · Policy — PASS

Honest device selection (CPU raises, never falls back, with the measured feasibility cited);
the capability went into the shared module generically (a triangle-soup `Scene`, a per-row material
gather, a named alternative shading law, a texture filter mode, a sample offset — none of them
mentioning a rocket) rather than into an env hack; no fidelity ceiling is tuned, and the ones present
are marked UNMEASURED with the 3090 plan attached (see RC-4 for the residual concern). The Step-6
addendum ("backend implementation may differ by device, renderer semantics may not") is respected:
this is a different rasterizer implementing the *same* renderer semantics (TinyRenderer's camera,
geometry, law, texture sampling), declared at the visual tier rather than claimed as parity.

---

## What only the cluster run can verify

1. **That the six locomotion nvdiff cells are byte-identical.** I proved every input to the rasterizer
   is identical; `dr.rasterize`, `dr.interpolate` and `dr.texture` are CUDA-only.
   `scratch/step7/rocket-landing/nvdiff_identity.py` is the right instrument (same job, same card,
   `PYTHONPATH=baseline` vs `PYTHONPATH=branch`, sha256 per cell, per-pixel diff on mismatch).
   **This replay is the gate on M2 and it has not run yet.**
2. **That `PFRocketLanding` renders at all.** The cell is the first caller to combine `mipmap=False`
   (hence `grad_db=False`) with a textured scene *through `BatchRenderer`*. `scratch/nvdiff/probe_uvcheck.py`
   already ran that combination on a GPU (it sets `br.mipmap = False` post-construction and is cited in
   `docs/research/nvdiffrast.md`), so I expect it to work — but it is the first thing the GPU job
   should print.
3. **Every fidelity number**: `check_visual_fidelity` (4 draws × 64 states),
   `check_visual_fidelity_under_variation`, the 200-frame corpus replay, and therefore whether the
   declared `mean_abs 3.0 / within8 0.95 / ssim 0.95` survive. **Blocked by RC-1 today.**
4. **Every performance number**: `R`, `cores`, `BATCH_SCALES`, `NO_LARGE_BATCH_REGRESSION`, the tier,
   and `cpu_frac`/`HOST_BOUND` once RC-3 adds it. Also whether B = 4096 fits on a 24 GB 3090 given the
   ~55 MB/world footprint and the 2.83 GB uint8 output at 480 px.
5. **Mandatory conformance**: batch invariance, batch-size invariance, determinism,
   `check_state_dims_matter` — in particular that the six velocity dimensions, whose only pictorial
   effect is a single 1/240 s lag, all move pixels.
6. **The sub-pixel geometry claims at render time**: the 0.5 px corner offset, the nearest-sampled
   checker with `wrap`, the cap-triangulation residual, and the 61 px flame.
7. **That the reference env and the nvdiffrast context coexist** in one process (PyBullet DIRECT +
   EGL + a CUDA raster context) for the ~57 s numba JIT plus 4×64 reference renders.
