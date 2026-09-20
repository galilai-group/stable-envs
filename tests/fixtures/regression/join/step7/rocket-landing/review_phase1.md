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

---

# Appendix — formal re-review at HEAD `13ca91b` (2026-09-15)

Single commit `13ca91b` on top of `92d9b28`. Diff scope confirmed: `examples/results/.../findings.md`,
`scratch/step7/rocket-landing/measure_tier.py`, `stable_envs/envs/rocket_landing.py`,
`tests/unit/test_rocket_landing.py`. **`stable_envs/render/nvdiff.py` is byte-identical to `92d9b28`**
(`git diff --quiet 92d9b28..13ca91b -- stable_envs/render/nvdiff.py` → clean), so the bit-identity
result I established for the six locomotion `Scene`s still stands unmodified. GPU job 6395516 is
queued on this tree.

## VERDICT: **APPROVE-WITH-CHANGES (2 remaining defects, both RC-3-class)**

**RC-1 and RC-2 — the two hard blockers — are demonstrably fixed.** RC-4 and RC-5 are fixed. RC-3 is
fixed except for two items, both confined to `scratch/.../measure_tier.py` plus one sentence of
`findings.md`. Nothing in `stable_envs/` is defective any more, so the library change is approvable;
per Sami's ruling that RC-3 lands "before any performance interpretation", the two residuals must
land before the numbers from 6395516 are read. **Job 6395516 need not be cancelled** — the tier
arithmetic is correct and the numbers it produces will be valid; D1 only costs provenance (record the
SHA out of band) and D2 only produces a duplicate row.

### RC-1 — FIXED (re-ran my own probe, not the author's)

`se7-rocket-venv`, the exact seed `measure_tier.py` uses for draw 0:

```
env.rng = np.random.default_rng(1000); st = env.sample_states(64)
  dtype float32, max ‖q‖−1 = 2.9646e-08          (the input is still off-unit — the fix is downstream)
  per-state:  64 ok, 0 AssertionError / 64        (was 28 ok / 36 raised at 92d9b28)
  batch call: OK, (64, 480, 480, 3)               (was: raised)
```

and the frames are not merely produced, they are **the right frames**:

```
float32 states vs the same states renormalised in float64:  64/64 byte-identical, max diff 0 px
```

Corpus, shipped `reference_render`, 50 states (every 4th, incl. index 124 — the previous 1-px outlier):

| states | assertion failures | byte-identical | max differing px |
|---|---|---|---|
| `obs30[:, :13]` float64 | 0 / 50 | **50 / 50** | 0 |
| `state13` float32 | **0 / 50** (was 56/100) | 1 / 50 | 79 |

So physically converged states are no longer rejected, and the float32 path now reproduces the
float64 picture exactly for sampled states. The remaining float32-vs-frozen-frame gap is the
state-quantisation term against frames recorded from float64 physics — inherent, correctly reported
alongside, and the subject of D2 below.

The tolerance is sound: `1 − |q·q'| ≤ 1e-9` is θ ≤ 8.9e-5 rad = **0.027 px** at 480 px / 90° fov, and
`pos ≤ 1e-6 m` is ~1e-4 px at the 3 m camera offset. Both are far inside a pixel, so the check still
does the job it was written for. `lagged_pose`'s float64 renormalisation is applied before
`quat_to_mat`/`quat_mul`, so the render side and the reference side now describe the same rotation.

**Checkpointing verified by reading the code:** `out` is computed before any work, `checkpoint()`
dumps `R` after *each* of corpus-replay (per label), mandatory checks, throughput/batch-checks, tier,
every fidelity draw, and variation fidelity; each `check_visual_fidelity` draw is wrapped in
`try/except` and a failure is recorded in `R["visual_fidelity"]["failures"]` rather than raised;
`worst`/`spread` are computed only `if vf`. A late failure can no longer erase the run.

### RC-2 — FIXED (full suite, both sides, no deselection)

```
pristine worktree at 76dc5116, /Users/samib/se6-venv, MUJOCO_GL=glfw, -p no:randomly
  baseline:  2 failed, 823 passed, 6 skipped
  branch  :  2 failed, 855 passed, 7 skipped
  the two failures are IDENTICAL on both sides:
    test_examples.py::…[robocasa_visual/render_mjwarp.py]
    test_repo_contracts.py::test_every_cited_scratch_path_resolves
```

**No new failures.** +32 passing tests, +1 skip (`pybullet_data` absent in `se6-venv`). With
`test_examples.py` deselected: baseline 1 failed / 713 passed / 6 skipped, branch 1 failed / 745
passed / 7 skipped, and the citation failure now lists exactly the baseline's **9** paths with **no
`step7` entry**; `test_no_cited_probe_is_excluded_by_gitignore` passes.

*Minor, third occurrence of the same class of error:* `findings.md` now records "branch after the
review commit: **713 passed / 1 failed / 37 skipped**". The branch adds 32 tests, so its pass count
cannot equal the baseline's 713; I measure 745 passed / 7 skipped. The load-bearing claim (one
pre-existing failure, no step7 citation) is correct and verified — the totals beside it are not.
Please quote numbers from a run that can be reproduced with the interpreter named in the same table.

### RC-3 — mostly fixed; **two defects remain**

Fixed and verified by reading the diff:

| item | status |
|---|---|
| B = 1 excluded from certified-batch selection | **fixed** — `cert = [b for b in a.batches if b != 1]`, `best_b = max(cert, …)` |
| tier capped at `GPU_PARITY` when `BATCH_SCALES` fails (§6.7) | **fixed and correct** — capped only when `R ≥ 1`, so `GPU_REGRESSION` is preserved below 1; `BATCH_SCALES_FAIL` / `LARGE_BATCH_REGRESSION` / `BELOW_G4` / `HOST_BOUND` are emitted as a `qualifiers` list |
| `cpu_frac` measured, `HOST_BOUND` at ≥ 1.5 | **fixed** — `time.process_time()/wall` over the timed loop (process-wide CPU across threads, the right definition), per batch, `cpu_frac_at_best` carried into `R["tier"]` |
| one-core JPEG comparator pinned | **fixed** — `torch.set_num_threads(1)` around the decode loop, restored after, and `decode_cpu_frac` recorded so the "one core" claim is *evidenced* rather than asserted |
| threads / load average / SLURM job id | **fixed** — in `R` and echoed into `R["tier"]` |
| no local-M5 numbers in the tier calculation | **fixed** — the `throughput_vs {official_fps_local_M5, one_core_jpeg_decode_fps_local_M5}` block is gone (`grep 34.25\|1190.9\|local_M5` → none); both comparators are same-node, same-job |
| `R` / `cores` / class-S ladder / `BATCH_SCALES` / `NO_LARGE_BATCH_REGRESSION` | unchanged and still at the frozen §6.3/§6.6/§6.7 definitions |

**D1 (remaining, RC-3) — no commit SHA and no resolved-checkout provenance.**
`measure_tier.py` records host, gpu, python, slurm_job, threads, load_average and package versions,
but **not the git commit and not the `stable_envs` module actually imported**
(`grep -n "git\|commit\|sha\|profile_note" measure_tier.py` → nothing but an unrelated comment). This
is precisely the field `scratch/step6/profile_note.py` exists to carry — it has `git_commit()` and
`stable_envs_file()`, with the comment *"each leg records which checkout it loaded; the git commit of
the harness does not answer that"* — and it is precisely the failure AGENT_COMMON warns about on
Oscar, where a stale `$SCR/stable-envs` on `main` sits next to `$SCR/se7/rocket-landing`. The
author's own `nvdiff_identity.py` gets this right (it records `pythonpath` and `stable_envs_file`);
`measure_tier.py` does not. **Fix:** import or copy `git_commit()` / `stable_envs_file()` from
`scratch/step6/profile_note.py` into `R` (and, since the run is already queued, record 6395516's
SHA by hand alongside its output).

**D2 (remaining, RC-3) — the "two corpus replays" are one measurement, and the claim beside them is
wrong.** `measure_tier.py` now renders the corpus twice, labelled `float64_obs` ("the renderer's
residual alone") and `float32_state13` ("what a consumer gets"), and `findings.md` repeats the
claim. They are the same render. Verified:

```
obs30[:, :13].astype(np.float32) is BIT-IDENTICAL to state13   (np.array_equal on the uint32 views: True,
                                                                max |diff| 0.0)
rocket_landing.py:634  s = torch.as_tensor(states, device=…, dtype=torch.float32)   # every input is cast
measure_tier.py        torch.as_tensor(st[lo:lo+16], dtype=torch.float32, …)        # both labels, explicitly
```

so both loops feed the renderer identical float32 tensors and will print identical numbers. The
inline note — *"the float32 rounding … is a STATE change of its own"* / *"the two differ only through
`lagged_pose`'s float64 renormalisation of a float32 quaternion"* — is false: the renormalisation
happens **after** the float32 cast, in both cases. The renderer's residual cannot be isolated on the
*our* side at all, because `render()` is float32 by contract.

**Fix (pick one):**
(a) drop the duplicate and label the single replay honestly as containing **both** terms, pointing at
`check_visual_fidelity` — which puts both sides at the same float32 sampled states — as the
renderer-only number; **or**
(b) keep two rows but make the second one the *reference* side: `reference_render(state13)` against
the frozen frames, which is the state-quantisation term and which I measured locally as
**1/50 byte-exact, max 79 differing px, 0 assertion failures**. Then
`frozen ↔ reference_render(state13)` is the quantisation term and
`reference_render(state13) ↔ env.render(state13)` is the renderer term, and they add up.

### RC-4 — FIXED

`visual_fidelity` is `{mean_abs: None, within8_frac: None, ssim: None, mean_abs_max: None,
within8_min: None, ssim_min: None, measured_at: "UNMEASURED: no number here is a measurement…",
reference_versions: {PyFlyt 0.29.0, pybullet 3.2.7}}`. Run directly on this machine:
`check_fidelity_declaration` and `check_decoder_declaration` both pass with the `None` values (the
validator checks presence), and the new `test_the_visual_declaration_is_complete_and_marked_unmeasured`
pins it. I grepped every consumer of the numeric keys in the repo: only
`examples/robocasa_visual/calibrate_tier.py` (robocasa-specific) and a synthetic class in
`test_rendernet_identity.py` read them, so `None` cannot propagate into an arithmetic comparison for
this cell. A consumer printing `describe()` now sees `None` next to the UNMEASURED sentence and
cannot mistake a placeholder for a measurement.

### RC-5 — FIXED

`missing_backend()` now reports all four, verified on this machine:

```
"nvdiffrast: ModuleNotFoundError: No module named 'nvdiffrast'; pybullet_data: ModuleNotFoundError: …"
```

`__init__` raises `ImportError` with that list, `register(PFRocketLanding)` is gated on it, and the
checker's provenance is tied to the `pybullet 3.2.7` declaration in the docstring. Camera tolerance is
now `3e-3` with a written float32 rationale that matches the 1.009e-3 I measured against PyFlyt's live
`view_mat`.

### Minor (not required, worth one line each)

* **Import side effect.** `missing_backend()` runs at module import, so `import stable_envs` now
  imports `scipy.spatial` and `PIL.Image` on every machine (verified: both in `sys.modules`; total
  `import stable_envs` 0.87 s here). `nvdiff.py`'s stated convention is the opposite ("imports neither
  mujoco nor nvdiffrast at module scope"). Consider gating `register` on nvdiffrast alone and probing
  the rest lazily in `_nvd()`, with `missing_backend()` still reporting all four on demand.
* **Stale artifact.** `reference_trick_validation.json` (199/200, one 1-px frame, index 124) predates
  the renormalisation; index 124 is now byte-exact in my sample, so the recipe is probably 200/200.
  Re-run `validate_reference_trick.py` or mark the JSON as pre-RC-1. The findings sentence
  understates the result rather than overstating it, so this is bookkeeping, not a correctness issue.
* `test_registered_iff_nvdiffrast_imports` now branches on `missing_backend()`, not on nvdiffrast —
  rename it.
* `cert = [b for b in a.batches if b != 1]` raises `ValueError` on `max(())` if someone passes
  `--batches 1`; guard it.

### Unchanged from the main report

Everything under "What I verified, and how" (the bit-identical `Scene` hashes across both checkouts,
the TinyRenderer law including the 162/158 back-face case, the lag law on fresh rollouts, the camera
and link frames against PyBullet, the colour law against the swm source, the hull silhouette IoU, the
default-appearance check, hygiene) was re-checked for scope and is untouched by `13ca91b`. The
"what only the cluster can verify" list is unchanged, minus item 3's blocker: `check_visual_fidelity`
can now run.

---

# Appendix 2 — formal verdict at HEAD `cf355a3` (2026-09-15)

One commit on `aa54c2c`. Six-point verification as agreed.

## VERDICT: **APPROVE** — with one operational defect (one line) that must be fixed before job 6395516's artifacts are accepted as provenanced

D1 and D2 are fixed in substance, the minors are fixed, and the full unit suite on the branch is
clean against a pristine baseline. The single remaining item does not touch `stable_envs/`, the
tests, the corpora, or any number the job computes — it makes the provenance stamp read `unknown` on
Oscar, which is the one machine it exists for.

### The remaining defect — `sha` resolves to `"unknown"` on the cluster

`measure_tier.py` reads `$ROOT/.git_sha`, falling back to `git -C $ROOT rev-parse HEAD`:

```python
sha_file = os.path.join(ROOT, ".git_sha")
sha = (open(sha_file).read().strip() if os.path.exists(sha_file)
       else os.popen(f"git -C {ROOT} rev-parse HEAD 2>/dev/null").read().strip() or "unknown")
```

**Nothing in the branch writes `.git_sha`** (`grep -rn git_sha` finds only the two *readers*:
`measure_tier.py:55` and `oscar_gpu.sbatch:34`), and AGENT_COMMON's rsync recipe copies the worktree
**`--exclude .git`**. So on `$SCR/se7/rocket-landing` there is no `.git_sha` and no `.git`, the
`rev-parse` fails with its stderr suppressed, and the fallback fires. Demonstrated by reproducing the
Oscar copy locally:

```
rsync -a --exclude .git ... -> $SCR/oscarsim       (no .git, no .git_sha)
measure_tier.py's expression evaluated against it:  sha = 'unknown'
```

Every `gate()` line, `gates.txt`, and `R["provenance"]["sha"]`/`["baseline_commit"]` would then read
`unknown`, and `oscar_gpu.sbatch:34`'s banner prints `SHA ` (empty) because `cat … 2>/dev/null` hides
the miss. `stable_envs_file` still resolves and is the stronger of the two fields, so the artifact is
not unattributable — but the field the coordinator asked for is not populated where it matters.

**Fix (one line, either place):** write the SHA at rsync time —
`git -C /Users/samib/stable-envs-step7-rocket-landing rev-parse HEAD > .git_sha` before the rsync (and
record that step in `findings.md`'s Oscar row) — or have `oscar_gpu.sbatch` fail loudly rather than
silently: `SHA=$(cat $WS/.git_sha) || { echo "no .git_sha in $WS"; exit 2; }`. Since 6395516 is
already queued, record its SHA by hand beside its output if it lands before the fix.

*Cosmetic, same block:* `baseline_commit` is set to the same value as `sha`, i.e. this branch's HEAD.
`profile_note.py` uses that name for "the commit of the checkout measured", so it is consistent — but
in a Step-7 report where `$SCR/se7/baseline` (76dc511) is a real, different thing, the name invites a
misreading. Consider `measured_commit`, or record 76dc511 explicitly beside it.

### 1 · Scope — PASS

`git diff --stat aa54c2c..cf355a3`: `findings.md`, `reference_trick_validation.json`, a new
`corpus_replay_state_quantisation_<host>.json` + `corpus_replay_state32.py`, `measure_tier.py`,
`validate_reference_trick.py`, `stable_envs/envs/rocket_landing.py`, `tests/unit/test_rocket_landing.py`.
The `stable_envs/` change is **exactly the lazy probe** (`_optional_import_errors` only) — I read the
whole diff, it is 15 lines and touches nothing else. **`stable_envs/render/nvdiff.py` is still
byte-identical to `92d9b28`** (`git diff --quiet 92d9b28..cf355a3` → clean), so the `Scene`
bit-identity result for all six locomotion models carries forward unmodified.

### 2 · D1 provenance fields — PRESENT (see the defect above for where they resolve)

`R["provenance"] = {sha, baseline_commit, stable_envs_file (realpath of the imported package), host,
slurm_job, threads, load_average, gpu}`, spread onto every `R["gates"][name]` entry and stamped on
every `GATE` line written to stdout and to `gates.txt`. `stable_envs_file` is the field
`profile_note.py` calls "the only proof of which checkout was measured", and it is correct.

### 3 · D2 — FIXED, and fixed the stronger way

The duplicate replay is gone. There is now **one** row, `corpus_replay.ours_state13_vs_frozen`,
carrying `"contains": "renderer residual + float32 state quantisation (see state_quantisation_term)"`,
with a gate line that repeats "(reported; contains both terms)"; and a **separate**
`corpus_replay.reference_state13_vs_frozen` = `reference_render(state13)` vs the frozen frames, which
is the quantisation term on its own. `check_visual_fidelity` is named in the comment as the
renderer-only number. The false sentence is gone and the replacement comment states my finding
correctly and for the right reason:

> "render() casts states to float32, so obs30[:, :13] and state13 feed bit-identical tensors and there
> is exactly one measurement here."

The author's local re-take of the quantisation term (200/200, 0 rejections, 9 exact, max 79 px, mean
11.4 differing px) is consistent with my own sample at `13ca91b` (50 states: 0 rejections, 1 exact,
max 79 px) — same tail, same worst case, and the mean is the number my 50-state sample could not give.
`findings.md` carries the corrected wording.

### 4 · Findings totals — PASS, they are now reproducible

My full-suite run on `cf355a3`: **2 failed, 855 passed, 7 skipped** — identical to `13ca91b` and to
the pristine baseline's two failures (`823 passed / 2 failed / 6 skipped`). `findings.md` now records
"reviewer's interpreter — baseline 76dc511 full suite **823 passed / 2 failed**, branch **855 passed /
2 failed / 7 skipped**, the same two failures; branch with `test_examples` deselected **745 passed /
7 skipped**", and separately labels the author's own Mac configuration (713/1/37) with the reason the
skip count differs. Both sets are attributed and the interpreter is named. That is the right fix.
*Nit:* an earlier sentence in the same paragraph still says the citation test covers "12 Step-6
citations"; the test prints **9**.

### 5 · Lazy probe and registration gating — PASS

```
import stable_envs                      -> 0.92 s
sys.modules leak of scipy/scipy.spatial/PIL/PIL.Image/pybullet_data:  NONE
missing_backend()  -> "nvdiffrast: ModuleNotFoundError: No module named 'nvdiffrast'; pybullet_data: not installed"
registered         -> False
gate expression    -> "if missing_backend() is None:  register(PFRocketLanding)"
```

`importlib.util.find_spec` replaces the import probe for scipy / pybullet_data / PIL; nvdiffrast keeps
its real import probe (it must, since the failure mode there is a plugin build, not a missing file).
Registration gating is unchanged and all four dependencies are still named on demand. *Nit:* the probe
now checks the top-level packages (`scipy`, `PIL`) rather than the submodules actually used
(`scipy.spatial`, `PIL.Image`); the docstring says so deliberately and a present-but-broken scipy is
pathological, so this is acceptable.

### 6 · `reference_trick_validation.json` regeneration — PASS, and better than asked

`validate_reference_trick.py` no longer validates its own private copy of the recipe: it now calls
`RL._reference_frame` and `RL.lagged_pose` directly, so **the artifact validates the shipped code
path**, and it records `"recipe": "stable_envs.envs.rocket_landing._reference_frame (renormalised,
float32-aware)"`. The regenerated numbers improve as expected from the renormalisation:
`pos_err_max` 7.1e-13 → **3.55e-15**, `q_err_max` 2.69e-11 → **1.11e-16** (now the sign-free
`1−|q·q'|`). Result stays **199/200 exact, max 1 px** at index 124.

*Observation, not a defect:* my own 50-state sample at `13ca91b` found index 124 **exact**, where the
full 200-state sequential run finds it at 1 px. The two runs differ only in what PyBullet call
preceded it (I rendered 120→124, the validator 123→124), so at least one frame's last pixel depends on
the preceding call's state. It is 1 px on 1 of 200, the committed artifact is the conservative one,
and the fidelity numbers come from `check_visual_fidelity` rather than from this file — but it is
worth one line in the note that the reference bridge is not bit-deterministic across call order for
one frame, since every fidelity number is computed through it.

### Minors from Appendix 1 — all fixed

Test renamed away from `…_iff_nvdiffrast_imports`; `max(cert)` guarded; the import side effect
eliminated; the stale validator regenerated; findings totals reproducible.

### Unchanged and still valid

Every positive finding in the main report and Appendix 1 — the bit-identical `Scene`/texture hashes
across both checkouts, the TinyRenderer shading law including the measured 162/158 back-face case, the
lag law on fresh rollouts, the camera and link frames against PyBullet, the colour law against the swm
source, the hull silhouette IoU 0.9949, the default-appearance check to 1.19e-08, and the hygiene
checks. The "what only the cluster can verify" list stands, with item 3 unblocked.

---

# Appendix 3 — final pre-merge review at HEAD `7b2e45c` (2026-09-15)

Eight commits since `cf355a3`. The substantive change is `mvp_rows`/`_mm4` in `nvdiff.Scene.clip`
(M6a `dac40f9`, after M5a `d1e61a1` → revert `283865b` → re-apply under Sami's ruling) plus the
reference bridge's contact-band flag and the now-measured declaration.

## VERDICT: **APPROVE-WITH-CHANGES**

All nine of Sami's conditions are met **on the evidence**. One defect must be fixed before the merge
gate can be read as satisfied, and it is a labelling bug in the author's own comparator, fixable from
data already committed with no new GPU job. Three GPU-conformance failures on the new cell are open
certification items that need a coordinator ruling rather than author work.

## Sami's nine conditions

| # | condition | verdict | evidence |
|---|---|---|---|
| 1 | changes are ULP-level, not semantic | **PASS** | see below |
| 2 | `max_abs_pixel_delta` reported | **PASS** | 1 / 1 / 1 / 8 / 81 / 67 |
| 3 | fraction of pixels changed | **PASS** | 3.5e-6 / 2.2e-6 / 1.1e-6 / 8.7e-6 / 3.8e-6 / 4.9e-5 |
| 4 | fraction changed by > 1 level | **PASS** | 0 / 0 / 0 / 1.4e-6 / 8.1e-7 / 5.4e-7 (0/0/0/5/3/2 px) |
| 5 | official-reference fidelity unchanged or improved | **PASS in substance** — see the judgement below; the author's comparator says otherwise on a zero-tolerance sign test | `refreeze/compare_refreeze.json` |
| 6 | all Gate-A checks other than the replaced digest PASS | **PASS** | `refreeze/gatea_branch.json`, `all_pass: true` |
| 7 | no Step-5 representative surface materially changes | **PASS** | `refreeze/step5_surfaces_compare.json`, `all_untouched: true` |
| 8 | batch-size invariance FAIL → PASS | **PASS**, decisively | probe 6407151 vs 6410105 |
| 9 | determinism / order / chunk invariance PASS | **PASS** | `gatea_branch.json` |

### (1) ULP-level, not semantic — PASS, and it is an accuracy *improvement*

I read the diff. `mvp_rows` computes **the same product with the same association** —
`(proj @ view) @ M`, exactly as before — but in float64 via four unrolled elementwise products
(`_mm4`) and rounds **once** to `M.dtype`. So the new value is the correctly-rounded double-precision
product; the old one was a float32 cuBLAS batched GEMM. The new path is strictly *closer to the exact
value*, not a different rendering rule: no change to geometry, camera, visibility, materials or
lighting. Shapes check out for both `view (4,4)` and `view (B,4,4)` (`vp[None,None]` / `vp[:,None]`
broadcast against `M (B,G,4,4)`). The code is generic — no env reference anywhere in `mvp_rows`,
`_mm4` or `clip`. Cost is negligible (B·G 4×4 products; ~3.4 MFLOP fp64 at B=4096).

That the change *improves* fidelity on three cells and worsens it by ~1e-6 on three is exactly what
removing a rounding error of the same order as the residual produces: sign-random.

**Note the trade honestly:** the six locomotion cells were **already** batch-size invariant
(`MujocoHopper`, probe 6407151: 0 rows differ, max 0, pre-mvp). The mvp change is needed only by
`PFRocketLanding`, whose 13 geoms at 100–500 m coordinates cross cuBLAS's kernel-selection threshold.
A per-caller opt-in flag was available and was **not** taken — correctly, because a renderer whose
semantics depend on which env calls it is the exact fork the Step-6 policy addendum forbids. That is
why the six must be re-baselined, and it is the right call.

### (5) Is this a "fidelity regression" in Sami's sense? — my judgement: **no**

Three cells move in the "worse" direction on one metric each. Put beside the cells' own measurement
reproducibility:

| cell | mean_abs Δ | \|declared − this draw\| | ratio | ssim Δ | headroom to `ssim_min` | ratio |
|---|---|---|---|---|---|---|
| HalfCheetah | −2.7e-07 | 4.69e-02 | 172,883× | +2.5e-10 | 2.66e-02 | 1.1e+08 |
| Hopper | −2.7e-07 | 1.50e-01 | 553,243× | +2.9e-09 | 2.81e-02 | 9.7e+06 |
| **Walker2d** | 0.0 | 6.06e-02 | — | **−2.9e-09** | 2.72e-02 | 9.5e+06 |
| **Ant** | **+2.7e-06** | 1.81e-02 | 6,679× | **−2.6e-07** | 2.76e-02 | 1.1e+05 |
| **Humanoid** | **+3.5e-06** | 9.56e-02 | 27,098× | **−1.9e-07** | 3.10e-02 | 1.6e+05 |
| Swimmer | −1.6e-05 | 2.23e-03 | 141× | +7.1e-08 | 2.22e-02 | 3.1e+05 |

The "regressions" are **2 to 5 orders of magnitude below the gap between each cell's declared fidelity
value and what the same code measures on this node and draw** — i.e. far below the noise floor the
declaration itself was set against. Against the declared ceilings the margin is 0.73–1.46 in
`mean_abs` (5–6 orders above the deltas) and 1e5–1e8× in SSIM. `within8_frac` is **unchanged to the
last digit on five of six** cells. Attributing a −2.9e-09 SSIM move on Walker2d to the transform
rather than to the four pixels that flipped is not a claim the data supports.

On "visually meaningful": the worst cell changed **180 px of 3,686,400** (2.8 px per 240×240 frame);
the largest per-pixel delta (Humanoid, 81) occurs on **14 pixels across 64 frames** — 0.22 px/frame.
A large *value* on a tiny *count* is the signature of a silhouette/edge pixel swapping which surface
it samples after a sub-pixel boundary move, which is precisely a rounding change at a geometric
boundary and not a change in what is drawn. Nothing here is visible.

**So: unchanged-or-improved in the operational sense Sami's ruling is about.** Report it to Sami as:
three cells move by ~1e-6 in the unfavourable direction on one metric, three move the other way, all
six stay inside their declared thresholds by 3–5 orders, and the moves are 2–5 orders below the cells'
own draw-to-draw reproducibility.

## Required change

### RD-1 (blocking the merge gate, not the engineering) — the comparator mislabels three cells and reports `refreeze_ok: false`

`refreeze/compare_refreeze.json` — the artifact the merge gate reads — says:

```
refreeze_ok: false
MujocoWalker2d  verdict MEANINGFUL_CHANGE   (max delta 1, 4 px, 0 px >1 level)
MujocoAnt       verdict MEANINGFUL_CHANGE
MujocoHumanoid  verdict MEANINGFUL_CHANGE
MujocoSwimmer   verdict ULP_ONLY            (max delta 67, 180 px, 2 px >1 level)
```

Swimmer changes 45× more pixels than Walker2d with a 67× larger maximum delta and is labelled
`ULP_ONLY`; Walker2d changes 4 pixels and is labelled `MEANINGFUL_CHANGE`. The label has nothing to do
with the pixel evidence. `refreeze_locomotion.py:239–243, 266–269`:

```python
fid["regressed"] = bool(fid["mean_abs"]["delta"] > 0
                        or fid["within8_frac"]["delta"] < 0
                        or fid["ssim"]["delta"] < 0)
...
cell["verdict"] = ("IDENTICAL" if identical else
                   "ULP_ONLY" if not fid_bad and not ds_bad else "MEANINGFUL_CHANGE")
```

`regressed` is a **zero-tolerance sign test**: any strictly unfavourable delta, at any magnitude, flips
it, and the verdict is `MEANINGFUL_CHANGE` iff `regressed`. A −2.9e-09 SSIM move therefore prints the
word "MEANINGFUL". Under Sami's ruling ("any visually meaningful change or fidelity regression → stop
and escalate") the coordinator cannot merge against an artifact that says the re-freeze did not pass.

**Fix:** give `regressed` a stated tolerance and rename the verdict so the label describes the
evidence — e.g. regressed only if a metric moves unfavourably by more than a declared epsilon tied to
the measurement's own reproducibility (a per-metric epsilon at, say, 1e-3 of the declared value leaves
4+ orders of margin over everything seen here), and emit the magnitudes beside the verdict. **No GPU
job is needed:** every input to the classifier — the per-cell pixel counts, every fidelity delta, the
declared values and the downsample bounds — is already inside `compare_refreeze.json`, so the
regeneration is a pure post-process over the committed file. (The per-cell `*_<cell>.npz` frames are
*not* committed, but the classifier does not need them.) Update `findings.md` to match.

## Open certification items (coordinator ruling, not author work)

The job's GPU conformance run is **3 failed, 228 passed, 63 skipped** — and **all three failures are
`[PFRocketLanding]`; zero `Mujoco*` failures**, so the re-baselined six are clean.

1. **`test_color_knobs_apply_the_requested_color[PFRocketLanding]` is structurally unsatisfiable.**
   The test asserts the requested RGB triple appears verbatim in the frame. Under TinyRenderer's
   declared law every material colour is multiplied by
   `0.6 + 0.35·max(0,n·l) + 0.05·max(0,r_z)²`, whose **maximum over all normals is 0.993567** (I
   maximised it numerically over a 2001×4001 grid). So a requested channel of 200 can never render
   brighter than **198**, 33 → 32, 17 → 16 — for *any* correct implementation of this reference
   renderer. All five knobs fail, and all six locomotion cells pass because MuJoCo's additive law
   reaches and clamps at 1.0. I verified separately (Appendix 1) that the knobs are wired correctly:
   the cell's colour table matches `getVisualShapeData` to **1.19e-08** on every geom. **The test needs
   a law-aware form** (compare our knob response against the *official* frame's knob response, or
   allow the declared shading scale); it lives in policy-frozen `tests/gpu/`, which the author
   correctly did not touch.
2. **`test_resolution_downsample_agrees_within_the_measured_bound[PFRocketLanding]`** — the frozen
   bound table has no entry for a new cell. Needs an entry, i.e. a `tests/gpu/` edit under coordinator
   ownership.
3. **`test_visual_tier_matches_its_declared_fidelity[PFRocketLanding]`** failed in job 6410105 because
   the declaration was still the `None` placeholder at the measured tree `726241e`. The declaration was
   filled afterwards, in `7b2e45c`. **No GPU run exists at `7b2e45c`, so this test has never been
   observed green against the filled declaration.** It will pass — the four recorded draws are
   7.482–7.709 / 0.961–0.962 / 0.854–0.868 against ceilings 10.0 / 0.91 / 0.80 — but that is inference,
   not evidence, and the merge record should say so.

## Everything else verified

* **Scope / hygiene.** `examples/results/step4|5|6`: **0 files**. `tests/gpu`: **0 files**. Non-step7
  library files: still only `stable_envs/envs/rocket_landing.py`, `stable_envs/render/nvdiff.py` and
  the two test files. Trailer grep **0**; sole author `sami-bg <samibghanem@gmail.com>`. `nvdiff.py` is
  unchanged between the measured tree `726241e` and the tip, so the re-freeze evidence is not stale.
* **Unit suite.** Branch `7b2e45c`: **2 failed, 857 passed, 7 skipped**; pristine `76dc511`:
  **2 failed, 823 passed, 6 skipped** — the same two pre-existing failures
  (`render_mjwarp.py --help`, `test_every_cited_scratch_path_resolves`). No new failures.
* **(6) Gate A, all six on the branch:** determinism identical; batch invariance 0; **batch-size
  invariance 0 at (1, 3, 7, 8, 64, 256)**; order independence 0 (permutation and duplicate); chunked
  render 0 at chunks (1, 7, 64); render purity 0 on all four probes; no dead dims. `all_pass: true`.
* **(7) Step-5 surfaces:** PushT `[scene]`, DMCReacherEasy/Hard `[mjwarp, raycast3d]` —
  `digest_ours` equal on baseline, branch **and the frozen Step-5 value**; `digest_official` identical;
  mean pixel identical to the last digit; and a call counter on `Scene.clip` / `BatchRenderer.render`
  records **0 calls** on both legs, so nvdiff is not on those paths at all. `all_untouched: true`.
* **(8) The defect the change exists to fix:** pre-mvp (`cf355a3`, job 6407151, gpu2109),
  `PFRocketLanding` row 0 rendered alone vs inside a batch of 64 differs by **max 255 on 10 pixels**,
  and **52 of 64 rows** differ from their single-row render. Post-mvp (`726241e`, job 6410105,
  gpu2106): **0 rows differ; max 0, 0 px at every prefix (1,2,3,4,7,8,15,16,17,24,31,…)**.
  `MujocoHopper` was 0 on both legs.
* **Both legs, one job, one card:** `same_gpu: true`, `same_host: true`, job 6410105 on gpu2106, 64
  states per cell at 240 px = 3,686,400 px, and `states_sha256_match: true` on all six — the two legs
  compared the *same* states.
* **`downsample_exact_frac`:** moves ≤ 8.7e-06 with 1.2e-03–5.8e-03 of headroom;
  `bound_needs_update: false` on all six; the bound's source is correctly identified as
  `tests/gpu/test_reproducibility.py`, not a Step-4 artifact. I confirm the six cells have **no frame
  digest under `examples/results/step4`** — consistent with `examples/results/step4` being untouched.
* **Rocket declaration = the machine-generated block, exactly.** `visual_fidelity` matches
  `tier/job6410105/declaration_block.json` field for field; the headline numbers are the **worst of the
  four draws** (7.709024→7.71, 0.961288→0.961, 0.854496→0.854), and the draws are recorded. Ceilings
  vs the family's realised ratios: `mean_abs_max` ×1.297 (family 1.322–1.369 — slightly *tighter*),
  `within8_min` ×0.947 (family 0.942–0.963 — **inside the band**), `ssim_min` ×0.937 (family
  **0.967–0.979** — *looser* than any existing cell; see minors). `UNMEASURED` is gone; the
  `measured_at` string names GPU, node, job, tree, torch and nvdiffrast versions.
* **Tier, at the frozen definitions:** R = 241.51 (3374.4 f/s at B=4096 vs 13.97 f/s official,
  same node), cores = 2.606, `cpu_frac` 1.027 → no `HOST_BOUND`, `best_certified_batch` 4096 (never 1),
  and **`BATCH_SCALES` FAILS at 14.17 (bar 20) so the tier is correctly capped at `GPU_PARITY` with
  qualifier `BATCH_SCALES_FAIL`** — the RC-3 cap I asked for, working as specified.
* **Contact-band flag does not narrow `sample_states`.** `sample_states` is unchanged in the diff; the
  docstring states the distribution is not narrowed; the fallback returns swm's *synced* picture and
  flags it, and the shared `check_visual_fidelity` keeps those rows, which is the **conservative**
  direction (the synced picture differs from the lagged one by the whole substep lag, so keeping it can
  only make the measured fidelity worse). Measured share 1/960 over 15 seeds × 64 states; **0 rows in
  any of the four certification draws** (`failures: []`, `unreachable_rows_total: 0`).
  `_reference_frame` keeps the strict raising form for the validators.

## Minors (worth a line each, none blocking)

* **`ssim_min` is looser than the family.** `fill_declaration.py` uses ×0.94 for the SSIM floor; the six
  locomotion cells' realised floors are ×0.967–0.979. 0.80 vs a ×0.97 convention's 0.83 is 0.03 of
  unjustified slack on a declaration whose measured value is 0.854. Either use the family ratio or say
  in the comment why this cell needs a wider floor.
* **`last_reference_unreachable` is mutable class state** overwritten by each `reference_render` call
  and read out-of-band. It is on the reference side, so `check_render_is_pure` is unaffected, but the
  substitution is **invisible to the shared `check_visual_fidelity` API** — any future caller of the
  generic conformance function gets a synced frame silently. `measure_tier.py` does read the flag
  (`unreachable_rows`, `unreachable_rows_total`); a generic caller would not. Prefer returning the
  flags alongside the frames.
* **The baseline leg's `sha` is still `"unknown"`** in both `compare_refreeze.json` and
  `step5_surfaces_compare.json` (the baseline copy is a `git archive` with no `.git` and no
  `.git_sha`); the branch leg has a real SHA. The baseline side is attributed only by `pythonpath` /
  `stable_envs_file` — adequate, but the D1 fix should be applied to the baseline copy too.
* **Register for Sami:** `BATCH_SCALES` fails at 14.17 not because the renderer does not batch — it
  reaches 3374 f/s at B=4096 and R = 241 — but because `fps(1) = 238` is unusually *high* (the chunked
  path is efficient at B=1). §6.7 set that bar to catch "a renderer that is not actually batching
  (≈1×)". A 14× scaling cell is being capped at `GPU_PARITY` by a bar aimed at 1× cells. That is the
  frozen contract applied correctly and I am not proposing to change it, but it is a definitional edge
  case the coordinator should surface rather than let the cell carry a `GPU_PARITY` badge unexplained.
