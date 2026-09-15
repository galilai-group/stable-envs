# Step 7 / W6 — `swm/PFRocketLanding-v0`: feasibility gate (Phase 0) and implementation (Phase 1)

**Classification: PATCH (M6).** Phase 0 (below, unchanged) recommended GO in the visual tier; Phase 1
(M1-M3, authorised) is implemented, unit-tested and independently reviewed (APPROVE at `cf355a3`); M4
ran every cluster gate twice on one RTX 3090 and found two defects in `stable_envs/` code; M5 fixed
both, and the defect-A fix was reverted because it could not keep the six locomotion nvdiff cells
byte-identical — the hard condition it was first authorized under. **Sami ruled on 2026-09-15 that
mandatory batch-size invariance takes precedence over accidental float bit identity**, so M6 re-applies
the fix and replaces that condition with a measured CORRECTNESS RE-FREEZE (job 6410105): per-cell pixel
deltas, official-reference fidelity before and after, the full mandatory set on all six, and the two
Step-5 representative surfaces. Every criterion holds — at most 5 pixels in 3,686,400 move by more than
one level in any cell, every cell stays inside its declared fidelity, Gate A is green on all six, and
PushT / DMCReacher never enter the changed code. **The re-freeze needed no frozen-artifact edit**: the
six cells have no frame digest under `examples/results/step4`, and the one bound that names them does
not move, so that directory is not touched. `visual_fidelity` for `PFRocketLanding` is now **measured
and declared**. Read M6 for the current state; M4 and M5 are the runs that got there.

| | |
|---|---|
| baseline SHA | `76dc5116f432e61dd47fba8f9e879cf9212c8e46` (Step 6 closed) |
| branch | `step7/rocket-landing`, worktree `/Users/samib/stable-envs-step7-rocket-landing` |
| milestone commits | Phase 0 `c6772f7` · M1 `4bb93a4` · M2 `83db3cc` · M3 `13ca91b` (+ review D1/D2 `cf355a3`) · M4 `e74d0bd` · M5a `d1e61a1` (both defect fixes) · M5b `283865b` (reverted the defect-A change under the byte-identity condition) · M5 `7224222` (the job-6408817 evidence) · M6a `dac40f9` (re-applies the defect-A fix under Sami's ruling; `nvdiff.py` and `test_nvdiff_triangles.py` are byte-identical to `d1e61a1` again) · M6b `8ca2d72` + `726241e` (the re-freeze harness; `726241e` is the tree job 6410105 measured) · M6 = the commit carrying this file, the filled declaration and the flipped test |
| hardware | Apple M5 Mac, CPU only; PyBullet DIRECT (`connectionMethod: 2`) = TinyRenderer software rasteriser. **Local timings are not certification numbers.** |
| venv (local) | `/Users/samib/se7-rocket-venv` (Python 3.12.11): PyFlyt 0.29.0, pybullet 3.2.7, cvxpy 1.7.5, gymnasium 1.3.0, torch 2.14.0, Pillow 12.3.0, trimesh 5.1.0, imageio 2.37.4, numba 0.67.0, stable-worldmodel 0.1.1, opencv-python-headless 5.0.0.93 (it moved numpy 1.26.4 -> 2.5.3); unit tests run under `/Users/samib/se6-venv` with `MUJOCO_GL=cgl` (the env modules `setdefault` egl, which is invalid on macOS) |
| Oscar | sync recipe: `git rev-parse HEAD > .git_sha` in the worktree, then the rsync (which excludes `.git`) — the job refuses to start without the file and stamps every number with it as `harness_commit` (never the `$SCR/se7/baseline` copy). SSH restored during Phase 1: `$SCR/se7/rocket-landing/` (this branch, rsynced), `$SCR/se7/baseline/` (76dc511, `git archive`), venv build job `se7-rocket-venv` **6394857** (`scratch/step7/rocket-landing/oscar_venv.sbatch`); the GPU job `oscar_gpu.sbatch` is submitted after it completes (one GPU job at a time; 3090, `gpu`/`norm-gpu`, nodes gpu2108/2113/2110/2116 excluded) |
| affected configs | none existing; one new cell `PFRocketLanding` (480 px, swm's registered default: 480 x 480 x 3 = 691,200 B per observation, confirmed from `RocketLandingEnv.__init__(render_resolution=(480, 480))` and the frame the probe recorded) |
| files touched (Phase 1) | `stable_envs/render/nvdiff.py` (M2), `stable_envs/envs/rocket_landing.py` (M3, new), `tests/unit/test_nvdiff_triangles.py` (new), `tests/unit/test_rocket_landing.py` (new), `examples/results/step7/rocket-landing/{reference,phase0,findings.md}`, `scratch/step7/rocket-landing/` (`phase0_probe.py`, `phase0_probe_b.py`, `tiny_law_probe.py`, `tiny_law_probe2.py`, `build_reference_corpus.py`, `validate_reference_trick.py`, `validate_env_math.py`, `cpu_raster_probe.py`, `measure_tier.py`, `nvdiff_identity.py`, `probe_batch_size.py`, `probe_clip_ulps.py`, `probe_reference_rejections.py`, `fill_declaration.py`, `oscar_venv.sbatch`, `oscar_gpu.sbatch`; the `.log` files beside them are run output and are gitignored) |

## Phase 1 — what was built, what was measured, what is pending

### M1 · reference corpus (`4bb93a4`) — `examples/results/step7/rocket-landing/reference/`

200 states from 24 STEPPED rollouts under a random policy (ignition sticky Bernoulli(0.5), throttle
U(0.3, 1), fins/gimbal U(-0.3, 0.3)); 6 uniform step indices per episode plus one trigger per height
threshold (200/100/50/25/12/6/3 m). Frames are the frames those steps produced — the lagged picture
swm observes (ruling 3). `states.npz` (97 kB): `state13 = obs[:13]` (ang_vel body, quat xyzw, lin_vel
body, lin_pos), `obs30` float64, `style` (5 RGB + flame bit, corpus column order), `pose`, `world_vel`,
`cached_link0`, per-frame sha256; `frames.npz` (1.35 MB compressed); `meta.json`.

Validated per state (`meta.json: validation`): cached link-0 pose == `(pos - v dt, exp(-w dt) q)` from
the obs to **1.8e-15 m / 5.6e-16**; obs quaternion == PyBullet's to 6.7e-16; obs body-frame
velocities rotate to the world ones to 6.4e-14; the colour law to 3e-8. Heights: 109 states above
300 m, 91 spread over 0-300 m; ground visible in 45 %; attitudes tumbling (tilt to 174 deg,
|w| to 4.9 rad/s) with a near-upright minority; lateral drift to 190 m.

**Reference at an arbitrary state** (`reference_trick_validation.json`, `validate_reference_trick.py`):
a teleport syncs the links, so the lagged picture is produced by teleporting the base to the PREVIOUS
substep pose, setting velocities, and running one raw `stepSimulation` with gravity off — iterating
the SET velocity until the POST-step velocity equals the recorded one, because Bullet's substep is
semi-implicit (`pos' = pos_prev + v1 dt`, `q' = exp(w1 dt) q_prev`, both exact) but a spinning
composite multibody changes its own velocity by up to 0.03 m/s per substep (gyroscopic and offset-mass
terms; `resetBasePositionAndOrientation` must precede `resetBaseVelocity`, it zeroes the velocity).
**199/200 corpus frames byte-identical, the last off by 1 px, base landing error 3.6e-15 m and
rotation error 1.1e-16 — regenerated under the renormalised recipe now in the module
(`reference_trick_validation.json`, `validate_reference_trick.py` calling `_reference_frame`) — FROM
THE FLOAT64 OBSERVATIONS.** The quantisation term measured on its own
(`tier/corpus_replay_state_quantisation_cis1b1zs2j3.json`, tree `aa54c2c` plus this commit's edits,
which do not touch `_reference_frame`): PyBullet at the float32 `state13` vs the frozen frames over all
200 states — 0 rejections, 9 exact, max 79 px, mean 11.4 differing px per frame, mean |diff| 0.0016/255. The env holds float32 states: the float32 rounding of a 400 m position
(~3e-5 m) and of a quaternion (|q| = 1 ± 3e-8) is a state change of its own, and the reviewer measured
what it costs (56/100 assertion failures and 3/100 exact frames under the first, float64-only tolerance).
`lagged_pose` now renormalises the quaternion in float64 and the reference's pose check compares
rotations sign-free with a float32-aware tolerance. The corpus replay is therefore TWO measurements
with two renderers, not two labels on one: (i) our render of the float32 `state13` vs the frozen
frames, which contains the renderer's residual PLUS the float32 state quantisation (``render`` casts
its input to float32, so `obs30[:, :13]` and `state13` feed bit-identical tensors and cannot be
told apart there); (ii) PyBullet's own `reference_render(state13)` vs the same frozen frames, the
quantisation term alone (`scratch/step7/rocket-landing/corpus_replay_state32.py`,
`tier/corpus_replay_state_quantisation_<host>.json`); `check_visual_fidelity`, where both sides see
the same float32 state, is the renderer-only number. This is `PFRocketLanding.reference_render`. One caveat, recorded: the bridge is not bit-deterministic
across PyBullet call ORDER for one frame in 200 — corpus index 124 is byte-exact in the reviewer's
50-state sample and 1 px off in the sequential 200-state run (both under the same recipe); the
committed artifact is the conservative one, and the tier's fidelity does not flow through this
artifact but through `check_visual_fidelity`, where reference and render see the same state.

### M2 · nvdiff extensions (`83db3cc`) — generic, defaults unchanged

`stable_envs/render/nvdiff.py`: `Scene.from_triangles` (a scene from a triangle soup + appearance
table, the same members and the same downstream path as the `MjModel` constructor, which now shares
`_assemble`); `prepare_texture_array` (split out of `prepare_texture`); `render(..., mat_rgb=)` per-row
`(B, G, 3)` material colour through `gather_material`; `TinyShading` + `shade_tiny` (world-fixed
directional light, `ambient + diffuse*max(0,n.l) + specular*max(0,r_z)^e` MULTIPLIED into the material
colour, reflection from the UNCLAMPED `n.l`); `BatchRenderer(..., scene=, shading=, tiny=, tex_filter=,
sample_offset=)` with `shading="mujoco"`, `tex_filter="linear"`, `sample_offset=(0,0)` as defaults —
the existing `_shade`, mipmapped texture and `S.mat_rgb[gid]` paths run as before (`offset_clip` at
`(0,0)` returns the input object; `gather_material` on a 2-D table is the same indexing).

`tests/unit/test_nvdiff_triangles.py` (14 tests, CPU): the soup constructor's members and its sparse
transform against a dense one; per-row vs shared gather; the shading law pinned to MEASURED TinyRenderer
pixels — 239 (default law, horizontal white), 220 / 190 (diffuse-only / specular-only, tilt 0), 197 / 112
(tilt 10 deg), **162 / 158 on faces turned AWAY from the light** (`tiny_law4_backface.json`; the clamped
law says 153 — measured after the first version of this test assumed it); `offset_clip`; `QUANT` trunc;
the texture preparation. Local suite: `tests/unit` **805 passed, 36 skipped, 3 failed** — the three are
(a) my back-face expectation, since fixed (29/29 on the two nvdiff files), (b)
`test_examples[...render_mjwarp.py]` and (c) `test_every_cited_scratch_path_resolves` (9 Step-6
citations of Oscar scratch paths under the `sboughan` home), **both of which fail identically on the untouched baseline** (`/Users/samib/stable-envs` at 76dc511,
same interpreter). Measured totals, interpreter named because the skip count is a property of it (a package that is
absent skips, one that is present runs): reviewer's interpreter — baseline 76dc511 full suite
**823 passed / 2 failed**, branch **855 passed / 2 failed / 7 skipped**, the same two failures;
branch with `test_examples` deselected **745 passed / 7 skipped**. My `se6-venv` on the Mac
(`MUJOCO_GL=cgl`, many optional packages absent): branch with `test_examples` deselected **713
passed / 1 failed / 37 skipped**, the one failure the pre-existing dead-citation test with no
`step7` path in its list. The review found that my own findings text had twice re-introduced dead
citations (a glob over this workstream's scratch directory that also matched gitignored run
logs, and a job-script name that was never written) — fixed.

### M3 · the cell — `stable_envs/envs/rocket_landing.py`, `PFRocketLanding`

`state_dim = 13` (`obs[:13]`), style = the variation space `{rocket.body/fin/leg/booster_color,
pad.color (RGBBox), flame.ignited (Discrete 2)}` whose flat layout is ALPHABETICAL (`gymnasium.spaces.Dict`
sorts keys) and is addressed by path everywhere (`style_layout`, `corpus_style_to_env`); defaults =
the official appearance of a reset without a variation request (URDF yellow/grey/black, a WHITE pad,
flame unignited). `sky_color` is not offered (dead upstream). `default_size = 480`, supersample 1.
`device="cpu"` raises `NotImplementedError` naming the missing CPU textured-triangle rasteriser
(ruling 1); registered only when nvdiffrast imports, `missing_backend()` says why otherwise.

The scene: 13 geoms in draw order (ground quad with `plane100.obj`'s UVs and the checker flipped for
nvdiffrast's row convention; body / fuel tank / booster / 4 fins / 3 legs / flame from the URDF joint
origins; the pad) = **382 vertices, 486 triangles**. Cylinders are `tiny_cylinder`: Bullet's
`btShapeHull` — the support points of the 2 x 32 ring in the 42 fixed unit-sphere directions
(`btShapeHull.cpp`, verbatim table), pushed out by the shape's 1 mm margin, normals = normalised
positions (`TinyRendererVisualShapeConverter.cpp` 211-290, 655-705); **42 vertices / 80 triangles per
cylinder, silhouette IoU 0.995 against TinyRenderer's top-down render** (`env_math_validation.json`; the
residual is the margin and scipy's cap fan vs HullLibrary's). Boxes are `nvdiff._box`. Renderer:
`BatchRenderer(scene=..., shading="tiny", tex_filter="nearest", mipmap=False, quant="trunc",
sample_offset=(0.5, 0.5))`, background white, camera per row (`camera_pose`: PyFlyt's recipe — link-0
lagged pose, offset `R·(-1,0,3)`, Euler pitch +30 deg, tracking target, gluLookAt; verified against
PyBullet's own `computeViewMatrix` on 200 random attitudes to 4.4e-4 = its float32 arithmetic, and
Euler/quaternion conventions to 1e-15), links at `t - dt` and the body at `t` (`geom_poses`, verified
against `getLinkState` on the loaded URDF to 1.9e-6). `sample_states` is a documented mixture covering
the corpus ranges (half high-altitude "rocket on white", half log-uniform descent; 30 % near-upright).

Declarations: `fidelity_tier = "visual"`, `parity_interior_min = None`, `visual_fidelity` **PROVISIONAL
AND MARKED UNMEASURED** in `measured_at` (Phase-0 estimate mean_abs 3.0 / within-8 0.95 / SSIM 0.95,
thresholds 6.0 / 0.85 / 0.90) with `reference_versions = {"PyFlyt": "0.29.0", "pybullet": "3.2.7"}`,
`visual_note` naming the residual; the measured numbers and thresholds are copied by hand from
`measure_tier.py`'s output after the 3090 run. `tests/unit/test_rocket_landing.py` (14 tests, CPU):
the state contract against the corpus (lag from the float64 obs to 1e-9, from the float32 state to
1e-3 m), the layout/converter, the colour law, the hull, the scene arrays, the poses, the camera, the
Euler round trip, `sample_states`, the default style, registration-iff-nvdiffrast; **13 passed, 1
skipped** (the checker texture needs `pybullet_data`, absent from the se6 venv).

**Local end-to-end preview of the residual** (`scratch/step7/rocket-landing/cpu_raster_probe.py`, a
third rasterizer driving the SAME `geom_poses`/`camera_pose`/`material_table`/`tiny_cylinder`, against
the frozen corpus frames, 32 states): **95.15 % of pixels bit-exact, mean 5.87/255, median frame
4.99, within-8 95.6 %, SSIM 0.863, worst frame 11.8**; 90 % of the differing pixels differ by more
than 8 — silhouettes and the sub-pixel far-ground checker under nearest sampling, i.e. the class of
residual Phase 0 predicted, not a shading error. A preview by construction, not the tier's number.

### CPU feasibility (ruling 1) — ONE bounded measurement, reported and stopped

`cpu_raster_probe.py`: a vectorised torch CPU rasteriser of the 486-triangle scene at 480 px, one
thread, near/far clipping, extent-sorted bounding-box chunks, `scatter_reduce(amin)` depth,
perspective-correct attributes, TinyRenderer's law, nearest texture. **1.92 f/s (521 ms/frame) on
one M5 core = 0.056 x the 34.25 f/s official (bar: S_cpu >= 5, i.e. >= 171 f/s) and 0.0016 of one
core of JPEG decode (bar: >= 1191 f/s).** Both bars missed by two to three orders of magnitude for
this naive attempt; a real CPU triangle backend (tile binning, compiled kernels) is a different
program and a coordinator/Sami decision. The cell declares CPU coverage MISSING.

### Planned for Oscar (one GPU job, `oscar_gpu.sbatch`) — as written before the runs; the results are in section M4

1. M2 on the card: the nvdiff unit tests on the node; nvdiff byte-identity of its existing users —
   the six locomotion visual cells (Fetch renders through mujoco_warp and the frozen Step-4 corpus
   holds no nvdiff-backed config, so the replay is 64 `sample_states` per cell rendered under
   `PYTHONPATH=$SCR/se7/baseline` and `=$SCR/se7/rocket-landing` in the same job, digests compared
   (`nvdiff_identity.py`); identity cannot be shown locally (nvdiffrast is CUDA-only); plus their own
   `test_visual_tier_matches_its_declared_fidelity`.
2. M3: `measure_tier.py` — `check_visual_fidelity` over 4 x 64 states against PyFlyt+TinyRenderer at
   480 px, `check_visual_fidelity_under_variation`, the frozen-corpus replay, the mandatory checks;
   then the conformance suite restricted to `PFRocketLanding`.
3. M4: throughput at batches 1/16/64/256 (same job, same card), against the official 34.25 f/s and
   the 1,190.9 f/s one-core decode measured on the Mac (re-measure both on the node for a
   same-machine ratio).

Gate A checks runnable locally: everything in `tests/unit/test_rocket_landing.py`; pending cluster:
render-side checks (batch invariance, batch-size invariance, determinism, state dims, resolution
consistency, visual fidelity).

### Review (independent, on `92d9b28`): APPROVE-WITH-CHANGES, landed as one commit

RC-1 (blocking): float32 quaternions failed the reference's float64 pose assertion (36/64 of draw 0)
and the fidelity loop aborted before writing anything — fixed as above; the loop now records a failed
draw instead of aborting and checkpoints the JSON after every stage. Evidence after the fix
(`reference/rc1_evidence.json`, `scratch/step7/rocket-landing/validate_rc1.py`): the exact `measure_tier`
draw-0 seed (`default_rng(1000)`, 64 float32 `sample_states`, |q| off by up to 3.0e-8) gives **0
rejections**; the first 100 corpus states from the float64 obs reproduce **100/100 frames exactly**
(0 rejections); the same 100 from float32 `state13` give 0 rejections and **2/100 exact, up to 79 px,
mean 11.5 px** — that is the float32 rounding of the STATE (3e-5 m at 400 m, 3e-8 in q), a term that
sits inside a consumer's residual and is therefore reported alongside the float64 replay, never
folded into it. RC-2 (blocking): my findings
text re-cited two dead scratch paths and claimed parity with the baseline's failures — corrected here
with measured numbers. RC-3: `measure_tier.py` never certifies on B = 1, caps the tier at `GPU_PARITY`
when `BATCH_SCALES` fails, measures `cpu_frac` (process CPU / wall of the timed loop; `HOST_BOUND` at
≥ 1.5) with threads, load average and the SLURM job id, pins threads for the one-core decode, and
reports only same-node comparators. RC-4: the placeholder `visual_fidelity` numbers are `None` under
an `UNMEASURED` marker (the validator checks presence), with a local test that the declaration
registers. RC-5: `missing_backend()` names nvdiffrast, scipy, pybullet_data and Pillow; the checker's
provenance rests on the pybullet 3.2.7 declaration. Minor: the camera test's tolerance is stated as
float32 (`computeViewMatrix` is float32 at 400 m; 1.009e-3 measured against the live `view_mat`).
Verified independently by the reviewer, no action: nvdiff defaults unchanged (Scene inputs hashed
bit-identical for all six locomotion models on both checkouts), the shading law, the lag law, the
colour law, the geometry and camera numbers.

### M4 · the cluster gates (jobs 6395707, 6406061; diagnostics 6407151, 6407256) — BLOCKER: two code defects, declaration NOT filled

**Classification for this milestone: BLOCKER.** Every gate ran, on the `cf355a3` tree, on one RTX 3090
(`gpu2109`), twice — the two full runs agree to the last digit on every GPU number. Two of the seven
gates fail for reasons that are in `stable_envs/` code (the batch-size invariance of the frame, and the
reference bridge rejecting states the cell's own `sample_states` draws near the ground). Under the M4
rule — a code defect stops the milestone, it is not fixed here — `visual_fidelity` stays `UNMEASURED`
in `stable_envs/envs/rocket_landing.py`, and the numbers that WOULD be declared are recorded below with
their provenance so the coordinator can fill the declaration in one edit once the defects are ruled on.

Provenance common to every number in this section unless a line says otherwise: tree `cf355a3a2a7d75891f85a13e5a4b46febc282418`
(the job refuses to start without `.git_sha`, and stamps it into every gate line and every JSON as
`sha`/`harness_commit`, with `stable_envs_file` = `$SCR/se7/rocket-landing/stable_envs/__init__.py`),
host `gpu2109.oscar.ccv.brown.edu`, NVIDIA GeForce RTX 3090, `$SCR/venvs/se7-rocket` (torch
2.14.0+cu126, nvdiffrast 0.4.0, PyFlyt 0.29.0, pybullet 3.2.7, stable-worldmodel 0.1.1, numpy 1.26.4,
scipy 1.17.1), 4 threads, `--cpus-per-task=4 --mem=48G`, `gpu`/`norm-gpu`. Node load average at
measurement: job 6395707 `17.4 / 18.8 / 19.5`, job 6406061 `37.3 / 37.8 / 39.0` (the node was shared;
the GPU numbers did not move, the two CPU comparators moved ~5 %). Job 6395707 ran the committed
harness (its `check_batch_size_invariance` call lacked the `states` argument and errored, and its
pytest output was cut to `tail -25`); job 6406061 is the same tree with those two harness fixes
(`scratch/step7/rocket-landing/measure_tier.py`, `oscar_gpu.sbatch`) and is the run of record. Its
artifacts are `tier/` and `nvdiff_identity/` directly; job 6395707's are kept under
`tier/job6395707/` and `nvdiff_identity/job6395707/`; the SLURM logs are `logs/se7-rocket-gpu_<job>.out`
verbatim. In the `.txt` copies of cluster output the scratch root is written `$SCR` (the
Oscar scratch root, AGENT_COMMON's alias) so that `tests/unit/test_repo_contracts.py`'s
citation scanner does not read a cluster path as a dead `scratch/` citation; the `.out` logs and `.json`
files are byte-verbatim.

#### The seven gates

| # | gate | result | evidence (job 6406061 unless noted) |
|---|---|---|---|
| 1 | six locomotion nvdiff cells, baseline (76dc511) vs branch, same job, same card | **PASS** | `nvdiff_identity/compare.json`: `all_identical: true`, `same_gpu: true`; per-cell sha256 of 64 sample-state frames at 240 px identical for HalfCheetah/Hopper/Walker2d/Ant/Humanoid/Swimmer (`baseline.json` = `branch.json`), in both jobs. The locomotion cells' own `test_visual_tier_matches_its_declared_fidelity` did not run: `mujoco.GLContext` fails on gpu2109 (`nvdiff_identity/egl_probe.txt`, `EGLError`), the node-level EGL problem the coordinator already recorded for gpu2104; it is not needed by this cell (TinyRenderer is CPU). |
| 2 | first successful `PFRocketLanding` render | **PASS** | 480x480x3 uint8, 19,720 non-white px, 161 distinct colours, `tier/first_render.png` (state in `gpu2109.json: first_render`); byte-identical when re-rendered after the PyBullet/EGL contexts were created. |
| 3 | 4 x 64 visual fidelity + under variation | **PASS as a measurement** — see defect B for why the suite's own test aborts | draws (`default_rng(1000..1003)`, 64 float32 `sample_states`, default appearance, 480 px, ss 1, vs PyFlyt 0.29.0 / pybullet 3.2.7 TinyRenderer): mean_abs **7.482 / 7.678 / 7.573 / 7.709**, within-8 **0.9622 / 0.9618 / 0.9620 / 0.9613**, SSIM **0.8632 / 0.8545 / 0.8684 / 0.8595**, exact 0.959-0.961, interior-exact 0.967-0.969; **0 rejections**; worst of four: **mean_abs 7.709, within-8 0.9613, SSIM 0.8545**. Under variation (seed 7, drawn appearance per state): mean_abs 7.646, within-8 0.9580, SSIM 0.8961, exact 0.903. Identical in both jobs. |
| 4 | corpus replay | **PASS** | ours(`state13` float32) vs the 200 frozen frames: mean_abs **6.968**, exact 0.9574, within-8 **0.9607**, SSIM **0.8901**, interior-exact 0.9668, worst frame 13.51 (index 85); by height 260-500 m: 3.89 / 0.970, 50-260 m: 6.37 / 0.951, 12-50 m: 6.99 / 0.946, 0-12 m: 7.28 / 0.945 (`gpu2109.json: corpus_replay`). The state-quantisation term alone, PyBullet's own `reference_render(state13)` vs the same frames: 200/200 rendered, 0 rejections, 1 exact, mean |diff| 0.0017/255, mean 13.7 differing px, max 118 px. |
| 5 | mandatory conformance | **FAIL (batch-size invariance) — defect A**; the other three PASS | `check_batch_invariance` 0; `check_determinism` identical; `check_state_dims_matter` no dead dims — the six velocity dims `ang_vel_x/y/z`, `lin_vel_x/y/z` all move pixels (their only pictorial effect is the one-substep lag). `check_batch_size_invariance` (64 states, windows 1/7/64): **max 255, "row 0 in a batch of 1"**; the suite's `test_frame_i_is_independent_of_batch_size` (sizes 1/3/7/8/64/1024): the same, max 255. |
| 6 | GPU sweep, tier | **PASS as a measurement; BATCH_SCALES FAIL; NO_LARGE_BATCH_REGRESSION PASS; not HOST_BOUND** | f/s at B = 1 / 16 / 64 / 256 / 1024 / 4096: **249.7 / 2069.8 / 3038.1 / 2738.8 / 3218.7 / 3407.5** (job 6395707: 252.4 / 2076.9 / 3038.3 / 2716.3 / 3207.4 / 3407.9); `cpu_frac` 0.993 / 0.993 / 0.993 / 0.994 / 1.156 / 1.054. `BATCH_SCALES` fps(4096)/fps(1) = **13.65** (bar 20) FAIL; `NO_LARGE_BATCH_REGRESSION` fps(4096)/fps(256) = **1.244** (bar 0.9) PASS; `cpu_frac` at the best certified batch **1.054** < 1.5, so no `HOST_BOUND` qualifier — but 1.05 says the wall clock IS one host core: the per-row float64 kinematics and camera in `render()` and the two host->device uploads are the ceiling at 4096, which is also why fps(1) is as high as 250 and the ratio misses 20. Same-node comparators: official TinyRenderer through the swm env **13.72 f/s** (14.51 in job 6395707; 34.25 on the M5), one-core JPEG-q90 decode of the env's own frames **1075.6 f/s** (1054.2; 14.7 kB/frame). Class S: **R = 248.4** (234.8), **cores = 3.17** (3.23) at B = 4096; tier under the frozen definitions **`GPU_PARITY`, qualifier `BATCH_SCALES_FAIL`** (capped from the R/cores tier by section 6.7). |
| 7 | PyBullet DIRECT + EGL + CUDA raster coexistence across the numba JIT | **PASS** | in one process: reference env created (DIRECT, JIT paid), `mujoco.GLContext` created (`egl: ok` inside the Python process even though the sbatch's standalone probe failed), then the same state re-rendered byte-identical (`render_after_identical: true`). |

Conformance suite on the card (`tier/conformance_PFRocketLanding.txt`, full output): **29 passed, 9 skipped, 4 failed**.
The four, each with its mechanism:

* `test_frame_i_is_independent_of_batch_size` — **defect A**, below.
* `test_visual_tier_matches_its_declared_fidelity` — not the `None` thresholds: `check_visual_fidelity`
  raised inside `_reference_frame` ("the reference did not reach the requested pose (pos 2.662e-01 m,
  1-|q.q'| 7.850e-06)") on pytest's own 64-state draw — **defect B**, below.
* `test_color_knobs_apply_the_requested_color` — all five RGB knobs, both targets: the requested colour is
  never literally in the frame. Mechanism: TinyRenderer's law MULTIPLIES the material colour by
  `0.6 + 0.35·max(0, n·l) + 0.05·max(0, r_z)^e`, which is < 1 for every pixel (with `l` the normalised
  `(-50, 30, 100)`, `n·l = 1` forces `r_z = l_z = 0.864`), and truncates — `int(255·c·f) < 255·c` for
  every channel. The official PyFlyt frame does not contain the exact colour either (Phase 0: the brightest
  body pixel is `0.991·int(255·c)`). The check's premise is flat shading; on a lit cell it measures the
  shading law, not the knob wiring (which `test_variations_all_do_something` and the fidelity numbers under
  variation cover). A test-suite decision, `tests/` is frozen — reported, not changed.
* `test_resolution_downsample_agrees_within_the_measured_bound` — `DOWNSAMPLE_EXACT_FRAC` has no
  `PFRocketLanding` entry (the test says so). Measured on the card at the test's own protocol (8 states,
  `default_rng(render_worker.STATE_SEED)`, default appearance): **2S->S exact 95.614 %, mean 0.9920/255,
  max 255** (job 6406061, gpu2109, RTX 3090, torch 2.14.0+cu126, nvdiffrast 0.4.0; identical in 6395707).
  The number to record is `0.95614`, with that provenance, by whoever edits `tests/`.

#### Defect A — the frame depends on the batch size (mandatory gate; `stable_envs/render/nvdiff.py` / `stable_envs/envs/rocket_landing.py`)

Measured (`tier/probe_batch/probe_batch_size_6407151_gpu2109.json` (and `_6407256_gpu2264.json`, identical), job 6407151, same node/card, `scratch/step7/rocket-landing/probe_batch_size.py`),
64 states from `default_rng(11)`:

* the 64-batch render is deterministic (rendered twice: identical);
* row 0 rendered in a prefix batch of **1, 2, 3, 4, 7, 8, 15, 16, 17, 24** equals its single render
  exactly and differs from row 0 of the 64-batch by **10 px (max 255)**; in a prefix batch of **31, 32,
  33, 40, 48, 63, 64** it equals the 64-batch render exactly and differs from the single render by the
  same 10 px. Two families, one threshold between B = 24 and B = 31, nothing in between;
* rows 0-31 rendered alone == rows 0-31 of the 64-batch (0 px), rows 32-63 alone == theirs (0 px): the
  cell's chunk loop is not involved (chunk = 116 rows at 480 px, so every batch here is one chunk);
* **52 of the 64 rows** differ between the full batch and their single renders; in a 1024-batch, row 0
  differs from its single render by 12 px and 53 of the first 64 rows differ;
* the 10 pixels of row 0 (`tier/probe_batch/row0_single.png`, `row0_in64.png`, `row0_diffmask.png`,
  rows 187-252, columns 208-466 — the rocket, not the ground): three silhouette pixels flipping between
  body yellow `(153,122,0)` and the white sky, two flipping between the body and the grey tank/booster
  seen through the open tube, five one-step truncation flips in the shaded body (`245,196,0 ->
  244,195,0`). That is the signature of an ulp-level change in clip-space vertex positions and
  interpolated normals — an edge crossing a pixel centre, a depth order at an internal edge, and
  `quant="trunc"` on a value at an integer boundary;
* control: **MujocoHopper** (the unchanged nvdiff path, same `Scene.clip`) on the same job, same
  protocol: 0 differing rows at every size, 0 px on row 0 for every prefix.

Where the ulps come from (`tier/probe_batch/probe_clip_ulps_6407256_gpu2264.json`, job 6407256,
`scratch/step7/rocket-landing/probe_clip_ulps.py`): with the inputs `xpos`, `xmat` and the view matrix bit-identical for row 0 at every B, the dense batched transform `mvp = (proj @ view)[:, None] @ M` — one cuBLAS batched GEMM over B·G = 13·B 4x4 products — is identical for B = 24 and differs for B = 31, 32, 64 in **46 of row 0's 208 entries, max |diff| 1.5e-5** (float32 ulps at this cell's 100-500 m coordinates); the three `torch.sparse.mm` stages are exact given the same input (SpMM alone on the B = 1 `mvp` replicated: 0 differing elements at every B; `nrm` and `wp`, which take `xmat` directly, 0 at every B), the dense GEMM alone with row 0 replicated reproduces the split exactly (0 at B = 24, the same 46 entries at 31/32/64), the 46 entries propagate to **1282 of row 0's 1528 clip-space `pos` elements** (max 1.5e-5), and `dr.rasterize` fed identical positions replicated to B = 24/32/64 returns identical coverage. TF32 is off (`allow_tf32 False`, `float32_matmul_precision 'highest'`), so this is cuBLAS choosing a different batched-GEMM kernel — a different FMA/summation order — once the batch of tiny matrices crosses a size between 13·24 = 312 and 13·31 = 403, not a reduced-precision path. Reproduced bit-for-bit on a second 3090 node (gpu2264, job 6407256: the same 10 px, the same 46 entries; `row0_in64.png` identical across the two nodes).

Why this cell shows it and the locomotion cells do not: its edges are crisp against a flat white sky
(no anti-aliasing, `supersample 1`, TinyRenderer has none), its interior has the open tube's internal
edges, its quantisation is truncation (any ulp on an integer boundary flips a level; the MuJoCo cells
round), and its far ground is a nearest-sampled sub-pixel checker. An ulp that is invisible on a
mipmapped, rounded, smoothly shaded Hopper is a 255 here. It is a real batch dependence of the public
`render()` — the same state gives different pixels in a loader's last partial batch — and the mandatory
gate and `test_frame_i_is_independent_of_batch_size` are right to fail it. The fix belongs in the
transform's numerics (a batch-invariant per-row path, or the transform in float64 then cast — 382
vertices, trivial), and touches `nvdiff.py`'s shared `Scene.clip` or the cell's use of it; not made here.

#### Defect B — the reference bridge rejects states the cell's own `sample_states` draws near the ground (`stable_envs/envs/rocket_landing.py`)

The suite's fidelity test aborted at pos error 0.266 m. Census (`tier/reference_rejections_local.json`,
`scratch/step7/rocket-landing/probe_reference_rejections.py`, CPU, Apple M5, `/Users/samib/se7-rocket-venv`,
PyBullet DIRECT): 15 seeds x 64 `sample_states` = 960 states through `_reference_frame` — **1 rejection**
(seed 1, index 56: height **2.07 m**, tilt 3.4 deg, |v| 10.4 m/s, |w| 1.3 rad/s, pos error 0.071 m); the four
measure_tier seeds (1000-1003), the under-variation seed and `STATE_SEED` have 0. Mechanism: at ~2 m a
near-upright rocket is standing on its legs, so the recipe's one gravity-off `stepSimulation` from the
previous-substep pose includes a ground/pad contact impulse and the requested lagged pose is unreachable
by construction (the corpus's lowest state is 1.9 m because that is where its episodes ended, on the
ground); `sample_states` draws heights log-uniformly down to 1.9 m, so roughly one state in a thousand
sits in the contact band — about one 64-state draw in fifteen aborts `check_visual_fidelity` (job 6406061's
pytest draw did, with a larger error, presumably a tumbling attitude intersecting the ground). It is
RC-1's class of failure on a state class RC-1's fix did not cover. Two honest fixes, either in
`stable_envs/`: keep `sample_states` above the contact band (the picture is unchanged, states there are
crashes), or make `reference_render` skip/record contact states rather than raise. Not made here.

#### What would be declared, and why it is not

Worst of four draws (gate 3): **mean_abs 7.71, within-8 0.961, SSIM 0.854** (spread 7.48-7.71 /
0.9613-0.9622 / 0.8545-0.8684; interior-exact 0.967-0.969; exact 0.959-0.961; under variation 7.65 /
0.958 / 0.896). Under the family's convention (DMC: ceiling ~1.3x the worst draw, floors ~0.93-0.95x,
round numbers) the thresholds would be **`mean_abs_max` 10.0, `within8_min` 0.91, `ssim_min` 0.80**,
`measured_at` naming 480 px / ss 1 / default appearance / worst of 4 x 64 / RTX 3090 gpu2109 / jobs
6395707 + 6406061 / tree cf355a3, `reference_versions` `{"PyFlyt": "0.29.0", "pybullet": "3.2.7"}`,
`measured_with` `{"nvdiffrast": "0.4.0", "torch": "2.14.0+cu126"}`. Plausibility against Phase 0: within-8
0.961 is above the > 0.95 estimate; mean_abs 7.7 and SSIM 0.85 are worse than the 1-4 / > 0.95 estimate
and match the M3 local preview (5.87 / 0.863 on the corpus; 6.97 / 0.890 for the corpus on the card) —
the residual is the predicted class (silhouettes, checker edges, far ground), concentrated: 4 % of the
pixels carry a mean of ~190/255 each. **Not declared**, because (i) the M4 rule stops on a code defect,
and (ii) a renderer whose pixels depend on the batch size does not yet have ONE number to declare
(the 4 x 64 draws render in batches of `env.B` = 16, the suite's test in a batch of 64: the ~10 px per
frame do not move these aggregates, but the principle is the point). Also to note before the fill:
`tests/unit/test_rocket_landing.py::test_the_visual_declaration_is_complete_and_marked_unmeasured` pins
the placeholder (every number `None`, `measured_at` starting `UNMEASURED`) and must be flipped to
assert the measured declaration in the same commit that fills it — a `tests/` edit, not made here.

#### Tests run for M4 (local)

`tests/unit` under `/Users/samib/se6-venv` (`MUJOCO_GL=glfw`, `test_examples.py` deselected):
**745 passed, 7 skipped, 1 failed** — the one is `test_every_cited_scratch_path_resolves` with its 9
pre-existing Step-6 citations (baseline count 9; no new dead citation from this milestone).
`tests/unit/test_rocket_landing.py` under `/Users/samib/se7-rocket-venv`: **17 passed** (pytest 9.1.1
was installed into that venv with `uv pip install --python .../se7-rocket-venv/bin/python pytest`; it had
none). On the card (`tier/unit_tests_gpu_node.txt`): the two nvdiff test files and the rocket file, 46 passed.

SLURM: **6395707** (se7-rocket-gpu, 00:03:31), **6406061** (se7-rocket-gpu, 00:03:11, run of record),
**6407151** (se7-rocket-probe, 00:00:11), **6407256** (se7-rocket-probe) — all `gpu`/`norm-gpu`, one at a
time, gpu2109; none cancelled; no simclr job touched.

### M5 · the two defect fixes on the card (job 6408817) — defect B FIXED, defect A fixed then REVERTED, declaration still NOT filled

**Classification for this milestone: BLOCKER (narrowed).** Both defects were fixed (`d1e61a1`), the
whole chain was re-run once on one RTX 3090, and **every gate that was failing for a code reason now
passes** — `check_batch_size_invariance` is 0 over a 4,096-state stack, the reference bridge no longer
rejects a state its own `sample_states` draws, and the four fidelity draws complete with 0 rejections.
But the defect-A fix was authorized under a hard condition — *the six locomotion nvdiff cells must
remain byte-identical baseline-vs-branch on the same card* — and that condition **failed**. Under the
condition's own rule (if any pixel moves: stop, revert, report) the mvp change is reverted in `283865b`;
`stable_envs/render/nvdiff.py` and `tests/unit/test_nvdiff_triangles.py` are byte-identical to `e74d0bd`
again. The defect-B fix is kept. `visual_fidelity` therefore **stays `UNMEASURED`**: on the shipped tree
the frame is batch-dependent again, and a renderer whose pixels depend on the batch size does not have
one number to declare. What would be declared is recorded below, measured, with its provenance.

Provenance common to every number in this section: measured tree **`d1e61a1efafdfa2eb144909b034db5f735466008`**
(stamped into every gate line and every JSON as `sha`; `stable_envs_file` = `$SCR/se7/rocket-landing/stable_envs/__init__.py`),
SLURM job **6408817** (`se7-rocket-gpu`, `gpu`/`norm-gpu`, `--gres=gpu:1 --constraint=geforce3090
--cpus-per-task=4 --mem=48G`, elapsed **00:03:39**, `COMPLETED`), host **gpu2109.oscar.ccv.brown.edu**,
**NVIDIA GeForce RTX 3090**, `$SCR/venvs/se7-rocket` (torch 2.14.0+cu126, nvdiffrast 0.4.0, PyFlyt 0.29.0,
pybullet 3.2.7, stable-worldmodel 0.1.1), **4 threads**, node **load average 31.32 / 32.54 / 32.95** at job
start and **32.96 / 32.83 / 33.04** at the tier measurement (the node was shared; the GPU numbers moved
< 1 %, the two CPU comparators moved 3-6 % against job 6406061). Artifacts: `tier/job6408817/`,
`nvdiff_identity/job6408817/`, log `logs/se7-rocket-gpu_6408817.out` (verbatim; in the `.txt` copies the
Oscar scratch root is written `$SCR`). The EGL messages at job exit are cosmetic. One GPU job at a time;
no `simclr` job touched; nothing cancelled.

#### The seven gates

| # | gate | result | evidence (job 6408817) / mechanism on failure |
|---|---|---|---|
| 1 | six locomotion nvdiff cells, baseline (`76dc511`) vs branch, same job, same card | **FAIL** | `nvdiff_identity/job6408817/compare.json`: `all_identical: false`, `same_gpu: true`. Per cell, 64 states at 240 px, differing px of 3,686,400 and max abs: HalfCheetah 13 / 1, Hopper 8 / 1, Walker2d 4 / 1, Ant 32 / 8, Humanoid 14 / 81, Swimmer 180 / 67. **Mechanism:** the batch-invariant mvp computes ONE rounding of `proj @ view @ M` (float64, cast once); the baseline's frames are the rounding cuBLAS's batched 4x4 GEMM happened to produce at the batch width each call used. The two differ by <= 1 ulp of the transform, which flips a few edge pixels. This is not a bug in either result — it is the same ulp mechanism as defect A, seen from the other side. The cells' own `test_visual_tier_matches_its_declared_fidelity` could not be run as a cross-check: `mujoco.GLContext` fails on gpu2109 (`nvdiff_identity/job6408817/egl_probe.txt`), the node EGL problem already recorded for gpu2104. |
| 2 | first successful `PFRocketLanding` render | **PASS** | 480x480x3 uint8, 19,721 non-white px, 161 distinct colours, `tier/job6408817/first_render.png`; printed before any measurement. (19,720 px on the pre-fix tree — the one-pixel difference is the same ulp term.) |
| 3 | 4 x 64 visual fidelity + under variation | **PASS** | `default_rng(1000..1003)`, 64 float32 `sample_states`, default appearance, 480 px, ss 1, vs PyFlyt 0.29.0 / pybullet 3.2.7 TinyRenderer at the lagged pose. mean_abs **7.4816 / 7.6781 / 7.5727 / 7.7090**, within-8 **0.96225 / 0.96175 / 0.96200 / 0.96129**, SSIM **0.8632 / 0.8545 / 0.8684 / 0.8595**, exact 0.9594-0.9606, interior-exact 0.9675-0.9688. **0 rejections, 0 unreachable rows** in all four draws. Worst of four: **mean_abs 7.709, within-8 0.9613, SSIM 0.8545** (identical over all rows and over reachable rows, because no row was flagged). Under variation (seed 7): 7.646 / 0.9580 / 0.8961, exact 0.9030, 0 unreachable rows. |
| 4 | corpus replay (200 frozen frames) | **PASS** | ours(`state13`) vs the frozen frames: mean_abs **6.9677**, exact 0.95736, within-8 **0.96067**, SSIM **0.8901**, interior-exact 0.96677, worst frame 13.505 (index 85). The state-quantisation term alone, `reference_render(state13)` vs the same frames: 200/200 rendered, 0 rejections, 1 exact, mean 0.002/255, max 118 differing px. |
| 5 | mandatory conformance | **PASS (all four)** | `check_batch_invariance` 0; **`check_batch_size_invariance` `{'max_abs_diff': 0, 'batch_sizes': (1, 7, 64, 4096), 'n': 4096}`** — was 255 at `e74d0bd`; `check_determinism` identical; `check_state_dims_matter` no dead dims (the six velocity dims all move pixels). The suite's own `test_frame_i_is_independent_of_batch_size` (sizes 1/3/7/8/64/1024) **passes**. Prefix-batch probe (`tier/job6408817/probe_batch/`): row 0 vs its single render is **0 px at every prefix 1, 2, 3, 4, 7, 8, 15, 16, 17, 24, 31, 32, 33, 40, 48, 63, 64 and in a batch of 1024**; **0 of the 64 rows** differ from their single renders; 64-render deterministic; `row0_diffmask_6408817.png` is empty. MujocoHopper control: 0 everywhere, as before. |
| 6 | GPU sweep, tier | **PASS as a measurement; `BATCH_SCALES` FAIL; `NO_LARGE_BATCH_REGRESSION` PASS; not `HOST_BOUND`** | f/s at B = 1 / 16 / 64 / 256 / 1024 / 4096: **232.2 / 2028.3 / 3015.4 / 2712.1 / 3195.6 / 3381.1**; `cpu_frac` 0.991 / 0.992 / 0.994 / 0.994 / 1.156 / **1.054** at the certified batch (< 1.5, so no `HOST_BOUND`, but 1.05 says the wall clock is one host core: the per-row float64 kinematics and camera in `render()` plus two host->device uploads). `BATCH_SCALES` fps(4096)/fps(1) = **14.56** (bar 20) **FAIL**; `NO_LARGE_BATCH_REGRESSION` fps(4096)/fps(256) = **1.247** (bar 0.9) PASS. Same-node comparators: official TinyRenderer through the swm env **14.52 f/s**, one-core JPEG-q90 decode of this env's own frames **1312.7 f/s** (`decode_cpu_frac` 0.993, 14,741 B/frame). Class S: **R = 232.81**, **cores = 2.576** at B = 4096; tier **`GPU_PARITY`, qualifier `BATCH_SCALES_FAIL`** (capped from the R/cores tier by section 6.7). |
| 7 | PyBullet DIRECT + EGL + CUDA raster coexistence across the numba JIT | **PASS** | one process: reference env created (DIRECT, ~57 s JIT paid), `mujoco.GLContext` created (`egl: ok` inside the process even though the standalone probe failed on this node), the same state re-rendered **byte-identical**. |

Conformance suite on the card (`tier/job6408817/conformance_PFRocketLanding.txt`, full `-rA` output):
**30 passed, 9 skipped, 3 failed** — down from 4 failed at `cf355a3`; the batch-size failure is gone.
The three, each with its mechanism:

* `test_visual_tier_matches_its_declared_fidelity` — **not a fidelity failure**: the declaration is still
  the `UNMEASURED` placeholder, so the comparison is `7.709 <= None`. The measured numbers are in gate 3
  and clear the ceilings that would be declared with 23-30 % margin. It passes as soon as the
  declaration is filled, which is what gate 1 is blocking.
* `test_color_knobs_apply_the_requested_color` — **known, escalated to Sami, not acted on**: TinyRenderer's
  law MULTIPLIES the material colour by `0.6 + 0.35*max(0, n.l) + 0.05*max(0, r_z)^2`, which is < 1 at
  every pixel (with `l` the normalised `(-50, 30, 100)`, `n.l = 1` forces `r_z = 0.864`), and truncates,
  so `int(255*c*f) < 255*c` for every channel and the requested colour is never literally in the frame.
  The official PyFlyt frame does not contain it either (Phase 0: the brightest body pixel is `0.991*int(255*c)`).
  The check's premise is flat shading; on a lit cell it measures the shading law, not the knob wiring
  (which `test_variations_all_do_something` and the under-variation fidelity number cover). `tests/` is frozen.
* `test_resolution_downsample_agrees_within_the_measured_bound` — **known Step-8 freeze item, not acted on**:
  `DOWNSAMPLE_EXACT_FRAC` has no `PFRocketLanding` entry. Measured on the card at the test's own protocol
  (8 states, `default_rng(render_worker.STATE_SEED)`, default appearance): **`DOWNSAMPLE_EXACT_FRAC` =
  0.95614**, mean 0.9920/255, max 255, on the RTX 3090 at torch 2.14.0+cu126 / nvdiffrast 0.4.0. `tests/` is frozen.

#### Defect A — fixed, measured, and reverted

The fix (`d1e61a1`, `stable_envs/render/nvdiff.py`): `mvp_rows(proj, view, M)` replaced
`(proj @ view)[:, None] @ M` in `Scene.clip`. It computes both 4x4 products as unrolled elementwise
products summed in a fixed order, in float64, cast once to float32 — no matmul, no reduction kernel,
so no kernel choice and no dependence on how many rows share the launch. Generic, the default path for
every caller, no per-env branch, plus two CPU unit tests (row-wise bitwise equality across prefix
batches 1..4096 at the rocket's 100-500 m coordinate scale, and equality to the correctly rounded
float64 product; `Scene.clip` row 0 alone == row 0 of 64).

**It works.** Gate 5 above: the mandatory check is 0 at windows 1/7/64/4096 over 4,096 states, the
prefix probe is 0 px everywhere, and the suite's own test passes. That is the whole of defect A.

**It cannot satisfy the condition it was authorized under, and neither can any fix of this shape.**
The baseline's locomotion frames *are* the output of the batch-dependent GEMM. A batch-invariant mvp
computes one fixed rounding; cuBLAS's rounding at whatever batch width a call used is a different one
for some entries; the gap is <= 1 ulp of the transform and flips a handful of edge pixels (gate 1:
4-180 px of 3,686,400 per cell, i.e. <= 4.9e-5 of the frame area, max |diff| 1-81). "Batch-invariant"
and "bit-identical to a batch-dependent baseline" are incompatible propositions, and the authorization
forbids reconciling them with a per-env branch. So this is a contract decision, not an implementation
one, and it is put to the coordinator rather than taken here:

* **(a) accept the re-baseline.** Take the fix, and re-verify the six locomotion cells' declared
  fidelity on a node with working EGL (their declarations carry 1.3x ceilings and the change is
  <= 4.9e-5 of pixels, so they will pass — but that is an argument, not a measurement, and gpu2109's
  EGL failure means it was not made in this job). Every cell in the repo then renders batch-invariantly
  and `PFRocketLanding` certifies.
* **(b) keep bit-compatibility.** `PFRocketLanding` keeps failing a mandatory gate and cannot be
  registered at any tier; the batch dependence stays latent in every nvdiff cell (invisible today only
  because the locomotion cells' ~1 m coordinates, rounded quantisation and smooth shading hide the ulp).

The reverted patch is recoverable at `d1e61a1`; `283865b` restores `e74d0bd`'s `nvdiff.py` byte for byte.

#### Defect B — fixed, kept, and declared as an exclusion

`_reference_frame_flagged(env, state, style_row, contact_fallback=True)` returns `(frame, unreachable)`.
When the velocity fixed point does not reach the requested pose **and** the requested pose is in the
contact band — any link collision AABB within `CONTACT_BAND_MARGIN` = 5 cm of the pad top
`PAD_TOP_Z` = 0.152 m / the ground — the lagged picture is unreachable by this bridge by construction
(the recipe's one gravity-off substep contains a ground contact, which Bullet resolves with a
split-impulse position correction the fixed point cannot absorb). The frame returned is then swm's
**synced** picture of the requested pose, which is swm's own goal-image convention, and the row is
flagged in `PFRocketLanding.last_reference_unreachable`. A failure **outside** the band still raises;
`_reference_frame` keeps the strict form for the validators. `sample_states` is **not** narrowed.

How the exclusion reaches the numbers: `check_visual_fidelity` in `stable_envs/testing/conformance.py`
is not edited. It calls `cls.reference_render(ref, states)` and gets a full stack back, so it can no
longer abort; it then computes over **all** rows, flagged ones included, which is the conservative
direction (a synced picture differs from the lagged one by the one-substep lag). The measurement
harness `scratch/step7/rocket-landing/measure_tier.py` mirrors that function line for line in
`fidelity_draw()` — same rng consumption, same reference, same metrics — reads the flags immediately
after the call, and reports both: the all-rows numbers the shared check computes and the
reachable-rows numbers the declaration would carry, with the flagged count on every draw.

Measured share, to be declared with the mechanism: **`REFERENCE_UNREACHABLE_SHARE` = 1/960**
(`tier/reference_rejections_local.json`: 15 seeds x 64 `sample_states`, one rejection — seed 1 index 56,
height 2.07 m, tilt 3.4 deg, |v| 10.4 m/s). Verified on this machine at `d1e61a1`
(`/Users/samib/se7-rocket-venv`, PyBullet DIRECT): that state is now **flagged, not raised**, classified
"in the contact band"; all 63 other states of that seed and all 64 of seeds 1000 and 1001 are unflagged;
`reference_render` returns `(64, 480, 480, 3)` with `last_reference_unreachable` true at index 56 only.
On the card, the four certification draws and the under-variation draw flagged **0 rows** and rejected
**0 states** (gate 3) — consistent with 1/960. Unrelated and pre-existing, recorded again: one frame per
64-state seed is not bit-reproducible across PyBullet call order (the reviewer's index-124 observation);
it is not a function of this change.

#### What would be declared, and why it is still not

Worst of four draws over the reachable rows (which is also the worst over all rows, since none was
flagged): **mean_abs 7.709, within-8 0.9613, SSIM 0.8545**; spread 7.4816-7.7090 / 0.96129-0.96225 /
0.8545-0.8684; interior-exact 0.9675-0.9688; exact 0.9594-0.9606; under variation 7.646 / 0.9580 / 0.8961.
Under the family's convention (DMC: ceiling ~1.3x the worst draw, floors ~0.93-0.95x, round numbers,
never tuned to the number) the declaration would be **`mean_abs_max` 10.0** (1.297x), **`within8_min`
0.91** (0.947x), **`ssim_min` 0.80** (0.936x), with `measured_at` naming 480 px / ss 1 / default
appearance / worst of 4 x 64 over the rows the bridge reaches / RTX 3090 gpu2109 / job 6408817 / tree
`d1e61a1`, `reference_versions {"PyFlyt": "0.29.0", "pybullet": "3.2.7"}`, `measured_with`
{"nvdiffrast": "0.4.0", "torch": "2.14.0+cu126"}`, and `reference_unreachable_share` 1/960 with the
contact-band mechanism. `scratch/step7/rocket-landing/fill_declaration.py` prints that block from the
tier JSON for a human to paste; it does not edit the module, and the ceilings it prints are the
convention applied to the measurement (`round(1.3 * worst)`, `round(0.95 * worst, 2)`,
`round(0.94 * worst, 2)`) rather than numbers chosen to clear it — run on this job's JSON it emits
exactly 10.0 / 0.91 / 0.80 (`tier/job6408817/declaration_block.json`).

**Not filled**, for one reason only: gate 1. The numbers above were measured on `d1e61a1`, and `283865b`
reverted the change that made the frame batch-independent. On the shipped tree the same state renders
differently in a batch of 24 and a batch of 31, so there is no single number to declare, and
`tests/unit/test_rocket_landing.py::test_the_visual_declaration_is_complete_and_marked_unmeasured` is
left as it is. The moment the coordinator rules on option (a), the declaration is one paste and one
test flip away and needs no further GPU time: these numbers already are the numbers.

#### Tests run for M5 (local, at `283865b`)

`tests/unit` under `/Users/samib/se6-venv` (`MUJOCO_GL=glfw`, `test_examples.py` deselected):
**747 passed, 7 skipped, 1 failed** — the one is `test_every_cited_scratch_path_resolves` with its 9
pre-existing Step-6 citations, unchanged. `tests/unit/test_rocket_landing.py` under
`/Users/samib/se7-rocket-venv`: **17 passed**. On the card at `d1e61a1`
(`tier/job6408817/unit_tests_gpu_node.txt`): the two nvdiff test files and the rocket file, **48 passed**
(46 at `cf355a3`; the two extra were the reverted mvp tests).

SLURM: **6408817** (`se7-rocket-gpu`, 00:03:39, `COMPLETED`, gpu2109) — one GPU job at a time, on
`gpu`/`norm-gpu`, excluding gpu2108/gpu2110/gpu2113/gpu2116 (and gpu2104); none cancelled; no `simclr`
job touched.

### M6 · the correctness re-freeze (Sami's ruling; job 6410105) — measured, and it needed no frozen-artifact edit

**Classification for this milestone: PATCH.** Sami ruled on 2026-09-15 that *mandatory batch-size
invariance takes precedence over accidental float bit identity*, authorised the batch-invariant mvp,
and directed that the six locomotion nvdiff cells be re-baselined as a CORRECTNESS RE-FREEZE provided
the evidence below holds. The M5b revert is therefore itself reverted (`dac40f9`), the harness was
written (`8ca2d72`, `726241e`) and one GPU job on a node with working EGL produced every number the
ruling asks for. **All of it holds.** The one thing the ruling anticipated that did not happen is the
re-freeze edit itself: measured rather than assumed, **no frozen artifact under
`examples/results/step4` needs to change**, and the reason is recorded below so a reviewer can check
the claim rather than take it.

Provenance common to every number in this section: measured tree
**`726241e8d03aec59789562839d658f87c8ae2891`** (stamped into every JSON as `sha`, with
`stable_envs_file` resolved from the package actually imported — `$SCR/se7/rocket-landing/...` for the
branch leg and `$SCR/se7/baseline/...` for the baseline leg, both recorded), baseline
**`76dc5116f432e61dd47fba8f9e879cf9212c8e46`**, SLURM job **6410105** (`se7-rocket-refreeze`,
`gpu`/`norm-gpu`, `--constraint=geforce3090 --cpus-per-task=4 --mem=48G`, elapsed **00:06:37**,
`COMPLETED`), host **gpu2106.oscar.ccv.brown.edu**, **NVIDIA GeForce RTX 3090**, **4 threads**, node
**load average 15.36 / 16.05 / 16.44** at job start and **15.79 / 16.07 / 16.43** at the measurement.
Both legs ran **in the same job on the same card**. Interpreters: `$SCR/venvs/se7-rocket` (torch
2.14.0+cu126, nvdiffrast 0.4.0, mujoco 3.12.0, gymnasium 1.3.0, PyFlyt 0.29.0, pybullet 3.2.7) for
everything nvdiff, `$SCR/venvs/se6-verify` (pymunk, gym_pusht, dm_control, mujoco_warp) for the Step-5
surfaces. Artifacts: `refreeze/`, `tier/job6410105/`, log `logs/se7-rocket-refreeze_6410105.out`
verbatim. The `EGLError` tracebacks in the log are the cosmetic teardown at interpreter exit
(`GLContext.__del__` -> `glCheckError`), three of them, none inside a measurement.

One prior attempt is recorded rather than hidden: job **6409703** landed on gpu2264 and exited 2 after
six seconds when `RasterizeCudaContext(device='cuda')` raised `invalid argument to exchangeDevice`.
That was an initialisation-order failure of the probe itself (the bare string `'cuda'` resolved against
a current device nothing had set), fixed in `726241e`; the guard cost six seconds instead of a
three-hour slot, which is what the guard is for.

#### What the six cells are, and what a re-freeze could even touch

Established before measuring anything, because it decides the scope of the authorized Step-4 edit:

* the six cells have **no frozen frame digest anywhere in `examples/results/step4`**. They are not
  among its **49** configs, they have no `states/<config>.npz`, and they appear in **none** of the five
  `conformance/<shard>.json` files (dmc 14, fetch 12, ogb 11, raster 6, swm 6 — 49, none of them a
  locomotion nvdiff cell);
* the **only** Step-4 artifact that names them is `certification.json`'s
  `bounds.downsample_exact_frac.value`, which is a mirror of
  `tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC` (six entries at lines 530-535) and is
  asserted equal to it by `tests/unit/test_step4.py:437`. It is a **fraction with a 0.005 margin**, not
  a digest;
* so the whole Step-4 surface at risk is those six floats, and the question "does the re-freeze need to
  move them" is a measurement, made below.

The six cells' own `VISUAL_FIDELITY` declarations in `stable_envs/envs/mujoco_locomotion.py` are the
other thing a re-freeze could touch, and they are judged by the before/after table below.

#### Per-cell pixel deltas — baseline `76dc511` vs branch, same job, same card

64 states per cell from `sample_states(64, rng=default_rng(0))` (the draw `nvdiff_identity.py` uses,
written down rather than inherited), rendered at each cell's default 240 px in chunks of 16.
**3,686,400 pixels per cell** (64 x 240 x 240); a pixel counts as changed if any of its three channels
moved.

| cell | max abs Δ | px changed | frac changed | px Δ > 1 level | frac Δ > 1 | verdict |
|---|---|---|---|---|---|---|
| `MujocoHalfCheetah` | 1 | 13 | 3.53e-06 | **0** | 0 | ULP only |
| `MujocoHopper` | 1 | 8 | 2.17e-06 | **0** | 0 | ULP only |
| `MujocoWalker2d` | 1 | 4 | 1.09e-06 | **0** | 0 | ULP only |
| `MujocoAnt` | 8 | 32 | 8.68e-06 | 5 | 1.36e-06 | ULP only |
| `MujocoHumanoid` | 81 | 14 | 3.80e-06 | 3 | 8.14e-07 | ULP only |
| `MujocoSwimmer` | 67 | 180 | 4.88e-05 | 2 | 5.43e-07 | ULP only |

The shape of this is the mechanism, not an accident: at most **five pixels in 3.7 million** move by
more than one level in any cell, and the largest `max abs Δ` (81 on Humanoid) comes from **three**
pixels. A ≤ 1-ulp change in the transform moves a silhouette edge across a pixel centre; where that
edge separates two similar colours the pixel moves by one level, and where it separates a limb from the
background it moves by a lot — but it is the same handful of edge pixels either way. Nothing shifts:
no cell has a region that moved, no cell has a systematic direction, and the aggregate metrics below
move in both directions across the six.

#### Official-reference visual fidelity, BEFORE and AFTER — the number that had to be unchanged or improved

`check_visual_fidelity(env, n=64)` against the **official gymnasium MuJoCo renderer through EGL**, with
`env.rng` reseeded identically in both legs so the two sides draw the same 64 states. This is the
measurement the cells' declarations are made of.

| cell | mean_abs before → after (Δ) | within8 before → after (Δ) | SSIM before → after (Δ) | inside its declaration |
|---|---|---|---|---|
| `MujocoHalfCheetah` | 4.0421026 → 4.0421023 (**−2.7e-07**, better) | 0.8755037 → 0.8755037 (0) | 0.96661498 → 0.96661498 (**+2.5e-10**, better) | yes (ceiling 5.5 / floors 0.83 / 0.94) |
| `MujocoHopper` | 4.4960767 → 4.4960764 (**−2.7e-07**, better) | 0.8408841 → 0.8408841 (0) | 0.94805810 → 0.94805811 (**+2.9e-09**, better) | yes (5.8 / 0.80 / 0.92) |
| `MujocoWalker2d` | 3.9216290 → 3.9216290 (**0**) | 0.8683502 → 0.8683502 (0) | 0.94724463 → 0.94724463 (−2.9e-09) | yes (5.2 / 0.83 / 0.92) |
| `MujocoAnt` | 3.4481182 → 3.4481209 (+2.7e-06) | 0.8918359 → 0.8918359 (0) | 0.97758157 → 0.97758132 (−2.6e-07) | yes (4.6 / 0.85 / 0.95) |
| `MujocoHumanoid` | 4.4825600 → 4.4825635 (+3.5e-06) | 0.8359768 → 0.8359768 (0) | 0.95101683 → 0.95101664 (−1.9e-07) | yes (5.8 / 0.79 / 0.92) |
| `MujocoSwimmer` | 1.9697748 → 1.9697591 (**−1.6e-05**, better) | 0.9443349 → 0.9443351 (**+2.7e-07**, better) | 0.98219508 → 0.98219515 (**+7.1e-08**, better) | yes (2.7 / 0.91 / 0.96) |

**Every cell stays inside its declared thresholds**, by three to five orders of magnitude of margin
(the tightest is Swimmer's `mean_abs` 1.9698 against a ceiling of 2.7). `within8_frac` is unchanged to
the last digit on five of six. Three cells improve on every metric, one is exactly unchanged on two of
three, and **three show a regression in the 7th to 10th significant figure**: Walker2d SSIM −2.9e-09,
Ant `mean_abs` +2.7e-06 and SSIM −2.6e-07, Humanoid `mean_abs` +3.5e-06 and SSIM −1.9e-07.

Those three are reported prominently rather than absorbed, because the ruling's stop condition names
"any fidelity regression", and on the most literal reading these are regressions. What they are in
fact: `mean_abs` is a mean over 3,686,400 pixels, so a delta of 2.7e-06 is arithmetically **exactly
the ten level-steps** the pixel table already counted for Ant, divided by the pixel count — the two
tables are the same event measured twice. There is no separate effect. The direction is not
systematic (three cells better, three worse, and the two "worse" cells are the two with the largest
single-pixel flips), which is the signature of rounding noise rather than of a renderer that got
worse. And the criterion's own governing clause is "a **visually meaningful** change": a change of
2.9e-09 in SSIM is not one under any reading.

**So the substantive answer is "unchanged or improved" and the literal answer is "three cells moved in
the 7th figure", and the decision does not turn on which one is taken** — because of the next section.

#### What the re-freeze actually had to edit: nothing

The downsample bound, measured on both legs at the protocol's own eight states
(`conformance.downsample_protocol_states`, `render_worker.STATE_SEED`, default appearance):

| cell | exact_frac before | after | Δ | frozen bound | bound − margin | after clears | needs update |
|---|---|---|---|---|---|---|---|
| `MujocoHalfCheetah` | 0.765822 | 0.765818 | −4.3e-06 | 0.758811 | 0.753811 | yes | **no** |
| `MujocoHopper` | 0.695799 | 0.695799 | **0** | 0.694054 | 0.689054 | yes | **no** |
| `MujocoWalker2d` | 0.761356 | 0.761356 | **0** | 0.755514 | 0.750514 | yes | **no** |
| `MujocoAnt` | 0.770716 | 0.770707 | −8.7e-06 | 0.771469 | 0.766469 | yes | **no** |
| `MujocoHumanoid` | 0.682871 | 0.682871 | **0** | 0.681649 | 0.676649 | yes | **no** |
| `MujocoSwimmer` | 0.910595 | 0.910597 | +2.2e-06 | 0.909481 | 0.904481 | yes | **no** |

Every cell clears its frozen bound on both legs with 1.2e-03 to 5.8e-03 of headroom, and the largest
movement is 8.7e-06 — three orders of magnitude inside the 0.005 margin the bound was frozen with.
**So `tests/gpu/test_reproducibility.py:DOWNSAMPLE_EXACT_FRAC` is untouched, and so is its mirror in
`examples/results/step4/certification.json`.** With no frame digest existing for these cells and no
declaration exceeded, **`examples/results/step4` is not edited at all in this milestone** — the
authorization to edit it is left unused, which is the right outcome for an authorization of that kind,
and `tests/unit/test_step4.py:437` (which asserts the mirror equals the table) still passes untouched.

The one `tests/` change in this milestone is the rocket's own declaration test, below. No regression
expectation anywhere pinned the six cells' frames, because none existed to pin.

#### Gate A on the branch — the full mandatory set, all six cells

| cell | determinism | batch invariance | batch-size invariance (1/3/7/8/64/256, n=256) | order independence | chunk invariance (1/7/64) | render purity | state_dims_matter |
|---|---|---|---|---|---|---|---|
| `MujocoHalfCheetah` | identical | 0 | **0** | 0 / 0 | 0 | 0/0/0/0 | no dead dims |
| `MujocoHopper` | identical | 0 | **0** | 0 / 0 | 0 | 0/0/0/0 | no dead dims |
| `MujocoWalker2d` | identical | 0 | **0** | 0 / 0 | 0 | 0/0/0/0 | no dead dims |
| `MujocoAnt` | identical | 0 | **0** | 0 / 0 | 0 | 0/0/0/0 | no dead dims |
| `MujocoHumanoid` | identical | 0 | **0** | 0 / 0 | 0 | 0/0/0/0 | no dead dims |
| `MujocoSwimmer` | identical | 0 | **0** | 0 / 0 | 0 | 0/0/0/0 | no dead dims |

`all_pass: true` (`refreeze/gatea_branch.json`). Each check is judged by the condition the suite itself
asserts, transcribed rather than invented: `identical` for determinism, `dead_dims` empty for state
dims, all four purity deltas zero, both order deltas zero, `max_abs_diff == 0` for the three
invariances.

#### The Step-5 representative surfaces are untouched — stated explicitly, and proved twice

`PushT` carries Step 5's LeWM gate (`swm/PushT-v1`, `examples/results/step5/pusht/gate_h.json`) and
`swm/ReacherDMControl-v0` carries the DINO-WM gate (`examples/results/step5/reacher/qualification.json`),
which is `DMCReacherEasy` / `DMCReacherHard` in this repo. **Neither renders through nvdiff.**

| config | backends (frozen shard) | module | nvdiff `Scene.clip` calls | `BatchRenderer.render` calls | frozen digest | baseline | branch |
|---|---|---|---|---|---|---|---|
| `PushT` | `["scene"]` (raster2d, `render/scene.py`) | `stable_envs.envs.pusht` | **0** | **0** | `cecda14accd3f029b…` | **=** | **=** |
| `DMCReacherEasy` | `["mjwarp", "raycast3d"]` | `stable_envs.envs.dmcontrol` | **0** | **0** | `4d684da4ce6b1677e…` | **=** | **=** |
| `DMCReacherHard` | `["mjwarp", "raycast3d"]` | `stable_envs.envs.dmcontrol` | **0** | **0** | `04e3961cabb2a57d4…` | **=** | **=** |

`all_untouched: true` (`refreeze/step5_surfaces_compare.json`). The digests are `oracle.digest_ours`
recomputed through `examples/fidelity_step4.py`'s OWN `reseed` / `corpus_for` / `oracle` — imported,
not reimplemented — against each config's frozen corpus (`PushT` sha `0aa3bacd24cf91ce…`, both
Reachers `f23ef2d51147a467…`, loaded and never redrawn), and they equal the frozen values on **both**
legs, so the frozen Step-4 numbers reproduce exactly and the branch changes nothing.

The structural half is checked by **call**, not by import, and that distinction matters: a
`sys.modules` check would be vacuous here, because `mujoco_locomotion.py` imports `..render.nvdiff` at
module scope, so the module is present in any interpreter where the six cells register (the JSON
records `nvdiff_module_imported: true` for exactly that reason). So the probe wraps
`nvdiff.Scene.clip` — the function the change is in — and `nvdiff.BatchRenderer.render` with a counter
and renders the config: **zero calls** for all three, on both legs. The changed code is not on their
path, by execution rather than by reading.

#### PFRocketLanding: the seven gates again, on the re-applied fix

| # | gate | result | evidence (job 6410105) |
|---|---|---|---|
| 1 | six locomotion cells vs baseline | **PASS (re-freeze criteria)** | the four tables above: ≤ 5 px in 3.7 M move by more than one level, fidelity inside every declaration, no Step-4 bound moved, Gate A all green. The byte-identity condition is withdrawn by Sami's ruling and replaced by these. |
| 2 | first successful render | **PASS** | 480x480x3 uint8, 19,721 non-white px, 161 distinct colours, `tier/job6410105/first_render.png` |
| 3 | 4 x 64 fidelity + under variation | **PASS** | worst of four **mean_abs 7.709 / within-8 0.9613 / SSIM 0.8545** (draws 7.4816 / 7.6781 / 7.5727 / 7.7090), **0 rejections, 0 unreachable rows**; under variation 7.646 / 0.9580 / 0.8961. Identical to job 6408817 on a different 3090 to every digit declared. |
| 4 | corpus replay | **PASS** | ours(`state13`) vs the 200 frozen frames: 6.9677 / 0.96067 / 0.8901, interior-exact 0.96677; quantisation term alone 0.002/255, max 118 px |
| 5 | mandatory conformance | **PASS (all four)** | `check_batch_size_invariance` **`{'max_abs_diff': 0, 'batch_sizes': (1, 7, 64, 4096), 'n': 4096}`** — **FAIL → PASS**, it was 255 at `e74d0bd`. Prefix probe: row 0 is **0 px** at every prefix 1..64 and at 1024; **0 of 64** rows differ from their single renders; Hopper control 0. |
| 6 | GPU sweep, tier | **PASS as a measurement; `BATCH_SCALES` FAIL; `NO_LARGE_BATCH_REGRESSION` PASS; not `HOST_BOUND`** | see the tier line below |
| 7 | DIRECT + EGL + CUDA coexistence | **PASS** | reference env (DIRECT, numba JIT), `mujoco.GLContext`, same state re-rendered byte-identical |

GPU conformance subset on the card for **the six cells AND the rocket**
(`tier/job6410105/conformance_six_plus_rocket.txt`): **228 passed, 63 skipped, 3 failed** in 2:39. All
six locomotion cells' `test_visual_tier_matches_its_declared_fidelity` **PASS** on the branch. The
three failures are all `PFRocketLanding` and all previously recorded:
`test_visual_tier_matches_its_declared_fidelity` (it compared against the `None` placeholder — fixed by
the declaration this milestone fills), `test_color_knobs_apply_the_requested_color` (TinyRenderer's
multiplicative law; escalated to Sami, `tests/` frozen, not acted on), and
`test_resolution_downsample_agrees_within_the_measured_bound` (`DOWNSAMPLE_EXACT_FRAC` has no
`PFRocketLanding` entry; measured **0.95614**, a Step-8 freeze item, not acted on).

Tier, frozen §6 definitions, same node, same job: f/s at B = 1 / 16 / 64 / 256 / 1024 / 4096 =
**238.1 / 2021.7 / 3006.5 / 2694.0 / 3177.6 / 3374.4**; official TinyRenderer **13.97 f/s**, one-core
JPEG-q90 decode **1295.0 f/s**; **R = 241.51**, **cores = 2.606**, `cpu_frac` 0.994 / 0.994 / 0.995 / 0.995 / 1.088 / **1.027**, i.e. **1.027**
at the certified B = 4096 (< 1.5, so no `HOST_BOUND`, though 1.03 still says the wall clock is one
host core); `BATCH_SCALES` **14.17** (bar 20) **FAIL**; `NO_LARGE_BATCH_REGRESSION`
**1.253** (bar 0.9) PASS; tier **`GPU_PARITY`, qualifier `BATCH_SCALES_FAIL`**.

#### The declaration, filled

`PFRocketLanding.visual_fidelity` now carries the measurement: **`mean_abs` 7.71, `within8_frac` 0.961,
`ssim` 0.854**, with `exact_frac` 0.959 and `interior_exact_frac` 0.967 reported for honesty, the draw
spread, the under-variation row, and **`reference_unreachable_share` = 1/960** with the contact-band
mechanism (defect B) declared beside the numbers rather than left for a reader to discover — none of
the four certification draws drew such a row. Thresholds **`mean_abs_max` 10.0, `within8_min` 0.91,
`ssim_min` 0.80**: the DMC family's convention applied to the worst draw (1.297x / 0.947x / 0.936x),
which is what `scratch/step7/rocket-landing/fill_declaration.py` prints from the tier JSON
(`tier/job6410105/declaration_block.json`) — computed from the measurement, never fitted to it. The
`UNMEASURED` marker is gone, and
`tests/unit/test_rocket_landing.py::test_the_visual_declaration_is_complete_and_marked_unmeasured` is
replaced by `test_the_visual_declaration_is_complete_and_measured`, which pins the SHAPE of an honest declaration (nothing
None, `measured_at` naming resolution/protocol/reference build/card, every threshold on the correct
side of its measurement with margin, the exclusion declared at its measured share) and deliberately
does not restate the values — those are asserted on a card by the GPU suite.

#### Tests run for M6 (local, at the M6 tip)

`tests/unit` under `/Users/samib/se6-venv` (`MUJOCO_GL=glfw`, `test_examples.py` deselected):
**747 passed, 7 skipped, 1 failed** — the one is `test_every_cited_scratch_path_resolves` with its 9
pre-existing Step-6 citations, unchanged and the only allowed failure.
`tests/unit/test_rocket_landing.py` under `/Users/samib/se7-rocket-venv`: **17 passed**.

SLURM: **6409703** (`se7-rocket-refreeze`, exited 2 on gpu2264 at six seconds, probe guard), **6410105**
(`se7-rocket-refreeze`, 00:06:37, `COMPLETED`, gpu2106) — one GPU job at a time, `gpu`/`norm-gpu`,
excluding gpu2108/2110/2113/2116 plus gpu2104, gpu2109 (broken `mujoco.GLContext`) and gpu2264; none
cancelled; no `simclr` job touched.

---

# Phase 0 — the feasibility gate (as delivered, unchanged)

**Recommendation: GO, in the VISUAL tier, through the existing nvdiffrast raster backend.** All four
gates pass on real frames, and two of the three premises of the Step-1 NO-GO are false on measurement:
the scene is not a mesh scene and the decoder is not rebuilt per episode.

## What the env actually is (census, not the Step-1 reading)

* **Not a mesh scene.** The rocket URDF has **no mesh assets**: 11 visual primitives — 2 cylinders
  (body l 4.77 r 0.185, booster l 0.5 r 0.25), 4 fin boxes, 3 leg boxes, 1 "flame" cylinder
  (l 2 r 0.15, rgba `(1, 0.8, 0, 0.1)`), plus a white cylinder l 1 r 0.1 that PyBullet synthesises
  from the fuel-tank *collision* shape (it has no `<visual>`; it is visible through the open tube, see
  below). The pad is one cylinder (l 0.1 r 2) fixed at `(0, 0, 0.1)`. The only mesh in the scene is
  pybullet_data's ground: **2 triangles**, 200 m square, one 512x512 RGBA checker texture tiled 100x
  (2 m tiles). TinyRenderer tessellates cylinders with `numSteps = 32` and boxes as fixed cubes; the
  whole scene is roughly **4 x 128 + 7 x 12 + 2 ≈ 600 triangles**, one texture.
* **The decoder is not rebuilt per episode.** `_modify_rocket_urdf` exists but is **never called**
  by `reset`; rocket colours are applied by `p.changeVisualShape` after the reset. Only the pad goes
  through a temp URDF, and that rewrite adds a `<material>` (the base pad has none) with the colour
  quantised to 3 decimals. Six resets (seeds 0,1,2,3,4,0; `variation=["all"]`, none, one leaf):
  **geometry, mass, local pose, mesh file and texture id identical on all 14 visual shapes; only
  `rgba` changes** (`phase0.json: G3.resets`).
* **`environment.sky_color` is a dead leaf**: sampled per reset, read nowhere; the background is
  TinyRenderer's clear colour `(255,255,255)` on every reset regardless of the sampled value.
  Without any variation the pad is white `(1,1,1,1)`; any variation touching `rocket.*` or `pad.*`
  writes the pad colour (`init 200/255 = 0.784`).
* **Camera** (`PyFlyt Camera`, link 0 = fuel tank, fixed at the base origin): eye = `link_pos +
  R·(-1, 0, 3)`, orientation = link euler with pitch +30° (not gimballed), **tracking** the link
  centre, up = `R·z`; `computeProjectionMatrixFOV(fov 90, aspect 1, near 0.1, far 255)`. The view
  matrix is recomputed from `getLinkState` on every access. **The rocket covers 10.0 % of the frame at
  every altitude** (23,000 px), because the camera rides with it.
* **Far plane 255 m, start height 350–475 m**: the ground is beyond the far plane for the first
  ~200 m of every descent, so the frame is *rocket on white* for roughly half the episode; the ground
  enters at ~255 m and fills the frame below ~48 m; the pad grows from 11 px (258 m) to 50,453 px
  (1.5 m) (`phase0.json: coverage.per_height`).
* **The tube is open**: TinyRenderer's cylinder has side walls only, so looking down the body you see
  its interior back faces (drawn with the outward vertex normals) and the fuel-tank cylinder inside
  (`preview_g1_pose_A.png`, `preview_descent_h2.png`).
* **cvxpy** is imported only by `expert_policy.py`, which `stable_worldmodel.envs.rocket_landing.
  __init__` imports unconditionally; the env itself does not use it. Reference construction needs
  cvxpy installed; the renderer does not.
* **PyFlyt's numba JIT costs 57.2 s per process** on the first physics steps (`@nb.njit` without
  `cache=True` plus a `jitclass`; 0.016 s with `NUMBA_DISABLE_JIT=1`). This is physics setup, not
  render setup, and a state -> frame decoder pays none of it. `gym.make` 1.5 s, later resets 0.07 s.

## The gates

### G1 · purity — PASS, with two named, reconstructible hidden bits

Measured on 480x480 frames (`phase0.json: G1`, `phase0_b.json`, `phase0_link_lag.json`):

| test | differing px |
|---|---|
| same state rendered twice | **0** |
| different history, teleported to the same base pose (teleport vs teleport) | **0** |
| velocities x100 + offset | **0** |
| fresh env instance, same seed, same pose | **0** |
| default reset recoloured by hand (`changeVisualShape`, exact rgba) to another episode's 5 colours, same pose — at 420 m and at 6 m with ground+pad in view | **0 / 0** |
| same pose after an ignition history | **61** (the flame, bbox 8x10 px) |
| **frame taken right after `env.step` vs the same reported base pose after a teleport** | **18,160** |

Two mechanisms, both measured to the bit:

1. **One-substep link lag.** After `stepSimulation`, `getBasePositionAndOrientation` is
   post-integration but the links' cached world transforms — what TinyRenderer draws for the 10 links
   *and where the camera link sits* — are one 1/240 s substep behind: 0.129–0.139 m at 30–33 m/s
   (`a_teleport.link_cached_pos_change_after_teleport_max_abs`). The official frame is therefore the
   body cylinder at pose *t* seen from a camera at pose *t−dt* with fins/legs/booster/flame at
   *t−dt* (`preview_b_diff_mask.png`). **The lagged pose is exactly `(pos − v·dt, exp(−ω·dt)·q)`**
   with world-frame velocities: position error **0.0**, quaternion error **1.1e-16** over 40 random
   steps (`phase0_link_lag.json`). So the official picture is an exact function of
   (pos, quat, lin_vel, ang_vel) — 13 floats — all of which the env's own 30-float observation
   carries (velocities in the body frame; convertible with the quaternion). This is RoboCasa §17.3's
   stale-kinematics trap in PyBullet form, and unlike RoboCasa it is *reproducible* rather than
   something to forward away.
2. **Flame colour is a 1-bit episode flag.** `RocketBaseEnv.step` sets link 9 to `(1, 0.3, 0.3,
   0.8)` on every ignition *transition* (off->on and on->off) and never restores it; after the first
   ignition the flame stays that colour for the episode. Worth 61 px (0.013 % of the frame) at the
   probe pose. Stored as one style bit, it is pure.

### G2 · compactness — PASS, by 4 orders of magnitude

| | bytes | ratio to the 691,200 B frame |
|---|---|---|
| synced picture: pose 7 f32 | 28 | 1 / 24,686 |
| official (lagged) picture: pose + world velocities, 13 f32 | 52 | 1 / 13,292 |
| per-episode: 5 colours x 3 uint8 + flame bit | 16 | amortised, negligible |
| the env's own observation, 30 f64 | 240 | 1 / 2,880 |

### G3 · shared decoder — PASS

Geometry is a constant of the family (six resets, 14/14 shapes identical); the per-reset change is
five RGB triples applied at runtime. Everything else a decoder needs is constant and measured:

* **Shading law** (bullet3 `TinyRenderer.cpp` `Shader::fragment`, and confirmed empirically):
  `out_i = int( 255·c_i · (A + shadow·(D·max(0, n·l) + S·max(0, r_z)^spec)) · L_i )`, clamped at 255,
  **truncated not rounded**, per-vertex normals barycentrically interpolated, no gamma, and the
  specular multiplies the material colour (unlike MuJoCo's additive white). Defaults **A = 0.6,
  D = 0.35, S = 0.05, L = (1,1,1), shadow off, light direction the normalised world-fixed
  `(−50, 30, 100)`** (the z-up branch of `TinyRendererVisualShapeConverter::render`; the
  `(−5, 200, −40)` member initialiser is *not* what a default render uses). Each of these passed
  explicitly reproduces the default frame with **0 px**; `(−50, 100, 30)` (y-up branch) differs on
  221,008 px, `shadow=1` on 143,406 px at 6 m and 0 px at 420 m (`phase0_light_sweep.json`).
  Ambient-only body pixels equal `int(255·c)` exactly; the darkest default body pixel equals
  `int(0.6·255·c)`; the brightest is 0.991·`int(255·c)` (`phase0_b.json: b_light`).
* **Trap recorded:** a `lightDirection` passed to `getCameraImage` **persists in the client** for
  all later default renders (`default_after_explicit_vs_explicit_diff_px = 0`). The first probe's
  lighting ablations after its first explicit call were contaminated by this and are superseded by
  the sweep file; the note is in `phase0.json: notes`.
* Texture: nearest-texel sampling of one 512² checker, tiled; no mipmap (TinyRenderer has none).
* Camera and projection as above; background `(255,255,255)`; cylinder 32 steps, no caps, both faces
  drawn; box 12 triangles; ground 2 triangles.

### G4 · economics — PASS on the loop, and the official renderer is slower than reading

One M5 core, 480x480, `phase0.json: G4`:

| | f/s | vs one core of JPEG-q90 decode of the env's own frames (1,190.9 f/s, 62.3 kB/frame) |
|---|---|---|
| official (TinyRenderer, one process) | **34.25** | **0.029 cores** — 35x slower than reading |
| official at the drone's 128² capture | 1,310.9 | — |

Render share of an env step: the agent step is **0.62 ms** without a camera and **3.41 ms** with the
128² capture PyFlyt performs inside every step when `render_mode` is set (**81.8 % of the step**); a
480² frame per step would be **97.9 %** of the step. This is the Griddly shape — the official renderer
is the bottleneck of data generation and is slower than the dataset it would replace — so any GPU
decoder that draws the picture wins G4 by construction; the nvdiff backend at 224 px on locomotion
runs 4.4–10.9 k f/s on an H200 (`when_does_this_pay_off.md` table) against 1,191 f/s of decode here,
and this scene is ~600 triangles against locomotion's 8.5 k–579 k vertices.

### G5 · fixed cast — PASS trivially (12 rocket shapes + pad + ground, no variable entity count).

## The smallest renderer, and the tier it can declare

**URDF primitives through `stable_envs/render/nvdiff.py`**, which already provides the batched
clip-space rasterisation, per-geom appearance gather, `Lights`/`Camera` dataclasses with a batched
camera pose, a texture path, and `QUANT(mode="trunc")`. What it lacks is exactly four small things,
each a PATCH_CANDIDATE against `nvdiff.py` (not written in Phase 0):

1. **A MuJoCo-free scene constructor** — `Scene` is built from an `MjModel`; this scene needs a
   `from_triangles(vert, norm, uv, face, geom_of_face, ...)` entry that fills the same members
   (`vert/norm/uv/vgeom/face/fgeom/mat_rgb/...`). The tessellation must be TinyRenderer's: 32-step
   **open** cylinders drawn double-sided with outward normals, 12-triangle boxes, the 2-triangle quad.
2. **Per-row material colour** — `render()` gathers `base = S.mat_rgb[gid]`; a `(B, G, 3)` colour
   tensor makes that `mat_rgb[b, gid]`. This is the one capability the visual-tier locomotion cells
   deliberately do not offer (their docstring says why: colour is baked in `Scene`), and here it is the
   whole variation space, so it has to be a tensor, not a per-value `Scene`.
3. **A `shading="tiny"` law** beside the MuJoCo one in `_shade`: `c·(A + D·ndl + S·spec)` with the
   specular multiplying the material colour and using the infinite-viewer reflection `r_z`, truncation
   quantisation, directional light only. Tiny: the composition differs, the inputs do not.
4. **Nearest, non-mipmapped texture sampling** for the checker (`filter_mode="nearest"`; the pyramid
   is what this backend exists for elsewhere and must be *off* here).

Plus the env cell itself (`stable_envs/envs/rocket_landing.py`, Fetch/locomotion pattern): 13 state
dims (pos, quat, world velocities) or 7 if the synced picture is declared, a style tensor of 5 RGB
triples + the flame bit, `variation_space` = the four rocket colours + pad colour (**not**
`sky_color`, which is dead upstream; `start_*` are dynamics and belong to `sample_states`, not
appearance), `sample_states` from the env's own descent distribution (heights 1.5–475 m, ±2 m lateral,
±0.2 rad tilt, ignition on/off), and `reference_render` driving PyFlyt+TinyRenderer at the exact
lagged pose. Links cannot be placed at *t−dt* from Python (a teleport syncs them), so the reference
corpus must be produced by *stepping* — recorded rollouts carrying velocities — not by teleporting.

**Tier: visual.** Parity is implausible: a different rasteriser with its own edge rule, pixel-centre
convention and float path against a software renderer that truncates — silhouette and checker-edge
pixels will differ, and the interior of a Gouraud-shaded tube will land within ±1/255 rather than
exactly. Visual-tier bounds are very likely strong, because ~90 % of every frame is either flat white
(the first half of the descent) or a crisp 2 m checker, and the mover is 10 % of the frame at all
times; the residual will concentrate on (a) silhouettes, (b) checker edges shifted by up to half a
pixel, (c) the far ground where tiles are sub-pixel and nearest sampling aliases differently under
any UV rounding difference, (d) the 61-px flame, whose alpha handling in TinyRenderer (0.1 vs 0 vs
0.8 all differ) is not yet fitted. My estimate before measurement: mean_abs 1–4/255, within-8 > 0.95,
SSIM > 0.95 — to be measured, not declared, in phase 1 (64 states x 4 draws, as the locomotion cells).

**CPU backend (policy addendum):** the scene is ~600 triangles at 480²; no CPU-native backend in this
repo draws an arbitrary textured triangle soup (cpu3d/raycast3d are MuJoCo-primitive raycasters).
Phase 1 must either measure a small CPU triangle rasteriser or have `device="cpu"` raise; it must not
silently use CUDA. This is the one open design question for the coordinator.

**Milestones (if GO):**
M1 reference corpus (local venv suffices — TinyRenderer is CPU; no EGL): 256 states from rollouts
with velocities, colours and the flame bit, official frames, frozen under `examples/results/step7/`.
M2 the four `nvdiff.py` extensions above, each with a unit test, reviewed independently.
M3 the env cell + `reference_render`; Gate A; `check_visual_fidelity` at 480 px on Oscar (3090, per
AGENT_COMMON); declare `visual_fidelity` from the worst of four draws.
M4 throughput (Step-6 machinery) at batch 1/16/64/256 on the same card; CPU decision; tiers reported
honestly (PERFORMANCE_MEASURED, no CPU_FAST/GPU_FAST requirement).

## Findings the coordinator should relay to Sami (and possibly upstream)

* FINDING (upstream, stable_worldmodel): `environment.sky_color` is in the variation space and has no
  effect; `_modify_rocket_urdf` is dead code; `rocket.booster_color` is applied only via
  `changeVisualShape` (fine) — the Step-1 "per-reset URDF rewrite" describes the pad only.
* FINDING (upstream, PyFlyt): the ignition flame is recoloured on every transition and never reset —
  a rendering state leak that makes the frame history-dependent by 61 px.
* FINDING (upstream, PyFlyt/PyBullet): the rendered frame after `step` draws links and the camera one
  physics substep behind the base; the swm goal image (rendered after a teleport) is the synced
  picture, so the goal and the observations are drawn under two different conventions.
* FINDING (this repo): Step-1's NO-GO rested on "mesh assets" (false), "per-reset URDF rewrite of the
  rocket" (dead code; pad only, colour only), and "no PyBullet backend" (not needed: the scene is 600
  triangles under a five-constant shading law). An unmeasured gate was a gap, not a verdict — the
  document's own rule.

## Tests, replay, tiers, pixels

* Tests run: none in `stable_envs/` (no code changed). Probes ran end to end (`phase0_probe.log`,
  `phase0_probe_b.log`).
* Step-4 replay: not applicable (no existing config touched). Tiers before/after: n/a. Pixels
  changed: none.
* Semantic follow-up: none in Phase 0; a DINO tripwire is the right check once frames exist (the tier
  docs already warn that a visually close render can sit far in embedding space).
* SLURM job IDs: none (Oscar SSH denied at auth).

## Pending cluster items

1. `visual_fidelity` measurement and throughput on a 3090 (`--partition=gpu --qos=norm-gpu`) once M2/M3
   exist; nvdiffrast is needed (locomotion's `_Nvd` path documents the import probe).
2. A PyFlyt+cvxpy venv on Oscar under `$SCR/venvs/se7-rocket` (numba JIT: 57 s per process, budget it
   in reference jobs; `NUMBA_CACHE_DIR` under `$SCR`).
3. Nothing here needs EGL: TinyRenderer is the official path and is CPU-only.
