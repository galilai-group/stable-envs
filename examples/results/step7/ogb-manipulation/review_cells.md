# Independent review — W5 (ogb-manipulation) CELLS: `stable_envs/envs/ogb_manip.py` (`OGBCube`, `OGBScene`)

Reviewer: independent (did not write the code). Scope: branch `step7/ogb-manipulation`, tip `1550ebdb`,
baseline `76dc5116` (40 commits, all `sami-bg`, 0 AI trailers, `examples/results/step4|5|6` untouched).
The shared `mjwarp.py` helpers were reviewed separately (`review_shared.md`, APPROVE conditional) and are
not re-reviewed here; only their USE by the cell is.

Local environment: `/Users/samib/se7-ogb-venv/bin/python` (torch 2.14 CPU, mujoco 3.12.0, ogbench 1.2.1)
with `PYTHONPATH=/Users/samib/stable-worldmodel` — `stable_worldmodel` is NOT installed in that venv but
imports cleanly from the source checkout, so BOTH swm envs were constructible locally and every model-side
claim below was tested on the real models, not inferred from artifacts. `MUJOCO_GL=glfw`. No CUDA, no
`mujoco_warp`: the MJWarp launch, the segmentation buffer, per-world `geom_rgba`/`mat_rgba`, the fidelity
numbers and the tiers are cluster-only (list at the end). Throwaway baseline worktree `<scratchpad>/base76`
at 76dc5116 (removed).

## VERDICT: REJECT

The declared numbers are real (they equal the committed JSONs) and the hygiene is clean, but the cell does
not implement the design its declarations describe, one of its two cells is silently wrong in ~40 % of
processes, its seeded sampler is not reproducible (a repo contract), the committed "pool" is not the pool
the corpus was drawn from, the official Gate A check would report a dead dimension the author's custom
probe was written around, and there is not a single test for the cell. Fixing B1 changes pixels, so the
fidelity numbers, thresholds, corpora, Gate F and tiers all have to be re-measured — that is a
resubmission, not a conditional approve.

### Blocking defects (each reproduced locally, commands in §9)

**B1. The composite does not use the plate on the floor: `mv = seg >= 0` composites EVERY geom MJWarp
drew, including the 3 (Cube) / 14 (Scene) static geoms that stay in MJWarp's enabled groups.**
`fetch.py` builds a mover LUT (`lut[movers+1]`, `fetch.py:846-851, 760`); `ogb_manip.py:552` uses
`seg >= 0`. `prepare_model` on the context copy hides 0 geoms (only zero-alpha ones), so the floor plane
and the UR5e base meshes are rendered by MJWarp and win the composite. The committed frames prove it
(`oscar/m3fid/OGBCube_frames.npz`, 4 states): of the 75.2 % of pixels that are static across all four
official frames, `ours == official` on **28.4 %**, while `ours` is itself static on 99.97 % of them — the
cell's floor is a FIXED image that is not MuJoCo's (signed diff on static px: 0: 47.8 %, +1: 14.0 %,
−1: 7.1 %, +2: 4.9 %, +3: 2.8 % — MJWarp's lighting residual). Bottom-quarter rows exact: **2.3 %**
(Cube), 13.8 % (Scene). The shipped `interior_exact` 0.351/0.367 and `all_exact` 0.278/0.269 are the RAW
corrected-MJWarp figures (`m3_corrections.json`: exponent=0 row `interior_exact 0.3507, all_exact 0.274`);
the plate contributes only skybox pixels. The m1b "composited 0.923-0.951" was measured with
`np.isin(seg, movers)` (`probe_m1b_residual.py:96`), i.e. the design, not the shipped code.
Consequently the module docstring ("Only the 10-19 % of pixels that are movers are MJWarp's to draw ...
Measured whole-frame exactness of the COMPOSITE is 0.92-0.95, and that is what a caller gets"), the
`_RESIDUAL_NOTE` ("Whole-frame exactness of the composite that actually ships is 0.92-0.95") and the
"floor ... exact BY CONSTRUCTION" sentence are FALSE for the shipped cell, and `_check_plate_is_constant`/
`_bake_plate` guard a plate that is never shown where it matters. Fix: the Fetch LUT (or move the statics
to `HIDDEN_GROUP` on the context copy); then re-measure everything downstream.

**B2. `OGBScene` button colours are wrong for every state whenever the reference env's first reset lands
on task 1 or 3.** `_reference_env` calls `env.reset(seed=0)`; ogbench's `ManipSpaceEnv.reset` picks the
task with the GLOBAL `np.random.randint(1, 6)` (unseeded), and swm's task table has init
`button_states` `[1,1]` (task 1) and `[1,0]` (task 3). `_bundle.pristine` is snapshotted at that point;
`_bake_plate` restores it; `_context` deep-copies it; `_ctx_button_tables` then computes
`out[code] = base + (T[bits] − T[(0,0)])` with `base` = pristine — correct only if pristine IS bits (0,0).
Reproduced with `reset(seed=0, options={'task_id': 1})`: code 0 (bits 00) gets `drawer_handle`
(0.90,0.90,0.90) instead of swm's (0.96,0.26,0.33); code 3 gets (0.84,1.54,1.47) — out of range; every
code wrong; the "disjointness" assertion is silent because `base_m == pristine` on the handle rows. Task
picks for `np.random.seed(s), s=0..9`: `[5,4,1,3,3,4,3,5,4,5]` → 4/10 processes wrong. The M3/Gate F/M4
jobs evidently drew a (0,0) task; nothing in the code or artifacts records which. Fix: capture the tables
relative to the ACTUAL bits at snapshot time (or force bits (0,0) before snapshotting, or key `base` on the
recorded bits), and assert `pristine == T[bits_at_snapshot]`.

**B3. `sample_states(rng)` is not reproducible — two calls with `default_rng(0)` differ (Cube max |Δ|
5.97, Scene 1.0), and still differ with `np.random.seed(0)` before each.** Sources: `CubeEnv.reset` →
`reset_variation_space(..., seed=None)` resamples the default variations from fresh entropy (the author
recorded this in findings §3 and then built the sampler on it); ogbench's global-RNG task pick; the
action-space RNG consumed inside `reset`'s goal phase. This violates
`tests/gpu/test_reproducibility.py::test_seeded_sampling_reproduces_in_a_fresh_process` (runs over
`sorted(se.REGISTRY)`) and is why B4 happened. Fix: derive the task (`options={'task_id'}`), the variation
values (`options={'variation_values'}` or `u.variation_space.seed(int)`) and the action seed from `rng`.

**B4. The committed pools are not the pools the corpora were selected from.** `probe_corpus_gatef.py`
draws `pool = env.sample_states(256, rng(STATE_SEED))`, saves it, then calls `corpus_for`, which draws its
OWN 256 (B3 makes them different). Measured: `states == pool[meta.indices]` **False** (max |Δ| 6.44 /
1.0); **0 of 64** kept rows occur anywhere in `pool_OGB*.npz`; `meta.indices` point into a pool that does
not exist. The commit message's "a selection whose pool is gone cannot be re-derived" describes exactly the
state the branch is in. The frozen `states/*.npz` themselves are well-formed (schema, sha256 match) and
usable, but "64 of 256" and the selection provenance are unverifiable. Fix: after B3, regenerate pool and
selection in one draw and assert `states == pool[indices]` before writing.

**B5. Gate A "0 inert dims" is the author's probe, not the repo's.** `conformance.check_state_dims_matter`
substitutes donor values, and falls back to nudges `+25, −25, +1, +0.5`. `mocap_quat` is the identity in
EVERY state the registered default distribution produces (corpus: 1 unique row, per-dim std 0 on all four
components, both cells) so donors "prove nothing", and the four nudges on `mocap_quat[0]` give
(26,0,0,0), (−24,0,0,0), (2,0,0,0), (1.5,0,0,0) — all the identity rotation after normalisation (verified
in MuJoCo: `xquat` identical; the author's own finding that +0.05 on w is inert shows the MJWarp path
normalises too). The official check therefore reports `dead_dims = ['mocap_quat[0]']` on both cells; the
custom rotation probe was written to avoid that verdict. Either the state layout changes (a yaw scalar, or
drop `mocap_quat` — the registered default never rotates the goal; `cube.goal_yaw` does not reach
`mocap_quat`) or the shared gate gains a quaternion rule by coordinator decision — not a per-env probe.

**B6. No tests.** `tests/` contains nothing that imports `ogb_manip` (only the three mjwarp helper test
files were added). `_split`/state layout, `_button_tables`, `_ctx_button_tables` (B2 is a five-line unit
test on CPU), the CPU raise, `_movers`, and the declaration shape are all testable without CUDA.

### Required changes (exact)

R1. `render()`: replace `mv = seg >= 0` with a mover LUT as in `fetch.py:846-851, 760` (or hide
    `statics` on the context copy). Re-measure M3 (3×64), Gate A, corpora, Gate F arm D and M4 on a 3090;
    re-fill `_MEASURED`; rewrite the module docstring and `_RESIDUAL_NOTE` to describe the composite that
    ships and quote only numbers the shipped cell produces.
R2. `_ctx_button_tables`: base the delta on the bits actually applied at snapshot time (record
    `u._cur_button_states` into `_Bundle`, assert `pristine == button_tables[bits]`), and add a CPU unit test
    that builds the bundle on `task_id=1` and checks every code equals swm's table.
R3. `sample_states`: make it a pure function of `rng` (task id, variation values, action seed all drawn
    from `rng`); prove with `frames_digest` equality across two processes.
R4. Regenerate `states/*.npz` and `pool_*.npz` in ONE draw after R3; assert `states == pool[indices]`;
    commit the sha256 of both.
R5. Gate A: run `conformance.check_state_dims_matter` itself and commit its output; resolve
    `mocap_quat[0]` by layout change or coordinator ruling on the shared gate. Do not ship a per-env probe
    as the Gate A evidence.
R6. `visual_fidelity`: add `reference_versions = {"mujoco": "3.12.0", "ogbench": "1.2.1"}` and
    `measured_with = {"mujoco_warp": "3.12.0", "warp": "1.17.0"}` (Oscar provenance dicts), name them in
    `measured_at` (required by `test_every_declared_reference_version_is_the_one_measured_at_names`), and
    identify the swm reference by source-checkout digest (the venv uses the checkout, not PyPI 0.0.6). Add
    `interior_frac`, `interior_exact_frac`, `all_exact_frac`, `draw_spread` as `dmcontrol.py` does.
R7. Thresholds: the rule in `_fidelity` (`mean_abs × 1.5`, `floor(·,0.05) − 0.05`) is looser than the
    DM Control precedent (× 1.3, −0.04/−0.05: 5.01→6.5, 8.07→10.5, 0.882→0.84, 0.826→0.77) and than
    Fetch's (0.9988→0.99); Cube's `ssim_min` 0.85 sits 0.093 below its measurement. Re-derive after R1 with
    the DM Control margins, or justify the wider ones in the declaration.
R8. Docstrings: class `__doc__` says "State is (qpos, qvel, mocap_pos, mocap_quat …)" — qvel is not in the
    state; `_MEASURED` cites `m3/fidelity.json` (file is `oscar/m3fid/m3_fidelity.json`); the class doc says
    residuals are "declared in visual_fidelity and parity_note" — there is no `parity_note`, they are in
    `visual_note` (correct for a visual cell; fix the sentence).
R9. Tests (B6): at minimum state layout/`_split` round-trip, `_button_tables` four combos differ on
    exactly `btngeom_0/1`, `drawer_handle`, `window_handle`, R2's table test, `device="cpu"` raises
    `ValueError` naming CPU3D-OGB-1, `_movers` includes the mocap goal geom, declaration shape.
R10. Provenance: commit the slurm log of the fidelity job (6398733) and of the first-run jobs
    (6398193/6398530) the findings cite — none is committed (committed ids: 6395175, 6395461, 6395736,
    6396447, 6396646, 6396748, 6397397, 6397454, 6397621, 6397885, 6398026, 6398427, 6398465, 6399587).
    `m3_fidelity.json` carries host/GPU but no job id or SHA. Record which task/bits the reference env sat
    on during the measurement (B2).
R11. Artifacts: 13 `.log` files are committed next to their `.log.gz` twin; `probe/` and `oscar/` hold
    parallel copies of the same probes (772 K + 2.2 M of a 3.9 M directory). Keep one. Sizes are otherwise
    modest (largest 382 K).
R12. `m4_tiers.json` was produced by code that predates `1550ebdb` (its Scene `cpu_error` quotes the
    Cube's 193,113 triangles; the commit then changed the message). Message-only, no pixel effect — but say
    so, or re-run.

## 1. Declarations vs artifacts

| claim | artifact | verdict |
|---|---|---|
| Cube `_MEASURED` (4.797169, 0.807268, 0.942822) | `m3_fidelity.json` worst_of_draws: max mean_abs draw 0 = 4.797168965, min within8 draw 0 = 0.8072684, min ssim draw 1 = 0.9428223 | equal |
| Scene (10.004062, 0.704476, 0.902295) | draws: 10.004062 (d1), 0.704476 (d1), 0.9022947 (d1) | equal |
| spreads 4.751-4.797 / 9.942-10.004 | JSON | equal |
| thresholds 7.2/15.1, 0.75/0.65, 0.85/0.85 | `_fidelity` formula | computed as stated; rule is the author's, looser than precedent (R7) |
| `reference_versions`, `measured_with` | absent from `visual_fidelity` (`reference_versions_of(cls) == {}`) | **missing** (R6) |
| `interior_frac` declared | absent; note says "interior mask is 99.2 % floor", JSON says interior_frac 0.41/0.28 | **missing** (R6) |
| alpha marker note (0.61 % / 0.15 %) | in `_RESIDUAL_NOTE` (visual_note) | present; wording in `__doc__` wrong (R8) |
| Scene bake residual per geom | `_SCENE_BAKE_NOTE`: `btn_metal`, geoms `//unnamed_geom_8`, `//unnamed_geom_20`, mean (0.523,0.516,0.512), 0.139 % px, interior 1.0 | matches my local `bake_unsamplable_textures` record on the context copy: material 6 `btn_metal`, texture 2 `T_btn` cube, geoms [65, 77], mean (0.522564, 0.516068, 0.512419), `n_removed 1` |
| CPU: honest raise, no CUDA fallback | `__init__` raises `ValueError` before any device work when `device != "cuda"`; `m4_tiers.json` `cpu_raises: true` both | ok (note `"cuda:0"` also raises) |
| GPU tier | not declared in any class/registry field; only findings §20 + `m4_tiers.json` (`GPU_PARITY (capped: BATCH_SCALES 5.02x/6.30x)`, `batch_scales_pass false`, raw rows present) | consistent with §6.7 (`step3_benchmarks.md:178`); coordinator to say where the record lives |
| two official-renderer numbers | findings §20: 2,012/1,526 f/s (GPU node, hardware EGL) vs 6.31/5.17 f/s (batch node, llvmpipe), roles stated | labelled; `m4_tiers.json` carries only the GPU-node one, fine |
| "whole-frame exactness of the composite 0.92-0.95" | shipped all_exact 0.278/0.269 | **false** (B1) |

## 2. State contract

- swm `SceneEnv.set_state(qpos, qvel, **kwargs)` reads `button_state_<k>` kwargs (`scene_env.py:222-231`)
  and calls `_apply_button_states()` then ogbench `set_state` (`mj_forward`); `CubeEnv` uses ogbench's
  `set_state(qpos, qvel)`. Nothing in swm's API takes mocap; the cell writes `data.mocap_pos/quat` directly
  and `mj_forward`s — correct, and the mocap body is the goal marker (`_cube_target_mocap_ids`).
- qvel: `mj_forward` with random qvel leaves `qpos` and `geom_xpos` bit-identical (max |Δ| 0); scene
  flags are MuJoCo defaults (no CONTACTFORCE); Cube render bit-identical under random qvel. Dropping qvel is
  sound. (The Scene's local glfw render is itself non-deterministic by ~4 px between identical calls — the
  author's §3 saw the same on this host — so the Scene qvel comparison is inconclusive locally; the
  mechanism proof stands.)
- Layout: `state_names` = 21/25 `qpos` + 3 `mocap_pos` + 4 `mocap_quat` (+2 bits) = 28/34, `_split` and
  `reference_render` consume the same offsets. Registered distribution: `mocap_quat` constant identity
  (both cells), `mocap_pos` takes 5 (Cube) / 4 (Scene) distinct values — the 5 task goals; `qpos[4]` std 0
  on Cube. 4-5 of the declared dims never vary in the distribution the corpus samples (B5).
- `sample_states`: reset + 0-25 random steps, `visualize_info=True` (registered default, confirmed
  `u._visualize_info True`, `ob_type 'states'`). Not reproducible (B3).
- Gate A probes: determinism/batch/order/sufficiency are the standard ones and pass on the cluster JSON;
  the per-dim probe is custom (B5). The bit-flip rule for buttons is legitimate; the 0.3-rad rotation is
  physically reasonable but is not the repo's check.

## 3. Button bits

- `_apply_button_states` is CALLED, not reimplemented (`_button_tables`), on swm's own env; the four
  combinations differ exactly on `btngeom_0` / `btngeom_1` (`geom_rgba`) and `drawer_handle` /
  `window_handle` (`mat_rgba`) — matches swm's rule (`scene_env.py:443-481`; the joint-damping edits are
  not pixels). Must-differ assertion fires correctly.
- Applied as a per-world delta over the bake with a disjointness assertion — but the base is assumed to be
  bits (0,0) and never asserted (B2). With the reference at (0,0) the four context tables equal swm's; with
  the reference at task 1 they are all wrong and the assertion is silent.
- Per-world `geom_rgba`/`mat_rgba` via `put_model(batch_sizes=...)` and `_upload_button_colours`
  (`index_copy_` per code) — cluster-only; the committed Gate A shows the two bits move 181/180 px.

## 4. `_context` deep copy

`copy.deepcopy(b.model)` then `prepare_model` + `bake_unsamplable_textures` on the copy: sha256 over every
`MjModel` array of the swm env's model unchanged (`060a6d45719fef11` Cube, `97cbd41f868f268a` Scene,
before == after). `_bake_plate` mutates `geom_group` and the colour arrays transiently and restores from
`keep`/pristine — digest identical after `_check_plate_is_constant`. Both cells' `_check_plate_is_constant`
pass locally at 128 px. Caveat: `_bake_plate` restores `geom_rgba`/`mat_rgba` to `pristine`, which is the
button state of the random first reset (B2), and the plate is used only on skybox pixels (B1).

## 5. Corpora

`states/OGBCube.npz` / `OGBScene.npz`: keys `states` (64×28 / 64×34 float32) + `meta` with the Step-4 key
set (`env, indices, kept, n_extreme, n_spread, n_typical, pool, seed, selection, sha256, state_dim,
state_names`) — identical to `examples/results/step4/states/Acrobot.npz`. `sha256` recomputed with
`conformance.frames_digest`: `55222c0339b5…ca65` and `4d64960ca7e2…0f11`, both match. Button bits ∈ {0,1}
(combos 00:36, 10:16, 11:12; 01 never sampled). `corpus_for` reads them (`p.exists()` branch). Pool: B4.

## 6. Gate F

swm registers both with no kwargs → `ob_type='states'` (`CubeEnv.__init__` default, `SceneEnv` default);
observation 28-D/40-D float64 vector, confirmed locally and in `corpus_gatef.json`. There is no pixel
observation to swap; arm D ("render() is the certified artifact") is the right reading. Arm-D numbers
(4.6345/0.8139/0.9494, 9.8025/0.7025/0.8982 over 24 swm-driven states) are consistent with the corpus
numbers — but both were produced by the B1 composite (and, for the Scene, by whatever task the process
landed on), so they must be re-run after R1/R2.

## 7. Regression

- Registry: base 76 → branch 78, added `['OGBCube', 'OGBScene']`, removed none — the same in the
  se7-ogb venv (+swm source) and in se6-venv without swm, because `ogb_manip` imports `stable_worldmodel`
  lazily inside `_reference_env` and so registers everywhere (`import_errors()` has no `ogb_manip` entry).
  No existing family changes.
- `tests/unit` (full, `MUJOCO_GL=glfw`, se6-venv): branch **873 passed / 2 failed / 6 skipped**, baseline
  **823 / 2 / 6**; the two failures are identical and pre-existing
  (`test_examples.py::…[robocasa_visual/render_mjwarp.py]`, `test_repo_contracts.py::test_every_cited_scratch_path_resolves`).
  +50 = the three mjwarp test files; 0 for the cell.
- `git diff --stat 76dc5116..HEAD -- examples/results/step4 step5 step6`: empty. 40 commits, all
  `sami-bg <samibghanem@gmail.com>`, 0 `Co-Authored-By`/`Generated with` trailers.
- Artifact dir 3.9 M (R11).

## 8. Policy

No env-specific code in shared files (the shared review's AST map stands; the cell only calls
`prepare_model`, `bake_unsamplable_textures`, `create_render_context`, `gl_rounding`, `HIDDEN_GROUP`).
No fidelity ceiling is tuned to pass (they are generous, R7). The Scene's weaker numbers are declared and
attributed. The bake residual is declared per geom. What is NOT honest is the mechanism text (B1) and the
"0.92-0.95" figure.

## 9. Commands (all from this review; scripts in `<scratchpad>/`)

```
git -C /Users/samib/stable-envs-step7-ogb-manipulation log --format='%h %an' 76dc5116..HEAD | sort -k2 | uniq -c -f1   # 40 sami-bg
git log 76dc5116..HEAD --format='%B' | grep -ciE 'co-authored-by|generated with'                                      # 0
git diff --stat 76dc5116..HEAD -- examples/results/step4 examples/results/step5 examples/results/step6                # empty
git show 1550ebdb -- stable_envs/envs/ogb_manip.py                                                                   # visible_tri added after M4
MUJOCO_GL=glfw se6-venv/python -m pytest tests/unit -q       # branch 873/2/6, base 823/2/6 (worktree <scratchpad>/base76)
PYTHONPATH=/Users/samib/stable-worldmodel MUJOCO_GL=glfw se7-ogb-venv/python -c 'import stable_envs as se; print(sorted(se.REGISTRY))'  # both trees, diffed: +OGBCube +OGBScene
se6-venv/python corpus_check.py          # schema, sha256 match, states != pool[indices], 0/64 rows in pool, constant dims
se6-venv/python  (frames analysis)       # m3fid frames: static px 75.2 %, ours==official on them 28.4 %, ours static 99.97 %
PYTHONPATH=… se7-ogb-venv/python local_checks.py   # digest before/after copy-prep, movers/statics, qvel, sample_states x2, plate constancy, tables at (0,0)
PYTHONPATH=… se7-ogb-venv/python local_checks2.py  # task picks per global seed, task_id=1 -> wrong tables, set_state kwargs, qvel isolation
PYTHONPATH=… se7-ogb-venv/python (mocap_quat nudges)  # w+25/-25/+1/+0.5/+0.05 all identity rotation; x+25 not
```

## 10. Cluster-only (cannot be verified on this Mac)

- That MJWarp's `seg_data[...,0]` is `-1` only off-geometry and ≥ 0 on the floor plane (B1 is inferred
  from the committed frames + the model side; the direct proof is one `seg` dump on a 3090).
- Per-world `geom_rgba`/`mat_rgba` actually reaching the megakernel via `put_model(batch_sizes=…)` and
  `index_copy_` on `wp.to_torch(c.wm.*)` (the committed Gate A bit-flip deltas are the only evidence).
- The graph capture inside `gl_rounding()` replaying the patched packer; determinism/batch/order/sufficiency.
- All fidelity numbers, thresholds' headroom, Gate F arm D, tiers (fps, official EGL, decode) — all to be
  re-measured after R1-R4 anyway.
- Which task/bits the reference env sat on in jobs 6398733/6399587 (B2) — not recorded anywhere.
- The official `check_state_dims_matter` verdict on the real render path (B5 is a MuJoCo-side proof plus the
  author's own +0.05 observation).

---

# Re-review at tip `7820af1c` (coordinator request)

Commits since `1550ebdb`: 7, all `sami-bg`, 0 AI trailers; `examples/results/step4|5|6` untouched;
`stable_envs/render/` and the three mjwarp test files byte-identical to the approved shared tip `9923594`.
Non-artifact files touched: `stable_envs/envs/ogb_manip.py`, `tests/unit/test_ogb_manip.py` (new),
`scratch/step7/ogb-manipulation/{13_remeasure_chain.sbatch, probe_remeasure_chain.py, probe_corpus_gatef.py}`.
Same local environment as above (se7-ogb venv + swm source checkout, `MUJOCO_GL=glfw`, no CUDA).

## Verdict: APPROVE-WITH-CHANGES (three declaration/import defects; none needs a re-measurement)

### B1-B6 re-run on the new tree — all closed

| defect | construction at 7820af1c | result |
|---|---|---|
| B1 composite | `m3/fidelity/OGB*_frames.npz` (job 6407605): pixels static across the 4 official frames, `ours == official` on them | Cube **0.9772** (was 0.284), bottom-quarter rows exact **1.000** (was 0.023); Scene **0.9754** (was 0.269), bottom quarter 0.972; interior_exact 0.957 / 0.921 recomputed from the frames. `render()` uses `lut.index_select(0, seg+1)` over `_movers` exactly as fetch.py; `test_the_lut_rule_selects_movers_and_lets_everything_else_fall_through` pins the arithmetic. |
| B2 button base | reference env forced to `task_id` 1..5 (start bits [1,1], [0,0], [1,0], [0,0], [0,0]); `_context`'s forced-zero block emulated; `_ctx_button_tables(..., device="cpu")` vs swm's tables | all four codes equal swm's table on **every** task start; env bits restored after the copy; negative control (copy taken without forcing zero at [1,1] / [1,0]) now raises `AssertionError: the context model was copied at button bits other than all-zero`. |
| B3 reproducibility | `sample_states(6, default_rng(20260909))` in three separate interpreters + twice in one | `frames_digest` identical across all runs (Cube `88e86061…`, Scene `e5964831…`); different rng differs; global `np.random` state restored; `ManipSpaceEnv.action_space` property restored (fresh Box per access again after the block). |
| B4 pool | `states/OGB{Cube,Scene}.npz` + `_pool.npz` | `pool[meta.indices] == states` **True** both; sha256 recompute matches (`35dd61dd149d…`, `5d14aa863431…`); 64/256; meta = Step-4 key set + `provenance` (superset; `corpus_for` loads it). Selection by `fidelity_step4.select` (same function), written by the chain script rather than `corpus_for` itself. |
| B5 mocap_quat | corpus per-dim std; remeasure.json `gate_a_repo_checks` | state 24 / 30 dims, **no constant dims** in either corpus; repo `check_state_dims_matter` → `dead_dims: []`, `check_determinism` identical, `check_batch_invariance` 0, `check_batch_size_invariance` [1,7,64] 0. (`check_order_invariance` is "absent in this build" — the module has no such check; order invariance is therefore not asserted by the repo machinery, only by the retired custom probe at the old tip.) |
| B6 tests | `tests/unit/test_ogb_manip.py` | **26 passed** (se7+swm); full `tests/unit` with swm present **885 passed / 2 failed / 20 skipped**, without swm 873 / 2 / 7; the two failures are the pre-existing ones. |
| distribution | goal positions old vs new corpus/pool | unchanged: 5 (Cube) / 4 (Scene) unique `mocap_pos`, identical xy ranges — passing `variation_values` did not move the registered-default distribution (task-mode goals still come from `task_infos`). |
| staleness fixes | reset the swm env, then `_Bundle.model`, `_renderer` | `_Bundle.model is u.model` after the rebind; `_renderer` returns a new Renderer bound to the current model; the popped Renderer is not auto-closed (`mujoco.Renderer` has no `__del__`). |
| declarations vs JSON | `_MEASURED` vs `remeasure.json` | `mean_abs`/`within8_frac`/`ssim` equal the worst-of-three to 1e-6 on both cells; `draw_spread` equals the JSON min/max; thresholds equal the JSON `thresholds` and the rule: `round(1.3×mean, 1)` → 3.2 / 7.0, `floor(20×worst)/20` → 0.90/0.90, 0.85/0.85. `reference_versions {"mujoco": "3.12.0"}`, `measured_with {"mujoco_warp": "3.12.0"}`, named in `measured_at`. |
| hygiene | | 0 `.log`/`.log.gz` twins; `oscar/` duplicates of m1/m1b/m3 removed; dir 3.4 M; logs 6398193/6398530/6398733/6399587/6407605 committed under `m3/fidelity/`; provenance carries host, GPU, job, versions, `ogb_manip_md5 f168b2d3…`, swm file digests (`git_sha` still ""). §19/§20 marked SUPERSEDED. |

### Required changes (declaration / import only — no pixel, corpus or tier changes)

C1. **The false composite sentence was NOT deleted.** §21 says "the '0.92–0.95' sentence is deleted rather
    than restated", but it survives in three places at 7820af1c: the module docstring (lines 36-46:
    "exact BY CONSTRUCTION … Only the **10-19 %** … whole-frame exactness of the COMPOSITE is **0.92-0.95**,
    and that is what a caller gets. The 0.35 …"), `_bake_plate`'s docstring (line 610: "worth 0.92-0.95 while
    MJWarp alone is worth 0.35"), and — the one that ships — `_RESIDUAL_NOTE` (line 983: "Whole-frame
    exactness of the composite that actually ships is 0.92-0.95"), which is the registered `visual_note`.
    The measured composite is all-exact 0.872 / 0.775 and interior-exact 0.958 / 0.922. Replace all three
    with the measured figures (or delete), and drop the "0.35 interior" framing, which no longer describes
    this renderer.
C2. **`register()` now builds both swm envs at import.** `_make_classes` evaluates
    `len(_bundle(cell).state_names)` inside the class `__doc__` f-string, so `import stable_envs` calls
    `gymnasium.make(swm/OGB*)` + `reset(seed=0)` for both cells (3.2 s, `_REF_CACHE` populated at import) and
    — in any interpreter without `stable_worldmodel` — the module now fails to import
    (`ModuleNotFoundError: stable_worldmodel`) so the two ids vanish from the registry (se6-venv: 76, not
    78; at 1550ebdb they registered lazily everywhere). Registration must stay side-effect free: write the
    dimension from `SPEC` (nq 21/25 → 24/30, or `nq + 3 + nbuttons` as constants) and keep `_bundle` out of
    import time.
C3. **`measured_at` says "the declared value is the WORST of the three", but `interior_exact_frac`,
    `all_exact_frac` and `interior_frac` are the MEAN of the three draws** (Cube interior_exact declared
    0.958115 = mean; worst draw 0.958007; Scene 0.921712 = mean, worst 0.917805; likewise all_exact and
    interior_frac). Either declare the worst, or say in `measured_at`/a comment that the three
    non-thresholded fields are means. Small, but it is exactly the kind of sentence the declaration test
    exists to keep true.

### Recommended (non-blocking)

- Fractional floors: `floor(20×worst)/20` leaves margins of 0.0155 / 0.0149 (Cube within8 / ssim) and
  **0.0068** (Scene within8) below the worst draw, against DM Control's 0.04-0.06. The spreads are small
  (0.0006-0.0085), so these are not tripwires on the evidence, but they are tighter than the precedent the
  rule cites; a floor one further 0.05 step down on the two 0.90s and the Scene 0.85 would match it.
- `reference_versions` omits `ogbench` although provenance records `ogbench 1.2.1` and the MJCF is
  ogbench's; the stated reason ("nobody measured under a named version") is contradicted by the provenance
  dict. Add `"ogbench": "1.2.1"` and name it in `measured_at`.
- Class `__doc__` still says residuals are "declared in visual_fidelity and parity_note" — they are in
  `visual_note`.
- `remeasure.json`/corpus `provenance.git_sha` is "" (rsync tree); the md5 of `ogb_manip.py` is the tree
  identifier — fine, but say so in the README.
- The old `m3/fidelity/m3_fidelity.json` (pre-B1 numbers) sits beside `remeasure.json` with no marker in the
  file or README; rename or annotate it.

### Cluster-only (unchanged list, now with the author's evidence for each)

MJWarp `seg` semantics on the floor (now indirectly proven by the 97.7 % static-pixel agreement), per-world
`geom_rgba`/`mat_rgba` reaching the megakernel (Gate A bit flips; repo `dead_dims []` includes the two bits),
graph capture inside `gl_rounding`, all fidelity/Gate F/tier numbers (job 6407605, gpu2109, md5-verified
tree), and `tests/gpu/test_reproducibility.py` itself (the cross-process digest equality above is the same
check run by hand).

### Commands (re-review)

```
git log --oneline 1550ebdb..HEAD; git log --format='%B' 1550ebdb..HEAD | grep -ciE 'co-authored|generated with'   # 7 commits, 0
git diff --stat 9923594..HEAD -- stable_envs/render tests/unit/test_mjwarp_*.py                                   # empty
MUJOCO_GL=glfw se6-venv/python -m pytest tests/unit -q                                                             # 873/2/7
PYTHONPATH=swm MUJOCO_GL=glfw se7-ogb-venv/python -m pytest tests/unit -q                                          # 885/2/20 (cell tests inside)
PYTHONPATH=swm MUJOCO_GL=glfw se7-ogb-venv/python -m pytest tests/unit/test_ogb_manip.py -q                        # 26 passed
se6-venv/python  (frames + corpus script)      # static-px agreement 0.9772/0.9754; pool[idx]==states; sha match; no constant dims
PYTHONPATH=swm … rereview.py main              # tasks 1-5 tables, negative control, in-process reproducibility, rebind tracking
PYTHONPATH=swm … rereview.py digest  (x2)      # cross-process digests equal
PYTHONPATH=swm … (declared-vs-json script)     # worst-of-three equality, x1.3 rule, MEAN fields
MUJOCO_GL=glfw se6-venv/python -c 'import stable_envs…'   # 76 ids, ogb_manip ModuleNotFoundError (C2)
grep -n '0.92-0.95\|10-19\|parity_note' stable_envs/envs/ogb_manip.py                                              # C1, wording
```

---

# Final check at tip `f62625e9` (coordinator request)

One commit on `7820af1c` (`sami-bg`, 0 trailers). Files: `stable_envs/envs/ogb_manip.py`,
`tests/unit/test_ogb_manip.py`, `scratch/step7/ogb-manipulation/check_import_without_swm.py` (new),
`findings.md`, `m3/fidelity/m3_fidelity.json` (+1 line: a `SUPERSEDED` key naming job 6398733 and the
`seg >= 0` defect). `git diff --stat 7820af1c..HEAD -- examples/results`: only those two text edits — no
frames, corpus, pool, tier or fidelity JSON changed. `examples/results/step4|5|6` untouched;
`stable_envs/render/` and the mjwarp tests still byte-identical to `9923594`.

## Verdict: APPROVE

| item | check | result |
|---|---|---|
| C1 | `grep '0.92-0.95\|10-19\|BY CONSTRUCTION\|0.35'` | one surviving `0.92-0.95` at line 704, a comment naming the defect; module docstring (line 45), `_MEASURED` comment (852) and `_RESIDUAL_NOTE` (1004) now quote 0.351 → 0.958 / 0.922 interior and 0.872 / 0.775 whole-frame; registered `visual_note` contains no `0.92-0.95` and does contain 0.958/0.922. |
| C2 | `_make_classes` call graph; import in both venvs | `_bundle(` is called only from `_context`, `__init__`, `reference_render`, `_renderer`; `SPEC` carries `state_dim` 24/30. se6-venv (no swm): `import stable_envs` 1.09 s, registry **78**, both cells registered, `import_errors()` has no `ogb_manip` entry, `_REF_CACHE == {}` and `_BUNDLE_CACHE == {}` after import; se7+swm: 0.93 s, `_REF_CACHE []`. `check_import_without_swm.py` prints `OK: registry 78, both cells registered, no stable_worldmodel needed`. |
| C3 | declaration vs `remeasure.json` | `mean_abs`/`within8_frac`/`ssim` equal worst-of-three (1e-6) and `draw_spread` equals JSON min/max; `interior_exact_frac` 0.958007 / 0.917805, `all_exact_frac` 0.870342 / 0.771891, `interior_frac` 0.411172 / 0.276900 — all now the WORST draw; `measured_at` names the three fields and says "WORST of the three". |
| floors | stated rule `floor(20·(worst − 0.04))/20` | Cube within8 0.85 (margin 0.0655), ssim 0.85 (0.0649); Scene within8 0.80 (0.0567), ssim 0.85 (0.0428); `mean_abs_max` 3.2 / 7.0 = `round(1.3·mean, 1)`. Rule reproduces every declared floor. |
| versions | `reference_versions` / `measured_at` | `{"mujoco": "3.12.0", "ogbench": "1.2.1"}`, both named in `measured_at`; `measured_with {"mujoco_warp": "3.12.0"}`. |
| wording | class `__doc__` | no `parity_note`; dims 24/30 stated from `SPEC`. |
| tests | `tests/unit` | without swm **873 passed / 2 failed / 7 skipped**; with swm **887 / 2 / 20**; cell tests **28 passed**; the two failures are the pre-existing `test_examples[robocasa_visual/render_mjwarp.py]` and `test_every_cited_scratch_path_resolves` on this Mac (the author's 859/21/2 is the Oscar count). |

Nit, non-blocking: the comment above `_REFERENCE_VERSIONS` (lines 910-912) still says "swm and ogbench
are deliberately absent" while the dict now includes ogbench — update the sentence when the file is next
touched. Nothing else outstanding; the cluster-only list from the re-review is unchanged.
