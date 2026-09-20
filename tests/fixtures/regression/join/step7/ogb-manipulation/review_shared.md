# Independent review — W5 (ogb-manipulation) SHARED changes to `stable_envs/render/mjwarp.py`

Reviewer: independent (did not write the code). Scope: branch `step7/ogb-manipulation`, tip `4264f202`,
baseline `76dc5116`; shared-file commits `4deb265`, `053a48f` (texture_gaps), `ba8b6000` (spot),
`fd5fbb4d` (bake). Standard: no registered family's output may change.

Local environment: `/Users/samib/se6-venv/bin/python` (mujoco 3.12.0, dm_control, torch CPU, no
mujoco_warp, no CUDA), `MUJOCO_GL=glfw`. `/Users/samib/se7-ogb-venv` has ogbench 1.2.1 but NOT
`stable_worldmodel`, so `swm/OGBScene-v0`/`OGBCube-v0` were not constructible here; the positive control
could only be checked from the committed artifacts and the mujoco_warp source.
Throwaway baseline worktree: `<scratchpad>/base76` at 76dc5116 (removed after the review).

## VERDICT: APPROVE-WITH-CHANGES

The three helpers are additive, generic, and provably inert on every registered family I could build
(and on the CUDA replays the author committed). Nothing OGB-specific is in `mjwarp.py`. But the spot
correction does not implement the MuJoCo semantics it claims (it leaves MJWarp's hard cone in place, which
MuJoCo discards), one of its own unit cases (`cutoff=-5`) asserts a behaviour MJWarp would not deliver,
the texture_gaps/bake mechanism is bound to `mujoco-warp <= 3.12.0` and is fixed upstream, the bake
record misattributes the offending texture in the multi-role case, and the spot byte-identity proof has
no committed provenance (no log, no card/node/SHA in the JSON, frames deleted). None of these changes any
registered family's pixels, which is why this is not a REJECT.

### Required changes (exact)

R1. `neutralise_illegal_spots`: also write `m.light_cutoff[i] = 180.0` for every neutralised light,
    and save/restore `light_cutoff` in the `restore` dict (`restore_model` already pattern-matches on
    keys). Reason (measured, see §1): MuJoCo drops the *whole* spot term for an out-of-range cutoff — a
    cutoff-100 spot aimed straight up (every visible surface > 90 deg off-axis) renders byte-identical to
    cutoff 180, and cutoff -5 renders identical to 180. MJWarp with `exponent=0` still applies
    `cos_theta < cos(cutoff)`: for cutoff in (90, 180) surfaces beyond the cutoff angle stay unlit; for a
    negative cutoff the retained cone is |cutoff| degrees (cos(-5 deg) = +0.996 -> a 5 deg cone). The
    author's own M3 job measured `exponent=0 + cutoff=180` at 0.3507 vs 0.3506 interior-exact on OGBCube
    (`oscar/m3/m3_corrections.json`), and upstream `render.py` uses `light_cutoff` only in that cone test,
    so the extra write costs nothing there and closes the semantic gap. Update the docstring sentence
    "the cone itself needs no change ... rejects essentially nothing" (false in general) and the test
    `test_an_out_of_range_cutoff_loses_its_angular_term` to assert the cutoff write too; drop or reword
    `test_the_cutoff_itself_is_left_readable` (its rationale conflicts with R1; the original is in
    `restore`). If the author insists on not touching the cutoff, the alternative is to document the
    residual precisely (which angles remain unlit) and remove `-5` from the "neutralised" parametrisation,
    since the test then asserts a MuJoCo equivalence the fix does not achieve.

R2. Version-bind the crash claim. `mujoco_warp` main guards the MESH branch
    (`if texcoord_offset >= 0:`), added by upstream commit `58d8c77` "Fix indexing error for meshes without
    texcoords (#1608)" on 2026-08-26. The Oscar pin is `mujoco-warp==3.12.0`, released on PyPI 2026-08-20,
    i.e. six days before the fix; 3.13.0 (2026-09-09) contains it. The author's mechanism is therefore
    correct for the pinned version and wrong for the current one. Required: in the `texture_gaps`
    docstring, `unsamplable_texture_materials`, `bake_unsamplable_textures` and findings §11/§17 state
    "unguarded in mujoco-warp <= 3.12.0; guarded upstream since #1608 (3.13.0)", and replace "the
    zero-pixel alternative is to fix mujoco_warp's sampler, ~3 lines" with "already fixed upstream; the
    zero-pixel alternative is a venv bump to >= 3.13.0, to be measured". On >= 3.13.0 condition 3 becomes a
    fidelity gap (uv stays (0,0), one texel) not a crash, so the bake's "crash half" framing must say it is
    conditional on the pinned version. The `texture_gaps` flag itself stays correct either way (a UV-less
    textured mesh renders one texel, the same class as condition 1).

R3. `bake_unsamplable_textures` record: in the multi-role case (2-D albedo in `mjTEXROLE_RGB`, cube map in
    another role) the helper clears ALL roles and records `texture = <the 2-D albedo>, tex_type = 0` while
    the reason for the bake was the cube map in role 4 (reproduced below, §3). Record every removed
    texture id/type (e.g. `removed: [(role, tex, tex_type), ...]`) and name the offending one(s), or only
    detach the non-2-D roles unless the mesh is UV-less. No registered family is affected; this is a
    declaration-correctness fix.

R4. Proof provenance for the spot byte-identity (job 6397885): no slurm log is committed for it, the
    `.npy` frames are deleted by the sbatch (`rm -f "$OUT"/*.npy`), and `m2/spot/byte_identity_spot.json`
    carries no host/GPU/SHA fields. Commit `slurm_6397885.log.gz` (as done for the other jobs) or add
    `host`, `gpu`, `pythonpath_before/after`, `git_head` to the JSON. Also state the sweep denominator
    honestly: "40 of 88 registry ids (Oscar) / 76 (local) have an MjModel the sweep reaches"; the 12 Fetch
    ids are covered only by the two CUDA replays and the 5 DMC trackcom locomotion ids (Cheetah, Hopper,
    Humanoid, Walker, Swimmer) by nothing in the author's artifacts (I covered them model-side, §2).

R5. The conditionally registered MJWarp families — RoboCasa, MetaWorld, HumanoidBench (`register()` when
    their packages import; absent from both sweeps because neither venv has them) — call `prepare_model`
    and (RoboCasa) `texture_gaps`. Their light cutoffs were checked by nobody. Before integration, run
    `sweep_spot_semantics.py`/`sweep_texture_gaps.py` in a venv where they build, or grep their MJCF lights
    statically, and record the result. (RoboCasa only stores `s.gaps`; nothing reads it, so a newly
    non-empty list changes no behaviour. MetaWorld already runs its own `_neutralise_cube_textures`.)

Recommended (non-blocking): declare the POINT==off finding as "MuJoCo 3.12.0 classic GL renderer"
behaviour (the type is copied to `mjvLight` unchanged in `engine_vis_visualize.c`; the classic renderer
simply does not draw POINT/IMAGE lights, which the new renderer does) — it is not a property of "MuJoCo".
Fix `m1/m1_mujoco_spot_cutoff.json`: its `exponent_sweep` table (exponent 10 -> 0 differs on 2.4 % of
pixels, `spot_off` 2.8 %) contradicts the seeded `light_forms` table (0 pixels) in the same file; the
note says the tables come from different unseeded resets — regenerate or delete the stale table.

## 1. MuJoCo semantics, verified with the CPU GL renderer (mujoco 3.12.0, glfw)

Script: `<scratchpad>/spot_probe.py` — one spot (`exponent=10`, dir (0.3,0,-1)), headlight off, 160 px,
frac of pixels differing / max abs / frame means. Shadows OFF (pure lighting law):

| comparison | frac diff | max | note |
|---|---:|---:|---|
| 100 vs 100 (noise floor) | 0 | 0 | deterministic |
| 100 vs 180 | 0 | 0 | **100 == 180** |
| 100 vs 90.0001 / 91 / 120 / 179 / -5 | 0 | 0 | all out-of-range values collapse to 180 |
| 100 vs 90 | 0.849 | 123 | 90 is honoured (mean 103.4 -> 38.0) |
| 100 vs 89.9 | 0.849 | 123 | |
| 100 vs 0 | 0.850 | 163 | 0 is honoured (frame black) |
| 100/exp10 vs 100/exp0, 100/exp50 | 0 | 0 | exponent inert at an illegal cutoff |
| 180/exp10 vs 180/exp0 | 0 | 0 | exponent inert at 180 |
| 90/exp10 vs 90/exp0 | 0.849 | 123 | exponent live at a legal cutoff |
| light off vs POINT | 0 | 0 | **POINT == off** (classic GL renderer) |
| **dir UP, 100 vs 180** | **0** | **0** | every visible surface > 90 deg off-axis is still lit at cutoff 100 |
| dir UP, 100 vs light off | 0.850 | 163 | i.e. the light is on |

So GL's rule is exactly "legal iff 0 <= cutoff <= 90 or cutoff == 180"; the fix's predicate
`not (0 <= cutoff <= 90)` treats 180 as illegal, which is harmless (exponent is inert at 180 in both
renderers and cos(180 deg) rejects nothing). The 90/90.0001 boundary test matches MuJoCo. With shadows ON,
100 vs 180 differs on 0.11 % of pixels (max 141) and -5 on 2.9 %: the spot's *shadow frustum* still reads
the raw cutoff, so "100 renders identically to 180" is exact only for the lighting term (author's own
numbers: 1.4e-4 of pixels, max 1). OGBCube's illegal spot has `castshadow=0` (census), so no issue there.

The residual that motivates R1: MuJoCo lights surfaces beyond the cone at an illegal cutoff (dir-UP row);
MJWarp after the fix still rejects `angle > cutoff` (100 deg here) and, for the test's `-5`, `angle > 5 deg`.

## 2. texture_gaps and prepare_model on real models, both trees

`<scratchpad>/reg_sweep.py` run at 76dc5116 and at 4264f202 in the same venv: registry 76 ids locally
(60 + 16 DMC; Fetch registers only with gymnasium_robotics, ogbench ids need ogbench, so 15 build here:
11 DM Control + 4 Mujoco). For all 15: `texture_gaps` identical, `prepare_model` hidden geoms/sites
identical, `light_exponent` untouched, `neutralised_spots == []`, `restore_model` exact, and on the branch
`unsamplable_texture_materials == {}`, `bake_n == 0`, no model field changed, restore exact. 8 of the 15
carry a spot light, all `cutoff=45, exponent=10` (7 DMC + MujocoPusher) — the correction is non-vacuous
and did not fire. 11 DMC models contain skybox textures (tex_type 2) no geom wears — not flagged.

`<scratchpad>/loco_probe.py` (dm_control `suite.load`, the models the trackcom cell hands to
`MW.prepare_model` at `dmcontrol.py:2128`): cheetah-run, hopper-hop, walker-walk, humanoid-stand,
swimmer-swimmer6, plus quadruped-walk and manipulator-bring_ball — identical base/branch on every field;
all spots cutoff 45; no gaps; no bake; quadruped carries a CUBE texture (tex_type 1) no rendered geom
wears — correctly not flagged.

Synthetic cases (branch): 2-D on mesh-with-UVs -> []; 2-D on plane -> []; cube on mesh -> 1; cube on
plane -> 1; cube on box -> 1; skybox worn by a mesh -> 1; 2-D on UV-less mesh -> 1; untextured UV-less
mesh -> []; 2-D on box -> 1 (unchanged condition 1). Baseline `texture_gaps` returns [] for cube-on-mesh,
2-D-on-UV-less-mesh AND cube-on-UV-mesh (the false all-clear is real). The author's 21+16+9 = 46 unit cases
pass (46 passed). The new test files cannot be collected against the baseline (ImportError on the new
names), so the unit-level positive control is the sweep script's verbatim old implementation plus the
committed OGBScene control (`old 0 -> new 2`, geoms 65/77, both the button meshes) — consistent.

## 3. bake helper

Generic: no env names in code paths (OGB appears only in docstrings/comments). Mean colour matches numpy
over `tex_data` exactly: cube checker red/blue -> [0.5, 0, 0.5] (helper identical); 2-D gradient ->
[0.1850, 0.1850, 0] (identical). Groups respected (a group-4 wearer is ignored, but the material is
detached for all wearers, which is fine since group-4 is not drawn). A textured BOX is left to
`bake_primitive_textures` (returns `n_materials=0`). Registry: touches none of the 15 local + 7 dm_control
models; author's sweep: none of 40. Multi-role reproduction (mat_texid `[-1,0,-1,-1,1,...]`, tex 0 = 2-D,
tex 1 = cube): record says `texture=0 ('a'), tex_type=0, mean of the 2-D` and all roles are cleared -> R3.

## 4. Regression and diff hygiene

`tests/unit` (ignore `test_examples.py`), same venv: baseline 713 passed / 1 failed / 6 skipped; branch
759 passed / 1 failed / 6 skipped; the single failure is identical on both
(`test_repo_contracts.py::test_every_cited_scratch_path_resolves`) — no new failures, +46 = the new cases.
AST map of the `mjwarp.py` diff (`<scratchpad>/ast_diff.py`): ADDED `GL_SPOT_CUTOFF_MAX`,
`TEXTURE_SAMPLER_OK`, `neutralise_illegal_spots`, `unsamplable_texture_materials`,
`bake_unsamplable_textures`, `field` import; CHANGED `Prepared` (one defaulted field + `summary`),
`prepare_model` (kw-only `spot_semantics=True`), `restore_model` (conditional key), `texture_gaps`.
Nothing else. No caller constructs `Prepared(...)` or reads `restore[...]` outside the module.
Files touched outside step7 artifacts: `stable_envs/render/mjwarp.py` and the three new test files only.

## 5. Proof quality

- `m2/texture_gaps_sweep.json` (local) and `m2/cuda/texture_gaps_sweep.json` (job 6397397, gpu2101,
  RTX 3090, log committed): `built 40, identical 40, changed 0`; the 40 ids enumerate 25 OGBench maze +
  11 DMC + 4 Mujoco; 36 (local) / 48 (Oscar) rows skipped as non-MuJoCo or not-constructible — the Fetch
  ids failed with `AssertionError` in the se7-ogb venv (mujoco 3.12 vs the 3.11 pin). Control fires
  (0 -> 2). The sweep's old implementation is a verbatim copy of 76dc5116's function (checked).
- `m2/cuda/byte_identity.json` DMCReacherEasy/PointmazeMediumNavigate and `byte_identity_fetch.json`
  FetchReach/FetchPickAndPlace (job 6397454, `FETCH before/after` sums equal, RTX 3090, log committed):
  I re-read the eight committed `.npy` frames — byte-equal, max 0, non-degenerate (std 36-77).
- `m2/spot/byte_identity_spot.json` (job 6397885): identical=True for DMCReacherEasy, DMCCartpoleSwingup,
  FetchReach; DMCPointMassEasy and MujocoPusher `missing` (no Step-4 states — admitted by the author;
  `08_spot_gap_close.sbatch` exists but no result is committed). No log, no frames, no host/card in the
  JSON -> R4.
- Positive control for the crash: `oscar/mjwarp/mjwarp_bisect_OGBScene.json` (illegal memory access,
  fixed by detaching the one cube texture) — consistent with the unguarded index in mujoco-warp 3.12.0.
- `m1/m1_mujoco_spot_cutoff.json`: internally inconsistent tables (see Recommended).

## 6. Hygiene over 76dc5116..4264f202

30 commits, all `sami-bg <samibghanem@gmail.com>`; 0 `Co-Authored-By`/`Generated with`/AI trailers
(grep over all bodies); `examples/results/step4|5|6`: 0 files touched.

## 7. What only the CUDA proofs cover (cannot be checked locally)

- That MJWarp's shader actually reads `light_exponent` per frame the way the docstring says and that
  `exponent=0` produces `pow(cos,0)=1` (the 0.0091 -> 0.3507 OGBCube improvement, jobs 6397454/6397621).
- That the crash is really the texcoord index (bisect job) and that OGBScene renders after the bake.
- Pixel identity of the Fetch family (3.11 stack) and of the two DMC replays — the model-side sweep shows
  the inputs to MJWarp are unchanged, but pixels were only asserted on Oscar (3090).
- Anything about `mujoco-warp >= 3.13.0` behaviour (not installed anywhere in this project).
- The 5 DMC trackcom locomotion cells and the 25 OGBench cells: model-side identity is shown (mine and the
  author's respectively); no pixel replay exists for them on the branch other than
  PointmazeMediumNavigate.

## Post-tip note (outside scope, checked because it bears on R4)

The worktree has moved to `e12c58f` ("Close the spot-correction proof gap with sampled states (job
6398026)"); the shared files and the three test files in the working tree are byte-identical to
`4264f202` (`git diff --stat 4264f202 -- <the four files>` is empty), so every number above applies to the
reviewed tip. `m2/spot/byte_identity_spot_sampled.json` now shows DMCPointMassEasy (8x84x84) and
MujocoPusher (8x240x240) identical, max 0, on `sample_states(8, seed 0)` generated once from the pristine
baseline — this closes the admitted replay gap, but with the same provenance defect as job 6397885: no
committed slurm log, no host/GPU/SHA in the JSON, frames deleted. R4 applies to job 6398026 as well.

## Commands (all from this review)

```
git worktree add --detach <scratchpad>/base76 76dc5116
MUJOCO_GL=glfw se6-venv/python spot_probe.py
MUJOCO_GL=glfw se6-venv/python reg_sweep.py <tree> <out.json>     # both trees, diffed
MUJOCO_GL=glfw se6-venv/python loco_probe.py <tree> <out.json>    # both trees, diffed
MUJOCO_GL=glfw se6-venv/python -m pytest tests/unit -q --ignore=tests/unit/test_examples.py  # both trees
se6-venv/python ast_diff.py base76/.../mjwarp.py <branch>/.../mjwarp.py
git log --format='%an|%ae' / '%B' 76dc5116..4264f202 | grep -ciE 'co-authored|generated with|claude|anthropic'
git diff --name-only 76dc5116..4264f202 | grep -E '^examples/results/step[456]/' | wc -l   # 0
```

---

# Re-verification at tip `9923594` (coordinator request)

Commits since `4264f202`: `e12c58f`, `a088fb5`, `527cc9f`, `9923594` — 4 commits, all sami-bg, 0 trailers,
`examples/results/step4|5|6` untouched (0 files). Non-artifact files touched: `stable_envs/render/mjwarp.py`,
`stable_envs/envs/ogb_manip.py` (the new cells, out of scope), the two test files. Worktree clean at HEAD.

## Verdict: APPROVE — conditional on job 6398530 (RoboCasa/MetaWorld/HumanoidBench model-side sweeps + Fetch control)

### R1 spot rule — verified, closed
Diff: `GL_SPOT_CUTOFF_OFF = 180.0`; `neutralise_illegal_spots` now saves `light_cutoff`, writes
`light_exponent = 0` AND `light_cutoff = 180` for `not (0 <= cutoff <= 90)`; `restore_model` restores both
arrays when present. Docstring replaced by the measured mechanism (cone test, -5 -> 5-degree cone, straight-up
probe, 0.3506 vs 0.3507). My constructions at 9923594 (`prepare_model` on the 3-light model):

| cutoff | neutralised | after (cutoff, exp) | MJWarp cone rejects | legal spot untouched | type unchanged | restore both arrays exact |
|---|---|---|---|---|---|---|
| -5 | [0] | (180, 0) | angle > 180 (never) | yes | yes | yes |
| 100 / 91 / 179 / 180 / 90.0001 | [0] | (180, 0) | never | yes | yes | yes |
| 0 / 45 / 90 / 89.9999 | [] | unchanged | own cone | yes | yes | yes |

`restore` keys: `geom_group, site_group, light_exponent, light_cutoff`. `spot_semantics=False` leaves (100, 10).
Tests now assert the cutoff write on all four bad values, the -5 case against `np.cos` (== -1 after), the
90/90.0001 discontinuity, restore of both arrays, and opt-out keeps the cutoff. The conflicting
`test_the_cutoff_itself_is_left_readable` was replaced by `test_prepare_model_restores_the_cutoff_too`.

### R2 version-binding — verified, closed
`texture_gaps` condition 3 and `bake_unsamplable_textures` now say "mujoco_warp <= 3.12.0; guarded upstream from
3.13.0 (58d8c77 / #1608, 2026-08-26); on such a pin a crash"; the "~3-line patch" sentence now reads "already
fixed upstream; a venv bump, TO BE MEASURED". POINT==off is stated as "MuJoCo's CLASSIC GL renderer at 3.12.0 ...
a property of that renderer at that version".

### R3 bake record — verified, closed
Multi-role reproduction (2-D albedo role 1 + cube role 4): record now carries `removed_textures =
[{tex 0, role 1, 2d, unsamplable False}, {tex 1, role 4, cube, unsamplable True}]`, `n_removed = 2`; the
single-role cube case records `[{tex 0, role 1, cube, unsamplable True}]`. Restore exact. Two new tests cover it.

### R4 provenance — closed for the identity proofs, residual on stamps
Committed: `m2/spot/slurm_6397885.log.gz` (gpu2106, RTX 3090), `slurm_6398026.log.gz` (gpu2109, RTX 3090),
`slurm_6398465.log.gz` (gpu2107, RTX 3090). Each log shows both legs in one job with `PYTHONPATH=.../se7/baseline`
then `.../se7/ogb-manipulation`, equal per-config pixel sums, and the `BYTE-IDENTITY` rows. Job 6398465 (after the
cutoff change): DMCReacherEasy, DMCReacherHard, DMCCartpoleSwingup, DMCCartpoleBalance, DMCAcrobotSwingup,
DMCPendulumSwingup (Step-4 states), DMCPointMassEasy, MujocoPusher (sampled seed 0) — all 8 spot-carrying
registered families `identical: True, max_abs 0`; FetchReach leg failed on a missing `_states.npy` (job-script
omission, re-run in 6398530 — pending). `m1_mujoco_spot_cutoff.json` `exponent_sweep` now reads 0 px on every row
with `verdict_exponent_tables_agree: True`. The eight CUDA `.npy` frames I byte-checked earlier were dropped in
`9923594` (size); my verification of them stands on the pre-drop commit. Residual (non-blocking): the spot and bake
sweep JSONs carry a `provenance` dict (host, python/mujoco/mujoco-warp/torch/warp versions, slurm_job, git_sha),
but the Oscar copies have `git_sha = ""` (rsync tree has no `.git`); the texture_gaps sweep JSONs and the two
identity JSONs carry no stamp; the three logs record node and card but no SHA (the tree is identified by
PYTHONPATH only). Recommend stamping `git_sha` from a file written at rsync time.

### R5 conditional families — pending 6398530
Cross-venv sweep (job 6398427, node2321, CPU): se-step5 15/88 reachable, se-step6 40/76, se7-ogb 40/88 — all
correct/identical, bake touches none, control fires only where `stable_worldmodel` exists (se7-ogb). The
`families_present.json` import errors show robocasa/metaworld/humanoidbench refuse to import on those
interpreters (missing metaworld / no compiled RoboCasa cache, mujoco 3.3.1 pin / mujoco_warp-only families). They
remain unverified until 6398530 reports; the sweep denominators are now stated honestly (`registry_ids_total`,
`n_with_reachable_mjmodel`, `n_no_mjmodel`, `n_skipped_not_constructible`, coverage note on Fetch).

### Regression / identity at 9923594
`tests/unit` (ignore `test_examples.py`): 763 passed, 1 failed, 6 skipped — the same pre-existing
`test_every_cited_scratch_path_resolves` as at baseline (713/1/6); +50 vs baseline = the new cases. Two-tree
registry sweep: 15 local families identical to 76dc5116 on every recorded field, `neutralised == []`, bake 0,
restore exact; dm_control locomotion/trackcom/manipulator/quadruped models identical, no neutralisation, no
bake. AST map vs baseline: ADDED `GL_SPOT_CUTOFF_MAX`, `GL_SPOT_CUTOFF_OFF`, `TEXTURE_SAMPLER_OK`, three
functions, `field` import; CHANGED `Prepared`/`summary`, `prepare_model`, `restore_model`, `texture_gaps`. No
OGB/env name on any non-comment, non-docstring line of `mjwarp.py`.
