# Step 8 — Workstream B: VariationMaps for the twelve Fetch ids

**Verdict: `PATCH_CANDIDATE`**, re-submitted after an independent review returned `BLOCK` with five
required corrections. All five are addressed below under **Review follow-up**; the escalation
candidate (`background.color` / skybox) has been ruled on and is now classified `partial`, and five
upstream findings remain, none of which is a stable-envs defect.

- Baseline SHA: `43ac95238f7472137619f3e69e7636e11dd088b1`
- Branch: `step8/variations-fetch`
- Narrative and the full leaf table: `docs/research/step8_variations_fetch.md`
- Machine-readable classification with every measurement: `leaf_classification.json` (this directory)

## What was done

The twelve `swm/Fetch*-v3` ids now have VariationMaps, built as **one family**: the `Dense` and
`Dict` suffixes select a reward and an observation container and reach no pixel, and their variation
spaces were enumerated from the live `FetchWrapper` and found **identical per scene** (11 leaves on
Push/Slide/PickAndPlace, 8 on Reach). So the maps come from one vocabulary plus a per-scene
difference table, and a test asserts that the only field differing between `FetchPush`,
`FetchPushDense` and `FetchPushDict` is `swm_id`.

New module `stable_envs/compat/variation_maps/fetch.py` (plus a minimal package `__init__.py`), one
registration line in `stable_envs/compat/worldmodel.py`, and
`tests/unit/test_variation_maps_fetch.py`. No renderer file was touched and no Fetch performance work
was done.

## Per-leaf counts

12 ids, **123 leaves**, all classified, **0 undocumented**:

| classification | leaves | per-id (Reach / object scenes) |
|---|---|---|
| `mapped` | 33 | 2 / 3 |
| `partial` | 12 | 1 / 1 |
| `unsupported` | 12 | 1 / 1 |
| `dead_upstream` | 15 | 2 / 1 |
| `not_applicable` | 51 | 2 / 5 |

Mapped: `table.color` (identical), `light.intensity -> light.diffuse` (scaled by exactly 1.25),
`object.color` (identical, object scenes only).

Partial: `background.color -> floor.color` on all twelve — the rename is exact and the floor half is
reproduced exactly; the skybox half is not rendered. See **Review follow-up, correction 4**.

These labels come from `VariationMap.classification()`, so `check_mapping`, the Gate-F matrix and
every generated support table inherit them. The unmapped leaves are declared in the three typed
buckets (`unsupported` / `dead_upstream` / `not_applicable`) rather than in the undifferentiated
`swm_only`, and a test asserts `swm_only` stays empty for this family.

Unsupported: `rendering.transparent_arm` — live upstream at 9.3-9.4 % of pixels, and `mujoco_warp`'s
render kernel reads `color[0..2]` and packs a constant 255, so there is no alpha path at all. The
same mechanism already recorded in `FetchBase.parity_note` for the four translucent `robot0:mocap`
markers. `DOCUMENTED_LIMITATION`.

Dead upstream: `camera.angle_delta` on all twelve — the wrapper rotates `model.cam_pos` for the four
**model** cameras and `MujocoFetchEnv` renders through the **free** camera from
`DEFAULT_CAMERA_CONFIG`, so the frame is **0/255 over 0 % of the frame at +-10 deg** on all four
scenes while `cam_pos` really moves by 0.2257. Plus `object.color` on **Reach only**: `reach.xml`
compiles neither `block_mat` nor `puck_mat` onto a geom, so the write lands on an empty geom list —
measured 0/255 between two maximally different colours, against 240/255 on the other three scenes. We
have working camera knobs and deliberately did **not** map the dead leaf onto them: that would move
our frame where the reference's does not.

Not applicable: `agent.start_position`, `goal.start_position`, `block.start_position`, `block.angle`
are reset distributions in swm and dimensions of our state (`qpos + goal`) — the same classification
PushT already carries. `block.mass` is pure dynamics: 0/255 at 0.01 kg versus 50 kg on all three
object scenes.

Ours and not swm's: `goal.color`, `arm.color`, `camera.distance_scale`, `camera.azimuth_delta`,
`camera.elevation_delta`.

## The one unit shift

swm assigns the light's diffuse absolutely (`Box(0,1)`, declared default 0.7); ours multiplies the
model's own `light_diffuse`, which is **0.8** on all four scenes, with default 1.0. So
`ours = theirs * 1.25`. The factor is exactly representable, so swm's 0.8 round-trips to our 1.0 with
no error and the default-reproduction test uses `array_equal`, not `allclose`. swm's full `[0,1]`
range maps into our `Box(0,2)` with nothing clipped — **no supported range was widened.**

## Escalation (plan section 4) — RULED, and now classified `partial`

**`background.color`, all 12 ids.** swm's "background" is the `floor_mat` geoms **and** the skybox
texture. Measured split at 480 px: floor **9.2-9.4 %** of the frame, skybox a further
**61.7-73.0 %**. We render the floor half exactly and the skybox not at all, because `mujoco_warp`
uploads textures when a render context is built, so a per-value skybox is one render context per
value — which `FetchBase.make_variation_space` already records as the reason for not offering a
skybox knob.

**Ruling (Sami, via the Step-8 coordinator, 2026-09-16):** a feasible, semantically required leaf is
**not** declared unsupported merely because a renderer change is needed. The skybox half is a
renderer *capability* need and is aggregated with the other families' capability needs after Wave 1;
this workstream does not implement it. So the leaf is classified **`partial`** — the mapping stays,
the residual travels with it, and nothing generated can count it as full support. See **Review
follow-up, correction 4**.

## Upstream findings

1. **`_apply_visual_variations` permanently unbinds three materials.** `model.geom_matid[i] = -1` on
   the table, floor and object geoms the first time **any** of the six visual leaves is active, for
   the life of the env; those geoms then render at the compiler default 0.5 grey instead of their
   material colours (0.93 / 0.2 / 0.2). Measured with the dead `camera.angle_delta` held at its own
   default, so the whole difference is the unbind: **19.0-32.3 % of the frame, up to 102/255**.
   `DEFAULT_VARIATIONS` activates five visual leaves, so a plain `reset()` always triggers it.
   Consequence: a parity number computed through the swm Fetch wrapper measures the wrapper, not
   stable-envs.
2. **swm's declared leaf defaults are not the official appearance** — table 0.3 vs 0.93, object
   (0.8,0.1,0.1) vs 0.2 grey, background 0.1 vs floor 0.2 + skybox, light 0.7 vs 0.8,
   `agent.start_position` (1.3418,0.7491) vs the env's own initial gripper xy (up to 55/255 on Reach).
3. **`variation_values` entries are used unconverted**: a plain Python list for `background.color`
   raises `AttributeError: 'list' object has no attribute 'astype'`.
4. **`compat.worldmodel.check_mapping` cannot read a Fetch variation space** — it uses
   `.unwrapped.variation_space`, and `FetchWrapper` is a `gym.Wrapper` whose `.unwrapped` is the
   official env. `AttributeError` for all twelve ids. Ownership request below.
5. **gymnasium-robotics + numpy 2.5.3 + mujoco 3.12 cannot construct a Fetch env** —
   `model.jnt_type[i] in (mjJNT_HINGE, mjJNT_SLIDE)` is `False` for an `np.int32` needle because
   mujoco's pybind11 enum returns `NotImplemented`, while `==` is `True`. Worked around in a
   scratchpad probe shim only; **nothing in this repository changes for it**, and Oscar's
   `se-s3b-fetch311` (mujoco 3.11.0) — the interpreter the frozen Fetch corpora were measured in — is
   unaffected.

## Ownership requests — all three RESOLVED by the merge of `step8/integration`

1. ~~`stable_envs/compat/worldmodel.py::check_mapping` — needs to walk the wrapper chain rather than
   read `.unwrapped.variation_space`.~~ **Done centrally by Workstream A**
   (`worldmodel.swm_variation_space`), which this branch takes by merging `step8/integration`.
   `variation_maps.fetch.swm_variation_space` now delegates to it instead of carrying a second copy,
   and `test_check_mapping_reads_a_fetch_variation_space_through_the_public_function` is the
   regression test through the public entry point. This was review correction 3.
2. ~~`VariationMap` would be more truthful with a `partial` field.~~ **Done centrally by the
   coordinator** after A merged: `partial: Mapping[str, str]` (`swm_path -> what is NOT reproduced`),
   refused in `__post_init__` unless some mapping field reaches the leaf, and reported as `partial`
   by `classification()`. The exact extension this workstream needed and would otherwise have had to
   ask for is therefore delivered; the one line, for the record, was
   `partial: Mapping[str, str] = field(default_factory=dict)` with a reason per leaf.
3. ~~`mkdocs.yml` — one additive nav entry.~~ **Done**; both A's and B's nav entries are kept in the
   merge, and `test_every_docs_page_is_reachable_from_the_nav` passes.

## Measurement environments

**Model side and official renderer** — `/Users/samib/se8-fetch-venv` (new): python 3.12,
mujoco 3.12.0, gymnasium 1.3.0, gymnasium-robotics 1.4.2, numpy 2.5.3, torch (cpu),
opencv-python-headless, `stable_worldmodel` from `/Users/samib/stable-worldmodel` (176d545) installed
editable, `MUJOCO_GL=glfw`, Apple M5 / macOS 25.6.0. Every per-leaf pixel number in this bundle is
the **official** gymnasium-robotics renderer driven through the swm `FetchWrapper` at 224x224 — which
leaf reaches which array, and whether the reference itself moves.

**Our renderer** — Oscar, one GPU job, `$SCR/venvs/se-s3b-fetch311` (mujoco / mujoco_warp 3.11.0,
gymnasium-robotics 1.4.2), the interpreter the frozen Fetch corpora were measured in. See
`gpu_evidence.json` and `pixel_parity.json` in this directory for provenance (GPU model, host,
driver, job id, git SHA, mujoco / mujoco_warp versions).

## Tests

`tests/unit/test_variation_maps_fetch.py`, three tiers by what they need installed. Measured on the
merged tip, verbatim:

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q \
    tests/unit/test_variation_maps_fetch.py -p no:cacheprovider
10 passed, 49 skipped in 0.09s

MUJOCO_GL=glfw /Users/samib/se8-fetch-venv/bin/python -m pytest -q \
    tests/unit/test_variation_maps_fetch.py -p no:cacheprovider -rs
10 passed, 49 skipped in 2.64s

MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q \
    tests/unit/test_repo_contracts.py -p no:cacheprovider
1 failed, 10 passed in 0.39s
  FAILED test_repo_contracts.py::test_every_cited_scratch_path_resolves
```

**The 10/49 figure is the truth and the earlier "32 passed, 24 skipped" was not reproducible.** On
`se8-fetch-venv` all 36 live-wrapper checks skip with `AssertionError()` from
`gymnasium.make("swm/Fetch*-v3")` — upstream finding 5 below, the numpy 2.5.3 + mujoco 3.12 enum
comparison — and the author's 32-pass run needed a scratchpad shim that is deliberately not in this
repository. The independent reviewer measured the same 8/48 (now 10/49) and was right to flag the
discrepancy. Nothing in the maps depends on the shim; what the shim bought was a second measurement
of the leaf snapshot, and the snapshot is instead pinned by tier 1 and by `leaf_classification.json`.

`test_every_cited_scratch_path_resolves` fails on this branch and **fails identically at the merge
commit with every change of this workstream stashed** (verified). It is broken by cited `scratch/`
paths that live in other workstreams' worktrees (`scratch/step8/perf/*`, `scratch/step8/semantic/*`,
`scratch/step8/report.py`, …) plus the pre-existing baseline breakages (`scratch/sboughan/...`,
`scratch/carracing/logs/`). Pre-existing and not this workstream's to fix; the coordinator will see
it resolve as the remaining workstreams merge their scratch directories.

**A certification run can no longer pass by skipping.** `STABLE_ENVS_REQUIRE_LIVE_SWM=1` turns every
skip in tiers 2 and 3 into a failure, so an all-skipped capability tier cannot be reported as green.
Documented in the test module's docstring. Verified both ways on `se6-venv`, which has neither
dependency: `10 passed, 49 skipped` without the switch, `25 failed, 9 passed, 24 errors` with it.

## GPU evidence (SLURM job 6443480 on the merged tip; supersedes 6452376 and 6451948)

`gpu_evidence.json`, 128 px, 8 states per scene drawn from the env's own pool, on
**NVIDIA GeForce RTX 3090** / `gpu2264.oscar.ccv.brown.edu` / driver 575.57.08 / CUDA 12.6 /
torch 2.14.0+cu126 / **mujoco 3.11.0, mujoco_warp 3.11.0, warp 1.17.0** / python 3.12.9 —
`$SCR/venvs/se-s3b-fetch311`, the interpreter the frozen Fetch corpora were measured in.
Tested tree SHA `5940eb6866e2cb0a91c7589f2f0aca1d58997a43` (the merged tip), baseline
`43ac95238f7472137619f3e69e7636e11dd088b1`. `gpu_job_6452376.out.txt` beside it is the raw stdout of
the earlier clean re-run on `4b3d09f`, kept because correction 1 is about it; the current artifacts
carry job 6443480 in their own provenance blocks. **Verdict: PASS on all four scenes.**

- **Our registered default is byte-identical through the mapping.** Precisely: stable-envs'
  `env.default_style()` equals `style_from_variation(env, {})` exactly, and the frames rendered from
  the two are **0/255 over 0 % of the frame** on Reach, Push, Slide and PickAndPlace. This is a
  statement about THE MAPPING — an empty swm config selects our registered default and nothing else.
  **It is not equality with the SWM wrapper's registered reset appearance**, and must not be reported
  as SWM default parity; see the Gate-F status line below.
- **The inverse round trip is exact.** `variation_from_style(default)` gives
  `light.intensity = 0.800000011920929`
  — the model's own diffuse, *not* swm's declared 0.7 — and re-applying it returns the default style
  byte for byte.
- **The 1.25 factor lands where it should.** `light.intensity` set to the model's own 0.8 reproduces
  the default frame at **0/255**, which is the whole justification for the factor.
- **Every unmapped leaf moves nothing.** All 7 (5 on Reach) declared-unmapped leaves pushed through
  the mapping leave the style byte-identical and the frame at **0/255**:
  `camera.angle_delta`, `rendering.transparent_arm`, `block.mass`, `agent.start_position`,
  `goal.start_position`, `block.start_position`, `block.angle`, and `object.color` on Reach.
- **Every mapped leaf moves pixels, and only its own style slot.**

  | leaf | ours | frame changed (Reach / Push / Slide / PickAndPlace) | max |
  |---|---|---|---|
  | `table.color` | `table.color` | 10.03 / 9.81 / 23.36 / 9.92 % | 240/255 |
  | `background.color` | `floor.color` | 9.30 / 9.30 / 9.64 / 9.30 % | 194/255 |
  | `object.color` | `object.color` | — / 0.084 / 0.043 / 0.087 % | 194/255 |
  | `light.intensity` | `light.diffuse` | 14.96 / 14.97 / 20.94 / 14.88 % | 86-90/255 |

  Our `floor.color` moves **9.30-9.64 %** of the frame; the official renderer's floor half of
  `background.color` was measured independently at **9.18-9.45 %**. The two agree, which is the
  corroboration that the rename is the right one and that the residual really is the skybox.

Job 6451948 exposed two defects in the evidence harness itself, neither in the mapping: the tier-3
fixture errored instead of skipping on an interpreter with `mujoco_warp` and a CUDA device and no
gymnasium-robotics (24 setup errors), and `provenance()` shelled out to `git rev-parse` in a tree
`rsync` had stripped of `.git`, so `git_sha` came back null. Both fixed; job 6452376 is the clean
re-run. Its unit tier reported **8 passed, 48 skipped** — `$SCR/venvs/se6-verify` has no
`gymnasium_robotics`, so the 48 live-wrapper and GPU-fixture checks skip there. That, not a 32-pass
run, is the unit-tier truth for job 6452376. The committed artifacts are now from job 6443480 on the
merged tip (same card, same venv), where the same tier reports **10 passed, 49 skipped** — two new
tier-1 tests and one new tier-2 test.

## Pixel movement

**None expected and measured as none.** This workstream adds a module and one registration line; the
render path imports neither. `pixel_parity.json` renders the four scenes' default frames from this
branch and from `$SCR/se8/baseline` on the same card in the same job and compares them byte for byte:
**0/255 over 0 % of the frame on all four scenes** (8 states each at 128 px, RTX 3090 gpu2264,
job 6443480, branch SHA `5940eb6`). Verdict `IDENTICAL`. This is a random-state smoke check, not the
frozen-corpus replay; the replay is `frozen_corpus_replay.json` (review correction 2).

## Not done, deliberately

No Fetch performance work (plan section 2). No renderer change for `rendering.transparent_arm` or for
the skybox half of `background.color` — both are reported, not implemented. No widening of a
supported range. No re-freeze.


---

# Review follow-up

Independent review of `ce1fcb3` returned **BLOCK** with five required corrections
(`examples/results/step8/reviews/b.md`). Each is addressed below with the evidence, and the
merge-gate table at the end is restated against the current tip.

## Correction 1 — evidence provenance

*"Recover the actual clean GPU evidence or rerun it on Oscar... Do not simply edit the old artifact's
SHA."*

The reviewer was exactly right: `findings.md` described job 6452376 while the committed
`gpu_evidence.json` was still 6451948 with `git_sha: null`, and `unit_tests.txt` still ended in
`8 passed, 24 skipped, 24 errors`. The corrected artifacts existed only on the cluster.

- Job 6452376's artifacts were pulled from `$SCR/se8/variations-fetch/...` into a staging directory,
  verified, and committed, together with the job's raw stdout as `gpu_job_6452376.out.txt`.
  Provenance: `git_sha 4b3d09fae90270c59d1273cfb7d5fc37e041ed75`, host `gpu2264`, RTX 3090,
  driver 575.57.08, `se-s3b-fetch311` (mujoco / mujoco_warp 3.11.0).
- **The evidence was produced by the same implementation as the reviewed tip:**
  `git diff --stat 4b3d09f ce1fcb3` is `docs/research/step8_variations_fetch.md`,
  `findings.md`, `gpu_evidence.json`, `pixel_parity.json`, `unit_tests.txt` — five files, all docs
  or artifacts, no source file and no test file. Nothing measurable changed between the measured
  tree and the reviewed one.
- **Unit-tier truth, stated:** job 6452376 reports **8 passed, 48 skipped**, not 32 passed.
  `$SCR/venvs/se6-verify` has no `gymnasium_robotics`, so the live-wrapper and GPU tiers skip there.
- Everything has since been re-measured on the merged tip (`5940eb6`) in job 6443480, and those are
  the committed artifacts now. The 6452376 log is kept because this correction is about it.
- The baseline SHA `43ac95238f7472137619f3e69e7636e11dd088b1` is recorded in every artifact.

## Correction 2 — affected frozen-corpus replay

*"`pixel_parity.py` ... never loads the Step-4 frozen Fetch corpora. It supports a limited
unchanged-default smoke check, not merge-gate item 5."*

Correct, and the replay has been run: `frozen_corpus_replay.json`, produced by
`scratch/step8/variations-fetch/frozen_corpus_replay.py`.

- **What it replays:** the committed `examples/results/step4/states/Fetch{Reach,Push,Slide,PickAndPlace}.npz`
  — 64 frozen states each — at Step 4's own render configuration (480 px, supersample 1, batch 64,
  cuda). The arrays are read from ONE tree and handed to both interpreters on disk, so a difference
  can only come from the renderer.
- **Both trees, one job, one card:** this branch and `$SCR/se8/baseline` in SLURM job **6443480** on
  `gpu2264.oscar.ccv.brown.edu`, NVIDIA GeForce RTX 3090, `se-s3b-fetch311` — mujoco 3.11.0,
  mujoco_warp 3.11.0, torch 2.14.0+cu126, numpy 2.5.3, python 3.12.9, identical on both sides.
- **Verdict `IDENTICAL_AND_ORACLE_REPLAYS`**, on all four scenes:

  | scene | branch vs baseline | branch vs frozen Step-4 `digest_ours` | corpus sha256 vs Step-4 record |
  |---|---|---|---|
  | FetchReach | byte-identical, maxdiff 0 | identical | identical (`f5a519689d336623…`) |
  | FetchPush | byte-identical, maxdiff 0 | identical | identical (`93d8ca8025baf479…`) |
  | FetchSlide | byte-identical, maxdiff 0 | identical | identical (`17375ce1403e4548…`) |
  | FetchPickAndPlace | byte-identical, maxdiff 0 | identical | identical (`ec78e35a4db90ab5…`) |

  The `.npz` files are also asserted byte-identical between the two trees, so the input to the
  comparison is provably the same corpus on both sides.
- **The eight `Dense`/`Dict` siblings** are hashed rather than re-rendered: their frozen corpora are
  the same arrays (asserted) and Step 4's own `sibling_digest_matches` recorded that they draw the
  same picture (asserted, from the committed shard). Re-rendering them measures one kernel three
  times.
- **Tested SHA:** `.git` is not synced to the cluster, so each tree's SHA is stamped into a
  `.git_sha` file by the sync and recorded in the artifact — branch `13ef3789…` (the tip at replay
  time), baseline `43ac9523…`. The artifact additionally carries the source-identity argument
  directly, since a stamped SHA is a claim rather than a proof:

  | path | `git ls-tree` at 43ac952 | at the branch tip | identical |
  |---|---|---|---|
  | `stable_envs/envs` | `b6ba355d0d997ed4…` | `b6ba355d0d997ed4…` | yes |
  | `stable_envs/render` | `6abd350f218d5f41…` | `6abd350f218d5f41…` | yes |
  | `stable_envs/testing` | `0a6299acdbd0124f…` | `0a6299acdbd0124f…` | yes |
  | `examples/results/step4` | `15a0fc2f71259a70…` | `15a0fc2f71259a70…` | yes |
  | `examples/results/step7` | `93858185db77b459…` | `93858185db77b459…` | yes |

  Nothing that can reach a Fetch frame, and neither frozen result tree, differs between the baseline
  and this branch.

  *(Recorded because it cost a wrong answer once: the first replay reported four false corpus
  mismatches because it hashed the state buffer plainly, while Step 4 hashed corpora with
  `conformance.frames_digest`, which feeds `shape` and `dtype` in first. The driver now uses Step 4's
  recipe and the four sha256s match the recorded ones exactly.)*

## Correction 3 — wrapper discovery in `check_mapping`

*"Resolve wrapper discovery centrally in `worldmodel.check_mapping`, with a regression test through
the public function... Coordinate ownership with A/D; do not independently overwrite the shared
file."*

**Resolved by Workstream A's merge, not by this branch.** `stable_envs/compat/worldmodel.py` is
coordinator-owned and this workstream did not edit it. A's `worldmodel.swm_variation_space` walks the
wrapper chain and `check_mapping` calls it, which is exactly the fix; `step8/integration` is merged
into this branch (`git merge`, no rebase) and `worldmodel.py` here is byte-identical to A's.

What this branch adds on top:

- `variation_maps.fetch.swm_variation_space` now **delegates** to the central function instead of
  carrying a second copy of the walk, so the discovery rule lives in one place, and it closes the
  env on every path including failure;
- `test_check_mapping_reads_a_fetch_variation_space_through_the_public_function` asserts the fix
  through the **public** entry point on `swm/FetchPush-v3` — the family that exposed the defect —
  including that `check_mapping` returns no `undocumented` leaves, classifies `background.color` as
  `partial`, `camera.angle_delta` as `dead_upstream`, `rendering.transparent_arm` as `unsupported`
  and `block.mass` as `not_applicable`, and keeps `background.color` in `covered` (a partial leaf is
  still a mapped leaf, qualified rather than dropped).

The registration of the Fetch maps also moved: B's direct
`from .variation_maps.fetch import MAPS; VARIATION_MAPS.update(...)` line in `worldmodel.py` is gone,
and `fetch` is one entry in `variation_maps.__init__._MODULES`, installed by A's `install()` with its
duplicate-key refusal.

## Correction 4 — the `background.color` partial

*"`VariationMap.pairs()` and `check_mapping` classify the floor-only mapping as covered... automatic
completeness claims cannot treat it as full support."*

**Ruled and implemented.** The coordinator's ruling (Sami) is that a feasible, semantically required
leaf must **not** be declared unsupported merely because a renderer change is needed, and that the
skybox half is a renderer *capability* need aggregated with the other families' after Wave 1. So:

- `background.color` is declared in `VariationMap.partial` on all twelve ids, with the mechanism and
  the **measured residual (floor 9.2-9.4 % reproduced exactly, skybox 61.7-73.0 % not rendered)** in
  the reason;
- the runtime mapping is **unchanged** — `floor.color <- background.color` still applies, because the
  floor half is genuinely what swm writes;
- `classification()` reports the leaf as `partial`, so `check_mapping`, the Gate-F matrix and every
  generated support table inherit it and **none of them can count it as full support**;
- `leaf_classification.json` says `partial` (totals now 33 mapped / **12 partial** / 12 unsupported /
  15 dead upstream / 51 not applicable, 123 leaves, 0 undocumented), the docs table says `partial`,
  and tests assert all of it;
- **no renderer file was touched and the skybox is not implemented.**

The class extension this workstream needed was added centrally by the coordinator after A merged,
which is what the brief asked to be written down here; for the record the one line is
`partial: Mapping[str, str] = field(default_factory=dict)`, `swm_path -> what is NOT reproduced`, with
`__post_init__` refusing a `partial` leaf that no mapping field reaches.

A second correctness fix travelled with it: this module had put all seven unmapped leaves into the
undifferentiated `swm_only` bucket with the taxonomy shouted in the reason string. They now use A's
three typed buckets, so `classification()` makes the distinction without anybody parsing prose, and
the committed artifact is asserted to agree with the maps leaf by leaf
(`test_the_committed_classification_artifact_agrees_with_the_maps`) — nothing regenerates that file
on import, so it could otherwise drift silently.

## Correction 5 — terminology for the default-equality tests

*"The measured equality is stable-envs `default_style()` versus `style_from_variation(env, {})`, and
an inverse mapping round-trip. It is not equality to the registered SWM wrapper's default reset
appearance."*

Accepted verbatim and stated in all three places — the test module docstring, the docs page and this
document. What is measured, precisely:

1. `env.default_style()` == `style_from_variation(env, {})`, byte for byte, and the frames rendered
   from the two are 0/255 apart;
2. the inverse round trip `style -> variation_from_style -> style_from_variation -> style` is the
   identity, including the exact 1.25 light factor.

Both are statements about **the mapping**. Neither is SWM default parity, and neither may be reported
as such.

### Gate-F status: SWM registered-default appearance — `MISMATCH (upstream)`

| what | measured |
|---|---|
| stable-envs default vs empty swm config | **identical**, 0/255, all four scenes |
| stable-envs default vs the SWM wrapper's reset appearance | **NOT EQUAL**, and not this workstream's defect |
| cause 1 — material unbind | `_apply_visual_variations` sets `geom_matid = -1` on the table, floor and object geoms the first time any visual leaf is active; `DEFAULT_VARIATIONS` makes that true on every plain `reset()`. Those geoms then render at the compiler default 0.5 grey instead of 0.93 / 0.2 / 0.2. **19.0-32.3 % of the frame, up to 102/255.** |
| cause 2 — declared leaf defaults | swm's declared defaults are not the official appearance: table 0.3 vs 0.93, object (0.8,0.1,0.1) vs 0.2 grey, background 0.1 vs floor 0.2 + skybox, light 0.7 vs 0.8, `agent.start_position` (1.3418,0.7491) vs the env's own initial gripper xy (up to 55/255 on Reach). |
| consequence | a parity number computed *through* the swm Fetch wrapper measures the wrapper, not stable-envs. Gate F must report this row as an upstream mismatch rather than counting the mapping tests as SWM default parity. |

Both causes are upstream findings 1 and 2 below, unchanged; this restates them as an explicit
Gate-F status line so the matrix cannot silently inherit the wrong claim.

## Merge gate (plan section 18), restated against this tip

| # | item | status | evidence |
|---|---|---|---|
| 1 | Scope | **PASS** | twelve `swm/Fetch*-v3` ids; one declarative family map, one package entry, tests, docs, artifacts. No renderer change, no performance work, no re-freeze. |
| 2 | Before/after diagnosis | **PASS** | before: no maps, `check_mapping` raised for all twelve. After: 123 leaves classified, 0 undocumented, `check_mapping` returns a full table. The two open issues (skybox residual, SWM default mismatch) are labelled `partial` and Gate-F `MISMATCH (upstream)` rather than described in prose. |
| 3 | Targeted unit tests | **PASS** | `tests/unit/test_variation_maps_fetch.py`: 10 passed / 49 skipped on `se6-venv` and on `se8-fetch-venv`, 10/49 on Oscar in job 6443480. `STABLE_ENVS_REQUIRE_LIVE_SWM=1` makes an all-skipped tier fail, so a certification run cannot pass by skipping. |
| 4 | Affected Gate A / Gate F | **PASS** | `check_mapping` now works for Fetch (correction 3) and is regression-tested through the public function. Gate-A oracle: the frozen Step-4 digests replay identically (item 5). Gate F carries an explicit `MISMATCH (upstream)` row for the SWM reset appearance and `partial` for `background.color`. |
| 5 | Affected frozen-corpus replay | **PASS** | `frozen_corpus_replay.json`, job 6443480, `IDENTICAL_AND_ORACLE_REPLAYS` on all four scenes: branch == baseline byte for byte, and both == the frozen Step-4 `digest_ours`. Corpus sha256s match the Step-4 record. |
| 6 | Pixel movement | **PASS — `IDENTICAL`** | no renderer file touched; `git ls-tree` on `stable_envs/render`, `envs`, `testing` identical to the baseline. Measured anyway on frozen corpora (480 px, 64 states) and on sampled states (128 px, 8 states): 0/255 on all four scenes. The new non-default mappings intentionally move pixels and are measured per leaf in `gpu_evidence.json`. |
| 7 | Performance regression | **NOT APPLICABLE** | no hot-path code changed; the module is imported by the compat layer and by nothing on the render path. |
| 8 | Independent review | **OPEN** | the previous verdict was `BLOCK`; this is the resubmission. Only an explicit `APPROVE` may merge. |
| 9 | Semantic follow-up | **NONE REQUIRED** | the default render path is unchanged and proved so by the frozen-corpus replay. No prior downstream claim is extended, and in particular none is extended to SWM background or SWM default behaviour, which are recorded as upstream mismatches. |
| 10 | Dashboard | **COORDINATOR** | to record: this resubmission, the `background.color` ruling and its `partial` classification, the skybox capability carried into the post-Wave-1 aggregation, the five upstream findings, and the Gate-F `MISMATCH (upstream)` row for the SWM reset appearance. |

## Still open after this round

1. **The skybox capability itself** — feasible, not implemented here, aggregated with the other
   families' renderer-capability needs after Wave 1. `background.color` stays `partial` until then.
2. **`rendering.transparent_arm`** — `DOCUMENTED_LIMITATION`; the fix is an alpha path in
   `mujoco_warp`, a first-party DeepMind dependency, not a stable-envs change.
3. **Tier 2 is not exercised on any interpreter available here.** Upstream finding 5 blocks live
   Fetch construction on `se8-fetch-venv`, and Oscar's only pytest interpreter has no
   `gymnasium_robotics`. A venv with `gymnasium_robotics` + mujoco 3.11 + pytest on Oscar would let
   `STABLE_ENVS_REQUIRE_LIVE_SWM=1` actually run tiers 2 and 3 green; until then the strictness
   switch documents the gap rather than closing it.
4. **`test_every_cited_scratch_path_resolves`** fails on the merged tree because other workstreams'
   `scratch/` directories are not in this worktree. Pre-existing and coordinator-level.
5. **The five upstream findings** are reported to `stable_worldmodel`, not fixed here.
