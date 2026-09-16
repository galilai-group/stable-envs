# Step 8 — Workstream B: VariationMaps for the twelve Fetch ids

**Verdict: `PATCH_CANDIDATE`** — with one escalation candidate (`background.color` / skybox) and five
upstream findings, none of which is a stable-envs defect.

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
| `mapped` | 45 | 3 / 4 |
| `unsupported` | 12 | 1 / 1 |
| `dead_upstream` | 15 | 2 / 1 |
| `not_applicable` | 51 | 2 / 5 |

Mapped: `table.color` (identical), `background.color -> floor.color` (rename, with a measured
residual), `light.intensity -> light.diffuse` (scaled by exactly 1.25), `object.color` (identical,
object scenes only).

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

## Escalation candidate (plan section 4)

**`background.color`, all 12 ids.** swm's "background" is the `floor_mat` geoms **and** the skybox
texture. Measured split at 480 px: floor **9.2-9.4 %** of the frame, skybox a further
**61.7-73.0 %**. We render the floor half exactly and the skybox not at all, because `mujoco_warp`
uploads textures when a render context is built, so a per-value skybox is one render context per
value — which `FetchBase.make_variation_space` already records as the reason for not offering a
skybox knob. The leaf is mapped (the floor half is genuinely what swm writes) with the residual
stated. An implementation is mechanically feasible and architecturally forbidden, and the skybox is
the single largest appearance term in the family, so the decision belongs to the coordinator/Sami
rather than to this workstream.

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

## Ownership requests

1. `stable_envs/compat/worldmodel.py::check_mapping` — needs to walk the wrapper chain rather than
   read `.unwrapped.variation_space`. One line in a shared file this workstream does not own;
   `variation_maps.fetch.swm_variation_space` is the drop-in body. Without it `check_mapping` raises
   for all twelve Fetch ids and the Gate-F matrix cannot report them.
2. `VariationMap` (Workstream A owns it) would be more truthful with a `partial` / `caveats` field.
   `background.color -> floor.color` is a real mapping with a measured 61.7-73.0 % residual, and the
   current dataclass can only say "mapped" or "not carried". Recorded in this module's docstring and
   in `leaf_classification.json` instead; not blocking.
3. `mkdocs.yml` — one additive nav entry for the new docs page, required by
   `tests/unit/test_repo_contracts.py::test_every_docs_page_is_reachable_from_the_nav`. Workstreams
   A and C will need the same; additive conflict.

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

- `tests/unit/test_variation_maps_fetch.py`, three tiers by what they need installed.
- `se6-venv` (no `stable_worldmodel`, no CUDA): **8 passed, 48 skipped**.
- `se8-fetch-venv` (live swm wrapper, no CUDA): **32 passed, 24 skipped** — tier 2 re-enumerates all
  twelve variation spaces and finds no undocumented leaf.
- Full `tests/unit` on this branch: **1043 passed, 57 skipped, 2 failed**, and both failures
  (`test_examples.py::...[robocasa_visual/render_mjwarp.py]` and
  `test_repo_contracts.py::test_every_cited_scratch_path_resolves`) reproduce identically on a clean
  checkout of the baseline SHA — pre-existing, untouched by this workstream.

## GPU evidence (SLURM job 6452376, the clean re-run; supersedes 6451948)

`gpu_evidence.json`, 128 px, 8 states per scene drawn from the env's own pool, on
**NVIDIA GeForce RTX 3090** / `gpu2264.oscar.ccv.brown.edu` / driver 575.57.08 / CUDA 12.6 /
torch 2.14.0+cu126 / **mujoco 3.11.0, mujoco_warp 3.11.0, warp 1.17.0** / python 3.12.9 —
`$SCR/venvs/se-s3b-fetch311`, the interpreter the frozen Fetch corpora were measured in.
Tested tree SHA `4b3d09fae90270c59d1273cfb7d5fc37e041ed75`, baseline
`43ac95238f7472137619f3e69e7636e11dd088b1`. Raw stdout of the job is committed beside it as
`gpu_job_6452376.out.txt`. **Verdict: PASS on all four scenes.**

- **Registered default byte-identical through the mapping.** `style_from_variation(env, {})` equals
  `default_style()` exactly, and the rendered frames are **0/255 over 0 % of the frame** on Reach,
  Push, Slide and PickAndPlace.
- **Round trip exact.** `variation_from_style(default)` gives `light.intensity = 0.800000011920929`
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
re-run and **its** artifacts are the committed ones. The unit tier inside that job reports
**8 passed, 48 skipped** — `$SCR/venvs/se6-verify` has no `gymnasium_robotics`, so the 48 live-wrapper
and GPU-fixture checks skip there. That, not a 32-pass run, is the unit-tier truth for job 6452376.

## Pixel movement

**None expected and measured as none.** This workstream adds a module and one registration line; the
render path imports neither. `pixel_parity.json` renders the four scenes' default frames from this
branch and from `$SCR/se8/baseline` on the same card in the same job and compares them byte for byte:
**0/255 over 0 % of the frame on all four scenes** (8 states each at 128 px, RTX 3090 gpu2264,
job 6452376, branch SHA `4b3d09f`). Verdict `IDENTICAL`. This is a random-state smoke check, not the
frozen-corpus replay; the replay is `frozen_corpus_replay.json` (review correction 2).

## Not done, deliberately

No Fetch performance work (plan section 2). No renderer change for `rendering.transparent_arm` or for
the skybox half of `background.color` — both are reported, not implemented. No widening of a
supported range. No re-freeze.
