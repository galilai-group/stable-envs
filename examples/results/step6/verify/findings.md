# Step 6 — integration verification, FINAL (`step6/integration-verify`)

**Classification: PASS-WITH-KNOWN-DELTAS**, and after `5402387` the correctness ledger is clean:
**48 of 49 configurations PASS Gate A and the 49th is `PARTIAL`, which is a passing state — no FAIL,
no BLOCKED — when each configuration is judged under the reference pair its own declaration names.**

Certification of `step6/integration` as a whole. This workstream changed no renderer code; it ran the
existing machinery — `examples/fidelity_step4.py`, `scratch/step4/certify.py`, `tests/gpu`,
`tests/unit` — against the frozen Step-4 corpus and against the Step-5 baseline tree.

| | |
|---|---|
| baseline SHA | `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`) |
| **integration SHA certified here** | **`7e0862c573cf0e9d4b269f1bc2f94bd549cb03c6`** (adds `5402387`, the downsample protocol) |
| previous full pass | `83bd778` — superseded for Gate A only; every pixel result below still stands |
| carried on this branch as | `5dd5596` (worktree `$SCR/se6/verify5`) |
| earlier SHAs in this report | `e4d015d` (pre-gfxdraw), `f7b07d7` (+gfxdraw), `e0182f6` (+SWMNAV-1), `43e6bcc` (+RAYCAST-4/5, +VISIBILITY-1) |
| my branch | `step6/integration-verify` — `step6/integration` itself was never committed to |
| headline | 48 of 49 frozen configs replay **bit-identical** on cuda; the 49th moves 0.00024/255. Gate A **43/5/1 -> 44/3/1/1**. Every CPU configuration that the baseline could draw is **byte-identical**; three families that the baseline could not draw on CPU at all now draw. |

## 0. Which SHA measured what, and why that is defensible

Integration moved five times while this ran, so each part names its SHA rather than implying one.

| part | SHA | why it is still current |
|---|---|---|
| `dmc`, `ogb`, `swm` shards on cuda + Gate A | `43e6bcc` (job 6358407) | the only library diff `43e6bcc..83bd778` is the **new** `stable_envs/render/cpu3d.py` and three **added** methods on `DMControlTrackCom` (`_cpu_scene`, `_cpu_poses`, `_render_cpu`) plus their dispatch — verified with `git diff --stat` over `stable_envs/ tests/ examples/`. `raycast3d.py`, `swm_nav.py`, `raster2d.py`, `gfxdraw*.py` are byte-identical across the two. |
| `raster`, `fetch` shards on cuda | **`83bd778`** (job 6359361) | re-run at the final SHA rather than inferred |
| the 4 trackcom cells, cuda and CPU | **`83bd778`** (jobs 6359331, 6358995, 6359181) | this is what cpu-3d changed |
| CPU identity, 25 configs | `43e6bcc` (job 6358408) | same diff argument |
| CPU identity, 8 more configs | `e0182f6` (job 6353226) | gfxdraw/tilemap/scene paths, untouched since |
| full `tests/gpu` (9 files) | `f7b07d7` (6354203) + `e0182f6` (6355272) | returned **identical counts and identical failure names** on both, so the pass is reproducible rather than a single sample |
| `test_reproducibility.py` in full | `43e6bcc` (job 6358407) | the VISIBILITY-1 fix is in it |
| `tests/unit` | all four SHAs | 584 -> 611 -> 662 -> 698 -> 727 passed as workstreams added tests; always the same 2 pre-existing failures |

## 1. Interpreters — chosen to reproduce the frozen shards, not for convenience

Each frozen shard records the packages it was measured with. A digest reproduced in a *different*
dependency set answers a different question, so every replay ran in its shard's own interpreter:
`se-step3` for `dmc`/`ogb`/`raster` (dm_control 1.0.45, gym_pusht 0.1.6, ogbench 1.2.1, mujoco and
mujoco_warp 3.12.0 — an exact field-for-field match), `se-step4-swm` for `swm` (craftax +
stable_worldmodel, no mujoco at all), `se-s3b-fetch311` for `fetch` (gymnasium_robotics 1.4.2,
mujoco/mujoco_warp 3.11.0).

**That choice is what closed MJWARP-1** (section 5), so it is not bookkeeping.

Three venvs were created for this workstream. **None installs anything**: each is a
`python -m venv --without-pip` whose single `.pth` chains existing site-packages, and no existing venv
was modified.

| venv | chain | why it had to exist |
|---|---|---|
| `se6-verify` | `se-step6` + `se-step3` + `se-step5` + the SWM checkout | the only interpreter that can run the WHOLE `tests/gpu` suite in one process: `se-step6` has pytest and no `mujoco_warp` (so every MJWarp cell would have skipped and the suite would have gone green on a family it never constructed); `se-step5` has `mujoco_warp` and no pytest |
| `se6-verify-fetch311` | `se-s3b-fetch311` + `se-step6` | Fetch is 12 of the 49 and its conformance tests errored on a missing `gymnasium_robotics` everywhere else |
| `se6-verify-tex311` | `se-tex311` + `se-step6` | the declared 3.11 reference pair for the four visual-tier cells. `se-tex311` alone has no `gymnasium`, so `import stable_envs` died before a frame was drawn |

## 2. The frozen Step-4 replay — all 49 render-distinct configurations, cuda

**48 of 49 `digest_ours` bit-identical. 49 of 49 `digest_official` bit-identical. No configuration
lost a check; every one gained `check_task_object_visibility`. Nothing unexplained.**

| config | shard | status | digest_ours | official | style | declaration | checks | Gate A (frozen->replay) | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Acrobot | raster | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| AntmazeGiantNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| AntmazeLargeExplore | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| AntmazeMediumExplore | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| AntmazeTeleportExplore | ogb | measured | **CHANGED** digest_ours,digest_ours_default_style | identical | - | - | 21/21 identical +1 new | FAIL->PASS | expected-known |
| CartPole | raster | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| CraftaxClassic | swm | measured | identical | identical | - | - | 21/21 identical +1 new | PARTIAL->PARTIAL | identical |
| DMCAcrobotSwingup | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| DMCBallInCupCatch | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| DMCCartpoleBalance | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| DMCCartpoleSwingup | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| DMCCheetahRun | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | FAIL->FAIL | identical |
| DMCFingerSpin | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| DMCFingerTurnEasy | dmc | measured | identical | identical | style_digest,style_dim | changed | 21/21 identical +1 new | PASS->PASS | expected-known |
| DMCFingerTurnHard | dmc | measured | identical | identical | style_digest,style_dim | changed | 21/21 identical +1 new | PASS->PASS | expected-known |
| DMCHopperHop | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | FAIL->FAIL | identical |
| DMCHumanoidStand | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->BLOCKED | gate A improved |
| DMCPendulumSwingup | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| DMCReacherEasy | dmc | measured | identical | identical | style_digest,style_dim | changed | 21/21 identical +1 new | PASS->PASS | expected-known |
| DMCReacherHard | dmc | measured | identical | identical | style_digest,style_dim | changed | 21/21 identical +1 new | PASS->PASS | expected-known |
| DMCWalkerWalk | dmc | measured | identical | identical | - | - | 21/21 identical +1 new | FAIL->FAIL | identical |
| FetchPickAndPlace | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchPickAndPlaceDense | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchPickAndPlaceDict | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchPush | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchPushDense | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchPushDict | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchReach | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchReachDense | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchReachDict | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchSlide | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchSlideDense | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| FetchSlideDict | fetch | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| HumanoidmazeGiantNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| HumanoidmazeLargeNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| HumanoidmazeMediumNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | FAIL->PASS | gate A improved |
| MountainCar | raster | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| MountainCarContinuous | raster | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| Pendulum | raster | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| Piecewise | swm | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| PiecewiseZones | swm | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| PointmazeGiantNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| PointmazeLargeNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| PointmazeMediumNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| PointmazeTeleportNavigate | ogb | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| PushT | raster | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| SimplePointMaze | swm | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| TwoRoom | swm | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |
| TwoRoomTarget | swm | measured | identical | identical | - | - | 21/21 identical +1 new | PASS->PASS | identical |

### The six rows that are not plain `identical`

| config | what moved | numbers | classification |
|---|---|---|---|
| `DMCReacherEasy`, `DMCReacherHard`, `DMCFingerTurnEasy`, `DMCFingerTurnHard` | the DECLARATION only | `style_dim` 17 -> 18, `style_digest` moves; `digest_ours` and `digest_ours_default_style` byte-identical | **expected-known** (reacher-correctness's visibility leaf) |
| `AntmazeTeleportExplore` | `digest_ours` `06a2e039e1fd…` -> `483968227d62…` | mean pixel **187.26056 -> 187.26032**, i.e. **0.00024/255**; `digest_official` unchanged; Gate A FAIL -> PASS | **expected-known** (documented bake nondeterminism). The magnitude is the argument: a renderer change does not land three ten-thousandths of a grey level away. |
| `DMCHumanoidStand` | Gate A `PASS` -> `BLOCKED` | `check_visual_fidelity` blocked on the reference-version mismatch instead of judged against a drifted reference | **expected-known** (texture-fidelity's provenance gate) |

### Gate A

| | frozen Step 4 | final integration |
|---|---|---|
| PASS | 43 | **44** |
| PARTIAL | 1 (`CraftaxClassic`) | 1 (`CraftaxClassic`) |
| BLOCKED | 0 | 1 (`DMCHumanoidStand`) |
| FAIL | **5** | **3** |

Every move has a mechanism:

- `AntmazeTeleportExplore`, `HumanoidmazeMediumNavigate` **FAIL -> PASS**: `check_style_is_state`
  reported `max_abs_diff: 1` frozen and reports **0** now. raycast3d's determinism fix, measured on
  the check that was failing.
- `DMCHumanoidStand` **PASS -> BLOCKED**: its only open item was `check_visual_fidelity`.
- `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk` stay **FAIL**, but on ONE item instead of two:
  `check_visual_fidelity` moved out of `failures` into `blocked`, leaving
  `check_resolution_consistency`. That is texture-fidelity's separate finding — a bound measured on
  an H200 at mujoco 3.9 over 8 states, applied to a 64-state corpus — and its fix (`5402387`, the
  downsample protocol) is **held for Sami**. These three cannot clear until that is decided, and
  **they are not a Step-6 regression**: section 5 shows them failing identically on the baseline tree.

The Gate A verdicts were computed by `scratch/step4/certify.py` on Oscar and recomputed locally over
the merged five shards; the two agree on all 31 shared configurations, so the number is not an
artifact of where the certifier ran.

## 3. CPU identity — one batch node, one torch thread

`--partition=batch --qos=normal`, `OMP_NUM_THREADS=1`. The frozen corpus is cuda-only, so
"reproduces the frozen digest" is not available on CPU. What is available is baseline-vs-integration
on one node in one job, as a **pixel diff** rather than a hash comparison, so a digest that moved
would arrive with a magnitude. The last column is reported as information and is not a bar.

**33 of 33 comparable configurations byte-identical (`max_abs_diff` 0 over every pixel of every
frame). Six more are not comparable because the baseline cannot render them on CPU at all.**

| config | owner | baseline CPU | integration CPU | baseline vs integration | CPU == frozen CUDA digest | job |
|---|---|---|---|---|---|---|
| PushT | raster2d | **no CPU path** (ValueError: cuda device) | renders | n/a — baseline cannot render | False | 6353226 |
| SimplePointMaze | raster2d | **no CPU path** (ValueError: cuda device) | renders | n/a — baseline cannot render | True | 6353226 |
| TwoRoom | raster2d/swm_nav | renders | renders | **byte-identical** | True | 6358408 |
| TwoRoomTarget | raster2d/swm_nav | renders | renders | **byte-identical** | True | 6358408 |
| Piecewise | raster2d/swm_nav | renders | renders | **byte-identical** | True | 6358408 |
| PiecewiseZones | raster2d/swm_nav | renders | renders | **byte-identical** | True | 6358408 |
| CraftaxClassic | tilemap | renders | renders | **byte-identical** | True | 6353226 |
| Acrobot | gfxdraw | renders | renders | **byte-identical** | True | 6353226 |
| CartPole | gfxdraw | renders | renders | **byte-identical** | True | 6353226 |
| MountainCar | gfxdraw | renders | renders | **byte-identical** | True | 6353226 |
| MountainCarContinuous | gfxdraw | renders | renders | **byte-identical** | True | 6353226 |
| Pendulum | gfxdraw | renders | renders | **byte-identical** | True | 6353226 |
| AntmazeGiantNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| AntmazeLargeExplore | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| AntmazeMediumExplore | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| AntmazeTeleportExplore | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCAcrobotSwingup | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCBallInCupCatch | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCCartpoleBalance | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCCartpoleSwingup | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCFingerSpin | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCFingerTurnEasy | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCFingerTurnHard | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCPendulumSwingup | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCReacherEasy | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCReacherHard | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| HumanoidmazeGiantNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| HumanoidmazeLargeNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| HumanoidmazeMediumNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| PointmazeGiantNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| PointmazeLargeNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| PointmazeMediumNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| PointmazeTeleportNavigate | raycast3d | renders | renders | **byte-identical** | False | 6358408 |
| DMCCheetahRun | cpu-3d | **no CPU path** (ModuleNotFoundError: 'warp') | renders | n/a — baseline cannot render | False | 6358995 |
| DMCHopperHop | cpu-3d | **no CPU path** (ModuleNotFoundError: 'warp') | renders | n/a — baseline cannot render | False | 6358995 |
| DMCHumanoidStand | cpu-3d | **no CPU path** (ModuleNotFoundError: 'warp') | renders | n/a — baseline cannot render | False | 6358995 |
| DMCWalkerWalk | cpu-3d | **no CPU path** (ModuleNotFoundError: 'warp') | renders | n/a — baseline cannot render | False | 6358995 |

The six incomparable rows are the point of two workstreams, not a gap:

- `PushT`, `SimplePointMaze` — the baseline raises `ValueError: Expected a cuda device, but got: cpu`
  (the Triton-dispatch defect, finding 1 of `examples/results/step6/findings.md`). raster2d fixed it.
- the four trackcom cells — the baseline raises `ModuleNotFoundError: No module named 'warp'`, four
  times. cpu-3d gave them a CPU backend where there was none.

**The cpu-3d result is a demonstration, not a claim.** Those four cells were rendered in
`$SCR/venvs/se-step6`, in which `import mujoco_warp` fails — so the frames can only have come from
`stable_envs/render/cpu3d.py`. The job prints that check before it renders.

### The four trackcom cells on CPU, under both reference arms

| arm | mujoco / dm_control | Cheetah | Hopper | Humanoid | Walker |
|---|---|---|---|---|---|
| `se-step6` (3.12) | 3.12.0 / 1.0.45 | Gate A **BLOCKED** | **BLOCKED** | **BLOCKED** | **FAIL** (`check_resolution_consistency` only) |
| `se6-verify-tex311` (**the declared pair**) | 3.11.0 / 1.0.44 | Gate A **PASS** | **PASS** | **PASS** | **FAIL** (`check_resolution_consistency` only) |

Under 3.12 the visual checks are blocked by design. Under the pair these cells' `visual_fidelity`
was **declared** against, `check_visual_fidelity` is MEASURED and passes: mean |diff| 5.12 / 4.52 /
5.95 / 8.03 per 255, SSIM 0.662 / 0.687 / 0.881 / 0.825, all 14 mandatory checks measured on all
four. That is the same verdict texture-fidelity recorded for the MJWarp path — **three PASS and
Walker failing on the corpus-downsample bound alone** — reached through an entirely different
renderer on a different device. Job 6359181, node1843.

## 4. `tests/gpu` and `tests/unit`

### The nine GPU files

Unfiltered, one file at a time so a per-file count exists, with **collection asserted before
execution** — `test_fast_paths.py` carries raycast3d's and gfxdraw's blocks merged by hand, and a
file that fails to collect reports zero failures.

| file | collected | passed | skipped | failed | errors |
|---|---|---|---|---|---|
| `test_conformance.py` | 1676 | 845 | 387 | 7 | 437 |
| `test_reproducibility.py` (at `f7b07d7`/`e0182f6`) | 2039 | 1275 | 218 | 17 | 529 |
| **`test_reproducibility.py` (at `43e6bcc`, after the VISIBILITY-1 fix)** | 2040 | **1276** | 225 | **10** | 529 |
| `test_fast_paths.py` | 348 | **168** | 180 | **0** | 0 |
| `test_gfxdraw_classic.py` | 80 | **80** | 0 | **0** | 0 |
| `test_gfxdraw_fused.py` (in `se6-gfxdraw-box2d`) | 4 | **4** | 0 | **0** | 0 |
| `test_rendernet.py` | 10 | 0 | 9 | 1 | 0 |
| `test_crash_isolation.py` | 5 | **2** | 3 | **0** | 0 |
| `test_rollout_on_device.py` | 87 | **57** | 30 | **0** | 0 |
| `test_distributed.py` | 6 | 0 | 6 | 0 | 0 |
| `test_conformance.py -k Fetch` (in `se6-verify-fetch311`) | 1300 | **156** | 72 | **0** | 19 |
| `test_reproducibility.py -k Fetch` (same) | 1661 | **216** | 60 | **0** | 23 |

`test_distributed.py` is **not run**: `SKIPPED [3] needs 2 GPUs` twice. This job holds one GPU, as
AGENT_COMMON requires. Recorded in `gpu/distributed_reasons_gfx.results.txt`.

### The 966 errors are one thing

Every error in both files is `ModuleNotFoundError` at fixture setup: `gymnasium_robotics` for the 12
`Fetch*` and 4 `PointMaze*` cells, `minigrid` for the 6 `MiniGrid*` and `BabyAIGoToLocal`. 23 envs x
19 or 23 tests = 437 and 529. **`minigrid` is installed in no venv on Oscar at all**, so those 7
registered envs have never had their GPU conformance run on this cluster (they are outside the 49).
Fetch IS 12 of the 49, so it was re-run in `se6-verify-fetch311`, where **all 372 Fetch tests pass**.
Classification: **environment**.

### VISIBILITY-1 — found here, fixed, and confirmed fixed

The one genuine defect this workstream found on the integration branch.
`test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility` was **7 failed / 4 passed** over the
DM Control family: `_visibility_envs()` selected on the class flag `VISIBILITY_KNOB`
(`dmcontrol.py:889`, true for all eleven non-trackcom cells) while the leaf it then asserts on is
derived per cell by `_hideable()`, which is empty for the seven cells whose episode draws no task
object — exactly as `_hideable`'s own docstring says it should be. The assertion that fired was the
selector's, not a pixel comparison. No pixel was ever affected: those seven cells' digests, Gate A
and CPU frames were verified identical throughout.

It survived its own workstream because the reacher review ran the test on the four cells it was
written for and recorded "4/4 GPU". Only an unfiltered run over the family could see it.

**After the fix (job 6358407): 4 passed, 7 skipped, 0 failed.**

### The 10 remaining `test_reproducibility.py` failures, each measured on the BASELINE tree

| failure | cells | baseline | integration | classification |
|---|---|---|---|---|
| `test_resolution_downsample_agrees_within_the_measured_bound` | Cheetah, Hopper, Walker | 3 failed | 3 failed | **expected-known** — 52.223% vs a 53.385% bound measured on an H200 at mujoco 3.9 over 8 states; fix commit `5402387` held for Sami |
| `test_the_reference_renderer_on_this_machine_is_deterministic` | `PointmazeTeleport{Navigate,Stitch}` | 2 failed | 2 failed | **NODE-1** |
| `test_frames_are_bit_identical_in_a_fresh_process` | `AntmazeTeleport*` (3), `HumanoidmazeLarge*` (2) | see below | see below | **NODE-1, dependent** |

And from `test_conformance.py`, on both trees identically: `test_visual_tier_matches_its_declared_
fidelity` on Cheetah (6.541 vs 6.5), Hopper and Walker (6.456 vs 5.7) — **the MuJoCo 3.11 -> 3.12
floor-texture drift**, `test_state_is_sufficient` on the four Pointmaze Giant/Teleport cells —
**NODE-1**, and `test_rendernet.py::test_no_checkpoint_means_no_class` — **environment**, because
pytest 9.1.1's `importorskip` no longer swallows an `ImportError` raised inside a module that exists
(`stable_envs/envs/robocasa.py` exists; robocasa is not installed).

### NODE-1, stated correctly

The OFFICIAL OGBench renderer is nondeterministic at 1/255 for the Pointmaze Teleport and maze2d
cells. The repository already has machinery for this and it fires by itself:

```
_pytest.outcomes.Exit: NODE FAULT on gpu2113: the OFFICIAL PointmazeTeleportNavigate renderer
produced two different frames for the SAME states on gpu2113 (max 1/255), before any stable-envs
code was involved. ... Every parity number measurable here is meaningless
```

Observed on **gpu2108, gpu2110, gpu2113 and gpu2116** — four of the eleven `gpu`-partition 3090
nodes, always identically on the baseline tree. **It is not a bad host; it is a property of the
reference stack on this cluster**, and the exclusion list is deliberately not grown to chase it:
excluding every node that shows it would eventually exclude the partition.
`--exclude=gpu2108,gpu2113` stays in every GPU sbatch here for one reason only — those two
additionally broke `test_state_is_sufficient`.

The dependent `fresh_process` failures behave the same way: they fire on `gpu2108`, `gpu2113` and
`gpu2116` inside the full-suite process but PASS on `gpu2113` when those five tests are the only ones
selected (job 6354905) — the EGL-context pressure `test_state_is_sufficient`'s own docstring
documents. Our `PointmazeTeleportNavigate` frames replayed bit-identical to the frozen oracle on
`gpu2104`, which is what says the renderer is not the variable.

### `tests/unit`

| SHA | node | job | passed | skipped | failed |
|---|---|---|---|---|---|
| `e4d015d` | node1831 | 6352886 | 584 | 3 | 2 |
| `f7b07d7` | node2333 | 6353226 | 611 | 3 | 2 |
| `e0182f6` | batch | 6355274 | 662 | 3 | 2 |
| `43e6bcc` (`se-step6`) | batch | 6358408 | 698 | 18 | 2 |
| **`83bd778`** (`se-step6`) | node2329 | 6358995 | **727** | 19 | **2** |

The two failures are the two named in advance and no others:
`test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]` and
`test_every_cited_scratch_path_resolves`. The count rises as workstreams add tests; the failures do
not.

## 5. MJWARP-1, closed

The four MJWarp trackcom cells, on cuda, **both venv arms and both trees in one allocation on one
card** (job 6359331, gpu2106):

| arm | baseline vs integration | vs the frozen Step-4 digests |
|---|---|---|
| `se-step5` (mujoco_warp 3.12) | **byte-identical**, `max_abs_diff` 0 over all 451,584 pixels x 4 cells | all four differ |
| `se-step3` (the venv the `dmc` shard was FROZEN in) | **byte-identical**, `max_abs_diff` 0 | **all four reproduce `digest_ours` AND `digest_official` exactly** |

So the earlier non-reproduction was the interpreter, not MJWarp nondeterminism and not any branch,
and cpu-3d moved no GPU pixel for these cells. Keeping both arms is why a real regression here could
never be mistaken for that again.

## 6. Verdict

**PASS-WITH-KNOWN-DELTAS.**

`step6/integration` at `83bd778`:

- draws the same pixels as the frozen Step-4 oracle on **48 of 49** configurations, and the 49th
  moves by 0.00024/255;
- reproduces **49 of 49** `digest_official`;
- loses **no check** on any configuration, and gains one on all of them;
- improves Gate A from **43 PASS / 5 FAIL / 1 PARTIAL** to **44 PASS / 3 FAIL / 1 PARTIAL / 1
  BLOCKED**;
- is **byte-identical to the baseline on CPU** on all 33 configurations the baseline could draw, and
  draws six more that it could not;
- has no failing test anywhere that does not also fail on the Step-5 baseline tree on the same card.

The deltas, all of them:

| delta | configs | classification |
|---|---|---|
| `style_dim` 17 -> 18, declaration only | 4 DMC cells | expected-known |
| `digest_ours` 0.00024/255 | `AntmazeTeleportExplore` | expected-known (bake nondeterminism) |
| Gate A PASS -> BLOCKED | `DMCHumanoidStand` | expected-known (reference-version gate) |
| Gate A FAIL on `check_resolution_consistency` | Cheetah, Hopper, Walker | expected-known; **fix held for Sami** (`5402387`) |
| `visual_tier` / `resolution_downsample` test failures | same three | expected-known; identical on the baseline tree |
| OGBench reference nondeterminism + dependent fresh-process | Pointmaze Teleport, maze2d, Antmaze Teleport, Humanoidmaze Large | **NODE-1 / environment**; identical on the baseline tree; seen on gpu2108, gpu2110, gpu2113, gpu2116 |
| 966 fixture errors | Fetch, MiniGrid, PointMaze, BabyAI | **environment**; Fetch re-run clean in its own interpreter, `minigrid` exists in no venv on the cluster |
| `test_no_checkpoint_means_no_class` | — | **environment** (pytest 9 `importorskip`) |
| 2 `tests/unit` failures | — | **pre-existing**, named in the brief |

**No regression was found that survives comparison with the baseline tree.** The one defect this
workstream did find, VISIBILITY-1, was a test selector, was fixed, and is confirmed fixed.

### Open items for Sami

1. **`5402387`, the downsample protocol.** Three Gate A FAILs (`DMCCheetahRun`, `DMCHopperHop`,
   `DMCWalkerWalk`) rest on `check_resolution_consistency` alone and cannot clear until this is
   decided. It is a change to which states a check is applied, not to any bound value.
2. **NODE-1.** The official OGBench renderer is nondeterministic at 1/255 on at least four of eleven
   3090 nodes, on the baseline tree as much as on any branch. Any Pointmaze parity number matters
   less than the fact that the suite currently aborts itself on those nodes.
3. **`minigrid` is in no venv on Oscar**, so 7 registered envs have never had GPU conformance run
   here. Outside the 49; worth a Step-8 ticket.
4. `DMCHumanoidStand` is BLOCKED rather than PASS. That is the provenance gate working as designed,
   but it means one covered configuration is currently unjudged on its visual tier under 3.12 — while
   **passing under the declared 3.11 pair on CPU** (section 3).

### SLURM jobs

| job | partition / node | SHA | what |
|---|---|---|---|
| 6352885 | gpu / gpu2104 | `e4d015d` | 49-config replay, all 5 shards, + Gate A |
| 6352886 | batch / node1831 | `e4d015d` | CPU identity 28 configs + `tests/unit` |
| 6353226 | batch / node2333 | `f7b07d7` | CPU identity 33 configs (+5 gfxdraw) + `tests/unit` |
| 6354203 | gpu / gpu2108 | `f7b07d7` | raster replay + full 9-file `tests/gpu` |
| 6354905 | gpu / gpu2113 | `f7b07d7` | Fetch under the 3.11 stack; the 7 failures baseline-vs-integration; `test_gfxdraw_fused` under Box2D; distributed reasons |
| 6355272 | gpu / gpu2113 | `e0182f6` | raster replay + 4 GPU files after SWMNAV-1 |
| 6355274 | batch | `e0182f6` | CPU identity + `tests/unit` after SWMNAV-1 |
| 6358407 | gpu / gpu2116 | `43e6bcc` | dmc+ogb+swm replay + Gate A + full `test_reproducibility.py` + VISIBILITY-1 |
| 6358408 | batch | `43e6bcc` | CPU identity, 21 raycast3d + 4 swm_nav, + `tests/unit` |
| 6358995 | batch / node2329 | **`83bd778`** | trackcom on CPU, `se-step6` arm + baseline probe + `tests/unit` |
| 6359181 | batch / node1843 | **`83bd778`** | trackcom on CPU, the declared 3.11 arm |
| 6359331 | gpu / gpu2106 | **`83bd778`** | trackcom on cuda, both venv arms x both trees + Gate A |
| 6359361 | gpu / gpu2106 | **`83bd778`** | raster + fetch shards + Gate A |

### Where things are

- report: `examples/results/step6/verify/findings.md`
- 49-row table: `examples/results/step6/verify/compare_final/compare49.{md,json}`
- merged Gate A: `examples/results/step6/verify/replay/final/certification.json`
- raw shards: `examples/results/step6/verify/replay/`, CPU: `.../cpu/`, test outcomes: `.../gpu/`
- scripts: `scratch/step6/verify/` — `replay.sbatch`, `gputests.sbatch`, `cpu_verify.sbatch`,
  `postmerge.sbatch`, `gaps.sbatch`, `final.sbatch`, `trackcom_cpu.sbatch`, `trackcom_gpu.sbatch`,
  `compare49.py`. All take `TREE=`, `SHARDS=`, `CONFIGS=`, `FILES=`, `ARMS=`, `CPUVENV=`, `TAG=`.



---

## 7. After `5402387` (the downsample protocol) — integration `7e0862c`

`check_resolution_consistency` now measures the judged number on the bound's own 8-state protocol
draw and carries the 64-state corpus number beside it as `frozen_corpus`. Two arms, because the four
visual-tier cells declare their numbers against mujoco 3.11 / dm_control 1.0.44 and the other 45
against 3.12.

### Gate A over all 49, judged under each configuration's DECLARED reference

| | frozen Step 4 | `83bd778` | **`7e0862c`, declared arm** |
|---|---|---|---|
| PASS | 43 | 44 | **48** |
| PARTIAL | 1 | 1 | 1 (`CraftaxClassic`, a passing state) |
| BLOCKED | 0 | 1 | **0** |
| FAIL | 5 | 3 | **0** |

`examples/results/step6/verify/replay/final_declared/certification.json`. The four visual cells are
carried in a shard of their own (`dmcvis.json`) so they travel with the 3.11 provenance they were
measured under: `certify.py` reads the reference version off the SHARD, so copying those rows into
the 3.12 document instead makes the certifier block the very numbers it was handed. It did, the first
time this was assembled, and the 48 became 44 — worth writing down, because that failure mode looks
exactly like a real BLOCKED.

The four cells PASS under the declared pair on BOTH devices, in two independently built
interpreters: cuda in `se6-verify-vis311` (mujoco 3.11.0, mujoco_warp 3.11.0, dm_control 1.0.44, job
6365966 on gpu2109) and CPU in `se6-verify-tex311` (same pair, no `mujoco_warp` at all, job 6365967
on node2320) — the CPU arm running through `cpu3d.py`. Walker's protocol `exact_frac` is **0.62879**
on CPU against a floor of 0.6290 − 0.005, with the corpus number 0.61854 recorded beside it as
`frozen_corpus`, which is exactly what `5402387` was for.

### A discrepancy the coordinator should see before this is handed to Sami

The merged `examples/results/step6/texture-fidelity/certify312_protocol/` record reports the four
cells as **BLOCKED with no `check_resolution_consistency` failure** under 3.12. My 3.12 arm reports
**three FAILs** on that check. Same node (`gpu2109`), same GPU and driver, same python/torch, same
mujoco 3.12.0 / mujoco_warp 3.12.0 / dm_control 1.0.45 / warp 1.17.0, and `git diff stable_envs/`
between the two SHAs is **empty**. The difference is the interpreter, and the digests say which:

| cell | frozen `digest_ours` | texture-fidelity's 3.12 arm | my 3.12 arm (`se-step3`) | protocol `exact_frac`: theirs / mine |
|---|---|---|---|---|
| `DMCCheetahRun` | `1fc511eaec27` | `d314a7eab012` | `1fc511eaec27` | 0.53385 / **0.52223** |
| `DMCHopperHop` | `d65978480077` | `80eb493872a6` | `d65978480077` | — / 0.53655 |
| `DMCHumanoidStand` | `539c71a22769` | `c32a08293794` | `539c71a22769` | — / 0.60390 |
| `DMCWalkerWalk` | `13b88aad7d82` | `1801687041bc` | `13b88aad7d82` | 0.62895 / **0.61258** |

`d314a7ea / 80eb4938 / c32a0829 / 1801687` are precisely the digests `se-step5` produced in job
6359331, and `se-step5` is the venv MJWARP-1 identified as the one that does NOT reproduce the frozen
corpus. So the merged 3.12 record was taken in the interpreter whose frames differ from the frozen
Step-4 oracle, and my arm was taken in the one that reproduces it byte for byte.

`exact_frac` is a knife-edge statistic — the two interpreters differ by ~0.012 on it, against a
margin of 0.005 — so **the 3.12 downsample verdict flips with the venv**: PASS in `se-step5` (0.53385,
the recorded floor to five decimals), FAIL in `se-step3` (0.52223). Neither is wrong; they are two
different renders.

**This does not change the verdict**, because the contract judges these cells under 3.11, where both
of my arms and texture-fidelity's own `certify311_protocol` agree on PASS ×4. But "no FAIL under
3.12" cannot be asserted, and the merged `certify312_protocol` record should say which interpreter it
was taken in — otherwise the page claims a clean 3.12 arm that the frozen-corpus interpreter does not
reproduce.

### Everything else at `7e0862c`

- `dmc` shard on cuda in `se-step3`: **14 of 14 `digest_ours` identical** to the frozen oracle,
  trackcom cells included. No pixel moved.
- `tests/unit` (job 6365967, `se-step6`): **732 passed, 21 skipped, 2 failed** — the same two
  pre-existing failures.
- Gate A under the 3.12 arm alone, for the record: 44 PASS / 1 PARTIAL / 3 FAIL / 1 BLOCKED
  (`replay/final_step312/certification.json`), with the three FAILs subject to the venv dependence
  above.

### Jobs

| job | partition / node | what |
|---|---|---|
| **6365966** | gpu / gpu2109 | `dmc` shard + Gate A under 3.12 (`se-step3`); the four visual cells + Gate A under the declared 3.11 pair (`se6-verify-vis311`) |
| **6365967** | batch / node2320 | the four visual cells on CPU under the declared 3.11 pair (`se6-verify-tex311`) + `tests/unit` |


---

## 8. SWMNAV-2/2b, the fused swm_nav GPU kernel — integration `83aa187`

New file `stable_envs/render/navdot.py` plus `swm_nav.py`. Jobs **6367990** (gpu / gpu2264, RTX 3090)
and **6367991** (batch).

| what | result |
|---|---|
| `swm` shard frozen replay on cuda | **6 of 6 configs measured, nothing moved** — `digest_ours`, `digest_ours_default_style`, `digest_official`, the corpus-style digest and every check status identical to the frozen oracle, under the fused kernel |
| `tests/gpu/test_navdot_fused.py` | 24 collected, **24 passed, 0 failed** |
| `tests/gpu/test_fast_paths.py` | 397 collected, **213 passed, 184 skipped, 0 failed** (up from 348/168 — the kernel adds its own fast-path cases) |
| CPU at 1 thread (`se-step6`) | **25 of 25 configs byte-identical** to the baseline tree; the four swm_nav configs' CPU digests unchanged AND still equal to their frozen CUDA digests |
| `tests/unit` | **732 passed, 22 skipped, 2 failed** — the same two pre-existing failures |

The CPU leg covered 25 configs rather than the four asked for, because `RAY=` on the sbatch line did
not clear the default: `${RAY:-...}` treats an explicitly empty value as unset. Harmless here — it
bought 21 extra byte-identical raycast3d rows at this SHA for free — but a subset flag that quietly
ignores you is not one to certify with, so `cpu_verify.sbatch` now uses `${RAY-...}`.

**SWMNAV-2/2b changed no pixel on either device.**
