# Step 6 — integration verification (`step6/integration-verify`)

**Classification: PASS-WITH-KNOWN-DELTAS.** One genuine defect found on the integration branch
(a test selector, section 3); no renderer pixel moved anywhere that was not intended.

| | |
|---|---|
| what this is | certification of `step6/integration` **as a whole**, not of any one workstream. No renderer code was changed by this workstream. |
| baseline SHA | `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`) |
| integration SHA verified, pass 1 | `e4d015d31f686395113ee5234081ae86883b19d5` — texture-fidelity, reacher-correctness, raycast3d, tilemap, raster2d-through-`b0c94ad` |
| integration SHA verified, pass 2 | `f7b07d7` (merge `62558d5`, gfxdraw), carried on this branch as `736e642` |
| branch | `step6/integration-verify` (never committed to `step6/integration`) |
| Oscar copies | `$SCR/se6/verify` = `e4d015d`, `$SCR/se6/verify2` = `736e642`. Both carry `GIT_COMMIT`; every artifact records the SHA through `STABLE_ENVS_GIT_COMMIT`, because rsync drops `.git`. |

## How the interpreters were chosen, and why it matters

Each frozen Step-4 shard was measured in a different interpreter, and
`examples/results/step4/conformance/<shard>.json` records exactly which packages were present. A
digest reproduced in a *different* dependency set answers a different question, so every replay here
runs in the interpreter its shard was frozen in:

| shard | frozen provenance | interpreter used here | match |
|---|---|---|---|
| `dmc`, `ogb`, `raster` | dm_control 1.0.45, gym_pusht 0.1.6, ogbench 1.2.1, mujoco 3.12.0, mujoco_warp 3.12.0, no SWM, no craftax (jobs 6176998 / 6178140) | `$SCR/venvs/se-step3` | exact, field for field |
| `swm` | craftax present, stable_worldmodel present, **no mujoco at all** (job 6180921) | `$SCR/venvs/se-step4-swm` | exact |
| `fetch` | gymnasium_robotics 1.4.2, mujoco 3.11.0, mujoco_warp 3.11.0 (job 6176999) | `$SCR/venvs/se-s3b-fetch311` | exact |

For the **pytest** suites no existing interpreter can run the whole of `tests/gpu` in one process:
`se-step6` has pytest and no `mujoco_warp` (so every MJWarp cell would have skipped as
`No module named 'warp'` and the suite would have reported green on a family it never constructed),
and `se-step5` has `mujoco_warp` and no pytest. So `$SCR/venvs/se6-verify` was created. **It installs
nothing**: it is a `python -m venv --without-pip` whose single `.pth` chains the existing
site-packages of `se-step6`, then `se-step3`, then `se-step5`, then the `stable_worldmodel` checkout.
No existing venv was modified. `craftax` is still absent, exactly as it was for every previous
workstream's GPU run, so `CraftaxClassic`'s official-reference checks stay `blocked_environment`.

## 1. The frozen Step-4 replay — all 49 render-distinct configurations, on cuda

`examples/fidelity_step4.py`, all five shards, RTX 3090 `gpu2104`, driver 575.57.08 — the same card
model and the same driver the frozen corpus was taken on. Job **6352885**, tree `e4d015d`. Raw shards
in `replay/`, the field-by-field diff in `compare/compare49.json`, Gate A in
`replay/certify/certification.json`.

**48 of 49 `digest_ours` bit-identical to the frozen oracle. 49 of 49 `digest_official` bit-identical.
No configuration lost a check. Nothing is unexplained.**

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

### The five rows that are not plain `identical`, with numbers

| config | what moved | classification |
|---|---|---|
| `DMCReacherEasy`, `DMCReacherHard`, `DMCFingerTurnEasy`, `DMCFingerTurnHard` | `style_dim` 17 -> 18 and `style_digest`, i.e. the DECLARATION. `digest_ours` and `digest_ours_default_style` are byte-identical, so the default picture did not move. | **expected-known** — reacher-correctness's visibility leaf |
| `AntmazeTeleportExplore` | `digest_ours` `06a2e039e1fd…` -> `483968227d62…`; mean pixel **187.26056 -> 187.26032**, a difference of **0.00024/255**. `digest_official` unchanged. | **expected-known** — the documented bake nondeterminism. The magnitude is the evidence: a real renderer change does not land three ten-thousandths of a grey level away. |
| `DMCHumanoidStand` | Gate A `PASS` -> `BLOCKED`, because `check_visual_fidelity` is now blocked on the reference-version mismatch instead of judged against a drifted reference. | **expected-known** — texture-fidelity's provenance gate |

### Gate A, frozen vs this branch

| | frozen Step 4 | `step6/integration` @ `e4d015d` |
|---|---|---|
| PASS | 43 | **44** |
| PARTIAL | 1 (`CraftaxClassic`) | 1 (`CraftaxClassic`) |
| BLOCKED | 0 | 1 (`DMCHumanoidStand`) |
| FAIL | 5 | **3** |

Every move is in the right direction and each one has a mechanism:

- `AntmazeTeleportExplore` and `HumanoidmazeMediumNavigate` **FAIL -> PASS**: `check_style_is_state`
  reported `max_abs_diff: 1` in the frozen run and reports **0** here. That is raycast3d's
  determinism fix landing, measured on the check that was failing.
- `DMCHumanoidStand` **PASS -> BLOCKED**: its only outstanding item was `check_visual_fidelity`,
  which is now blocked rather than judged.
- `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk` stay **FAIL**, but for ONE reason instead of two:
  `check_visual_fidelity` moved from `failures` into `blocked`, and what is left is
  `check_resolution_consistency`. That is texture-fidelity's separate, smaller finding — the bound
  applied to a 64-state corpus it was not measured on — and the commit that addresses it
  (`5402387`, the downsample protocol) is **held for Sami**, not merged. So these three rows are
  expected to stay FAIL until that decision is made, and they are not a Step-6 regression.

### One thing the brief predicted that did NOT happen, and it is good news

The brief warned that the four MJWarp trackcom cells' digests "do not reproduce at baseline either".
**They reproduce here, exactly.** `DMCCheetahRun`, `DMCHopperHop`, `DMCHumanoidStand` and
`DMCWalkerWalk` all return their frozen `digest_ours` AND their frozen `digest_official`, byte for
byte. The difference between that result and the reacher workstream's is the INTERPRETER: this replay
ran in `$SCR/venvs/se-step3`, which reproduces the frozen shard's package set field for field, and
the earlier non-reproduction was measured in `$SCR/venvs/se-step5`. So MJWARP-1 is, on this evidence,
a venv difference rather than MJWarp nondeterminism. That is a finding for the coordinator, not a
verdict: it says where to look, and it says the frozen dmc digests are reproducible.


## 2. CPU identity — one batch node, one torch thread

`--partition=batch --qos=normal`, `OMP_NUM_THREADS=1`, node1831 (job **6352886**, `e4d015d`) and
node2333 (job **6353226**, `736e642`, which adds the five gfxdraw configs).

Every digest in the frozen Step-4 corpus was taken on cuda, so "reproduces the frozen digest" is not
a statement CPU can make. The statement it CAN make is baseline-vs-integration on one node in one
job, and that is a PIXEL DIFF rather than a hash comparison, so a digest that moved would come with
a magnitude. The cpu-vs-cuda column is reported beside it as information and is not a bar.

**31 of 31 comparable configurations byte-identical (`max_abs_diff` 0 over every pixel of every
frame). Two more are not comparable because the baseline cannot render them on CPU at all.**

| config | owner | baseline CPU | integration CPU | baseline vs integration | CPU digest == frozen CUDA digest |
|---|---|---|---|---|---|
| PushT | raster2d | ERROR (no CPU path) | renders | n/a — baseline cannot render | False |
| SimplePointMaze | raster2d | ERROR (no CPU path) | renders | n/a — baseline cannot render | True |
| TwoRoom | raster2d | renders | renders | **byte-identical** | True |
| TwoRoomTarget | raster2d | renders | renders | **byte-identical** | True |
| Piecewise | raster2d | renders | renders | **byte-identical** | True |
| PiecewiseZones | raster2d | renders | renders | **byte-identical** | True |
| CraftaxClassic | tilemap | renders | renders | **byte-identical** | True |
| Acrobot | gfxdraw | renders | renders | **byte-identical** | True |
| CartPole | gfxdraw | renders | renders | **byte-identical** | True |
| MountainCar | gfxdraw | renders | renders | **byte-identical** | True |
| MountainCarContinuous | gfxdraw | renders | renders | **byte-identical** | True |
| Pendulum | gfxdraw | renders | renders | **byte-identical** | True |
| AntmazeGiantNavigate | raycast3d | renders | renders | **byte-identical** | False |
| AntmazeLargeExplore | raycast3d | renders | renders | **byte-identical** | False |
| AntmazeMediumExplore | raycast3d | renders | renders | **byte-identical** | False |
| AntmazeTeleportExplore | raycast3d | renders | renders | **byte-identical** | False |
| DMCAcrobotSwingup | raycast3d | renders | renders | **byte-identical** | False |
| DMCBallInCupCatch | raycast3d | renders | renders | **byte-identical** | False |
| DMCCartpoleBalance | raycast3d | renders | renders | **byte-identical** | False |
| DMCCartpoleSwingup | raycast3d | renders | renders | **byte-identical** | False |
| DMCFingerSpin | raycast3d | renders | renders | **byte-identical** | False |
| DMCFingerTurnEasy | raycast3d | renders | renders | **byte-identical** | False |
| DMCFingerTurnHard | raycast3d | renders | renders | **byte-identical** | False |
| DMCPendulumSwingup | raycast3d | renders | renders | **byte-identical** | False |
| DMCReacherEasy | raycast3d | renders | renders | **byte-identical** | False |
| DMCReacherHard | raycast3d | renders | renders | **byte-identical** | False |
| HumanoidmazeGiantNavigate | raycast3d | renders | renders | **byte-identical** | False |
| HumanoidmazeLargeNavigate | raycast3d | renders | renders | **byte-identical** | False |
| HumanoidmazeMediumNavigate | raycast3d | renders | renders | **byte-identical** | False |
| PointmazeGiantNavigate | raycast3d | renders | renders | **byte-identical** | False |
| PointmazeLargeNavigate | raycast3d | renders | renders | **byte-identical** | False |
| PointmazeMediumNavigate | raycast3d | renders | renders | **byte-identical** | False |
| PointmazeTeleportNavigate | raycast3d | renders | renders | **byte-identical** | False |

`PushT` and `SimplePointMaze` are the two `scene`-backend configs, and on the baseline tree both
raise `ValueError: Expected a cuda device, but got: cpu` — the Triton-dispatch defect recorded as
finding 1 in `examples/results/step6/findings.md`. On the integration tree both render. That is the
raster2d fix, and it is the one place where the integration branch is not identical to the baseline
on CPU because the baseline had nothing to be identical to.


## 3. The full `tests/gpu` suite

All nine files, unfiltered, one file at a time so a per-file count exists and a crash in one file
cannot hide the rest. `$SCR/venvs/se6-verify`, RTX 3090 `gpu2113`/`gpu2108`, job **6354203** (main
pass, `736e642`) and **6354905** (the gaps, both trees on one card). **Collection was asserted before
execution** — `tests/gpu/test_fast_paths.py` carries raycast3d's and gfxdraw's blocks merged by hand,
and a file that fails to collect reports zero failures.

| file | collected | passed | skipped | failed | errors | verdict |
|---|---|---|---|---|---|---|
| `test_conformance.py` | 1676 | 845 | 387 | **7** | 437 | 7 failures, all reproduced on the baseline tree or explained below; 437 errors are one missing import |
| `test_reproducibility.py` | 2039 | 1275 | 218 | **17** | 529 | one real finding (7 of the 17), the rest environment |
| `test_fast_paths.py` | 348 | 168 | 180 | 0 | 0 | **clean** — the hand-merged file collects and runs |
| `test_gfxdraw_classic.py` | 80 | 80 | 0 | 0 | 0 | **clean** |
| `test_gfxdraw_fused.py` | 4 | 4 | 0 | 0 | 0 | **clean** in `$SCR/venvs/se6-gfxdraw-box2d` (job 6354905). In `se6-verify` all four skip for want of Box2D and the repo's own EMPTY-RUN guard correctly calls that red. |
| `test_rendernet.py` | 10 | 0 | 9 | **1** | 0 | identical failure on the baseline tree — pre-existing |
| `test_crash_isolation.py` | 5 | 2 | 3 | 0 | 0 | **clean** |
| `test_rollout_on_device.py` | 87 | 57 | 30 | 0 | 0 | **clean** |
| `test_distributed.py` | 6 | 0 | 6 | 0 | 0 | **needs 2 GPUs** — `SKIPPED [3] needs 2 GPUs` twice, recorded in `gpu/distributed_reasons_gfx.results.txt`. Not run; this job holds one GPU, as AGENT_COMMON requires. |
| **Fetch, in `$SCR/venvs/se6-verify-fetch311`** | | | | | | |
| `test_conformance.py -k Fetch` | 1300 | **156** | 72 | **0** | 19 | all 12 Fetch cells clean; the 19 errors are `MiniGridPOVFetch`, which `-k Fetch` also matches |
| `test_reproducibility.py -k Fetch` | 1661 | **216** | 60 | **0** | 23 | same |

### The 966 errors are ONE thing, and it is not this branch

Every error in both files is `ModuleNotFoundError` at fixture setup — `gymnasium_robotics` for the 12
`Fetch*` and the 4 `PointMaze*` cells, `minigrid` for the 6 `MiniGrid*` and `BabyAIGoToLocal`. 23
envs x 19 (conformance) or 23 (reproducibility) tests = 437 and 529. **`minigrid` is installed in no
venv on Oscar at all**, so that family's GPU conformance has never been runnable here; it is also not
among the 49 frozen configurations. `Fetch` IS 12 of the 49, so it was re-run properly in
`$SCR/venvs/se6-verify-fetch311` — built like `se6-verify`, installing nothing, chaining
`se-s3b-fetch311` (the interpreter the fetch shard was frozen in) and then `se-step6` for pytest —
where all 372 Fetch tests pass. Classification: **environment**.

### The one genuine finding: `test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility`

**7 failed, 4 passed** over every DM Control cell (job 6354905,
`gpu/baked_visibility_gfx.results.txt`):

| cell | result | declares a `visibility.*` leaf |
|---|---|---|
| `DMCReacherEasy`, `DMCReacherHard`, `DMCFingerTurnEasy`, `DMCFingerTurnHard` | **PASSED** | yes |
| `DMCAcrobotSwingup`, `DMCBallInCupCatch`, `DMCCartpoleBalance`, `DMCCartpoleSwingup`, `DMCFingerSpin`, `DMCPendulumSwingup`, `DMCPointMassEasy` | **FAILED** | no |

The assertion that fires is the test's own selector, not a pixel comparison:

```
AssertionError: DMCAcrobotSwingup advertises VISIBILITY_KNOB but declares no visibility.* leaf
assert []
```

The mechanism. `_visibility_envs()` selects on `getattr(cls, "VISIBILITY_KNOB", False) and
hasattr(cls, "_hideable")`. `VISIBILITY_KNOB` is a CLASS flag set once on `DMControlSuite`
(`stable_envs/envs/dmcontrol.py:889`) and turned off only for `DMControlTrackCom`
(`dmcontrol.py:2223`), so the predicate is true for all eleven non-trackcom cells. But the leaf is
DERIVED per cell from `_hideable()`, whose own docstring says so: *"the other eleven draw nothing —
so eleven cells gain no leaf and their `describe()` does not move."* The seven cells whose episode
draws no task object therefore have `variation_space` with no `visibility.*` path, and the test
asserts they must have one.

**Nothing is wrong with the renderer.** Those seven cells' `digest_ours`,
`digest_ours_default_style` and `digest_official` are all bit-identical to the frozen oracle (section
1), their Gate A is unchanged, and their CPU frames are byte-identical to the baseline's (section 2).
The defect is in the test's selector, which claims in its docstring to find *"registered envs that
declare a `visibility.*` leaf"* and in fact finds envs whose class advertises the knob.

Why nobody saw it: the reacher-correctness review ran this test on the four cells it was written for
and recorded "4/4 GPU". It is red on the other seven, and only an unfiltered run over the family can
see that. **Classification: regression (test defect), introduced by the `step6/reacher-correctness`
merge, no pixel effect.** The smallest fix is in `_visibility_envs()` — select on `cls._hideable()`
being non-empty, which is a classmethod and needs no env built, exactly as the current predicate
promises. Owner: `step6/reacher-correctness`.

### The other failures, each measured on the BASELINE tree on the same card

| failure | cells | baseline | integration | classification |
|---|---|---|---|---|
| `test_visual_tier_matches_its_declared_fidelity` | `DMCCheetahRun` 6.541 vs 6.5, `DMCHopperHop`, `DMCWalkerWalk` 6.456 vs 5.7 | **3 failed** | **3 failed** | **expected-known / environment** — the MuJoCo 3.11 -> 3.12 floor-texture drift (texture-fidelity verdict B). Identical on both trees. |
| `test_resolution_downsample_agrees_within_the_measured_bound` | same three | — | 3 failed | **expected-known** — 52.223% vs a 53.385% bound measured on an H200 at mujoco 3.9 over 8 states. texture-fidelity's separate finding; the commit that fixes it (`5402387`, downsample protocol) is **held for Sami**. |
| `test_state_is_sufficient` | `PointmazeGiant*`, `PointmazeTeleport*` (4) | session **aborted with NODE FAULT** | session **aborted with NODE FAULT** | **environment (node)** |
| `test_the_reference_renderer_on_this_machine_is_deterministic` | `PointMaze{Large,Medium,Open,UMaze}`, `PointmazeTeleport{Navigate,Stitch}` | **2 failed + 4 errors** | **2 failed + 4 errors** | **environment (node)** |
| `test_frames_are_bit_identical_in_a_fresh_process` | `AntmazeTeleport*` (3), `HumanoidmazeLarge*` (2) | **10 passed, 5 skipped** | **10 passed, 5 skipped** | **environment (node/flake)** — failed on `gpu2108`, passes on `gpu2113` on BOTH trees |
| `test_no_checkpoint_means_no_class` | `test_rendernet.py` | **1 failed** | **1 failed** | **environment** — `pytest.importorskip("stable_envs.envs.robocasa")` does not swallow an `ImportError` raised inside a module that exists, under pytest 9.1.1 |

The node finding deserves its own sentence, because the repository already has machinery for it and
that machinery fired: on `gpu2113` the suite **aborted the session itself** with

```
_pytest.outcomes.Exit: NODE FAULT on gpu2113: the OFFICIAL PointmazeTeleportNavigate renderer
produced two different frames for the SAME states on gpu2113 (max 1/255), before any stable-envs
code was involved. ... Every parity number measurable here is meaningless; rerun with --exclude=gpu2113
```

That is the reference renderer, on the node, before our code runs, and it is identical on the
baseline tree. It also explains the four `test_state_is_sufficient` failures and the five
fresh-process failures on `gpu2108`: same family, same 1/255, same class of fault. **Two `gpu`-
partition 3090 nodes, `gpu2108` and `gpu2113`, produce a nondeterministic official OGBench renderer.**
Our own `PointmazeTeleportNavigate` digest replayed bit-identical on `gpu2104` (section 1), so this is
about where parity is measured, not about the renderer. Recommend the coordinator carry this as its
own ticket.

## 4. `tests/unit`

| tree | node | passed | skipped | failed |
|---|---|---|---|---|
| `e4d015d` | node1831 (job 6352886) | 584 | 3 | **2** |
| `736e642` (integration + gfxdraw) | node2333 (job 6353226) | **611** | 3 | **2** |

Both failures are the two the brief names in advance and no others:
`test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]` (hardcoded
path) and `test_every_cited_scratch_path_resolves`. The count rose 584 -> 611 because the gfxdraw
merge adds 27 unit tests, all passing.

## 5. Verdict

**PASS-WITH-KNOWN-DELTAS.**

`step6/integration` at `f7b07d7` draws the same pixels as the frozen Step-4 oracle on 48 of 49
configurations, draws the same pixels as the baseline tree on every CPU configuration that the
baseline could draw at all, improves Gate A from 43/5/1 to 44/3/1/1, and loses no check anywhere.

One item is a genuine defect on the integration branch and it is a test, not a renderer:
`test_a_baked_scene_does_not_inherit_the_last_bake_s_visibility` is red on 7 of 11 DM Control cells
because of its own selector. It cannot change a pixel, and every pixel those seven cells draw is
verified identical here — but the suite is red on the integration branch until it is fixed, so this
is not a clean PASS.

Everything else that is not green is one of: a declaration change this step intended, a documented
reference-version drift whose fix commit is deliberately held, a dependency absent from the
interpreter, or a node whose official renderer is nondeterministic — each one demonstrated by running
the identical test against the baseline tree on the same card.

### For the coordinator

1. **VISIBILITY-1** (blocking a clean suite): fix `_visibility_envs()` in
   `tests/gpu/test_reproducibility.py` to select on `cls._hideable()` rather than on the class flag.
   Owner `step6/reacher-correctness`. No pixel effect; re-run
   `-k a_baked_scene_does_not_inherit` to confirm 11/11.
2. **MJWARP-1 can probably be closed.** The four MJWarp trackcom cells reproduce their frozen
   `digest_ours` AND `digest_official` byte for byte in `$SCR/venvs/se-step3`, the interpreter the
   dmc shard was frozen in. The earlier non-reproduction was measured in `$SCR/venvs/se-step5`. That
   points at the venv, not at MJWarp nondeterminism.
3. **NODE-1** (new): the official OGBench renderer is nondeterministic at 1/255 on `gpu2108` and
   `gpu2113` for the Pointmaze Teleport and maze2d cells, on the baseline tree as much as on this
   one. The repo already aborts the session over it. Any Pointmaze parity number taken on those two
   nodes should be re-taken elsewhere.
4. **`minigrid` is installed in no venv on Oscar**, so 7 registered envs have never had their GPU
   conformance run on this cluster. Outside the 49, but worth a ticket.
5. Three Gate A FAILs remain (`DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk`) on
   `check_resolution_consistency` alone. They cannot clear until the held downsample-protocol commit
   is decided.

### SLURM jobs

| job | partition / node | what |
|---|---|---|
| **6352885** | gpu / gpu2104 (RTX 3090) | frozen Step-4 replay, all 5 shards, all 49 configs, cuda, + Gate A at `e4d015d` |
| **6352886** | batch / node1831 | CPU identity 28 configs + `tests/unit` at `e4d015d` |
| **6353226** | batch / node2333 | CPU identity 33 configs (incl. the 5 gfxdraw) + `tests/unit` at `736e642` |
| **6354203** | gpu / gpu2108 (RTX 3090) | raster shard replay + full 9-file `tests/gpu` suite at `736e642` |
| **6354905** | gpu / gpu2113 (RTX 3090) | Fetch under the 3.11 stack; the 7 failures baseline-vs-integration on one card; `test_gfxdraw_fused` under Box2D; distributed skip reasons |

### Re-running a subset

Every script takes a tree and a subset, so the coordinator's next two merges do not need new code:

```bash
sbatch --export=ALL,TREE=$SCR/se6/verify3,SHARDS='swm raster',TAG='_swmnav'     scratch/step6/verify/postmerge.sbatch
sbatch --export=ALL,TREE=$SCR/se6/verify3,RASTER='PushT,SimplePointMaze,TwoRoom,TwoRoomTarget',RAY='',TAG='_swmnav'     scratch/step6/verify/cpu_verify.sbatch
sbatch --export=ALL,TREE=$SCR/se6/verify3,SHARDS='ogb dmc',TAG='_raycast5'     scratch/step6/verify/replay.sbatch
```

