# Step 6 — integration verification (`step6/integration-verify`)

**Classification: _pending_ — this file is written as each job returns; a row with no number in it
has not been measured and must not be read as a pass.**

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


<!--TESTS-->
