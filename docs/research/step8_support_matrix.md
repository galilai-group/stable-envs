# Step 8 — support matrix for all 42 `stable-worldmodel` ids

!!! info "Generated, not written"

    `python scratch/step8/report.py` regenerates this page and
    `examples/results/step8/support_matrix.json` from committed artifacts. Every column is
    read out of a file that is in the repository and cited below. Editing this page by hand
    makes it wrong at the next regeneration, which is the point: a hand-maintained matrix
    drifts from the evidence silently.

Baseline `43ac95238f7472137619f3e69e7636e11dd088b1` · generated at `20a8039406b375d20ca534b4d39d92d58372cb7a`

## The five words

Plan section 3 keeps these apart and this table never collapses them into a generic
failure, because each carries a different obligation:

| word | meaning | what it obliges |
|---|---|---|
| `PASS` | the claim holds and there is evidence | nothing |
| `UNSUPPORTED_BY_DESIGN` | absent on purpose, and the env says so | nothing — it is closed |
| `UNMEASURED` | nobody measured it | work, or an explicit decision not to |
| `DOCUMENTED_LIMITATION` | known, measured, bounded | a record, not a campaign |
| `UPSTREAM_FINDING` | the defect is SWM's or a dependency's | report upstream |

`FAIL` is deliberately NOT one of the five: a real failure is a real failure, and this
table reports it as such rather than dressing it as a limitation.

The two tier columns use a sixth word, `MEASURED`, and only those columns do. A tier is a
GRADE, not a gate: `CPU_REGRESSION` is a well-measured tier, and calling it `PASS` would
read as "fast enough" while calling it `FAIL` would read as a defect. Neither is what a
tier says. The tier string carries the grade; the word beside it says only whether a
measurement exists at all.

## Counts

```text
rows                 42
pixel-producing      40
symbolic             2
render covered       40
gate A               {"FAIL": 3, "PASS": 36, "UNMEASURED": 1, "UNSUPPORTED_BY_DESIGN": 2}
cpu tier status      {"MEASURED": 24, "UNMEASURED": 15, "UNSUPPORTED_BY_DESIGN": 3}
gpu tier status      {"MEASURED": 40, "UNMEASURED": 2}
variation map        {"complete": 26, "n/a": 2, "partial": 12, "unknown": 2}
backend class        {"R": 12, "S": 28, "n/a": 2}
```

## The matrix

| SWM id | cell(s) | class | kind | render | backend(s) | devices | fidelity tier | Gate A | reference/version | CPU tier | GPU tier | VariationMap | unsupported/dead leaves | Gate F | physics | reproducibility | semantic | limitations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `Acrobot` | R | pixel | covered | `gfxdraw` | — | `parity` | **PASS** | — | CPU_REGRESSION (MEASURED) | GPU_FASTER (MEASURED) | complete | `physics.link_com_pos_1`, `physics.link_com_pos_2`, `physics.link_length_1`, `physics.link_length_2`, `physics.link_mass_1`, `physics.link_mass_2`, `physics.link_moi` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/AcrobotDMControl-v0` | `DMCAcrobotSwingup` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.lower_arm_density`, `agent.upper_arm_density`, `agent.upper_arm_locked`, `floor.color`, `gravity.x`, `gravity.y`, `gravity.z`, `target.color`, `target.shape` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/BallInCupDMControl-v0` | `DMCBallInCupCatch` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.density`, `ball.color`, `ball.density`, `ball.size`, `floor.color`, `gravity.x`, `gravity.y`, `gravity.z`, `rendering.render_target`, `target.color`, `target.shape` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/CartPoleControl-v1` | `CartPole` | R | pixel | covered | `gfxdraw` | — | `parity` | **PASS** | — | CPU_PARITY (MEASURED) | GPU_FASTER (MEASURED) | complete | `physics.force_mag`, `physics.gravity`, `physics.length`, `physics.masscart`, `physics.masspole`, `physics.tau` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/CartpoleDMControl-v0` | `DMCCartpoleBalance`, `DMCCartpoleSwingup` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `parity` | **PASS** | — | CPU_PARITY (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.cart_mass`, `agent.cart_mass`, `agent.cart_shape`, `agent.cart_shape`, `agent.pole_density`, `agent.pole_density`, `floor.color`, `floor.color`, `gravity.x`, `gravity.x`, `gravity.y`, `gravity.y`, `gravity.z`, `gravity.z` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `visual` | **FAIL** | mujoco 3.11.0 | CPU_FASTER (MEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | `agent.back_foot_density`, `agent.back_foot_locked`, `agent.torso_density`, `floor.color`, `floor.friction`, `gravity.x`, `gravity.y`, `gravity.z` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassic` | R | pixel | covered | `prng` | cpu:PASS, cuda:PASS | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FASTER (MEASURED) | n/a | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/CraftaxClassicSymbolic-v1` | — | n/a | symbolic | UNSUPPORTED_BY_DESIGN | — | — | — | **UNSUPPORTED_BY_DESIGN** | — | unknown (UNMEASURED) | unknown (UNMEASURED) | unknown | — | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |
| `swm/CraftaxPixels-v1` | `CraftaxFull` | R | pixel | covered | `prng` | cpu:PASS, cuda:PASS | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FASTER (MEASURED) | n/a | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | **DOCUMENTED_LIMITATION** resolution consistency |
| `swm/CraftaxSymbolic-v1` | — | n/a | symbolic | UNSUPPORTED_BY_DESIGN | — | — | — | **UNSUPPORTED_BY_DESIGN** | — | unknown (UNMEASURED) | unknown (UNMEASURED) | unknown | — | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |
| `swm/FetchPickAndPlace-v3` | `FetchPickAndPlace` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchPickAndPlaceDense-v3` | `FetchPickAndPlaceDense` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchPickAndPlaceDict-v3` | `FetchPickAndPlaceDict` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchPush-v3` | `FetchPush` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchPushDense-v3` | `FetchPushDense` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchPushDict-v3` | `FetchPushDict` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchReach-v3` | `FetchReach` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `object.color`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchReachDense-v3` | `FetchReachDense` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `object.color`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchReachDict-v3` | `FetchReachDict` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `object.color`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchSlide-v3` | `FetchSlide` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchSlideDense-v3` | `FetchSlideDense` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FetchSlideDict-v3` | `FetchSlideDict` | S | pixel | covered | `mjwarp` | — | `parity` | **PASS** | — | CPU_UNMEASURED (UNMEASURED) | GPU_PARITY (MEASURED) | partial | `camera.angle_delta`, `rendering.transparent_arm` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/FingerDMControl-v0` | `DMCFingerTurnHard` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `parity` | **PASS** | — | CPU_REGRESSION (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.fingertip_density`, `agent.proximal_density`, `floor.color`, `spinner.color`, `spinner.density`, `spinner.friction`, `target.shape` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `visual` | **FAIL** | mujoco 3.11.0 | CPU_FASTER (MEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | `agent.foot_density`, `agent.foot_locked`, `agent.torso_density`, `floor.color`, `floor.friction`, `gravity.x`, `gravity.y`, `gravity.z` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |
| `swm/HumanoidDMControl-v0` | `DMCHumanoidStand` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `visual` | **UNMEASURED** | mujoco 3.11.0 | CPU_FASTER (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.left_knee_locked`, `agent.right_lower_arm_density`, `agent.torso_density`, `floor.color`, `floor.friction`, `gravity.x`, `gravity.y`, `gravity.z` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |
| `swm/ManipulatorDMControl-v0` | `DMCManipulatorBringBall` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | cpu:PASS, cuda:PASS | `parity` | **PASS** | — | CPU_REGRESSION (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.hand_density`, `agent.upper_arm_density`, `agent.upper_arm_length`, `floor.color` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | **DOCUMENTED_LIMITATION** resolution consistency |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuous` | R | pixel | covered | `gfxdraw` | — | `parity` | **PASS** | — | CPU_PARITY (MEASURED) | GPU_FASTER (MEASURED) | complete | `physics.max_speed`, `physics.power` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/MountainCarControl-v0` | `MountainCar` | R | pixel | covered | `gfxdraw` | — | `parity` | **PASS** | — | CPU_PARITY (MEASURED) | GPU_FASTER (MEASURED) | complete | `physics.force`, `physics.gravity`, `physics.max_speed` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/OGBCube-v0` | `OGBCube` | S | pixel | covered | `mjwarp` | cpu:UNSUPPORTED_BY_DESIGN, cuda:PASS | `visual` | **PASS** | mujoco 3.12.0, ogbench 1.2.1 | CPU_MISSING (UNSUPPORTED_BY_DESIGN) | GPU_PARITY (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/OGBMaze-v0` | `PointmazeLargeNavigate`, `PointmazeLargeStitch` | S | pixel | covered | `raycast3d` | — | `parity` | **PASS** | — | CPU_FAST (MEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/OGBScene-v0` | `OGBScene` | S | pixel | covered | `mjwarp` | cpu:UNSUPPORTED_BY_DESIGN, cuda:PASS | `visual` | **PASS** | mujoco 3.12.0, ogbench 1.2.1 | CPU_MISSING (UNSUPPORTED_BY_DESIGN) | GPU_PARITY (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/PFRocketLanding-v0` | `PFRocketLanding` | S | pixel | covered | `nvdiff` | cpu:UNSUPPORTED_BY_DESIGN, cuda:PASS | `visual` | **PASS** | PyFlyt 0.29.0, pybullet 3.2.7 | CPU_MISSING (UNSUPPORTED_BY_DESIGN) | GPU_PARITY (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | **DOCUMENTED_LIMITATION** nvdiff x86 CPU sparse-stage one ULP; **DOCUMENTED_LIMITATION** check_colors_apply_exactly |
| `swm/PendulumControl-v1` | `Pendulum` | R | pixel | covered | `gfxdraw` | — | `parity` | **PASS** | — | CPU_PARITY (MEASURED) | GPU_FAST (MEASURED) | complete | `physics.dt`, `physics.g`, `physics.l`, `physics.m`, `physics.max_speed`, `physics.max_torque` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/PendulumDMControl-v0` | `DMCPendulumSwingup` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.mass_density`, `agent.mass_shape`, `agent.pole_density`, `floor.color`, `gravity.x`, `gravity.y`, `gravity.z` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/Piecewise-v0` | `Piecewise`, `PiecewiseZones` | R | pixel | covered | `navdot` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FASTER (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/PushT-Discrete-v1` | `PushT` | R | pixel | covered | `scene` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/PushT-v1` | `PushT` | R | pixel | covered | `scene` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/QuadrupedDMControl-v0` | `DMCQuadrupedWalk` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | cpu:PASS, cuda:PASS | `visual` | **PASS** | mujoco 3.11.0 | CPU_UNMEASURED (UNMEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | `agent.foot_back_left_density`, `agent.knee_back_left_locked`, `agent.torso_density`, `floor.color`, `floor.friction`, `gravity.x`, `gravity.y`, `gravity.z` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/ReacherDMControl-v0` | `DMCReacherHard` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FAST (MEASURED) | complete | `agent.arm_density`, `agent.finger_density`, `agent.finger_locked`, `floor.color`, `target.shape` | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` | R | pixel | covered | `scene` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FAST (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/TwoRoom-v1` | `TwoRoom`, `TwoRoomTarget` | R | pixel | covered | `navdot` | — | `parity` | **PASS** | — | CPU_FASTER (MEASURED) | GPU_FASTER (MEASURED) | complete | — | UNMEASURED | UNMEASURED | PASS | UNMEASURED | — |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | S | pixel | covered | `cpu3d`, `mjwarp`, `raycast3d` | — | `visual` | **FAIL** | mujoco 3.11.0 | CPU_FASTER (MEASURED) | GPU_EXCEPTIONAL (MEASURED) | complete | `agent.left_foot_density`, `agent.right_knee_locked`, `agent.torso_density`, `floor.color`, `floor.friction`, `floor.rotation_y`, `gravity.x`, `gravity.y`, `gravity.z` | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | — |

## What is UNMEASURED at this HEAD, and why

These are absences of evidence, not failures, and each names the workstream that closes it:

| column | why | closes with |
|---|---|---|
| Gate F / SWM API | no committed Step-8 Gate-F artifact exists yet | Workstream D |
| physics | the Step-7 closure matrix carries no per-row physics verdict; `check_step_matches_reference` is judged on its own axis and is not joined here | Workstream E, on the final pass |
| semantic | no committed Step-8 semantic artifact exists yet | Workstream H |
| VariationMap (`missing` rows) | the map was never built for that row | Workstreams A/B/C |
| CPU tiers marked `UNMEASURED` | the measurement was load-quarantined or never run | Workstream G |

## Sources

```text
census:
    scratch/census/census_macos-probe.json
    scratch/census/census_oscar-3090-fetch.json
    scratch/census/census_step8-craftax-gpu.json
    scratch/census/census_step8-craftax.json
    scratch/census/census_step8-fetch311.json
    scratch/census/census_step8-ogb.json
    scratch/census/census_step8-rocket.json
    scratch/census/census_step8-step5.json
    scratch/census/census_step8-step6.json
    scratch/census/census_step8-vis311.json
classifier:
    scratch/step3/certify.py:backend_class
coverage_matrix:
    examples/results/step7/closure2/coverage_matrix.json
coverage_matrix_integration_head:
    fec4f6eea606e390c22e2b206b3d0738d11468c8
step8_certification:
    examples/results/step8/certification/census_class_diff.json
    examples/results/step8/certification/colour_check_decision.md
    examples/results/step8/certification/colour_law_ceiling.json
    examples/results/step8/certification/corpus_index.json
    examples/results/step8/certification/downsample_provenance.json
    examples/results/step8/certification/gate_a_rejudged.json
    examples/results/step8/certification/nvdiff_ulp_classification.json
    examples/results/step8/certification/order_invariance_finding.json
    examples/results/step8/certification/oscar_paired_tests.json
    examples/results/step8/certification/replay49.json
    examples/results/step8/certification/universe8_check.json
```
