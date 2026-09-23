# Fetch family backend recovery

Starting branch `repair/environment-semantics`, clean at
`c3e03f198e91c632a42ea81e69b474093d0a7e84`. Implementation:
`0dd0271f47d5b01dde1630a8cdbb1ea0072a5f92`. No push.

## Scope and ownership

One parameterized `FetchAdapter` and `FetchSemantics` serve twelve declarations:
four physical tasks, sparse/dense reward configurations, and four Dict aliases.
The original `FetchPickAndPlaceAdapter` private name remains an alias. Public
step returns exactly full State, reward, success, terminated and truncated.
The representative's previous flat observation and SWM-style info packaging
were removed under the explicit ownership clarification. No full State is put
in `info["state"]`; no Observation abstraction or filtering was added.

Backend acceptance and **SWM observation integration: DEFERRED** are separate.
Historical flat observations are float32 concatenations of upstream observation
and desired_goal; `flatten=False` preserves the upstream observation dictionary.
Historical `info["state"]` is the flat learner observation even on Dict IDs.
These conventions are provenance, not backend acceptance requirements.
No stable-worldmodel production code or imports were added or changed.

## Authoritative source and state

`semantic_recovery_evidence/fetch_family/provenance.json` records exact source
paths, SHA256s, literal constructor/controller/reset/reward/TimeLimit symbols,
versions, task settings and registration kwargs. SWM commit is
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. CPU authority is Gymnasium 1.2.3,
Gymnasium Robotics 1.4.2, MuJoCo 3.8.1, NumPy 2.2.6, Torch 2.10.0,
TensorDict 0.14.2. Neither installed upstream packages nor SWM were patched.

All tasks use `MujocoFetch{Task}Env` from Robotics' `envs/fetch/` modules,
`MujocoFetchEnv`, `BaseRobotEnv`, the MuJoCo action helpers and Gymnasium TimeLimit.
The pinned SWM Dense registrations select `Fetch{Task}Dense-v4`, whose registration
kwargs specify `reward_type="dense"`. Dict selects the sparse upstream ID and
sets only SWM `flatten=False`.

| Physical task | qpos / qvel / warmstart widths | Source reset/task differences |
|---|---|---|
| PickAndPlace | 22 / 21 / 21 | Object; movable gripper; extra height .2; object/target ranges .15; goal may gain a sampled height up to .45 |
| Reach | 15 / 15 / 15 | No object; locked gripper; extra height .2; goal sampled in a 3D box of radius .15 around reset anchor |
| Push | 22 / 21 / 21 | Object; locked gripper; extra height 0; ranges .15; target on table |
| Slide | 22 / 21 / 21 | Puck; locked gripper; extra height -.02; object range .1, target range .3, target offset [.4,0,0]; scene-specific model/contact configuration |

Every full State additionally contains float64 `time[1]`,
`task.initial_gripper_xpos[3]`, `controller.position[3]`,
`controller.quaternion[4]`, `goal[3]`; int64 `lifecycle.elapsed_steps[1]`,
`rng.state_words[4]`, `rng.increment_words[4]`, `rng.has_uint32[1]`,
`rng.uinteger[1]`. PCG64's complete documented state is atomic. These are sufficient
named snapshots under the same persistent Variation, not a minimality claim.

The accepted restore order is unchanged: qpos/qvel/goal/time/reset anchor,
zero applied forces, `mj_forward`, warmstart, gripper controller cache, TimeLimit,
complete simulator PCG64. Reach/Push/Slide perform a blocked-gripper callback and
another forward after each step; their 32-step traces and eight-step continuation
from different histories pass this same restore order without adjustment.
Normal steps never restore the simulator. ctrl and mocap are overwritten by the
source action controller before stepping; the public API never injects forces.

All tasks clip actions to [-1,1], multiply simultaneous XYZ controls by .05,
and use 20 MuJoCo substeps. Gripper control is suppressed on Reach/Push/Slide.
Float32 multiplication is preserved before upstream concatenation; float64
controls remain float64. Reward uses the Euclidean achieved-goal distance `d`:
sparse `-(d > .05).astype(float32)`, dense `-d`. Success is
`(d < .05).astype(float32)`. At exactly .05 sparse reward is zero and success
is false. Reach's achieved goal is the gripper site; other tasks use the object
site. Termination is always false; TimeLimit truncates at elapsed >= 50.

## CPU evidence and coverage

Each `{Task}.json` and `{Task}Dense.json` has 12 cases: seeds 0/17/42,
batches 1/3, float32/64. Each covers independent snapshots, exact roundtrip,
32 sequential steps, 11 steps of different history before restore, eight further
steps, unseeded simulator reset replay, repeated seeded resets, action clipping
and simultaneous controls, compact-State refusal, and explicit CPU rendering
refusal. Values, shapes and dtypes are exact; no tolerances are used.

Sparse and dense gates additionally cover lifecycle 49/50/51 with restore at 49,
reset-anchor history replay, and bounded Variation. For object tasks, public
post-step distances include zero, the adjacent float64 values around .05,
exactly .05 and .2. Reach has no freely placeable object at the origin: its exact
boundary evidence directly exercises the same upstream reward/success functions;
its public step evidence records the actual achieved post-step distance separately.
The distinction is explicit in `Reach*.json`, with no false exact-.05 step claim.

`{Task}-alias.json` proves all three registrations against independent pinned
SWM oracles, comparing only backend outputs. It covers the same 12 seed/batch/dtype
combinations, reset equality, 32 steps, different-history restoration and eight
continuation steps, simulator RNG reset and RenderState equality. Dense has
identical next full State/success/done and exactly negative upstream distance;
Dict has identical reset, future and reward to sparse. Dict observation
containers are neither implemented nor asserted here.

`public-contract.json` checks all twelve public declarations' typed State,
action specification, execution readiness, persistent visual settings across
resets and explicit reset visual settings remaining until replaced. This is a
bounded backend gate, not a claim that the unchanged broad replacement inventory
is READY across every historical Variation leaf.

## Variation

The bounded object-task scope is agent reset anchor, block XY, block angle,
goal XYZ, reset/persistent mass and object color. Mass is applied after source
reset, matching SWM. Explicit reset mass overrides are cleared to configured
mass on the next reset; persistent mass is tested through repeated resets.
Mass8 changes physical traces (maximum proprio differences: PickAndPlace
.00025946916669317144, Push .0002594679128132271, Slide .000008029274063225622).
Positions/angle alter source physical trajectories. Reach supports agent reset
anchor, goal override and table color; changing its anchor changes future goal
sampling, not the initial robot qpos. There is no mass/object leaf for Reach.

Independent seeded visual sampling uses the source Gym space tree, including
its otherwise unexposed leaves for the same seed assignments. Reach omits the
absent block subtree. Seeds 0/17/42 reproduce exact selected colors while simulator
PCG64 is unchanged. Unseeded visual reset draws fresh external entropy in SWM;
no replay claim is made for that external entropy. Simulator RNG replay is
certified under fixed Variation. Visual color changes leave physical traces
unchanged. Persistent default renderer appearance remains the frozen appearance;
the entire historical default five-leaf randomization surface is not certified.

## Renderer gate

GPU gate status: **BLOCKED / PENDING at QOSGrpGRES, StartTime Unknown**.
No GPU output or terminal marker has arrived. The job remains queued.
Per the requested GPU-blocker stop condition, no new row is promoted.
Final matrix: **22 VERIFIED / 20 INCOMPLETE** (eleven Fetch candidates await this gate). Oscar job `6649686`, immutable stage
`fetch-family-0dd0271f`, archive SHA256
`9829af3404ba7458ecb2e5e8195c3ac98f943a1e8c1f83114fb5a7b4305c4bbb`.
The accepted runner uses `3090-gcondo` / `cs-3090-gcondo`, with the existing
`geforce3090|a5000|a5500` feature constraint for this small workload.

Frozen baseline is **6d80b68b8b3ccd07d4207be94cbf4f0760758a06**, not bf9101cb:
bf9101cb is the representative's compatible baseline/candidate test driver.
Neither baseline nor accepted fixtures changed. All 15 renderer function/class
ASTs match the accepted integration tip, including FetchBase. Only its public
semantic factory and documentation changed. CPU canonical PickAndPlace State
is byte-identical to the accepted representative bridge.

Four frozen rows per physical scene are compared on the same GPU/profile to the
baseline in a separate process. Each candidate additionally renders three full
CPU-authoritative snapshots and their private projections, verifies no snapshot
or live simulator mutation, refuses compact restoration, and checks a visual
change affects pixels only. Dense/Dict aliases must match the sparse canonical
and varied frames exactly. The manifest records scene/projection/default/renderer
identity as the reason the sparse frozen comparison covers the aliases.

CPU full rendering is `CPU_BACKEND_MISSING`. The GPU profile uses MuJoCo/MJWarp
3.11 and is a renderer-only claim. CPU physics parity does not imply GPU-profile
physics parity; no such claim is made.

## Regression and limitations

`start_audit.json` validates 94 original accepted evidence hashes and ten actual
terminal logs for all 22 accepted rows. `diff-review.json` records an independent
inspection of the final production diff: exactly three Fetch production files
changed, no shared contract/taxonomy/parity/promotion machinery, other accepted
family code unchanged, and frozen fixtures unchanged. Shared readiness,
full-State, semantic harness, Fetch Variation taxonomy, legacy Fetch source and
SWM promotion checks: **62 passed, 56 skipped** in the local general profile.
Optional reference/renderer tests are skipped there; physical parity and GPU
pixels have their explicit separate profiles/gates. Legacy renderer construction
tests now construct FetchBase directly, preserving their original renderer-only
assertions while all public registrations acquire the semantic mixin.

Post-metadata readiness/promotion rerun: **29 passed**, recorded in
`final-readiness.out`. No shared-contract review is requested. PushT remains separate and parked;
OGBench Cube/Scene, Craftax and Rocket are untouched. Full historical Variation,
arbitrary externally injected forces, nondefault upstream task kwargs,
upstream-image similarity and final paired benchmarks remain outside scope.
The retained `Reach-invalid-fixture.out` is a rejected test input outside SWM's
fixed-z goal Box, corrected before the passing evidence; no source patch was made.

## Reproduction

From the integration worktree, `PYTHONPATH=.:/Users/samib/stable-worldmodel`, with
`/private/tmp/fetch-authority-swm-lock/bin/python`:

```sh
python scripts/recovery/fetch_family_provenance.py
# For each task: PickAndPlace, Reach, Push, Slide
python scripts/recovery/fetch_public_gate.py --task TASK --output EVIDENCE/TASK.json
python scripts/recovery/fetch_public_gate.py --task TASK --suffix Dense --output EVIDENCE/TASKDense.json
python scripts/recovery/fetch_alias_gate.py --task TASK --output EVIDENCE/TASK-alias.json
python scripts/recovery/fetch_family_contract_gate.py
```

`EVIDENCE=docs/research/semantic_recovery_evidence/fetch_family`.
The public driver writes `canonical-TASK.json` beside its result. Internal
execution order was PickAndPlace Dense/alias, three sparse physical tasks,
their Dense/alias gates and representative sparse regression.

```sh
.venv/bin/python -m pytest -q tests/unit/test_replacement_readiness.py tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_variation_maps_fetch.py tests/unit/test_envs_fetch.py tests/unit/test_swm_matrix.py
.venv/bin/python -m pytest -q tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py
.venv/bin/ruff check stable_envs/_adapters/fetch_pickandplace.py stable_envs/envs/_fetch_semantics.py stable_envs/envs/fetch.py scripts/recovery/fetch_public_gate.py scripts/recovery/fetch_alias_gate.py scripts/recovery/fetch_renderer_gate.py scripts/recovery/fetch_family_provenance.py scripts/recovery/fetch_family_contract_gate.py tests/unit/test_envs_fetch.py
git diff --check
# Oscar scratch, after immutable archive hash verification:
sbatch fetch-family-0dd0271f/scripts/recovery/oscar_fetch_family_gate.sbatch fetch-family-0dd0271f
```

The sbatch runner executes `fetch_renderer_gate.py --task TASK --baseline` in the
frozen checkout and the candidate with `--fixture canonical-TASK.json`, then
requires exact frozen JSON equality and actual success markers for all four tasks.
