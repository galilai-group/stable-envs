# Cube/Scene upstream-based CPU compatibility checkpoint

Cube and Scene pass the bounded CPU semantic gates on the immutable SWM-lock
profile. This is a CPU implementation checkpoint for review, not promotion.
The matrix and replacement map remain byte-identical: **33 VERIFIED / 9 INCOMPLETE**.
Renderer/GPU acceptance has not started. No renderer or shared-contract redesign
is requested, and no push occurred.

## Checkpoint and scope

- Worktree: `/private/tmp/stable-envs-semantic-recovery`.
- Branch: `repair/environment-semantics`; started clean.
- Starting HEAD / dependency-boundary audit:
  `75ef812977423fa059b857b36542119e101a430f`.
- RNG-feasibility checkpoint: `434cc72974360ab98e47cad7aebcc8b1b62bb1de`.
- Documentation checkpoint: `03adc1ab79bae71e3e62db2e5c4cb19f20723166`.
- Immutable SWM oracle: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
- Both public production adapters were unimplemented at the starting checkpoint.
- This change implements only the empty-kwargs historical single Cube and Scene
  registrations, supported task/reward/reset options, and explicit stable-envs
  Variation anchors. Other environment families are unchanged.

Implementation commit: `14456339ec7f1b28ad698fc3158fe9e4ed8f6d1e`.
The implementation and evidence are separate commits. The final handoff supplies
both full SHAs and the ending SHA, since an evidence commit cannot contain its
own hash. The exact changed-file inventory is in
[closure.json](semantic_recovery_evidence/ogb_cpu_compatibility/closure.json).
No containing commit is needed: the evidence commit is the clean review tip.

The pre-code method plan is
[ogbench_cpu_compatibility_plan.md](ogbench_cpu_compatibility_plan.md).
Its four categories remain the boundary: excluded, existing hook/configuration,
delegated pre/post adjustment, and unavoidable compatibility algorithm.

## Implementation and ownership

The public `OGBCube`/`OGBScene` classes add `ManipulationSemantics` before the
unchanged historical renderer classes. They compose `OGBManipulationAdapter`.
The adapter instantiates this engine hierarchy:

```text
CompatibleCube(Compatibility, ogbench.manipspace.envs.cube_env.CubeEnv)
CompatibleScene(Compatibility, ogbench.manipspace.envs.scene_env.SceneEnv)
  -> ManipSpaceEnv -> CustomMuJoCoEnv -> Gymnasium Env
```

OGBench 1.2.1 is pinned in the existing extra and checked at engine construction.
The installed package, XML, meshes, task tables, controller and differential IK
are unchanged. No upstream fork was created. Production imports upstream OGBench
and stable-envs spaces, never stable-worldmodel. Only evidence/tests import SWM.

The public registration wiring changes only the two manipulation entries. Maze
classes/cache exports are unchanged. The frozen `manipulation.py` file is
byte-identical; the historical renderer implementation is not exercised by these
CPU classes. Public rendering and historical dataset state sampling refuse with
explicit pending/out-of-scope errors.

New production files:

| File | Physical lines | Responsibility |
|---|---:|---|
| `stable_envs/_adapters/ogbench_manipulation_compat.py` | 302 | Bounded source deltas, existing MJCF hooks, stable-envs Variation and owned RNGs |
| `stable_envs/_adapters/ogbench_manipulation.py` | 402 | Numeric State, validation, atomic reconstruction, batched step/reset |
| `stable_envs/envs/ogbench/manipulation_semantics.py` | 210 | Public Env/State/Variation/RenderState boundary |

The following is every method introduced by the compatibility mixin and Scene
specialization, including validation helpers. Physical counts include the `def`
and body, excluding decorators; normalized counts remove docstrings/comments and
use `ast.unparse` (which includes decorators). Exact source file/method hashes,
line locations, source counterparts for both families, all adapter/public methods,
and the resolved upstream MRO are in
[ownership.json](semantic_recovery_evidence/ogb_cpu_compatibility/ownership.json).
A missing upstream/SWM counterpart is explicitly null there, not invented provenance.

| Class/method | Category | Physical | Normalized | Reason/delegation |
|---|---:|---:|---:|---|
| `Compatibility.__init__` | 2 | 23 | 17 | Registered constructor/configuration and stable-envs Variation tree |
| `Compatibility._seed_owned` | 3 | 9 | 9 | Design A per-instance ownership, before any implicit reset |
| `Compatibility.action_space` | 3 | 5 | 6 | Memoize genuine upstream Box; own complete PCG64 |
| `Compatibility.validate_options` | 2 | 22 | 22 | Restrict options before mutation |
| `Compatibility.validate_task` | 2 | 33 | 24 | Validate supported numeric task payload |
| `Compatibility.reset` | 3 | 21 | 16 | Owned RNG/Variation prelude then delegate whole reset |
| `Compatibility.initialize_arm` | 4 | 12 | 11 | Only unavoidable reset algorithm body: Variation position replaces uniform draw |
| `Compatibility.initialize_episode` | 3 | 5 | 5 | Initialize immediate-output caches then delegate entire episode |
| `Compatibility._cube_colors_from_variation` | 2 | 6 | 6 | Configure registered cube and target colors |
| `Compatibility.post_step` | 3 | 3 | 3 | Delegate physics/success/button changes then apply cube color |
| `Compatibility.get_reset_info` | 3 | 5 | 5 | Delegate observations/info then adapt output keys |
| `Compatibility.get_step_info` | 3 | 2 | 2 | Delegate info then adapt output keys |
| `Compatibility._output_info` | 2 | 3 | 3 | Registered env_name/target/success output boundary |
| `Compatibility.modify_mjcf_model` | 2 | 38 | 37 | Existing MJCF hook: size, material, floor, camera, light and source dirty predicate |
| `Compatibility.compile_model_and_data` | 3 | 3 | 3 | Delegate compilation then track exact compiled configuration |
| `CompatibleScene._apply_button_states` | 3 | 7 | 7 | Delegate damping/forward then configure registered lock colors |

`initialize_arm` is the sole category-4 reset algorithm: **12 physical / 11
normalized lines**, with executable AST identical to both SWM families. Its only
source-level change relative to upstream is where `eff_pos` comes from: the
Variation value replaces the upstream position uniform draw. Calling `super()`
would consume those three draws before yaw and use the wrong arm position.
Yaw, Lie transformations, differential IK, joint assignment and `mj_forward`
remain the original sequence and call OGBench/MuJoCo implementations.

Configuration hooks necessarily adapt registered source deltas too. A conservative
upper bound counting **every function** in the entire compatibility module,
including new RNG ownership, validation, tree construction and forwarding wrappers,
is **247 physical / 191 normalized lines**. This is not a claim that those lines
were all copied. The copied algorithm total is 11 normalized lines; the remaining
configuration/adaptation surface is included in the 191-line bound. No complete
SWM file, reset/episode routine, observation builder, task table, controller, IK,
physics, asset or spaces implementation was copied.

Audited function-identity assertions prove normal `step`, task tables,
`add_objects`, observation/oracle observation, reward, success and object lookup
are inherited upstream unchanged. The CPU tests also assert direct `set_control`
ownership and record exactly one real `mj_step(nstep=25)` per action, advancing
0.05 seconds. Normal step never calls `set_state`.

Two subtle registered deltas are retained:

1. Python-list cube size goes through NumPy's float32 weak-scalar multiplication;
   typed float64 size remains float64. The historical default is the float32-derived
   value. Exact geom size, body mass and inertia agree with SWM, including endpoints.
2. The source agent-material dirty predicate is inverted (`allclose` marks dirty).
   An agent-only update can change MJCF without compiling. State therefore stores
   both compiled and pending MJCF configuration. Restoring only the latest Variation
   would silently change physical history. This branch has a dedicated oracle test.

Source info generation is delegated. The adapter removes upstream goal keys and
adds registered `env_name`, `target`, `success`. Scene button transitions, damping,
and forwarding run upstream; a narrow post-adjustment applies lock colors. Cube
post-step likewise delegates first, then applies registered cube/target color.

## Complete public State

State is a named numeric TensorDict with batch `[B]`: **139 leaves for Cube, 132
for Scene**. Every snapshot owns its tensor storage. No arbitrary objects or
pickles appear. The complete enumerated keys, shapes and dtypes are in
[state-schema.json](semantic_recovery_evidence/ogb_cpu_compatibility/state-schema.json).
All shapes below have a leading batch dimension in public State.

| Group/leaf | Cube shape | Scene shape | Type / purpose |
|---|---|---|---|
| `compatibility.signature` | 32 | 32 | uint8 SHA256 of schema/family/OGBench/MuJoCo/NumPy/Gym profile |
| `physics.time` | 1 | 1 | float64 integration time |
| `physics.qpos` | 21 | 25 | float64 generalized position |
| `physics.qvel`, `warmstart`, `qfrc_applied` | 20 each | 24 each | float64 official integration fields |
| `physics.ctrl` | 7 | 7 | float64 controls |
| `physics.xfrc_applied` | 156 | 228 | float64 body-force array, flattened |
| `physics.eq_active` | 3 | 3 | bool equality activation |
| `physics.mocap_pos`, `mocap_quat` | 3, 4 | 3, 4 | float64 goal mocap |
| `controller.site_xpos` | 1×3 | 5×3 | float64 pinch; Scene also drawer/window handles and targets |
| `controller.site_xmat` | 1×9 | 3×9 | float64 pinch; Scene also drawer/window handle rotations |
| `output.gripper_force` | 6 | 6 | float64 right-pad external force, read by immediate observation |
| `output.prev_qpos`, `prev_qvel` | 21, 20 | 25, 24 | float64 immediate info caches |
| `task.goal_observation` | 28 | 40 | float64 registered goal output |
| `task.id`, `reward_id` | 1 each | 1 each | int64; -1 represents None; valid concrete IDs 1..5 |
| `task.success` | 1 | 1 | bool source success cache |
| `lifecycle.reset_next_step` | 1 | 1 | bool, false for every initialized public checkpoint |
| `task.payload.init_xyz`, `goal_xyz` | 1×3 each | — | float64 Cube custom/predefined reset payload |
| `task.payload.{init,goal}_block_xyzs` | — | 1×3 each | float64 Scene payload |
| `task.payload.{init,goal}_button_states` | — | 2 each | int64 Scene payload bits |
| `task.payload.{init,goal}_{drawer,window}_pos` | — | 1 each | float64 Scene payload |
| `scene.{cur,prev,target}_button_states` | — | 2 each | int64 transition/target bits |
| `scene.target_{drawer,window}_pos` | — | 1 each | float64 target configuration |

The official integration union is asserted against `mjSTATE_INTEGRATION`:
TIME, QPOS, QVEL, ACT, HISTORY, WARMSTART, CTRL, QFRC_APPLIED, XFRC_APPLIED,
EQ_ACTIVE, MOCAP_POS, MOCAP_QUAT, USERDATA, PLUGIN. Nonempty components above
are stored using official get/set APIs. ACT/HISTORY/USERDATA/PLUGIN are empty in
these models; they are omitted, not guessed from another family.

Variation current values and persistent reset anchors are separate float64 leaves
under `variation.current` and `variation.anchor`, using literal dotted leaf names:

| Leaf | Shape | Cube | Scene |
|---|---|---|---|
| `agent.color` | 3 | yes | yes |
| `agent.ee_start_position` | 3 | yes | yes |
| `camera.angle_delta` | 1×2 | yes | yes |
| `cube.color` | 1×3 | yes | yes |
| `cube.size` | 1 | yes | yes |
| `cube.start_position`, `cube.goal_position` | 1×2 each | yes | excluded |
| `cube.start_yaw`, `cube.goal_yaw` | 1 each | yes | excluded |
| `floor.color` | 2×3 | yes | yes |
| `light.intensity` | 1 | yes | yes |
| `lock_color` | 2×3 | no | yes |

`model.compiled` and `model.mjcf` each contain float64 values for `cube.size`,
`agent.color`, `floor.color`, `camera.angle_delta`, `light.intensity` with these
same shapes. Model cube size records the actual effective geometry value, including
float32 endpoint rounding. `set_variation` changes persistent anchors; reset-scoped
options change that reset's values without changing anchors.

No diagnostic task name, root seed, last-seed metadata, temporary IK workspace,
or all-body force cache is stored. Their values are not read by the supported
future paths or immediate outputs. The action Box is eagerly materialized, so
there is no uninitialized RNG flag to restore. Controller-visible cached site
poses and right-pad force are retained because the source reads them before
recomputation. Warmstart and cached-pinch negative controls prove behavioral
relevance; preserving the complete State restores exact continuation.

## RNG ownership and encoding

Design A is independently re-proven on these upstream-derived production classes.
`SeedSequence(seed).spawn(3)` allocates action, Variation parent, and default-task
streams. The genuine memoized upstream Gym Box is seeded with the action child's
uint64 output. Default task selection uses a per-instance legacy MT19937 seeded
from the task child's uint32 output; the selected ID goes through upstream
`options['task_id']`. Gym's existing reset path owns the environment PCG64.
Cube's dropped-seed bug is repaired through that path; Scene forwarding is retained.

Every required PCG64 has four numeric leaves: 16-byte uint8 little-endian `state`,
16-byte uint8 little-endian odd `increment`, int64 `[1]` `has_uint32`, int64 `[1]`
`uinteger` in 0..2^32-1. The task stream has int64 `[624]` key, int64 `[1]` position
0..624, int64 `[1]` Gaussian flag, float64 `[1]` cached Gaussian. Both cache fields
are preserved, even when their flags are unset.

There are 20 PCG64 owners plus one MT19937 owner in Cube and 17 plus one in Scene:
environment, action, Variation parent, every required Dict node and active Box
leaf. `state-schema.json` enumerates their exact names. All Variation node seed
allocation uses Gym's original sorted tree. Scene keeps four inactive start/goal
leaves privately only to preserve that allocation; they are not sampled or exposed
as supported controls/State. Each reset overwrites their irrelevant private RNGs.

Ownership is installed before metadata/checker/compilation can trigger reset.
Properties retain the same genuine action Box. Supported reset/step/restore paths
do not mutate global NumPy, use a process-wide monkeypatch, or install a production
mock. Tests observe real sampling/stepping and use scoped failure injection only.
Arbitrarily wide integer seeds are supported and tested at `2**300 + 17`.

## Validation and atomic restore

All lanes and all nested containers/leaves are checked before simulator mutation:
exact batch/keys/shapes/dtypes, finite values, dependency/family signature, lifecycle,
nonnegative time, unit object/mocap quaternions, orthonormal cached rotations,
valid task/reward IDs and consistent predefined payloads, supported Variation/model
ranges, Scene bits/targets and RNG encoding constraints. Custom task arrays follow
the source's NumPy-array requirement; malformed Python-list payloads fail before reset.

Restore then performs:

1. Construct temporary independent upstream-based instances only after validation.
2. Build/compile the stored physical configuration, restore pending MJCF values,
   then restore current Variation and persistent anchors.
3. Restore task, reward, lifecycle, goal, and immediate previous-position/velocity fields.
4. Restore Scene button/target values and target-site positions; use upstream
   `_apply_button_states` for damping/forward, followed by registered colors.
5. Restore target visibility and registered cube colors without `post_step`.
6. Apply every official integration component; run `mj_forward`; reapply every
   integration component, preserving warmstart and other values forwarding can alter.
7. Reapply controller-visible site positions/orientations and right-pad output force.
8. Restore every owned RNG last.
9. Compare every resulting public State leaf exactly to the requested State.
10. Swap the entire batch only after all candidates succeed, then close old instances.

No reset, simulation step, RNG-consuming helper or button-toggling `post_step` runs
in restore. Constructor seed allocation creates temporary generators without
sampling them. A failure in the second lane's compilation leaves both original
lanes unchanged. Compact RenderState and partially unpacked RenderState cannot
restore dynamics. RenderState conversion only packs qpos, goal mocap and Scene
button bits as float32; it is not an acceptance claim for rendering.

## CPU acceptance evidence

The reproducible runner is
[run_checks.py](semantic_recovery_evidence/ogb_cpu_compatibility/run_checks.py).
It records exact argv, environment, directory, return codes and elapsed time in
[commands.json](semantic_recovery_evidence/ogb_cpu_compatibility/commands.json).
The terminal record and every per-case result are in
[cpu-tests.log](semantic_recovery_evidence/ogb_cpu_compatibility/cpu-tests.log).
Run from the recorded worktree:

```bash
.venv/bin/python docs/research/semantic_recovery_evidence/ogb_cpu_compatibility/run_checks.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. \
  /private/tmp/ogb-manip-authority-swm-lock/bin/python \
  docs/research/semantic_recovery_evidence/ogb_cpu_compatibility/audit.py
```

The physics command uses the authority interpreter and appends the recovery
`.venv/lib/python3.12/site-packages` only at the end for pytest. It does not replace
the authority physics dependencies. The exact authority profile is Python 3.12.11,
OGBench 1.2.1, MuJoCo 3.8.1, NumPy 2.2.6, Gymnasium 1.2.3, dm-control 1.0.41,
Torch 2.10.0, TensorDict 0.14.2 on macOS arm64.

**Profile limit:** the existing `ogbench` renderer extra still requires MuJoCo
`>=3.11,<3.13`. That range and other families' dependency requirements were not
changed. This checkpoint certifies exact CPU parity on the pinned SWM-lock profile;
it does not certify numerical equality on the renderer-install profile. A later
checkpoint must reconcile/test that profile before promotion. There is no GPU result.

| Gate | Result |
|---|---|
| Cube + Scene CPU suite | 86 passed; no skipped CPU cases |
| Shared State/harness/readiness/matrix/repository contracts | 52 passed |
| OGB family structure | 42 passed, 1 skipped, 13 deselected |
| Focused final unsupported-mode/custom-payload guards | 2 passed, 84 deselected |
| Ruff, format, source/documentation whitespace | PASS |
| Strict documentation build | PASS |
| Source/accepted-evidence preservation and ownership/schema audit | PASS |

The family-structure skip is the existing optional-dependency probe: the general
recovery `.venv` has no OGBench installed. The authority-profile CPU suite has no skips. The 13
deselections are renderer/lifecycle/dataset tests from the older rendering suite;
CPU lifecycle and state sampling refusal are explicitly covered by the new suite.
They are not presented as renderer acceptance. Test logs report zero EGL contexts
and zero renderers. Warnings concern optional ALE absence, source Box precision
casts and oracle Variation logging; the authority import also reports duplicate
SDL Objective-C classes from existing OpenCV/Pygame wheels. No renderer was created.

The 86 CPU cases cover:

- Both families × seeds 0/17/42 × default, explicit task IDs 1..5 and custom payloads;
  exact engine integration, output observations/info, model effects, and RNG equality.
- Reward-task IDs 0 (source alias to 2), 1 and 5; explicit/custom routes do not consume
  the default task stream. Actual reward/success/termination/truncation are compared.
- Fixed and sampled start/size/appearance Variation, canonical child allocation,
  persistent versus reset-scoped anchors, float32/float64/list size endpoints,
  recompilation/no-recompilation source branch.
- Float32 and float64 action bases, both signs, simultaneous and out-of-range
  components; upstream clipping/scaling/translation/yaw/gripper, 25 substeps.
- Complete batched independent restore, same-instance replay, different-history
  continuation, six-step and 30-step continuations, immutable snapshots, seeded and
  unseeded reset replay, malformed leaves and late-compilation atomic failure.
- Metadata/checker independence, interleaved forward/reverse order, four independent
  threaded instances, global NumPy nonmutation, all six genuine two-action settling
  tapes (Cube/Scene × seeds 0/17/42).
- Source success threshold neighbors, all Scene button patterns, and 101 raw steps
  with no invented 100-step World/TimeLimit truncation.

Each contact case runs 50 real source-matched steps; restoration at step 20 gives
30 exact continued steps, plus eight steps of same-instance replay. Measured
contact witnesses from the final log:

| Interaction | Evidence |
|---|---|
| Cube grasp/lift | Both gripper pads contact object; cube z 0.018936382303856303 to 0.17033701934243456 |
| Scene button 0 | Both pads contact button; joint minimum -0.028317843141690682; button state transitions |
| Scene drawer | Pad/coupler contacts; joint -0.012282001062037843 to 0.003344676371038167 (interaction, not a full-open task claim) |
| Scene window | Both pads contact; joint -0.00053202441303267 to 0.1774994133384465 |

Removing warmstart in the recorded Cube contact history changes next qpos by
`1.3010426069826053e-18` (an exact-bit witness, not a large physical displacement).
Perturbing Scene warmstart in its contact history changes next qpos by
`0.00014860199580035705`. Changing cached pinch position also changes next qpos.
Keeping these fields reproduces exact State, reward, success, done and continuation.

Development evidence is retained, including failures. Raw pytest logs preserve
their original trailing whitespace; the source/documentation whitespace check
excludes `*.log` artifacts rather than rewriting captured output. Initial execution lacked
pytest in the authority interpreter; the final bootstrap fixes only test discovery.
The 57-case development run had 8 failures (six invalid source EE Python lists,
two nested CompositeSpec errors). The next 65-case run had 7 failures (six
float32-size promotion mismatches and an insensitive Cube warmstart control).
Focused size and State regressions subsequently passed 6 cases each. The structural
run initially had one obsolete manipulation base-class assertion; only the two
manipulation assertions were updated. Maze expectations were untouched. The
pre-final 86-case full pass is also retained, before explicit dataset refusal and
custom-payload guards; final full results are separate. Temporary exploratory
contact searches informed the encoded tests but are not acceptance artifacts.

## Preservation, unsupported modes and next review

The closure audit compares all 2,026 starting tracked files. Only three existing
files are permitted to change: `pyproject.toml` (OGBench pin), the family package
registration wiring, and manipulation expectations in `test_envs_ogbench.py`.
The other **2,023 files are byte-identical**, including Maze, Fetch, DMC, classic
control, custom environments, shared State/RenderState/SimulatorAdapter contracts,
frozen renderer, matrix/map, prior documentation and research evidence.

All **170 accepted artifact hashes**, **11 exact terminal logs**, **75 Fetch
manifest entries**, **27 RNG-feasibility entries**, **15 dependency-audit entries**,
and **485 SWM/upstream source files** match. The oracle HEAD and tracked content
are unchanged. All 33 accepted registrations remain closed on their existing
evidence; no new GPU regression is claimed. There is no shared code change or
request for a shared-contract exception.

Unsupported historical APIs/modes are data collection, non-single Cube variants,
arbitrary constructor modes, pixels/multiview dataset collection, historical raw
`options['state']`, `render_goal`, convenience target setters as a public API, and
historical dataset `sample_states`. Scene's four inactive Cube start/goal controls
are rejected. Public custom task payloads must match the source numeric shapes and
NumPy array types. Closed simulator lifecycle and full-State compatibility checks
are explicit.

Remaining gates: CPU checkpoint review; renderer contract/integration and profile
compatibility review; authorized Oscar GPU/renderer correctness and regression
acceptance. Frozen renderer changes, GPU execution and promotion are outside this
checkpoint. The matrix deliberately still describes Cube/Scene as INCOMPLETE;
its adapter fields are not updated until a separately authorized status change.

No stable-worldmodel modification, production SWM import, upstream OGBench fork,
shared redesign, other-family recovery, GPU job, promotion or push occurred.
