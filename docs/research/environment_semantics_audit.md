# Environment semantics: Phase 0 source audit

This audit covers all **42 historical SWM registrations** in the frozen support matrix:
**30 WRONG_ABSTRACTION, 12 INCOMPLETE, 0 VERIFIED**. It does not certify runtime parity.
The 12 incomplete rows comprise 10 registrations with native movement implementations
and two symbolic Craftax registrations with no mapped replacement class.

The machine-readable companion is [environment_semantics_audit.json](environment_semantics_audit.json).
It records each registration's observation, simulator restoration interface, current public
State and RenderState, missing fields, actions, reset/RNG, reward/done/task semantics,
variation declarations, source references, dependency versions, and open questions.
References are resolved through its `sources` table, with file hashes and symbol line numbers.
Shared variation declarations are resolved through `variation_catalog`; expressions remain
expressions when they depend on runtime model construction. They are not invented numeric defaults.

## Pinned scope and baseline

| Item | Pinned value |
| --- | --- |
| Renderer baseline | `6d80b68b8b3ccd07d4207be94cbf4f0760758a06` (`v0-draft`) |
| Local baseline pointer | `baseline/renderer-v0` |
| Audit branch | `repair/environment-semantics` |
| Audit worktree | `/private/tmp/stable-envs-semantic-recovery` |
| SWM oracle | `176d54532b3c0a251a4f509b74f2fdacc143a63d` |
| SWM source checkout | `/Users/samib/stable-worldmodel` |
| Starting working directory | `sami/pre-step6-baseline`, `89fa5ca81341dc4df7186ec325cb691431a2ef17` |

The starting working directory was not the clean v0 product described by the brief.
The audit branches from the separate v0 worktree. Existing checkouts and their untracked
files were left intact. The SWM checkout had no tracked changes; its unrelated untracked
files were not used as oracle source.

Counts are **historical registration counts**, not unique renderer classes. Some registrations
map to several renderer variants; continuous and discrete PushT share one renderer.
The frozen crosswalk is evidence about renderer coverage, not authority for task identity.

| Family | Rows | WRONG_ABSTRACTION | INCOMPLETE | VERIFIED | Current implementation |
| --- | ---: | ---: | ---: | ---: | --- |
| DM Control | 12 | 12 | 0 | 0 | Renderer only |
| Fetch | 12 | 12 | 0 | 0 | Renderer only |
| OGBench | 3 | 3 | 0 | 0 | Renderer only |
| Craftax pixels | 2 | 2 | 0 | 0 | Renderer only |
| Craftax symbolic | 2 | 0 | 2 | 0 | No mapped class |
| Rocket | 1 | 1 | 0 | 0 | Renderer only |
| Classic control | 5 | 0 | 5 | 0 | Native motion; replacement contract incomplete |
| PushT | 2 | 0 | 2 | 0 | Native approximate motion; actions/task mismatch |
| Navigation | 3 | 0 | 3 | 0 | Native motion; reward/done/reset contract incomplete |
| **Total** | **42** | **30** | **12** | **0** | **No certified replacement execution** |

`NATIVE` and `DELEGATED` are successful execution labels in the brief, which requires parity
before awarding them. Consequently `execution` is null for the incomplete rows and
`execution_candidate` records the proposed implementation mechanism separately. Null is
explained, not an unreported measurement. The 30 renderer projections are `RENDER_ONLY`.

## Findings that affect the shared contract

1. **The actual get/set API differs from the proposed restoration API.**
   [Env.get_state](../../stable_envs/env.py) returns the packed float32 renderer row.
   `env.state` returns the TensorDict and can include live velocity fields.
   Classic control's `set_state` zeros omitted velocities. Thus
   `set_state(get_state())` is not a full dynamic-state roundtrip, even for families
   whose live TensorDict contains velocities. This requires an explicit shared API decision.

2. **RenderState decoding is unsafe as an implicit restore contract.**
   `unpack_render_state` returns a State-shaped TensorDict with omitted dynamics fields
   zero-filled. Classic control can transition such a value. Preserve compact renderer
   layouts, but define how callers distinguish a render reconstruction from a complete
   restorable snapshot. Do not silently infer zero velocities from missing evidence.

3. **All ten native-mapped registrations omit reference reward/done outputs.**
   [Classic control](../../stable_envs/envs/classic_control.py),
   [PushT](../../stable_envs/envs/pusht.py), and
   [navigation](../../stable_envs/envs/navigation.py) return state-only StepResults.
   Their source explicitly documents this choice. Existing equation or trajectory evidence
   does not certify SWM task/reward/horizon behavior.

4. **Several renderer identities are the wrong task identities.**
   SWM Humanoid defaults to `walk`; the crosswalk names `DMCHumanoidStand`.
   SWM Acrobot uses `Balance(sparse=True)`, while the renderer declares `swingup`.
   SWM Cartpole uses sparse swingup, while the crosswalk offers balance and dense swingup.
   Cheetah, Hopper and Walker use SWM custom task classes. A shared XML model or matching
   picture does not resolve these differences.

5. **Reference reset is not renderer state sampling.**
   The base stable-envs reset calls `sample_states`. DMC sampling resets upstream and
   advances 0–39 random steps before capturing the render row. Fetch uses a sampled
   rollout-state pool. Those are useful renderer corpora, not the reference initial-state
   distribution. Reset options and variation sampling order need separate parity tests.

6. **Lifecycle and RNG are observable state.**
   DMC's internal `_step_count` and `_reset_next_step` can trigger an implicit reset while
   SWM continues to return `truncated=False`; subsequent behavior then consumes task RNG.
   Fetch and classic control have inner Gymnasium TimeLimits. Rocket carries termination
   and truncation latches, fuel and noisy actuators. Craftax has both a wrapper PRNG key
   and the game state's `state_rng`.

7. **Some oracle quirks must be preserved or explicitly resolved.**
   SWM Cube accepts `seed` but does not forward it to its parent reset; OGBench task
   selection also uses global NumPy RNG. SWM Scene's `reset(options.state)` omits the
   button kwargs required by its own `set_state`, a source-level failing path needing
   live reproduction. SimplePointMaze's cached goal image and reward goal come from
   different values. A replacement must not silently fix these while claiming parity.

## Family appendix

### DM Control: qpos is insufficient, but qpos + qvel is not yet a complete resume proof

SWM's base wrapper explicitly restores `physics.data.qpos` and `physics.data.qvel` and
exports both in info. Stable-envs declares `qpos` plus optional episode render slots and
zeros `qvel` when reconstructing the reference scene for rendering. This establishes
WRONG_ABSTRACTION without needing a pixel experiment.

One public action is normalized to `[-1,1]`, cast to the reference action dtype, scaled
to upstream controls, and repeated twice. Reward is the sum of both rewards, treating
None as zero. Default tasks return `terminated=False`, `truncated=False`; the Reacher
`qpos_match` option has a special early-termination override. Cumulative reward affects
`info.score` and belongs in the explicit bookkeeping contract if info parity is claimed.

For Reacher, task target geometry is episode-persistent configuration. Preserve that
alongside qpos/qvel. Include lifecycle and RNG needed to reproduce internal time-limit
resets. Actuator, mocap and integration-state requirements beyond these remain candidates
for the two-history experiment; this audit does not prescribe copying every solver cache.
Runtime nq/nv/nu measurements remain blocked by the import failure below.

### Fetch: twelve task/observation variants, four physical scenes

SWM's `-v3` names instantiate Gymnasium Robotics `-v4` IDs. Each scene has sparse,
dense and sparse-Dict variants. Flat observations concatenate proprioceptive observation
and desired goal; they are not full MuJoCo snapshots. State currently contains only qpos
and goal. qvel and the inner TimeLimit counter are absent; mocap/control/integration
requirements need restoration tests. Fixed goal and physical model configuration can
live in persistent Variation if restoration supplies them explicitly.

Actions are four bounded controls: xyz delta scaled by 0.05 and gripper command, with
fixed end-effector rotation. There are 20 physics substeps. Sparse reward is negative
failure indicator; dense reward is negative distance. Success uses strict `< threshold`,
whereas sparse failure uses `> threshold`. Termination is false; the default inner
TimeLimit truncates at 50 steps. The wrapper applies visual defaults and object mass,
and active start/goal variation keys change physical reset state.

### OGBench: task and controller state accompany physics

The default maze is point/maze/large with state observations. Point control adds
`0.2 * action` to qpos, zeros qvel and runs five MuJoCo substeps. This detail means
velocity importance must be assessed against the exact target transition and info,
not assumed from the family name. Cube and Scene use normalized xyz/yaw/gripper
deltas, inverse kinematics and 25 default physics substeps. Target clipping occurs
after unnormalization; the controller does not explicitly clip the incoming action.

Manipulation renderer rows contain qpos and mocap positions; Scene adds two button
states. Task goals, reward mode, success timing, RNG, qvel and controller/integration
requirements are not certified. Default reward is success indicator and termination
occurs at success; `reward_task_id` changes reward semantics. SWM manipulation classes
implement task/initialization logic themselves, so merely instantiating an upstream
OGBench class would not prove wrapper equivalence.

### Craftax: preserve the complete game PyTree and the carried key

The JSON lists the exact upstream EnvState, Inventory, Mobs, EnvParams and StaticEnvParams
field names from installed Craftax 1.6.1. Full-state restoration needs the complete
PyTree and wrapper `_key`; local visible tiles and display inventory cannot encode
offscreen worlds, mob health/cooldowns, achievements, plants and timers.

SWM uses `auto_reset=False`, ignores reset options and declares no variation space.
Each step splits the wrapper key and calls the authoritative JAX environment once.
Reward combines achievement increments and health change; the full game weights
achievement increments. Upstream max-timestep exhaustion is reported as `terminated`,
with SWM `truncated=False`. The two symbolic registrations have no mapped class and
must not disappear from the replacement inventory because they produce no RGB frame.

### Rocket: SWM task logic plus PyFlyt dynamics

Current State is 13D attitude/velocity/position. Default SWM observation is 30D:
13 attitude entries, seven actions, nine auxiliary values and a contact flag. Neither
representation is a proven simulator snapshot. Booster source explicitly maintains fuel
ratio, ignition and throttle and consumes RNG during noisy throttle updates.

SWM subclasses RocketBaseEnv and owns its reset/reward/landing logic. Defaults include
40 Hz actions, three Aviary steps per action, fuel ratio 0.05, accelerated drop at
z velocity -30, and constrained horizontal spawn offset. Successful landing sets
`truncation` and `env_complete`; it is not a terminal-success flag. The timeout tests
`step_count > max_steps` before increment. Preserve those exact boundary conventions.
Bullet snapshots used for goal rendering are process-local and do not establish a
portable public State contract for Python-side actuator/RNG/bookkeeping state.

### PushT: both action variants need distinct semantics

SWM continuous actions are relative offsets scaled by 100 from current agent position.
The discrete registration first decodes `MultiDiscrete([16,16])` polar bin centers.
Stable-envs takes absolute targets in `[0,512]`, and the shared renderer mapping provides
no discrete adapter. SWM reward is negative full 7D distance to sampled goal_state;
success uses position and wrapped-angle thresholds. This differs from both a state-only
StepResult and gym_pusht coverage reward.

SWM's seven observation entries include agent velocity but omit block velocity. Its
`_set_state` advances physics by one dt and does not restore a complete Pymunk snapshot.
The two-history test must determine whether block/contact state must be exported; the
historical seven-vector cannot be declared sufficient just because SWM calls it state.
The existing fast approximate solver must retain its own declared accuracy claim.

### Classic control and navigation: motion is only one part of parity

Classic-control live TensorDicts already include physical velocities; Acrobot's internal
angles remain distinct from its trigonometric observation. Still missing are reset/restore
parity, reward/done outputs and inner TimeLimit state. CartPole additionally tracks
post-termination rewards. Native discrete actions floor/clamp floats, whereas reference
Gymnasium actions are validated discrete values. Valid-action parity and invalid-input
behavior should be deliberate parts of the contract.

TwoRoom and Piecewise return reward 0 and terminate within 16 units of the persistent
target. SimplePointMaze has `[-0.2,0.2]` controls, reward 1 at goal and -0.01 otherwise.
Its implementation also updates `variation_space.agent.position` on every step; that
leaf is dynamic state despite its location in the reference variation object. Native
movement ports do not yet implement these full task/reset/info contracts.

## Evidence limits and next gate

A CPU-only probe attempted to import/construct all 12 pinned SWM DMC wrappers using
`/Users/samib/se6-venv/bin/python`. Every attempt stopped at SWM's import dependency
`ModuleNotFoundError: No module named 'lancedb'`; no constructor completed. The audit
records installed package versions as source-inspection profiles, not validated runtime
profiles. No simulator import was stubbed and no substitute reference was used.

The inspected Fetch profile also contains MuJoCo 3.12.0, outside the v0 `robotics` extra's
declared `mujoco>=3.11,<3.12` constraint. The package documents a Gymnasium Robotics 1.4.2
setter incompatibility. That failure was not reproduced here; establish a compatible
isolated Fetch profile before attempting parity.

Artifact checks matched all 42 IDs against both the frozen matrix and the pinned SWM
registration source, validated every required row field/reference, and checked all 113
recorded source hashes. These checks validate the audit's inventory and provenance,
not the environments' dynamics.

No parity harness, production fix, renderer regression test, GPU run, oracle corpus or
benchmark was implemented/executed in Phase 0. Historical renderer certification remains
historical evidence, not a new `REGRESSION = 0` result. Existing frozen fixtures and
production files are unchanged. GPU validation belongs on Oscar after inspecting its
documented workflows when that phase begins.

Before family implementation, review this audit and resolve the shared full-State get/set
API, lossy RenderState guard, explicit task/RNG/lifecycle bookkeeping, and the task-identity
mismatches. Then build the shared parity harness and run Reacher's reset, roundtrip,
one-step, multi-step, reward/done and two-history tests in a working pinned-reference
profile. Family branches and delegated implementation begin only after that contract.
