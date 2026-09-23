# PushT family checkpoint: stopped at Phase A

Both `swm/PushT-v1` and `swm/PushT-Discrete-v1` remain **INCOMPLETE**.
Final matrix: **33 VERIFIED / 9 INCOMPLETE**. No implementation or promotion was
attempted after the contact-restoration gate failed. Stop here for review.

Starting branch: `repair/environment-semantics`, clean worktree.
Starting SHA and exact recovered **Fetch GPU-promotion SHA**:
`b45500ca84d8c6b3216d90196acac747e8fb3d5b`.
Ending branch: `repair/environment-semantics`.
Evidence commit: `6873b7d608cfe0fb7777fb35680e859e2160f511`.
The second commit contains this checkpoint; its exact SHA is reported in the final
handoff (a commit cannot embed its own hash). No accepted commit was amended.
No push was performed. No other family work or GPU job was started.

Only the new `semantic_recovery_evidence/pusht_phase_a/` directory and this
checkpoint changed. Production/shared files, tests, matrix/map, SWM production,
frozen renderer definitions and fixtures are unchanged. No shared-contract change
is requested. No public exhaustive State schema was finalized.

## Gate result and concrete blocker

The fresh experiment uses the pinned SWM checkout at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`, Pymunk **7.2.0**, NumPy **2.2.6**,
Gymnasium **1.2.3**, Python **3.12.11**, on macOS arm64. Pymunk release commit is
`3ef8f6e304fdc09e16c8ca2cfb1a3aa88a9399bc`, underlying Munk2D is
`ade7ed72849e60289eefb7a41e79ae6322fefaf3` (runtime engine
`2.0.1-ade7ed72849e60289eefb7a41e79ae6322fefaf3`). This corrects the dependency
profile of the earlier exploratory evidence, which used Pymunk 7.3.0.

All 12 candidate roundtrips agree exactly before continuation. Free-space and
independently constructed same-history controls agree for all four actions.
Different-history candidate restoration fails both isolated contact cases and
the combined case. The untouched source is never restored or rebuilt.

The smallest isolated wall witness produces block-y error
`0.5523790245538862` and reward error approximately `0.25934770324983` after one
zero action, despite matching candidate body/task/RNG fields. A stronger
combined agent/block + wall witness still differs after restoring every inspected
numeric arbiter/contact field, space scalars and live membership via a target-only
detection step: first block-x error `0.010771303883643668`, with divergence in
all four subsequent steps. That canonicalization attempt therefore does not
match the original source continuation. It is not an acceptable restore path.

The engine source confirms persistent body bias velocities beyond normal
velocity, opaque body/space structures, and active/cached contact bookkeeping.
The fresh combined-contact numeric cache fields match, but cached traversal
order differs. The experiment does not isolate which missing engine mechanism
causes each discrepancy. No safe complete restoration implementation has been
established; this is a bounded blocker, not an impossibility proof. The supported
`Space.copy()` path also fails before comparison while copying a pygame Surface
through the SWM callback graph on this profile. It is not fresh numerical parity
evidence. Details, field lists/dtypes, exact inputs/outputs, failed attempts and
source excerpts are in the [diagnostic README](semantic_recovery_evidence/pusht_phase_a/README.md).

## Source audit retained for the next reviewed attempt

The following pinned SWM files were read in full: `envs/pusht/env.py`,
`envs/pusht/env_discrete.py`, `envs/quantizer.py`, `envs/pusht/expert_policy.py`,
`envs/__init__.py`, and `world/world.py`. Reset spaces and Pymunk/Munk2D code were
also inspected. `audit.json` records exact source hashes. These findings explain
future required behavior; they do not substitute for the blocked acceptance tests.

**State and ownership.** Historical SWM observation is seven float64 values:
agent xy, block xy, block angle modulo 2*pi, agent xy velocity. Proprio is agent
position/velocity. That observation omits block velocity/angular velocity and
solver state. It is not exhaustive. Backend restoration must include every
future-relevant field; the diagnostic candidate additionally tracks both bodies'
angle/velocity/force/torque/COG, task goal, render goal, RNG/Variation state and
contact/space counters. Bias velocities and opaque solver state remain missing.
The probe uses Python/NumPy records, not a production TensorDict serializer.
SWM keeps learner observations, historical info state, proprio and goal images.
No observation container or filtering was added to stable-envs.

**Reset.** `reset` first calls Gym reset, then creates `self.rng` from the supplied
seed. Variation is re-seeded/reset and its selected leaves sampled. `_setup`
builds fresh physics/geometry; constructor COG/damping overrides follow. Unless
explicit `options['goal_state']` is supplied, the reward goal separately samples
agent/block start-position and block-angle spaces with `set_value=False`, plus
the current Variation agent velocity. `_set_state(goal)` sets agent velocity,
agent position, then block angle before position and performs one physics step.
The raw seven-number goal is retained while its goal image is rendered from the
post-step scene. Episode start comes from `options['state']` or current Variation
values; a second `_set_state` performs another physics step in the **same space**.
Block velocity/contact history from goal placement is not explicitly cleared.

Gym's RNG persists on reset(None), but `self.rng=default_rng(None)` and Variation
re-seeding request fresh entropy. Copying existing generator states alone has
not proved unseeded reset replay. Current and initial Variation leaves, constructor
configuration and explicit reset options must remain distinct candidate anchors.
Seed 0 is exercised here; reset parity for seeds 17/42, reset-anchor and seeded/
unseeded replay are **NOT_RUN_PHASE_A_STOP**.

**Goal and reward.** `goal_state` is the raw reward/success target. `goal_pose`
comes separately from goal-position/angle Variation and draws the target ghost.
The cached goal image, reset-start configuration and reward goal are different
objects. Dense reward is negative Euclidean distance over all seven numbers,
including agent velocity and the raw goal angle versus wrapped current angle.
Success is position distance over the first four entries strictly below 20 and
`min(abs(goal_angle-current_angle), 2*pi-abs(...)) < pi/9`. That expression does
not explicitly modulo arbitrary angles, so an angle difference above 2*pi can
produce a negative second term. Raw terminated equals success and raw truncated
is false. Declared reward_range `(0,1)` disagrees with the actual negative reward.
No normalization/correction was introduced. Exact boundary/neighbour, malformed
and non-finite input acceptance tests remain **NOT_RUN_PHASE_A_STOP**.

**Lifecycle decision.** Both pinned PushT registrations omit `max_episode_steps`.
`World.__init__` defaults it to 100 and `_make_env` passes it to `gym.make`.
Thus this particular 100-step cap is SWM orchestration/future adapter behavior,
not a raw PushT horizon. Under the accepted separation of family simulator state
and SWM wrapper integration, a raw delegated PushT backend must preserve false
truncation rather than silently acquire that cap. No elapsed counter is needed
for this raw source behavior. Four-step raw continuation is observed; a full
lifecycle/public-backend gate is **NOT_RUN_PHASE_A_STOP**. This differs from Fetch,
whose accepted horizon is declared by the historical registrations themselves.

**Actions.** Continuous Box bounds are [-1,1], shape (2,), declared float32.
Relative target is current agent position + action * 100, with no explicit clip
in step. PD uses kp=100, kv=20, dt=.01 and ten Pymunk steps per public step.
Sequential source stepping does not restore State. `latest_action` is retained
for the optional action rendering; dtype-sensitive source arithmetic must be
preserved. The probe uses float32 and float64 actions but is not the complete
boundary, out-of-range, mutation or malformed-input gate.

**Discrete.** `PushTDiscrete` subclasses PushT and dequantizes before its step.
MultiDiscrete([16,16]) uses PolarQuantizer(16,16,max_distance=1). Centers are
r=(radial+.5)/16 and theta=(angular+.5)*2*pi/16, converted to float32 XY.
The lowest radial bin is nonzero. Quantize uses clipped radius indices and
atan2 angle modulo 2*pi; dequantize has no explicit index validation in source.
The expert policy is an external producer of normalized/quantized actions; its
RNG is not simulator RNG. Complete 256-center mapping, invalid dtype/shape/index
behavior and continuous/discrete State/reward/done/frame equivalence remain
**NOT_RUN_PHASE_A_STOP**. A future implementation must share the same backend.

**Variation.** Default sampled leaves are agent.start_position,
block.start_position and block.angle. Full tree sampling order is background,
goal, block, agent, rendering. Source routing is:

| Leaves/configuration | Source effect; remaining proof |
|---|---|
| Agent start position/velocity | Episode reset values; position also sampled independently for reward goal. Velocity contributes to goal and reward. Runtime Variation gate not run. |
| Agent shape/scale/angle | Collision geometry/body construction. Circle is kinematic and its helper ignores angle; noncircle helpers create dynamic bodies. Not appearance-only. Gate not run. |
| Block start position/angle | Episode placement and independently sampled goal components. Gate not run. |
| Block shape/scale | Actual collision geometry. Default T helper hardcodes scale 30 despite exposed scale Variation; other helpers use supplied scale. Gate not run. |
| Goal position/angle | Rendered ghost pose, distinct from sampled reward goal. Gate not run. |
| Goal scale | Source-confirmed no-op in goal drawing, which reuses block shapes. Gate not run. |
| Agent/block/goal color, background | Drawing configuration; non-mutation witnesses not run. |
| rendering.render_goal | Drawing toggle; non-mutation witness not run. |
| Constructor damping/block_cog | Physics-affecting configuration; gate not run. |

Default T mass is 1, COG (0,45), moment 3000 from computing both inertia terms
using the bar polygon. Default circle radius is 15; damping is 0, gravity 0.
Setting `body.friction=1` does not set Shape friction, which remains 0. Damping
zero does **not** justify omitting block velocity: position integration consumes
it before the next velocity update, and bias velocity is separate. None of these
historical behaviors was “fixed” or used to excuse the failed restoration.

## Acceptance and preservation

| Gate | Result |
|---|---|
| Free-space restoration | Exact bounded four-action witnesses |
| Same-history independent environment | Exact bounded four-action controls |
| Different-history contact restoration | **FAIL** |
| Target contact reconstruction + numeric cache | **FAIL** for combined contact against untouched source |
| Final exhaustive State / public roundtrip | **NOT_PROVEN**; candidate roundtrip is insufficient |
| Reset/RNG replay, actions, reward boundaries, lifecycle, Variation | Source audit + limited Phase A use only; full gates **NOT_RUN_PHASE_A_STOP** |
| Discrete quantizer / registration equivalence | **NOT_RUN_PHASE_A_STOP** |
| CPU rendering, full/private projection, snapshot/live non-mutation, compact refusal | **NOT_RUN_PHASE_A_STOP**; reset's upstream render call is not renderer acceptance |
| GPU rendering/frozen baseline | **NOT_RUN_PHASE_A_STOP**; no GPU availability claim or job submission |
| Shared readiness/matrix/State contract/harness | **42 passed**, one warning |
| Original 33 evidence closure | **PASS**: 170 distinct hashes, 11 terminal logs, 75 Fetch-manifest files |
| Formatting / diff check | **PASS** |

All executed test commands are in the diagnostic README. No GPU physics,
renderer parity, full family CPU acceptance or additional-family result is
claimed. The independent final inspection compared the changed file list against
the starting commit; it contains only the new evidence and this checkpoint.
No production implementation exists to review in this checkpoint.

Evidence root: `docs/research/semantic_recovery_evidence/pusht_phase_a/`.
SHA256 values:

- `results.json`: `33ed794fa5f316130080f5c5baa60a84ea50cc47ab0a0ba30ea6ef10bdc42234`
- `audit.json`: `a50b740a47ff1090eb14f440ddf41dcec953d08765d869243b2a31c321b83b46`
- `SHA256SUMS`: `fdc64f70d2f379d008b38d33f4aa0fcf4339ad751a51b3158249061c064615fb`

No newly promoted registrations. Remaining incomplete:
`swm/PushT-v1`, `swm/PushT-Discrete-v1`, `swm/CraftaxClassicPixels-v1`,
`swm/CraftaxClassicSymbolic-v1`, `swm/CraftaxPixels-v1`, `swm/CraftaxSymbolic-v1`,
`swm/OGBCube-v0`, `swm/OGBScene-v0`, `swm/PFRocketLanding-v0`.

The next reviewed decision concerns whether to pursue an engine-supported,
safely serializable restoration mechanism for PushT. This checkpoint does not
authorize or propose shared abstraction changes, weakened tolerances, or promotion
from body-only State. **Shared-contract change requested: no. Push performed: no.**
