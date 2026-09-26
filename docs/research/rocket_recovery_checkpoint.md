# Rocket recovery checkpoint: rejected-control replay blocker

**39 VERIFIED / 3 INCOMPLETE. Rocket is not promoted.** The frozen renderer
preflight passed. Recovery stops for review because the proposed bounded replay
recipe does not cover the authoritative side effects of rejected controls.
No shared State/RenderState or renderer change was made; PushT was untouched.

## Exact starting checkpoint and worktree

Starting SHA and the commit containing `branch_replay_checkpoint.md`:
`3d0a799d329bb00fbecbf708e970be18a14ce611`.
Branch: `repair/environment-semantics`.
Worktree: `/private/tmp/stable-envs-semantic-recovery`.
The original start was clean. The previous tool authorization failure is resolved.
On resumption the temporary worktree's Git link and 528 tracked files were
missing. Git repaired the link, and only absent tracked files were restored from
that exact commit. Existing files and pending work were not overwritten.
[Repair inventory](semantic_recovery_evidence/rocket_resume_20260926/missing-tracked-files.json)
and [verification](semantic_recovery_evidence/rocket_resume_20260926/preservation-after-repair.json).

The previous 180-file sequence added 179 files totaling **2,591,455 bytes**;
the index added 601 net bytes. Net tree growth: 2,592,056 bytes. The read-only
single-branch pack comparison measured **606,809 incremental bytes**.
[Exact measurement](semantic_recovery_evidence/rocket_production_start/checkpoint-and-growth.json).
No history rewrite, deletion, repack or push was performed.

## Smallest decisive witness

Pinned SWM source, seed 17, default constructor, float64 actions. Each action
below raises `AssertionError` because gimbal-x is 2.0. Both histories contain
three rejected calls, and both finish with exactly the same control:

| Call | History A: fin-x | History B: fin-x | Other controls |
|---|---:|---:|---|
| 1 | 0.5 | 0 | `[0, 0, 0, 0, 2, 0]` |
| 2 | -0.4583333333333333 | 0 | `[0, 0, 0, 0, 2, 0]` |
| 3 | 0 | 0 | `[0, 0, 0, 0, 2, 0]` |

All **91 audited numeric fields** are exactly equal afterwards, including
actuator values, contacts, main RNG, physical pose/velocity, cached links,
action, reward and lifecycle fields. Public step count remains 0; physics count
is 20 and Aviary count is 10, unchanged from reset. Yet the next valid all-zero
action differs by **0.0004015837397325629** in the observation and
**0.0005432316374118784** in reward.

The mechanism is visible in official PyFlyt: `Aviary.step` calls
`Rocket.update_physics` before `stepSimulation` and before incrementing counters.
`update_physics` applies body and lifting-surface forces before validating the
gimbal and booster. An assertion therefore leaves accumulated Bullet forces
pending. The paired controls return the fin actuator values to zero but leave
different pending forces. A rejected-call counter and final caches do not
identify the history. Rejecting these controls before the engine call would
change this source behavior.

The default `max_steps + 2 = 1202` bound still covers successful physics-advancing
public calls. It does **not** bound side-effecting rejected calls: **4,096** were
executed without incrementing public/physics/Aviary counters or reaching a
terminal state. Arbitrary rejected prefixes cannot simply be appended to the
same 1,202-row history under a claim that the source guarantees the bound.

The [27-case gate](semantic_recovery_evidence/rocket_rejected_controls_gate/result.json)
covers ignition, throttle and gimbal rejection; seeds 0/17/42; float32/float64
for one-versus-two rejected calls; and equal-count cancellation witnesses for
float64. The migrated private task matches the source before and after these
calls and the next valid action. The consolidated
[equal-count NPZ](semantic_recovery_evidence/rocket_rejected_controls_gate/equal-count-witness.npz)
contains both action prefixes and both immediate/future numeric states.

This refutes the proposed recipe and the counter/final-cache shortcut for this
case. It does not prove that every conceivable bounded numeric force-restoration
scheme is impossible. No such alternative has been implemented or certified.
A review decision is required on restorable post-rejection behavior, or a further
family-local proof of a sufficient numeric pending-force representation. The
source behavior and error boundary have been preserved rather than changed to
make the gate pass.

## Implementation and source boundary

Exactly two production-tree files were added:

- `stable_envs/_adapters/rocket_task.py`: bounded SWM-local task migration,
  optional reset-anchor capture/injection hooks, and owned temporary-file cleanup.
- `stable_envs/_adapters/rocket_spaces.py`: Rocket-only source-compatible
  Box/RGBBox/Dict semantics, preserving the Python float default height ratio
  and source sampling order. Shared spaces are unchanged.

They form an **unregistered private candidate**, with no public State adapter or
changed registration/factory. No stable-worldmodel production import exists.
The root-level `THIRD_PARTY_NOTICES.md` accompanies the migration.
The source boundary is SWM commit
`176d54532b3c0a251a4f509b74f2fdacc143a63d`, task SHA256
`5ede3640ce1e2add32b6b0a6c36e190bb1130e38bb258ab182af12a6fbedee23`;
official **PyFlyt 0.29.0 / PyBullet 3.2.7**, NumPy 2.5.3, Gymnasium 1.3.0.
No experimental Bullet build or native monkeypatch is used. Source-file hashes
are recorded in the gate JSON. Production dependency-extra packaging is deferred
with the public adapter; no supported installation/promotion claim is made.

The [private task parity gate](semantic_recovery_evidence/rocket_task_layer_candidate/result.json)
passes **24 cases**: seeds 0/17/42, both action dtypes, four Variation selections,
reset and three mixed-control steps, all 91 numeric fields exact. This limited
migration check is not the requested full production semantic acceptance suite.

## Reset classification, Variation and goal image

[Reset metadata](reset_replay_classification.json) preserves all existing accepted
claims. Rocket's source requirement is
`SEEDED_REPLAYABLE_FRESH_UNSEEDED_ENTROPY`. Exact continuation must still restore
all owned RNG. No new shared-contract structure or protocol was required.

The [source reset gate](semantic_recovery_evidence/rocket_reset_boundary_source/result.json)
passes seeds 0/17/42 in `None` and `rgb_array` modes. Seeded observation, all-selected
Variation, RNG and goal-image reconstruction are exact. Consecutive unseeded
resets change all nine Variation leaf RNGs and values while retaining the owned
main RNG object. Those nine leaves include the source-unused sky color; this
is resampling evidence, not proof that sky changes pixels.

Both tested goal-image modes have **691,200 raw bytes** at 480×480×3. Their exact
reconstruction in this bounded matrix does not establish every required
render/Variation case. The cache has not been removed. No production State
schema, fixed ignition-state dtype, State byte count or serialized production
State size was frozen. The historical **767,077-byte numeric / 807,407-byte NPZ**
research candidate remains preserved and explicitly non-production.

## Oscar renderer preflight

Job **6721727**, `3090-gcondo` / `cs-3090-gcondo`, RTX 3090 `gpu2109`:
**COMPLETED, 0:0, 00:01:24**. Exact marker: `RENDERER_PREFLIGHT_PASS`.
[Raw scheduler evidence](semantic_recovery_evidence/rocket_renderer_preflight_gpu_6721727/scheduler-terminal.txt),
[results](semantic_recovery_evidence/rocket_renderer_preflight_gpu_6721727/results/result.json),
[archive/source hashes](semantic_recovery_evidence/rocket_renderer_preflight_submission/UPLOAD-SHA256SUMS).
The downloaded evidence archive SHA256 is
`2583d7798db6fe9942585c89d0167a21d8f7745d33060acf8d2851b4ec357228`.
Only one job was submitted; no duplicate active job or abandoned GPU job remains.

All **102 frames / 17 groups** meet the existing frozen visual-policy limits:
mean-absolute ≤10, within-eight ≥0.91, SSIM ≥0.80. Worst group measurements were
9.493016493055556, 0.9488802083333333 and 0.9360060152818963 respectively.
Singleton/batch frames match exactly and renderer inputs are unchanged.
Coverage includes reset, early/near-contact fall, crash/post-terminal,
success/post-success, and all-selected Variation with mixed controls.
Frozen renderer source and historical baselines remain byte-identical. This is
not full-State versus RenderState or restore-frame production acceptance; no
public production adapter exists. The earlier measured reset link-position
mismatch did not fail the visual policy.

## Deferred acceptance and performance

The blocker prevents freezing the full State recipe, atomic restore, public
wiring and production CPU/renderer acceptance. Maximum-horizon restoration,
all lifecycle/reward boundaries, resource ownership under failed restore,
per-leaf/persistent Variation and practical batching remain uncertified.
No Rocket restore timings at reset/25%/50%/75%/maximum/contact/success, B16 costs,
branch costs or rendered end-to-end production timings are claimed. Existing
performance reports are unchanged; no historical measurement was overwritten.
The GPU job's elapsed time above is execution provenance, not a benchmark result.

Shared State, semantic harness, readiness/taxonomy/promotion, repository,
registry and Rocket renderer math checks: **95 passed, 2 skipped**. The earlier
broader Variation/legacy renderer run had 253 passed, 61 skipped and 112 failures;
the same 112 test IDs failed on an extracted original starting commit. No broad
suite pass is claimed. Ruff, formatting, whitespace and strict docs are checked
again for this closing checkpoint.

All **39 accepted registrations**, **208 accepted artifact hashes**, **112 existing
production hashes**, matrix/map history, prior sealed evidence and terminal logs
are preserved. All original tracked files except the append-only research index
remain unchanged. [Closing verification](semantic_recovery_evidence/rocket_recovery_checkpoint/verification.json).
[Complete changed-file inventory](semantic_recovery_evidence/rocket_recovery_checkpoint/changed-files.json)
and the adjacent repository SHA256 seal record every file in this sequence.
No push occurred. Work stops here for review of the rejected-control blocker.

## Commits before this closing report

- `906bf85f6f6ea1341622eacf9d7674acfb4c507c`: docs: clarify reset entropy guarantees and seal Rocket starting checkpoint
- `3e24542c4453e5682d12481d6a458144e049d60f`: research: preserve Rocket source renderer preflight and pending handoff
- `b61cb7b42bc2dbfc4f9ba4980dcd0d1e55f869b3`: research: verify Rocket source reset entropy and goal reconstruction boundary
- `6a95e2825692884b919e4ab61098838f474aba03`: rocket: preserve source-derived private task candidate without public registration
- `70a99be9f249a9a9048e026a600da94f39aa9476`: rocket: preserve exact rejected-control counterexample to bounded replay
- `4c57219ad1a097286e519accb6a0a5a795adf1f0`: rocket: record passing Oscar renderer preflight and repaired checkpoint preservation

The closing report/seal commit follows these six commits; its exact SHA is
given directly in the final handoff. No push occurred.
