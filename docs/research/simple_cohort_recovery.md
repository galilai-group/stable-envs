# Remaining simple/native and source-aligned DMC cohort

This is one bounded CPU recovery cohort, opened from reviewed integration commit
`152cb2402bd44063f414865faff6d28e4ccb67c2`. It does not change the shared State,
SimulatorAdapter or parity contracts. No registration is promoted by opening it.

| Slice | Exact historical defaults in scope | Worktree branch |
| --- | --- | --- |
| Remaining classic control | AcrobotControl-v1, MountainCarControl-v0, MountainCarContinuousControl-v0 | audit/native-envs |
| Navigation | TwoRoom-v1, Piecewise-v0, SimplePointMaze-v0 | repair/native-navigation |
| Source-aligned DMC | BallInCupDMControl-v0, FingerDMControl-v0, ManipulatorDMControl-v0, QuadrupedDMControl-v0 | repair/dmcontrol |

All identifiers above have the historical `swm/` prefix. The exact DMC tasks are
BallInCup, Finger Turn Hard, Manipulator Bring Ball, and Quadruped Walk. Nearby
variants and Humanoid's unresolved Walk/Stand identity are outside this cohort.
PushT remains separate. Fetch, OGBench manipulation, Craftax and Rocket are not
started. No paired benchmark work is included.

Each registration must pass the established pinned-source exact gate: complete
State/output leaf comparisons, seeded reset, independent snapshots,
candidate-only restoration versus an uninterrupted reference, different-history
replay, one/32-step outputs, reward/done/lifecycle/RNG, meaningful Variation,
full/private render equality and non-mutation, and lossy-restoration refusal.
Hidden State fields require task-specific evidence. No tolerance or opaque
serialization is introduced. Native CPU or authoritative delegated CPU execution
must be labeled accurately. CUDA placement/rendering never implies GPU physics.

The coordinator owns legacy API tests, renderer-preservation drivers, recovery
matrix and merges. Family worktrees own only their assigned implementations,
additive independent oracles, bounded gates/tests and reports. Historical fixtures,
audit and performance evidence remain immutable. SimplePointMaze's goal-visibility
identity correction must preserve the historical renderer through a private
reference translation rather than keeping the wrong public default.

The preceding five CPU-passed registrations are still awaiting Oscar jobs
`6605941` and `6605999`. At this cohort's opening both are `PENDING (QOSGrpGRES)`;
no completion logs exist. Their staged archives and existing remote drivers are
left unchanged. Inspect actual completed gates and frame hashes before closing
that acceptance cohort. New cohort work does not confer acceptance on those rows.

The integration branch is `repair/environment-semantics`. Product `v0-draft` and
`baseline/renderer-v0` stay at their previously recorded commits.

## Remaining classic-control CPU checkpoint

Implementation `b3b6da79555020a3fe9ae046c3f6b4a2bef8a153` and its report
`3bebf6275a506746314d15e3f6fab26e16edbe05` are integrated only on the repair branch.
The independent coordinator gate passed exactly; all three frozen CPU renderer
cells match the immutable baseline. Combined current API, full-State, prior-two
and new-three semantic suites passed **138 tests**. Readiness checks passed 8 tests.

Oscar job `6608718` tests the immutable implementation archive, CUDA State and
rendering, and the prior CartPole/Pendulum gate. Its separately versioned driver
and runner leave both older queued jobs unchanged. Hashes and acceptance
requirements are in `semantic_recovery_evidence/simple_cohort_oscar_pending.json`.
All three rows remain `INCOMPLETE / CPU_PASSED_OSCAR_PENDING`; accurate execution
metadata is `DELEGATED_CPU`. No acceptance is inferred from CPU evidence.

## Source-aligned DMC CPU checkpoint

Immutable implementation `e4d5c2cd99d035fc57d84dd5eab1ea70e9b26941` is integrated
on the repair branch. Independent coordinator execution of the complete CPU gate
passed, including positive sparse rewards, field-specific omission experiments,
and physical Variation divergence from default with exact varied-reference parity.
Combined DMC API, adapter, oracle and integration checks passed **128 tests** with
three existing Swimmer CPU skips. The four frozen CPU renderer cells match baseline.

Oscar job `6610032` gates the immutable source archive, all four CUDA State/render
cells, Reacher, the first DMC family, and the previous DMC continuation cohort.
All four new rows remain `INCOMPLETE / CPU_PASSED_OSCAR_PENDING`.
