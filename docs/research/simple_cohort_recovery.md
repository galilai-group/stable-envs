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
