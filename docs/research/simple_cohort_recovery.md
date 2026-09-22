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

## Native navigation CPU checkpoint

Immutable implementation `9fed5402631660c0764229182ed2a71c9a7a8e83` is integrated
on the repair branch. The final independent coordinator CPU gate passed all
18 seed/batch cells and the hidden-field and dtype-flag omission probes against
uninterrupted source outputs. Float16/32/64 actions, blocked dtype retention,
strict success thresholds, structural Variation divergence, and cached goal-image
bytes are covered. All three frozen CPU renderer cells match baseline.

SimplePointMaze now shows its reward goal by default. Its full State preserves
physical precision and the cached image's separate agent position. The historical
hidden-goal two-column renderer remains available only through the private
`_render_historical_rows` fixture translation. No historical fixture changed.
Its cached `info.goal` output requires the new `navigation` optional dependency
(`pip install -e '.[navigation]'`). CPU exact-byte evidence uses Matplotlib 3.10.9;
the queued Oscar profile has 3.11.1 and must pass its own exact source comparison.

Oscar job `6611139` gates the immutable navigation archive and a separately
versioned renderer driver/runner. None of the four older queued jobs was modified.
All three new navigation rows remain `INCOMPLETE / CPU_PASSED_OSCAR_PENDING`.

## Integrated validation and remaining acceptance

The combined CPU suite passed **304 tests**. It initially reported six skips:
three expected Swimmer CPU exclusions and three OGBench dependency skips. Running
the OGBench suite in its existing `/Users/samib/se7-ogb-venv` environment passed
all four tests (including the oracle hash check already counted in the main run).
Thus **307 distinct tests passed**, with only the three existing Swimmer CPU
exclusions remaining. Coverage includes Reacher, the earlier DMC families, the
earlier classic-control defaults, OGBench Maze, all ten new defaults, the shared
semantic harness, full-State/API contracts, and immutable oracle checks. The
maintained readiness inventory also passed its eight consistency checks.

The ten new defaults have exact CPU gates and byte-identical scoped frozen CPU
renderer comparisons. GPU acceptance is still open. At **2026-09-22 17:00 UTC**,
all five jobs were `PENDING (QOSGrpGRES)` with no execution or completion evidence:

| Job | Acceptance slice | Immutable source |
| --- | --- | --- |
| 6605941 | Earlier CartPole/Pendulum | `446f8608bc800c2d61701f675a2536df0dcbaa3f` |
| 6605999 | Earlier Cheetah/Hopper/Walker | `6470521e84a47d5f95a5bd5f0c5a338ab293619f` |
| 6608718 | Remaining classic control | `b3b6da79555020a3fe9ae046c3f6b4a2bef8a153` |
| 6610032 | Source-aligned DMC | `e4d5c2cd99d035fc57d84dd5eab1ea70e9b26941` |
| 6611139 | Native navigation | `9fed5402631660c0764229182ed2a71c9a7a8e83` |

Close the preceding five-registration cohort only after retrieving its actual
terminal `FAMILY_GPU_GATE_PASS` markers, exact semantic JSON and matching
baseline/candidate frozen renderer JSON. The DMC jobs additionally require their
Reacher and earlier-family regression outputs. Apply the same acceptance rule to
each new slice; job completion alone is insufficient. Immutable archive/driver
hashes and output requirements are recorded in the pending evidence files.

No registration was promoted: the semantic matrix remains **5 VERIFIED /
37 INCOMPLETE**, with 15 of those incomplete rows now CPU-passed/Oscar-pending.
PushT stays separate and unstarted in this cohort. Fetch, Craftax and Rocket were
not started. Humanoid Walk/Stand and the OGBench Scene constructor blocker remain
explicit. Shared contracts, the Phase 0 audit, historical render/performance
fixtures, `v0-draft`, and `baseline/renderer-v0` remain unchanged. No branch was pushed.
