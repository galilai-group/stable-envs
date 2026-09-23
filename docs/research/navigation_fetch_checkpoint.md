# Navigation acceptance and representative Fetch checkpoint

The requested Navigation stopping gate has completed. The integration matrix is
**21 VERIFIED / 21 INCOMPLETE**; DMC remains **12 / 12 historical defaults VERIFIED**
within its existing bounded scopes. Only the three Navigation registrations are
newly promoted. Fetch is not promoted.

## Navigation closed

Correction commit: `6ee8d8cb54031ff61eb1de24eafd1dd5c066fc3e`.
The family-local Piecewise lookup now divides its float64 offset by a float64
tensor denominator before scaling and truncating. This preserves the historical
zone selection on CUDA. No epsilon, coordinate special case, changed zone,
Variation, State, tolerance or shared parity change was introduced.

Full Navigation CPU gate and frozen baseline/candidate CPU rendering pass. The
expanded gate adds **42 boundary cases per profile**, covering x/y internal zone
planes, both sides, their intersection and the immediate float32 neighbours of
112. Each case runs eight continuation steps; batches 1/3 use distinct lane
positions. These accompany the original 18 trajectory cells (seeds 0/17/42,
batches 1/3, three domains), 32 steps and eight-step different-history tests.
All reported errors are exactly zero.

Complete Oscar job **6642897** ran immutable stage
`navigation-zone-fix-6ee8d8cb`, archive SHA-256
`185da82949590fb01ed45741f5a04b0c15ee6f1f62b0c80febb387581e976d00`.
The existing versioned runner/driver and source archive hashes were rechecked
before submission. The temporary VPN interruption was resolved before upload and
submission; no partial upload was used for acceptance.

The real terminal log contains both required markers:

```text
FAMILY_FROZEN_RENDERER_GATE_PASS navigation-zone-fix-6ee8d8cb 3 cells
FAMILY_GPU_GATE_PASS navigation-zone-fix-6ee8d8cb
```

Complete CUDA semantic JSON passes TwoRoom, Piecewise and SimplePointMaze,
strict reward/termination boundaries, 257 steps without source truncation,
meaningful physical Variation, snapshot/restore checks, and full-State/private
render equality with non-mutation. The failing Piecewise boundary now advances
exactly to [112,52]. All 42 expanded boundary continuations pass on CUDA too.
SimplePointMaze's cached goal image matches exactly under Oscar Matplotlib 3.11.1;
its distinct reward goal, dtype flag, blocked float64 action and float32/float64
nextafter success thresholds pass. CPU uses Matplotlib 3.10.9; comparisons are
within each profile, not a claim of cross-version pixel identity.

All baseline/candidate frozen renderer JSON files are byte-identical within CPU
and CUDA profiles. Each accepted class has four frozen rows:

| Class | CUDA frame SHA-256 |
| --- | --- |
| TwoRoom | ad6bde3489ecbf5a64173d71ceb442ef4ca27daf34d05910d03c32ed76eb62ab |
| Piecewise | 7b9c71849136dd8517cfc8e8f6f71352ac2a34ec19fef548aa9ce27bbb7a91a3 |
| SimplePointMaze | dc43fc1a7be5151af0880d2f21bced79a99a97ad609e61dbc2b0f8a75e36166d |

Promoted exactly `swm/TwoRoom-v1`, `swm/Piecewise-v0` and
`swm/SimplePointMaze-v0`, with execution `NATIVE_TORCH`. TwoRoomTarget,
PiecewiseZones, nondefault task kwargs and the full historical Variation surface
remain outside acceptance. Upstream unseeded reset draws fresh external entropy;
no restored-unseeded-reset replay claim is made. SimplePointMaze's frozen former
hidden-goal rows retain their private historical rendering projection.

The [acceptance manifest](semantic_recovery_evidence/navigation_zone_acceptance/acceptance.json)
contains immutable provenance, dependency profiles, raw log/JSON and checksums.
Earlier failed diagnostics remain unchanged.

## Representative Fetch evidence

Authority remains the unmodified pinned SWM source and direct upstream profile:
Python 3.12.11, NumPy 2.2.6, Gymnasium 1.2.3, Gymnasium Robotics 1.4.2 and MuJoCo
3.8.1. Only `swm/FetchPickAndPlace-v3`, mapped to sparse
`FetchPickAndPlace-v4`, is investigated. No sibling implementation or acceptance
is inferred. New bounded worktree `/private/tmp/stable-envs-fetch-proof`, branch
`repair/fetch-pickandplace-proof`, started from reviewed integration `c9eeec46`.

Source experiment commit `0381906f9916cba2fe42001714d37263731d7747` was
integrated as `5c887a44766d58e67dc459c80e095355200fc0aa`. The important new
State evidence is an explicit seven-component **controller cache**: gripper-link
`data.xpos[3]` and `data.xquat[4]` left by the final MuJoCo step. Forwarding the
restored final qpos recomputes different values; the next Fetch mocap controller
reads them. Restoring these supported named arrays after `mj_forward` recovers
exact continuation. This fits the existing State/SimulatorAdapter design.

Six source-only cases (seeds 0/17/42, float32/float64 actions) pass 32 steps after
restoration into a simulator with a different seed and eleven different actions.
The uninterrupted source is the oracle. Thirty-nine independent omission cases
cover thirteen groups and eight steps each. No opaque serialization or tolerance
is used. Evidence supports:

- qpos[22], qvel[21], goal[3]: omissions change public outputs.
- warmstart[21]: omission changes public proprio; the seed-zero first-step witness
  is 7.058087432865617e-18, retained as a failure under exact equality.
- controller position[3] and quaternion[4]: recomputation instead of restoration
  changes first-step public proprio by 0.0005323878641405599 and
  9.878135664997956e-7 respectively in the seed-zero probes.
- elapsed TimeLimit: omission loses truncation at steps 50/51.
- explicit PCG64 state: omission changes object/goal sampling on unseeded reset.

Mocap and control omission continuations are exact in the tested histories;
public control overwrites them. Applied forces are zero in both histories, so
these probes do not establish behavior for arbitrary nonzero external forces.
Time omission changes the omitted clock only; no independent public-output
necessity witness was found. The source experiment's superset must not be
mistaken for a final minimal schema. Its RNG replay disables visual variation;
SWM Variation-space randomness is not yet certified.

The [source proof report](fetch_representative_state_experiments.md) and raw
artifacts preserve each omission and the precise restore order.

## Private Fetch adapter checkpoint

Implementation commit `68c59316f1e637b2384bdd612d44243c682815a6` was integrated
as `d4bdf75f`; stricter output dtype/shape and oracle provenance assertions were
integrated as `3986fd3de17f1337c2147b1397ff308e07a5eeb9` from `c4d62d38`.
The private adapter runs an independent upstream CPU environment per lane,
accurately classified as `DELEGATED_CPU`, with no production SWM import.
Public Fetch registrations are not yet wired to it.

The **provisional**, bounded default State is qpos[22], qvel[21], warmstart[21],
controller.position[3], controller.quaternion[4], goal[3] (all float64), plus
elapsed_steps[1] and named PCG64 state/increment words[4 each], has_uint32[1]
and uinteger[1] (int64). Simulator RNG omission has a reset witness; necessity
of each cached-integer RNG component remains unproved. Time, mocap, ctrl and
applied forces are excluded within the tested supported-operation scope, not
certified for arbitrary force injection or untested Variation. There is no
final approved Fetch family schema yet.

The adapter-only probe passes **12 cases**: seeds 0/17/42, batches 1/3 and
float32/float64 actions. Each compares 32 steps and eight continuation steps after
candidate-only restoration into eleven steps of different history. Exact output
values, shapes and dtypes, all candidate State fields, seeded reset, repeated
seeded reset, independent snapshots and State roundtrip pass. Actions include
-1/+1, simultaneous XYZ/gripper commands and out-of-range values exercising
upstream clipping. Reward/success match on these trajectories; dedicated
threshold tests remain outstanding.

Unseeded simulator-reset RNG replay passes with visual variations disabled.
TimeLimit parity passes at 49/50/51, including restoration at 49 and continued
steps after truncation. Compact render rows are refused for restoration.
Physical/visual Variation has not passed a candidate gate: the adapter's mass
configuration method is preparatory only. Full rendering is also unaccepted.

The [adapter report](fetch_representative_adapter_initial.md) records the complete
provisional schema, scope and raw evidence. These are private adapter results,
not completed public-environment acceptance.

## Fetch renderer constraint

The existing Fetch module requires MuJoCo >=3.11 and MJWarp at import, while the
simulator authority here is 3.8.1. More fundamentally, its full mover renderer
has **no CPU backend**: `FetchBase._render` invokes MJWarp, and
`tests/unit/test_envs_fetch.py` explicitly records `CPU_BACKEND_MISSING` for the
18 mesh mover pass. Existing `render/cpu3d.py` supports analytic primitives,
not these meshes. Only the static plate can render locally; that is not a full
Fetch frame or frozen-render acceptance.

No dependency patch, alternate renderer, source monkeypatch, process-level
redesign or fixture rewrite was introduced to hide this limitation. Frozen full
CPU renderer preservation cannot be claimed from a plate comparison. This is an
existing renderer/profile limitation, not evidence that the shared State
contract cannot represent Fetch semantics.

The full representative gate is unfinished: public environment wiring, final
minimal State certification, reward-threshold candidate tests, physical/visual
Variation, full/private rendering equality, render non-mutation and frozen full
renderer preservation remain outstanding. No Fetch row is promoted.

## Preserved scope

PushT remains parked with its supported-interface blocker and C3 evidence intact.
No other simulator family or final benchmark was started. DMC was not reworked.
The shared State/RenderState, SimulatorAdapter, parity, taxonomy and promotion
contracts remain unchanged. Eight readiness tests pass. All 42 historical IDs,
exactly three new promotions, immutable audit hash, old evidence, product branch
and renderer baseline pointers are preserved. The containing commit is the final
integration checkpoint.
