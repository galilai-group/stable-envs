# Public representative Fetch recovery

Only `swm/FetchPickAndPlace-v3` is wired to the authoritative delegated CPU
adapter. Eleven siblings remain renderer-only. No shared contract changed.
CPU physics uses Python 3.12.11, NumPy 2.2.6, Gymnasium 1.2.3, Robotics 1.4.2
and MuJoCo 3.8.1. Production never imports stable-worldmodel. Public rendering
requires CUDA/MJWarp; CPU full rendering explicitly raises CPU_BACKEND_MISSING.
The module no longer requires GPU renderer dependencies to import CPU semantics.

## Source trace and sufficient named State

Inspected pinned SWM `envs/gymnasium_robotics/fetch.py` and `spaces.py`, plus
Robotics 1.4.2 `envs/fetch/pick_and_place.py`, `fetch_env.py`, `robot_env.py`,
`utils/mujoco_utils.py`, and Gymnasium 1.2.3 `wrappers/common.py` TimeLimit.

| Component | Classification and source reason |
| --- | --- |
| qpos[22], qvel[21] | Persistent physical state consumed by integration and observation |
| time[1] | Inexpensive documented simulator clock; retain for sufficient roundtrip, despite time-independent task reward |
| warmstart[21] | Solver continuation; earlier exact omission witness retained |
| controller.position[3], quaternion[4] | Final-step gripper body caches read by mocap controller before recomputation; earlier necessity witnesses retained |
| goal[3] | Task state read by observation, sparse reward and success |
| task.initial_gripper_xpos[3] | Reset anchor; SWM writes XY before upstream reset and later object/goal sampling reads it; persists across resets |
| lifecycle.elapsed_steps[1] | Gymnasium TimeLimit truncates at 50 and allows subsequent steps |
| Complete named PCG64 | Atomic simulator/reset RNG, including state, increment and cached integer fields |
| mocap pose, ctrl | Public controller overwrites before physics; ctrl uses qpos, mocap uses retained gripper cache |
| applied force arrays | Zero under supported public actions; explicitly reset to zero during restoration |
| activation/userdata/plugin state | Model has none |
| physical mass | Persistent Variation; same Variation required across restoration |
| visual sampling RNG | Separate Gymnasium Dict tree; source reseeds on every reset, including fresh entropy for seed=None |
| visual material/camera arrays | Renderer-only; bounded object color does not alter physics |

Restoration validates the full State, writes qpos/qvel/time, goal and reset
anchor, clears external forces, calls mj_forward, then writes warmstart and the
controller cache. Finally it restores TimeLimit and complete PCG64. No opaque
serialization is public, and normal sequential step never restores the simulator.
Source-backed explicit fields suffice; scalar-by-scalar RNG minimality is not a
requirement. Existing private-adapter and omission artifacts remain untouched.

## Bounded CPU gate

`scripts/recovery/fetch_public_gate.py` uses independent pinned SWM extraction
and checks source provenance before execution. Twelve public cases (seeds
0/17/42, batches 1/3, float32/float64 actions) compare all named State fields and
numeric output values/shapes/dtypes: reset including mocap, independent snapshots,
roundtrip, 32 steps, eleven different-history steps then eight restored steps,
repeated seeded reset and unseeded simulator reset under fixed Variation.
Sequential step is checked to avoid calling adapter set_state. Compact tensor
and decoded renderer rows are refused by public set_state.

Constructed post-action object positions give distances exactly zero, the
immediate float64 neighbours of 0.05, exactly 0.05, and 0.2. Reward is 0 at the
threshold while success is 0; above it reward is -1, below it success is 1.
Terminated remains false. TimeLimit 49/50/51 and restoration at 49 match exactly.
Action probes include simultaneous XYZ/gripper, exact +/-1 and values beyond
those bounds, preserving upstream clipping/scaling/dtype order.

Variation scope is explicit object mass 8, combined agent/block start, block
angle and goal start, object color [0.2,0.8,0.3], and seeded object-color sampling.
Mass changes the source trajectory (maximum proprio difference
0.00025946916669317144), and the candidate matches exactly, including restored
continuation. Visual-only color gives zero physical difference. Start-anchor
writes precede upstream sampling; object/goal overrides and mass writes follow
reset; mj_forward follows changed placements, matching SWM ordering.

Stable-envs retains its persistent default renderer Variation. CPU semantic
parity uses SWM options variation=[] except explicitly certified variation
probes. SWM's default five-leaf appearance randomization and its full theoretical
Variation surface are not certified by this bounded promotion. Seeded object
color follows the same independent Gymnasium seed tree; unseeded source visual
sampling draws fresh external entropy, so no visual-RNG replay claim is made.

The initial gate attempt used Python lists for explicit source goal variation;
SWM retains that list and then fails goal_distance because it lacks shape. The
failed log is preserved. Gate inputs now use float64 NumPy arrays accepted by
the unmodified source; no dependency or oracle patch was made.

## Separate renderer acceptance

Canonical full-State tensors from the CPU authority are serialized in
`semantic_recovery_evidence/fetch_public/canonical_state.json`. The Oscar gate
renders these in a separate MuJoCo >=3.11/MJWarp profile, compares public/full
and private/projected frames, checks snapshot/simulator non-mutation, checks
object color changes frames, and compares unchanged frozen renderer rows against
baseline/renderer-v0 in separate processes. Physics under the renderer profile
is not a semantic authority claim. Acceptance remains pending until the actual
Oscar terminal marker and complete renderer JSON are recorded.

The expanded `cpu-v3.json` gate additionally restores the reset anchor from a
simulator with a different agent-start history, then resets without explicit
variation overrides. Per-reset mass overrides return to the configured initial
mass; persistent `set_variation(block.mass=8)` matches source `init_value` across
two seeded resets and eight steps each. This source-order distinction is now
implemented explicitly. The initial pending Oscar job 6646358 was cancelled
before execution; its revision remains preserved and is not acceptance evidence.
Legacy Fetch source checks pass (10); seven renderer-construction tests skip in
the local general test environment without Robotics. CPU semantic tests run in
the separate pinned authority profile and do not skip.
