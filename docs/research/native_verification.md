# Native-environment semantic verification

This Track-B checkpoint audits all ten historical registrations that already had a native movement
candidate, and implements the first two exact historical defaults. The source oracle is the tracked,
clean checkout of `stable-worldmodel` at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`. The implementation checkpoint is
`446f8608bc800c2d61701f675a2536df0dcbaa3f` on `audit/native-envs`.

The shared Phase-0.5 contract required no change. Full State remains a complete independent
`TensorDict`; compact float32 renderer rows remain private and cannot restore dynamics. The recovery
matrix is coordinator-owned and is unchanged here. CartPole and Pendulum are CPU-gate-passed
candidates for `VERIFIED`; that status is awarded only after the coordinator's GPU/state-placement
and renderer-preservation acceptance. The other eight rows below remain unverified.

## Source audit of all native historical registrations

| Historical registration | Pinned reference identity and defaults | Finding at this checkpoint | Required correction or proof |
| --- | --- | --- | --- |
| `swm/CartPoleControl-v1` | `CartPoleWrapper` -> Gymnasium `CartPole-v1`; Discrete(2); 500-step `TimeLimit`; SWM applies every float32 physics default on reset | Exact default implemented in canonical `CartPole` as `DELEGATED_CPU`; compact `[x, theta]` renderer row retained | CPU exact gate passed. Await coordinator GPU/state-placement and renderer-preservation acceptance. |
| `swm/PendulumControl-v1` | `PendulumWrapper` -> Gymnasium `Pendulum-v1`; Box(1); 200-step `TimeLimit`; pre-step cost reward | Exact default implemented in canonical `Pendulum` as `DELEGATED_CPU`; compact `[theta, torque]` renderer row retained | CPU exact gate passed. Await coordinator GPU/state-placement and renderer-preservation acceptance. |
| `swm/AcrobotControl-v1` | `AcrobotWrapper` -> Gymnasium `Acrobot-v1`; Discrete(3); reward -1 until terminal then 0; 500-step `TimeLimit` | Existing `Acrobot` floors/clamps float actions, omits task outputs/lifecycle/RNG, and has only near-exact torch RK4 evidence | Extend the same explicit Gymnasium adapter pattern with physical state, PCG64, elapsed steps, strict discrete actions, reward/done, all seven float32 physics defaults, reset options, and exact boundary tests. Do not infer certification from the recovered DMC Acrobot task. |
| `swm/MountainCarControl-v0` | `MountainCarWrapper` -> `MountainCar-v0`; Discrete(3); reward -1; 200-step `TimeLimit` | Existing `MountainCar` omits reward/done/lifecycle/RNG and accepts floored/clamped floats | Add complete PCG64/TimeLimit State, strict integer actions, exact reset and terminal boundary, and float32 `gravity`/`force`/`max_speed` Variation. |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuousWrapper` -> `MountainCarContinuous-v0`; Box(1); terminal bonus minus action cost; 999-step `TimeLimit` | Existing native path has a known first-step dtype discontinuity and omits task/lifecycle/RNG | Delegate or reproduce Gymnasium's state-dtype transition exactly; preserve action clipping and cost from the original action, PCG64, elapsed steps, terminal rule, and float32 `power`/`max_speed` Variation. |
| `swm/PushT-v1` | SWM Pymunk `PushT(relative=True)`; Box(2) `[-1,1]`; target is `agent.position + 100*action`; ten `dt=.01` substeps | Existing `PushT` takes absolute `[0,512]` targets, uses an approximate solver, omits the 7D goal task/reward/done, and has no proven Pymunk resume State | Build a distinct continuous semantic adapter; recover complete body/contact continuation State experimentally; reproduce reset's goal construction and two `_set_state` physics advances; return full observation/info numerics, `-||goal-state||`, success, and `truncated=False`. |
| `swm/PushT-Discrete-v1` | `PushTDiscrete`; `MultiDiscrete([16,16])`; polar bin-centre decode followed by the same relative PushT controller | Both historical PushT IDs currently map to one continuous stable class; discrete decoding is absent | Keep a distinct public action contract. Decode `r=(r_bin+.5)/16`, `theta=(theta_bin+.5)/16*2pi`, emit float32 `(r*cos(theta), r*sin(theta))`, then use the exact continuous relative controller. The continuous gate does not certify this row. |
| `swm/TwoRoom-v1` | `TwoRoomEnv(render_target=False)`; clipped Box(2), speed-scaled motion, wall/door collisions; reward 0; success distance `<16` | Existing `TwoRoom` matches the default renderer identity but omits persistent target/task output, reset option/RNG proof, reward and termination | Put target and sampled geometry/speed outside or inside State according to restoration experiments, preserve variation sampling order and `state`/`target_state` options, then gate door-edge collisions and strict success boundary. `TwoRoomTarget` is a separate visual variant and gains no certification. |
| `swm/Piecewise-v0` | `PiecewiseEnv(render_target=False, grid_n=2)`; zone-specific bias after speed-scaled clipped action; reward 0; success distance `<16` | Existing `Piecewise` omits target/task output and reset/RNG proof | Preserve target, four zone biases, grid geometry and speed as persistent Variation; prove zone-boundary selection, clamping, reset options and strict success. `PiecewiseZones` is not the registered default. |
| `swm/SimplePointMaze-v0` | `SimplePointMazeEnv(show_goal=True)`; Box(2) `[-.2,.2]`; wall rejection then bounds clamp; reward 1 or -.01 | Existing canonical renderer uses `show_goal=False`; reset caches a goal image sampled at a random agent location while reward uses `goal.position`; dynamic `agent.position` is also written into the variation tree | Correct the default visual identity first. Preserve the two distinct goal notions, constructor-seeded wall defaults, variation draw order, collision inclusivity, dynamic agent position, reward and strict goal-radius termination. Do not silently make the cached goal image equal the reward goal. |

The audit is based on the actual registration table and wrapper sources under
`stable_worldmodel/envs/{gymnasium_control,pusht,two_room,piecewise}` and
`stable_worldmodel/envs/simple_point_maze.py`, plus installed Gymnasium 1.3.0. It does not promote a
row based on renderer similarity or an existing equation test.

## Exact recovered identities and execution

`CartPole` now represents the exact default `swm/CartPoleControl-v1` task, and `Pendulum` represents
the exact default `swm/PendulumControl-v1` task. Both hold one independent Gymnasium environment per
batch row. Physics always executes on CPU; `device="cuda"` controls State/output placement and the
existing renderer, and makes no GPU-physics claim. Production imports Gymnasium directly and never
imports `stable_worldmodel`.

Delegation was selected from measured evidence rather than convenience. Starting the retained torch
equations from identical pinned-wrapper State and applying SWM's float32 physics defaults produced
nonzero last-bit drift without tolerance:

| Probe | Exact steps out of 32 | Largest observed difference |
| --- | ---: | ---: |
| CartPole seed 17, alternating actions | 13 | theta `8.33e-17`; theta-dot `4.44e-16` |
| CartPole seed 42, alternating actions | 31 | theta-dot `5.55e-17` |
| Pendulum seed 17, seeded float32 torques | 30 | theta-dot `4.44e-16` |

Other sampled seeds happened to be bit-identical. That state dependence is observed evidence; the
gate does not claim a single unmeasured cause. Exact delegation removes the difference and preserves
Gymnasium lifecycle/action behavior without introducing tolerance. The former torch equations remain
as renderer-era provenance but are no longer reachable through public `step`; `transition`, `unroll`,
`sim_state` and `set_sim_state` explicitly refuse rather than expose the incomplete old model.

## Full State schemas

CartPole State contains:

```text
x, x_dot, theta, theta_dot                         float64[1]
lifecycle.elapsed_steps                            int64[1]
task.steps_beyond_terminated                       int64[1]
task.has_steps_beyond_terminated                   bool[1]
rng.state, rng.increment                           int64[4] uint32 limbs
rng.has_uint32, rng.uinteger                       int64[1]
```

Pendulum State contains:

```text
theta, theta_dot, torque                           float64[1]
task.has_last_u                                    bool[1]
lifecycle.elapsed_steps                            int64[1]
rng.state, rng.increment                           int64[4] uint32 limbs
rng.has_uint32, rng.uinteger                       int64[1]
```

The presence bits distinguish Gymnasium's `None` bookkeeping from numeric zero. Pendulum torque is
float64 in full State because Gymnasium accepts float64 arrays with a warning and retains their
precision in `last_u`; float64 also represents declared float32 actions losslessly. PCG64's two
unsigned 128-bit integers are four explicit base-2^32 limbs each rather than an opaque serialization.
Physical Variation is persistent configuration outside State and must match when resuming.

The exact restoration experiments include an independent snapshot mutation test, candidate-only
advance/restore while the reference remains uninterrupted, two candidates arriving from different
histories, continuation after restoring a near-TimeLimit snapshot, and a precise float64 Pendulum
torque restore. Raw two-column render rows and incomplete dictionaries are rejected. Full-State
rendering projects through the unchanged compact row and does not mutate the simulator.

## CPU gate evidence

`scripts/recovery/native_classic_gate.py` asserts the pinned commit and clean tracked reference
checkout, then runs seeds 0, 17 and 42 at batches 1 and 3. Every State leaf and numeric output is
compared exactly. The gate exercised:

- exact reset distributions and subsequent unseeded resets through restored PCG64 State;
- one-step and 32-step trajectories, rewards, observations and numeric `info.state`;
- CartPole termination and post-termination rewards `[1, 0, 0]`;
- CartPole truncation first at step 500 and Pendulum truncation first at step 200, plus post-truncation
  continuation;
- Pendulum's exact upright reward 0 and a nonzero negative cost;
- strict CartPole integer action rejection, Pendulum clipping, and accepted precise float64 torque;
- all six CartPole and all six Pendulum physical Variation leaves at nondefault values;
- full-State/private-render equality, render non-mutation, and compact/incomplete restoration refusal.

The independent frozen oracle under `tests/fixtures/semantics/native_classic/` was generated only by
the real pinned SWM wrappers. Its manifest records versions, array schemas and SHA-256 hashes; the
integration test verifies those hashes before replay. No older fixture or provenance file changed.

CPU validation at the implementation checkpoint:

```text
python -m pytest -q tests/unit/test_native_classic_semantics.py \
  tests/integration/test_native_classic_gate.py
8 passed

python -m ruff check <owned Python files>
All checks passed
```

The coordinator separately reported 119 affected classic/navigation/full-State checks passing
against this implementation in its integration harness. GPU state-placement and renderer-preservation
checks belong on Oscar; physics remains delegated CPU execution there as well.

## Remaining blockers and next corrections

PushT is the highest-risk native family. Action conversion can be corrected directly, but an exact
claim also needs a two-history experiment during active contact. Candidate State fields must begin
with both bodies' position, angle, linear/angular velocity and explicit controller/task bookkeeping,
then add solver continuation fields only when the experiment proves them necessary. Pymunk's cached
contact impulses are not exposed as a stable portable State API. If recreating the space and rebuilding
contacts cannot reproduce the next step exactly, that is a concrete restoration blocker; it must not
be hidden inside an opaque process snapshot or bypassed by retaining the approximate solver. Continuous
and discrete actions require separate gates even if they share the same Pymunk adapter and renderer.

The three remaining classic-control registrations can use the bounded adapter pattern established
here, but each still needs its own State schema and lifecycle/action/task proof. Acrobot's RK4 state
and discrete action semantics differ from both recovered rows; continuous MountainCar's dtype behavior
is specifically a restoration concern. They remain source-audited only.

Navigation has simple motion but nontrivial task configuration. TwoRoom and Piecewise need target,
geometry, speed/bias and reset sampling order reconciled with full State versus persistent Variation.
SimplePointMaze additionally has an actual default-identity mismatch (`show_goal=True`) and distinct
cached-image/reward goals. None of those rows inherits verification from its renderer variant.

