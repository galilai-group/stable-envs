# Historical DMC continuation cohort

This bounded recovery covers exactly three registrations from pinned
stable-worldmodel commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`:

| Historical registration | Public environment | Constructed task |
| --- | --- | --- |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | `CustomCheetah(goal='run', move_speed=10)` |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | `CustomHopper(goal='hop', hopping=True)` |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | `CustomPlanarWalker(goal='walk', move_speed=1)` |

For these defaults, each custom reward method takes its stock superclass branch.
The candidate therefore constructs the matching stock task on the wrapper's
modified, recompiled MJCF model. A census after reset compares 585 numeric
`MjModel` arrays exactly for each domain. The task control timesteps and DMC step
limits are Cheetah 0.01/2500, Hopper 0.02/1000, and Walker 0.025/1000. The public
wrapper repeats every normalized float64 action twice and sums the two rewards.
Finite values outside `[-1, 1]` retain the historical pass-through behavior;
invalid shapes and nonfinite actions are rejected before mutation.

All numeric outputs are preserved. Besides qpos/qvel, Cheetah info contains
speed, torso and foot heights, and angular momentum; Hopper contains speed,
height, and angular momentum; Walker contains speed, torso height/uprightness,
both foot heights, and both thigh heights. Observation, reward, success, score,
termination, and truncation match the pinned wrapper exactly.

## Full State evidence

The starting restore interface was the wrapper's qpos/qvel setter. Experiments
then added only state needed for exact continuation. The final State is qpos,
qvel, simulation time, qacc warmstart, DMC step count/reset-next-step, cumulative
reward, and the full MT19937 state. Snapshots are independent copies.

Warmstart necessity was tested by saving a candidate snapshot, advancing only
the candidate through a different history, restoring every saved field except
warmstart, and comparing the next eight public outputs to the same uninterrupted
SWM reference outputs. This produced nonzero observable errors outside the
warmstart leaf: Cheetah `2.7533531010703882e-14` after a five-step seed-0 prefix,
Hopper `2.1316282072803006e-14` after a five-step seed-0 prefix, and Walker
`1.6897594434794883e-13` after a two-step seed-0 prefix. Restoring warmstart made
all three continuations exactly zero-error against those same expected outputs.

Prior control is omitted because the next public action overwrites it; the gate
checks that different histories can retain different control values while their
next eight outputs remain exact after State restore. qacc is derived by forward
and step. All three models have `na=0`, so there is no activation state. Task
goal/speed/hopping values are fixed construction configuration. The Cheetah and
Hopper `_timeout_progress` members are only assigned/reset in the pinned source
and are not read by the default task path.

The CPU gate proves exact reset and 32-step parity at seeds 0/17/42 and batches
1/3, different-history continuation, positive reward states, lifecycle replay
through 1252 Cheetah or 502 Hopper/Walker public steps, implicit reset and RNG
change, full-State rendering through the private qpos projection without
mutation, and rejection of incomplete renderer rows. Gravity
`[1.25, -0.75, -7.5]` matches for 32 steps in each domain. A visual-only
variation preserves exact nominal `[0, 0, -9.81]` gravity and the next eight
reference steps. The production adapter imports dm_control and MuJoCo, never
stable-worldmodel.

The standalone gate is `scripts/recovery/dmc_continuation_gate.py`; it accepts
`--device cpu|cuda --output PATH`. The immutable reference-only corpus under
`tests/fixtures/semantics/dmc_continuation/` records seed 17, batch 3, 32 steps,
every numeric output leaf, dependency versions, source commit, and content hashes.
Its generator refuses to overwrite existing files.

## Remaining historical identities

The same pinned registration module contains five DMC defaults outside this
cohort. This table is a source inventory, not semantic certification:

| Historical registration | Pinned default construction | Existing public cell | Identity finding |
| --- | --- | --- | --- |
| `swm/HumanoidDMControl-v0` | `Humanoid(move_speed=1, pure_state=False)` (`task='walk'`) | `DMCHumanoidStand` | Contradiction: Stand cannot represent the historical Walk default. A future `DMCHumanoidWalk` public identity is required. |
| `swm/BallInCupDMControl-v0` | `BallInCup(random=seed)` | `DMCBallInCupCatch` | Name/task appears aligned; uncertified. |
| `swm/FingerDMControl-v0` | `Turn(target_radius=0.03)` (`task='turn_hard'`) | `DMCFingerTurnHard` | Name/task appears aligned; uncertified. |
| `swm/ManipulatorDMControl-v0` | `Bring(use_peg=False, insert=False, fully_observable=True)` | `DMCManipulatorBringBall` | Name/task appears aligned; uncertified. |
| `swm/QuadrupedDMControl-v0` | `Move(desired_speed=0.5)` (`task='walk'`) | `DMCQuadrupedWalk` | Name/task appears aligned; uncertified. |

Reacher hard, Acrobot balance, Pendulum swingup, and Cartpole swingup were
recovered in earlier bounded cohorts. No sibling tasks or untested Variation
routes are certified here.
