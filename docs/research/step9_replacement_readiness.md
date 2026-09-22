# Replacement-readiness of the 42 historical environment cells

Maintained inventory after the v0 architecture freeze. Semantic certification
is tracked separately in [the recovery matrix](semantic_recovery_matrix.json);
these architecture statuses do not certify task parity. Status is **derived** from
`tests/helpers/reference/readiness.py`, not from prose beside the table.

<!-- readiness-counts READY=0 PARTIAL=22 INCOMPLETE=18 SYMBOLIC=2 -->

`READY` means:

```text
execution in {NATIVE, DELEGATED_CPU}
UNSUPPORTED_MEANINGFUL = 0
PARTIAL = 0
```

A cell that only renders is `INCOMPLETE`, even if its VariationMap is tidy.
A cell that steps but still has meaningful missing knobs is `PARTIAL`.
Do not mark `READY` while listing a gap.

## Architecture freeze

Permanent for v0. Do not reopen as a product-role discussion.

```text
stable-envs owns the environment layer                       YES
stable-envs runtime knows about stable-worldmodel            NO
SWM IDs belong in stable-envs registry                       NO
stable_envs.interop                                           NO
Variation is appearance-only                                 NO
Variation includes meaningful persistent env configuration   YES
SWM API/names must be cloned                                  NO
SWM migration evidence may live in regression/research       YES
SWM may eventually consume stable-envs                        YES
```

Five nouns:

```text
Env        owns the relationship among the others
State      instantaneous dynamic world state
Action     intervention at a timestep
Variation  persistent / episode-level environment configuration
Frame      rendered observation

frame = f(state, variation)
where dynamics exist: next_state = g(state, action, variation)
```

## Execution mode

| mode | meaning |
|---|---|
| **NATIVE** | stable-envs implements `transition` / `step` itself |
| **DELEGATED_CPU** | stable-envs owns the Env API and calls the upstream CPU simulator; CUDA State/rendering does not imply GPU physics |
| **INCOMPLETE** | currently only enough capability to render |
| **SYMBOLIC** | no pixel cell |

Delegated execution is accepted. Exact historical defaults are recovered independently;
see the semantic recovery matrix for each tested task and its evidence. A sibling
renderer sharing the same historical registration can remain incomplete.

## Counts (42 historical SWM ids)

| status | count |
|---|---|
| READY | 0 |
| PARTIAL | 22 |
| INCOMPLETE | 18 |
| SYMBOLIC | 2 |

Execution: NATIVE 5 · DELEGATED_CPU 17 · INCOMPLETE 18 · SYMBOLIC 2.

## Per historical cell

The table reports implementation coverage. `PARTIAL` is a conservative architecture
status derived from the historical VariationMap overlay; it is separate from an
exact default task being `VERIFIED`. Full Variation surfaces remain uncertified.

| swm id | native class(es) | execution | status | remaining UM / PARTIAL |
|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `Acrobot` | DELEGATED_CPU | PARTIAL | Exact CPU default; full historical Variation surface remains uncertified |
| `swm/AcrobotDMControl-v0` | `DMCAcrobotBalance` | DELEGATED_CPU | PARTIAL | sparse Balance; complete historical Variation surface remains uncertified |
| `swm/BallInCupDMControl-v0` | `DMCBallInCupCatch` | DELEGATED_CPU | PARTIAL | Exact CPU historical default; complete Variation surface and sibling tasks uncertified |
| `swm/CartPoleControl-v1` | `CartPole` | DELEGATED_CPU | PARTIAL | Gymnasium CartPole-v1; full Variation surface remains uncertified |
| `swm/CartpoleDMControl-v0` | `DMCCartpoleSwingup` (sparse default only) | DELEGATED_CPU | PARTIAL | Balance and dense swing-up remain uncertified; complete Variation surface |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | DELEGATED_CPU | PARTIAL | historical custom run default; density/floor and other task variants |
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassic` | INCOMPLETE | INCOMPLETE | no SE step (delegate to Craftax/JAX); world layer only |
| `swm/CraftaxClassicSymbolic-v1` | — | SYMBOLIC | SYMBOLIC | no pixel cell |
| `swm/CraftaxPixels-v1` | `CraftaxFull` | INCOMPLETE | INCOMPLETE | no SE step (delegate to Craftax/JAX) |
| `swm/CraftaxSymbolic-v1` | — | SYMBOLIC | SYMBOLIC | no pixel cell |
| `swm/FetchPickAndPlace-v3` | `FetchPickAndPlace` | INCOMPLETE | INCOMPLETE | no SE step; `block.mass`; `transparent_arm`; skybox PARTIAL |
| `swm/FetchPickAndPlaceDense-v3` | `FetchPickAndPlaceDense` | INCOMPLETE | INCOMPLETE | same + dense reward only upstream |
| `swm/FetchPickAndPlaceDict-v3` | `FetchPickAndPlaceDict` | INCOMPLETE | INCOMPLETE | same; dict obs not in SE |
| `swm/FetchPush-v3` | `FetchPush` | INCOMPLETE | INCOMPLETE | Fetch family gaps |
| `swm/FetchPushDense-v3` | `FetchPushDense` | INCOMPLETE | INCOMPLETE | same |
| `swm/FetchPushDict-v3` | `FetchPushDict` | INCOMPLETE | INCOMPLETE | same |
| `swm/FetchReach-v3` | `FetchReach` | INCOMPLETE | INCOMPLETE | no `block.mass` on Reach; arm/skybox |
| `swm/FetchReachDense-v3` | `FetchReachDense` | INCOMPLETE | INCOMPLETE | same |
| `swm/FetchReachDict-v3` | `FetchReachDict` | INCOMPLETE | INCOMPLETE | same |
| `swm/FetchSlide-v3` | `FetchSlide` | INCOMPLETE | INCOMPLETE | same |
| `swm/FetchSlideDense-v3` | `FetchSlideDense` | INCOMPLETE | INCOMPLETE | same |
| `swm/FetchSlideDict-v3` | `FetchSlideDict` | INCOMPLETE | INCOMPLETE | same |
| `swm/FingerDMControl-v0` | `DMCFingerTurnHard` | DELEGATED_CPU | PARTIAL | Exact CPU historical default; complete Variation surface and sibling tasks uncertified |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | DELEGATED_CPU | PARTIAL | historical custom hop default; density/floor and other task variants |
| `swm/HumanoidDMControl-v0` | `DMCHumanoidStand` | INCOMPLETE | INCOMPLETE | no SE step; historical `walk` vs `stand` |
| `swm/ManipulatorDMControl-v0` | `DMCManipulatorBringBall` | DELEGATED_CPU | PARTIAL | Exact CPU historical default; complete Variation surface and sibling tasks uncertified |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuous` | DELEGATED_CPU | PARTIAL | Exact CPU default; full historical Variation surface remains uncertified |
| `swm/MountainCarControl-v0` | `MountainCar` | DELEGATED_CPU | PARTIAL | Exact CPU default; full historical Variation surface remains uncertified |
| `swm/OGBCube-v0` | `OGBCube` | INCOMPLETE | INCOMPLETE | no SE step; size/light/camera; reset poses |
| `swm/OGBMaze-v0` | `PointMazeLargeNavigate` (default only) | DELEGATED_CPU | PARTIAL | other maze configurations/tasks and complete appearance surface |
| `swm/OGBScene-v0` | `OGBScene` | INCOMPLETE | INCOMPLETE | **BLOCKED** default constructor omits required `env_type`; no global skip |
| `swm/PFRocketLanding-v0` | `PyFlytRocketLanding` | INCOMPLETE | INCOMPLETE | no SE step; start height/offset/tilt |
| `swm/PendulumControl-v1` | `Pendulum` | DELEGATED_CPU | PARTIAL | Gymnasium Pendulum-v1; full Variation surface remains uncertified |
| `swm/PendulumDMControl-v0` | `DMCPendulumSwingup` | DELEGATED_CPU | PARTIAL | SwingUp; complete historical Variation surface remains uncertified |
| `swm/Piecewise-v0` | `Piecewise`, `PiecewiseZones` | NATIVE | PARTIAL | Exact CPU default with native Torch dynamics; sibling variants and full Variation surface uncertified |
| `swm/PushT-Discrete-v1` | `PushT` | NATIVE | PARTIAL | `agent.shape` / `block.shape` / `agent.velocity`; start poses as Variation |
| `swm/PushT-v1` | `PushT` | NATIVE | PARTIAL | same; downstream task evidence exists |
| `swm/QuadrupedDMControl-v0` | `DMCQuadrupedWalk` | DELEGATED_CPU | PARTIAL | Exact CPU historical default; complete Variation surface and sibling tasks uncertified |
| `swm/ReacherDMControl-v0` | `DMCReacherHard` | DELEGATED_CPU | PARTIAL | hard/default only; easy/qpos_match and complete Variation surface |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` | NATIVE | PARTIAL | Exact CPU default with native Torch dynamics; sibling variants and full Variation surface uncertified |
| `swm/TwoRoom-v1` | `TwoRoom`, `TwoRoomTarget` | NATIVE | PARTIAL | Exact CPU default with native Torch dynamics; sibling variants and full Variation surface uncertified |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | DELEGATED_CPU | PARTIAL | historical custom walk default; density/floor and other task variants |

`agent.speed` (TwoRoom / Piecewise / SimplePointMaze) and Piecewise `zones.bias_*` are now native Variation leaves. They no longer appear as `UNSUPPORTED_MEANINGFUL`.

## Reward / task semantics

The recovered defaults expose exact upstream reward, termination, truncation,
observations and numeric info. CartPole and Pendulum explicitly refuse the former
physical-only `transition`/`unroll` path; callers use full-State restoration and
`step`. Native movement in the other eight registrations does not establish
reward, reset, action or lifecycle parity.

Acrobot now implements sparse **Balance**, despite using a historical swing-up
renderer asset. Cartpole implements only the historical sparse SwingUp default.
Humanoid's historical default is **walk**; the existing Stand renderer remains
uncertified as its replacement.

## Work remaining

Continue independent task recovery and restoration experiments. The active native
audit covers the remaining three classic-control registrations, PushT continuous
and discrete, and navigation. OGBench Scene remains blocked by its default
constructor and belongs to the later manipulation cohort. Fetch, Craftax and
Rocket remain unresolved. Benchmark eligibility comes only from the exact
`VERIFIED` scope in the semantic recovery matrix.

No additional shared abstraction redesign is justified by this inventory.
