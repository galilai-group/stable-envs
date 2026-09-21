# Replacement-readiness of the 42 historical environment cells

Maintained inventory after the v0 architecture freeze. Status is **derived** from
`tests/helpers/reference/readiness.py`, not from prose beside the table.

<!-- readiness-counts READY=0 PARTIAL=10 INCOMPLETE=30 SYMBOLIC=2 -->

`READY` means:

```text
execution in {NATIVE, DELEGATED}
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
| **DELEGATED** | stable-envs owns the Env API and steps by calling the canonical upstream simulator |
| **INCOMPLETE** | currently only enough capability to render |
| **SYMBOLIC** | no pixel cell |

`DELEGATED` is fully acceptable. Do not reimplement official contact dynamics to make the word "native" true. No cell is `DELEGATED` yet: DMC / Fetch / OGBench / Craftax / Rocket still need an `Env.step` that calls the upstream simulator.

## Counts (42 historical SWM ids)

| status | count |
|---|---|
| READY | 0 |
| PARTIAL | 10 |
| INCOMPLETE | 30 |
| SYMBOLIC | 2 |

Execution: NATIVE 10 · INCOMPLETE 30 · SYMBOLIC 2 · DELEGATED 0.

## Per historical cell

Coverage is what **stable-envs** owns standalone. Reward on NATIVE cells is omitted, not fabricated.

| swm id | native class(es) | execution | status | remaining UM / PARTIAL |
|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `Acrobot` | NATIVE | PARTIAL | `physics.link_*` masses/com/moi/lengths |
| `swm/AcrobotDMControl-v0` | `DMCAcrobotSwingup` | INCOMPLETE | INCOMPLETE | no SE step; `gravity.*`; task `swingup` vs historical `swingup_sparse` |
| `swm/BallInCupDMControl-v0` | `DMCBallInCupCatch` | INCOMPLETE | INCOMPLETE | no SE step; gravity/density/`ball.size` |
| `swm/CartPoleControl-v1` | `CartPole` | NATIVE | PARTIAL | `physics.{gravity,masscart,masspole,force_mag,tau,length}` |
| `swm/CartpoleDMControl-v0` | `DMCCartpoleBalance`, `DMCCartpoleSwingup` | INCOMPLETE | INCOMPLETE | no SE step; gravity/`cart_mass`; historical default was sparse swingup |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | INCOMPLETE | INCOMPLETE | no SE step; gravity/density/floor |
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
| `swm/FingerDMControl-v0` | `DMCFingerTurnHard` | INCOMPLETE | INCOMPLETE | no SE step; density/`target.shape` |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | INCOMPLETE | INCOMPLETE | no SE step; gravity/density/floor |
| `swm/HumanoidDMControl-v0` | `DMCHumanoidStand` | INCOMPLETE | INCOMPLETE | no SE step; historical `walk` vs `stand` |
| `swm/ManipulatorDMControl-v0` | `DMCManipulatorBringBall` | INCOMPLETE | INCOMPLETE | no SE step; arm/hand density |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuous` | NATIVE | PARTIAL | `physics.{power,max_speed}` |
| `swm/MountainCarControl-v0` | `MountainCar` | NATIVE | PARTIAL | `physics.{force,gravity,max_speed}` |
| `swm/OGBCube-v0` | `OGBCube` | INCOMPLETE | INCOMPLETE | no SE step; size/light/camera; reset poses |
| `swm/OGBMaze-v0` | `PointMazeLargeNavigate` (+ family) | INCOMPLETE | INCOMPLETE | no SE step; `light.intensity` / `floor.pixel_encoding`; arena configs |
| `swm/OGBScene-v0` | `OGBScene` | INCOMPLETE | INCOMPLETE | no SE step; size/light/camera/`lock_color` |
| `swm/PFRocketLanding-v0` | `PyFlytRocketLanding` | INCOMPLETE | INCOMPLETE | no SE step; start height/offset/tilt |
| `swm/PendulumControl-v1` | `Pendulum` | NATIVE | PARTIAL | `physics.{g,m,l,dt,max_speed,max_torque}` |
| `swm/PendulumDMControl-v0` | `DMCPendulumSwingup` | INCOMPLETE | INCOMPLETE | no SE step; gravity |
| `swm/Piecewise-v0` | `Piecewise`, `PiecewiseZones` | NATIVE | PARTIAL | `task.min_steps`; target draw; `render_bias_field`; agent.position-as-reset |
| `swm/PushT-Discrete-v1` | `PushT` | NATIVE | PARTIAL | `agent.shape` / `block.shape` / `agent.velocity`; start poses as Variation |
| `swm/PushT-v1` | `PushT` | NATIVE | PARTIAL | same; downstream task evidence exists |
| `swm/QuadrupedDMControl-v0` | `DMCQuadrupedWalk` | INCOMPLETE | INCOMPLETE | no SE step; gravity/density/floor |
| `swm/ReacherDMControl-v0` | `DMCReacherHard` | INCOMPLETE | INCOMPLETE | no SE step; density/`target.shape` |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` | NATIVE | PARTIAL | `goal.*` under `show_goal=False`; agent.position-as-reset |
| `swm/TwoRoom-v1` | `TwoRoom`, `TwoRoomTarget` | NATIVE | PARTIAL | `task.min_steps`; agent/target.position-as-reset |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | INCOMPLETE | INCOMPLETE | no SE step; gravity/density/floor |

`agent.speed` (TwoRoom / Piecewise / SimplePointMaze) and Piecewise `zones.bias_*` are now native Variation leaves. They no longer appear as `UNSUPPORTED_MEANINGFUL`.

## Reward / task semantics

NATIVE cells omit reward / termination / truncation rather than inventing them. `task.min_steps` and SPM `goal.*` (with `show_goal=False`) belong to a higher-level task layer unless that env intrinsically defines them.

INCOMPLETE cells that later become DELEGATED should surface the upstream simulator's reward/termination unchanged.

Historical DMC task mismatches (not fabricated, not silently "fixed"):

- Acrobot: certified `swingup` vs historical SWM `swingup_sparse`
- Cartpole: certified balance+swingup pair vs historical sparse-swingup default
- Humanoid: certified `stand` vs historical `walk`

## Implementation tasks remaining

These are capability work, not architecture:

1. Thread classic-control `physics.*` through `_dynamics` (and geometry leaves through the rasterizer) without breaking CUDA-graph capture or default bit-exactness.
2. Add `Env.step` that **delegates** to dm_control / Gymnasium Robotics / OGBench / Craftax-JAX / PyFlyt.
3. Expose DMC gravity/density/geometry through the Env Variation space onto the MuJoCo model (or document official-simulator-only configuration).
4. PushT: configurable pymunk shapes; treat instantaneous `agent.velocity` as State, not Variation; start poses as reset Variation.
5. Fetch: `block.mass` and `transparent_arm` via the MuJoCo model; skybox half of `background.color` is a re-freeze blocker.
6. OGBench Cube/Scene: object size, lighting, camera, `lock_color` via OGBench/MuJoCo configuration.
7. Rocket: start/reset pose as Variation; step via PyFlyt.
8. Navigation: `task.min_steps` only if the env intrinsically defines that task; otherwise leave it to the consumer.
9. Craftax: delegate `reset`/`step` to Craftax/JAX rather than a historical wrapper.

Closing `UNSUPPORTED_MEANINGFUL` and `PARTIAL` on NATIVE cells, and adding DELEGATED `step` on render-only cells, is the remaining work. There is no further architecture pass.
