# Replacement-readiness of the 42 historical environment cells

Maintained inventory after the v0 architecture freeze. This is **not** a claim that
`UNSUPPORTED_MEANINGFUL = 0`. The frozen VariationMap buckets
(`mapped` / `partial` / `unsupported` / `dead_upstream` / `not_applicable` / `swm_only`)
are unchanged; `VariationMap.replacement_class()` is the overlay used here.

Historical Step-5 paired-eval harnesses: [artifacts/](artifacts/README.md).

## Architecture freeze

Settled. Do not reopen as a product-role discussion.

```text
stable-envs is the canonical environment layer: YES

stable-envs knows about SWM at runtime: NO

SWM IDs in stable-envs registry: NO

stable_envs.interop: NO

Variation is appearance-only: NO

Variation represents persistent environment configuration: YES

stable-envs should absorb meaningful old environment capabilities: YES

stable-envs must clone SWM's API/names: NO

historical SWM correspondence remains in regression/provenance: YES

SWM may later depend on stable-envs: YES
```

`frame = f(state, variation)`. Where native dynamics exist, `next_state = g(state, action, variation)`.
Render fidelity and contact-solver fidelity stay separate claims.

## Kinds (42 SWM ids)

| kind | meaning | count |
|---|---|---|
| **A** | native `step` / `transition` in stable-envs | 10 ids (~12 classes: TwoRoom+Target, Piecewise+Zones, PushT serves two ids) |
| **B** | dynamics delegated to an **official** simulator (`gymnasium`, `dm_control`, OGBench, Craftax/JAX, PyFlyt) — not SWM | 30 |
| **C** | certification `reference_env` may still `gymnasium.make("swm/...")` | overlay on nav / rocket / OGB manip — not production `step` |
| **D** | symbolic / no pixel cell | 2 (`CraftaxClassicSymbolic-v1`, `CraftaxSymbolic-v1`) |

Kind B is the intended shape for MuJoCo-scale physics: own the user-facing Env, delegate contact
to the official sim. Kind C is a **certification** leftover, not a shipped runtime dependency.

## Per historical cell

Coverage is what **stable-envs** owns standalone. `✓` native · `↑` upstream sim only · `—` omitted
by design (not fabricated) · `~` partial.

| swm id | native class(es) | kind | render | reset | action | dynamics | reward | term. | variation | batch | status | gaps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `swm/AcrobotControl-v1` | `Acrobot` | A | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | `physics.*` not in variation |
| `swm/AcrobotDMControl-v0` | `DMCAcrobotSwingup` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | no SE step; task `swingup` vs swm `swingup_sparse` |
| `swm/BallInCupDMControl-v0` | `DMCBallInCupCatch` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | no SE step |
| `swm/CartPoleControl-v1` | `CartPole` | A | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | `physics.*` not in variation |
| `swm/CartpoleDMControl-v0` | `DMCCartpoleBalance`, `DMCCartpoleSwingup` | B | ✓ | ✓ | ↑ | ↑ | mismatch | — | ~ | ✓ | PARTIAL | swm default sparse swingup vs two non-sparse tasks |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | visual tier; no SE step |
| `swm/CraftaxClassicPixels-v1` | `CraftaxClassic` | B | ~ | ✓ | ↑ | ↑ | ↑jax | — | n/a | ✓ | PARTIAL | world layer only; no SE step; JIT vs un-jitted ref |
| `swm/CraftaxClassicSymbolic-v1` | — | D | — | — | — | — | — | — | — | — | READY (symbolic) | no pixel cell |
| `swm/CraftaxPixels-v1` | `CraftaxFull` | B | ✓ | ✓ | ↑ | ↑ | ↑jax | — | n/a | ✓ | PARTIAL | no SE step |
| `swm/CraftaxSymbolic-v1` | — | D | — | — | — | — | — | — | — | — | READY (symbolic) | no pixel cell |
| `swm/FetchPickAndPlace-v3` | `FetchPickAndPlace` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | no SE step; variation partial |
| `swm/FetchPickAndPlaceDense-v3` | `FetchPickAndPlaceDense` | B | ✓ | ✓ | ↑ | ↑ | ↑gym dense | — | ~ | ✓ | PARTIAL | same + dense reward only upstream |
| `swm/FetchPickAndPlaceDict-v3` | `FetchPickAndPlaceDict` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | dict obs not in SE |
| `swm/FetchPush-v3` | `FetchPush` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | Fetch family gaps |
| `swm/FetchPushDense-v3` | `FetchPushDense` | B | ✓ | ✓ | ↑ | ↑ | ↑gym dense | — | ~ | ✓ | PARTIAL | same |
| `swm/FetchPushDict-v3` | `FetchPushDict` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | same |
| `swm/FetchReach-v3` | `FetchReach` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | same; no `block.mass` on Reach |
| `swm/FetchReachDense-v3` | `FetchReachDense` | B | ✓ | ✓ | ↑ | ↑ | ↑gym dense | — | ~ | ✓ | PARTIAL | same |
| `swm/FetchReachDict-v3` | `FetchReachDict` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | same |
| `swm/FetchSlide-v3` | `FetchSlide` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | same |
| `swm/FetchSlideDense-v3` | `FetchSlideDense` | B | ✓ | ✓ | ↑ | ↑ | ↑gym dense | — | ~ | ✓ | PARTIAL | same |
| `swm/FetchSlideDict-v3` | `FetchSlideDict` | B | ✓ | ✓ | ↑ | ↑ | ↑gym | — | ~ | ✓ | PARTIAL | same |
| `swm/FingerDMControl-v0` | `DMCFingerTurnHard` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | no SE step |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | visual tier; no SE step |
| `swm/HumanoidDMControl-v0` | `DMCHumanoidStand` | B | ✓ | ✓ | ↑ | ↑ | mismatch | — | ~ | ✓ | PARTIAL | swm `humanoid-walk` vs `stand` |
| `swm/ManipulatorDMControl-v0` | `DMCManipulatorBringBall` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | no SE step |
| `swm/MountainCarContinuousControl-v0` | `MountainCarContinuous` | A | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | `physics.*` not in variation |
| `swm/MountainCarControl-v0` | `MountainCar` | A | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | `physics.*` not in variation |
| `swm/OGBCube-v0` | `OGBCube` | B | ✓ | ✓ | ↑ | ↑ | ↑ogbench | — | ~ | ✓ | PARTIAL | CUDA-only; no SE step; native OGBench constructor |
| `swm/OGBMaze-v0` | `PointmazeLargeNavigate` (+ family) | B | ✓ | ✓ | ↑ | ↑ | ↑ogbench | — | ~ | ✓ | PARTIAL | default large-point; arena configs uncovered |
| `swm/OGBScene-v0` | `OGBScene` | B | ✓ | ✓ | ↑ | ↑ | ↑ogbench | — | ~ | ✓ | PARTIAL | CUDA-only; no SE step; native OGBench constructor |
| `swm/PFRocketLanding-v0` | `PyFlytRocketLanding` | B+C | ✓ | ✓ | ↑ | ↑ | ↑pyflyt | — | ~ | ✓ | PARTIAL | CUDA-only; no SE step; ref may `make("swm/PFRocketLanding-v0")` |
| `swm/PendulumControl-v1` | `Pendulum` | A | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | `physics.*` not in variation |
| `swm/PendulumDMControl-v0` | `DMCPendulumSwingup` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | no SE step |
| `swm/Piecewise-v0` | `Piecewise`, `PiecewiseZones` | A+C | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | speed / `task.min_steps` / zone bias not in variation; ref `swm/Piecewise-v0` |
| `swm/PushT-Discrete-v1` | `PushT` | A | ✓ | ✓ | ✓ | ~ | — | — | ~ | ✓ | READY | GPU sim ≈ pymunk, not bit-exact; shapes/velocity gaps |
| `swm/PushT-v1` | `PushT` | A | ✓ | ✓ | ✓ | ~ | — | — | ~ | ✓ | READY | same; downstream task evidence exists |
| `swm/QuadrupedDMControl-v0` | `DMCQuadrupedWalk` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | visual tier; no SE step |
| `swm/ReacherDMControl-v0` | `DMCReacherHard` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | no SE step |
| `swm/SimplePointMaze-v0` | `SimplePointMaze` | A+C | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | speed / goal-as-task not in variation; ref `swm/SimplePointMaze-v0` |
| `swm/TwoRoom-v1` | `TwoRoom`, `TwoRoomTarget` | A+C | ✓ | ✓ | ✓ | ✓ | — | — | ~ | ✓ | READY | speed / `task.min_steps` not in variation; ref `swm/TwoRoom-v1` |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | B | ✓ | ✓ | ↑ | ↑ | ↑dmc | — | ~ | ✓ | PARTIAL | visual tier; no SE step |

Reward on Kind A cells is **omitted, not fabricated**. Kind B rewards exist only when stepping the
official simulator.

## Variation leaves (replacement overlay)

`PARTIAL` and `UNSUPPORTED_MEANINGFUL` are the work list. Dead/no-op stays `DEAD_UPSTREAM`.
Class-split flags (`rendering.render_target` / `render_zones`) can be `NATIVE` via the sibling class.

### PARTIAL

| family | leaf | residual |
|---|---|---|
| Fetch (12) | `background.color` | floor mapped; skybox not (re-freeze blocker) |

### UNSUPPORTED_MEANINGFUL (meaningful, not “N/A because 0 pixels”)

| family | leaves |
|---|---|
| **PushT** | `agent.shape`, `block.shape`, `agent.velocity`; reset `*.start_position` / `block.angle` as variation-config (they are State today) |
| **Fetch** | `block.mass`; `rendering.transparent_arm`; reset `agent/goal/block.start_position`, `block.angle` as variation-config |
| **Classic control** | `physics.*` (gravity, masses, `tau`, link lengths, …) |
| **DMC** | `gravity.*`, `*.density`, plus geometry that moves official pixels (`ball.size`, some `target.shape`, trackcom masses that move the camera) |
| **Nav (TwoRoom / Piecewise / SPM)** | `agent.speed`, `task.min_steps`, Piecewise `zones.bias_*`; SPM `goal.*` under `show_goal=False` |
| **OGB Cube / Scene** | `cube.size`, `light.intensity`, `camera.angle_delta`; Scene `lock_color`; reset/goal poses as variation-config |
| **OGB maze** | `light.intensity`, `floor.pixel_encoding` |
| **Rocket** | `environment.start_height_ratio` (and start offset / tilt) |

Do not relabel those as not-applicable because they change no pixel.

## Other capability gaps

- **No SE `step`** on most Kind B cells (DMC, Fetch, OGB, Craftax, Rocket). Delegation to the official sim is acceptable; a coherent Env that actually steps is still the replacement target.
- **Reward omitted, not fabricated**, on Kind A cells.
- **Three DMC task mismatches** vs historical defaults: Acrobot swingup vs `swingup_sparse`; Cartpole pair vs sparse swingup default; Humanoid stand vs `walk`.
- **OGB Cube / Scene and Rocket are CUDA-only.**
- **Craftax has no SE step**; Classic pixels omit the inventory strip.
- **Fetch** has no CPU backend; Gate-F reset-appearance mismatch is an upstream wrapper unbind.
- **OGBMaze** does not cover arena layouts.

Closing `UNSUPPORTED_MEANINGFUL` and `PARTIAL` is implementation work after this freeze, not another architecture pass.
