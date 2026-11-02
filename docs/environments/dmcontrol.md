# DM Control

These 17 classes port tasks of the [DeepMind Control Suite](https://github.com/google-deepmind/dm_control)
([Tassa et al., 2018](https://arxiv.org/abs/1801.00690)).

The renderers come in two groups:

- The fixed-camera tasks (Acrobot, Ball in cup, Cartpole, Finger, Manipulator, Pendulum, Point mass and
  Reacher) use the analytic ray caster `raycast3d`. The tests compare their frames with MuJoCo's
  frames pixel by pixel.
- The body-tracking tasks (Cheetah, Hopper, Humanoid, Quadruped, Swimmer and Walker) use `cpu3d` on CPU
  and MJWarp on CUDA. Their frames differ from MuJoCo's frames. `env.describe()["visual_fidelity"]`
  gives the measured agreement.

Each action dimension drives one actuator of the model, in the order of the MuJoCo model. The bounds
come from the actuator control range.

```python
import stable_envs as snv

env = snv.DMCReacher(num_envs=8)
state = env.reset(seed=0)
out = env.step(env.action_spec.sample(8))
```

The reward of each source control step is in [0, 1]. Each public action repeats twice and returns
the sum of both rewards. The suite does not define a success criterion. Ball
in cup is the exception: its `success` value is 1 when the ball is in the cup.

## Direct MuJoCo Warp physics

The 16 classes other than Reacher accept `simulation_backend="mjwarp"` for direct batched MuJoCo Warp
physics on CUDA. Reacher retains its existing native/MJX integration.
Install `stable-envs[dmcontrol-gpu]`; the reference profile uses MuJoCo/MuJoCo Warp 3.12 and Warp 1.17.
An explicit Warp request raises an error when CUDA or its dependencies are unavailable.

This implementation has passed the 16-class CUDA integration suite, but full numerical acceptance is
unfinished. The remaining source-agreement, replay and fault-sensitivity failures are recorded for
focused follow-up below. Automatic backend selection remains unchanged.

```python
with snv.DMCFingerTurnHard(num_envs=64, device="cuda", simulation_backend="mjwarp") as env:
    state = env.reset(seed=7)
    result = env.step(env.action_spec.sample(64))
    checkpoint = env.get_state()
    env.set_state(checkpoint)
```

`simulation_backend` selects physics. `backend` continues to select rendering. Automatic routing is
unchanged: Reacher follows the table below, and the other tasks use native CPU stepping. Native CPU
physics and Reacher's explicit `simulation_backend="mjx"` remain available.

Warp physics uses float32. Public State retains its existing shapes and dtypes, including float64
physical columns and a clock accumulated with native float64 increments. `set_state(s); get_state()`
preserves the supplied checkpoint exactly, including solver warmstart, activation, reset lifecycle,
cumulative reward, the complete source RNG, episode targets and physical variation. Snapshots own
their memory. This exact restoration does not imply bitwise identical trajectories after further
GPU integration; numerical agreement is measured separately for each task, field and horizon.

The adapter retains the registered native reset routines and RNG. A pending reset consumes one source
control round with zero reward. Other lanes continue through a compact Warp batch; cumulative wrapper
reward survives these implicit resets. Both action repetitions and every source physics substep are
preserved, including RK4 and acceleration-stage sensor timing.
A single scratch runtime is reused while the number of continuing lanes is unchanged. A different
count requires a new allocation and capture, so mixed-reset costs can exceed steady-state stepping.

Rendering still consumes State. Rendering a restored checkpoint and rendering the State captured with
an online frame must reproduce that frame exactly. Pixel differences caused by distinct CPU and GPU
trajectories are a separate measurement from renderer fidelity on identical States.

The explicit CUDA runner is `python scripts/ci/run_dmc_mjwarp.py`. Its `--smoke` mode checks all 16 supported
classes without numerical calibration; it is not full acceptance. Full validation requires a pinned
independent source checkout and frozen calibration evidence. The calibration tool is
`python -m tests.helpers.reference.dmc_mjwarp --help`, and matched performance measurements use
`python benchmarks/dmc_mjwarp.py --help`. Missing CUDA and skipped required cases fail the runner.

The numerical calibration workflow uses 50 seeds with three lanes, at one and eight public actions. For each task and field,
the source envelope combines nominal starts, reset boundaries, cached-Gaussian resets, reached
contacts or constraints, and the declared variation endpoints. Its absolute bound is 1.5 times the
largest measured error plus a float32 roundoff floor. Resumed-versus-uninterrupted Warp trajectories
have separate bounds. Held-out rollouts and deliberate repeat, control, contact and constraint faults
use the frozen bounds at both horizons; the validation run does not adjust them.

Contact admission can amplify very small distance differences into different trajectories. These
empirical bounds apply to the recorded cases and horizons. `benchmarks/dmc_mjwarp_pixels.py` measures
the resulting pixel drift through the same renderer separately from exact rendering of an identical State.

The implementation stays within `envs/dmcontrol/simulation`:

| Module | Ownership |
|---|---|
| `mjwarp.py` | `DMCWarpAdapter`, source reset owners, the two control rounds, checkpoints and result assembly |
| `mjwarp_runtime.py` | Batched model/data storage, Torch views, CUDA stream ordering, graph capture and physics substeps |
| `mjwarp_tasks.py` | Bound task observations, rewards and optional info over device arrays |

The task binding resolves model indices once. Cartpole and Finger variants share their domain
formulas. The adapter uses the existing factories, State fields and variation calculations; adding a
task does not require a separate environment registry. Native owners remain authoritative for reset
sampling and model changes, while ordinary snapshots read the live GPU checkpoint.

For MuJoCo Warp 3.12, each DMC runtime corrects two contact cases before constraint assembly:
plane–capsule friction tangents and the normal of intersecting capsule centerlines. The corrections
follow native MuJoCo's [primitive collision](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_collision_primitive.c)
and [frame construction](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_util_spatial.c)
rules. The capsule-intersection correction uses float64 scratch arithmetic; stored physics and
integration remain float32. Native-reference tests cover contact geometry, integrated dynamics and
isolation between runtime instances.

### Validation status

The final CUDA integration run on an RTX A6000 passed **53 checks with no skips or deselections**.
These cover all 16 classes, exact checkpoint ownership/restoration, captured-State rendering, sparse
outputs, the contact corrections and non-default streams. The existing DMC CPU profile passed
875 tests with 112 expected optional-device/family skips; Reacher's existing MJX suite passed all
six tests without changing its thresholds. The [CUDA integration report](../assets/dmc-mjwarp/integration-16.json)
records every required case and the evaluated code hashes.

The separate nominal-trajectory image diagnostic measured 960 frames (10 seeds × three lanes ×
16 tasks × two horizons). All 160 trials reproduced identical-State images exactly without changing
live State. CPU-versus-Warp images were identical for 239/480 frames after one public action and
160/480 after eight. The largest normalized RGB MSE was `6.13e-6` at one action and `7.14e-5` at
eight; individual edge pixels can still differ by 255. This is measured trajectory drift through the
same renderer, not a pixel acceptance bound or a comparison with the source renderer. See the
[per-task summary](../assets/dmc-mjwarp/pixel-drift-summary-16.json) and
[complete compressed measurement](../assets/dmc-mjwarp/pixel-drift-16.json.gz).

An earlier full numerical run passed 144 cases and failed nine, with no skips. That run included
Reacher and preceded the two contact corrections, so its frozen bounds do not certify the current
implementation. Remaining follow-up includes FingerSpin contact warmstart agreement,
Cheetah/Manipulator CUDA replay, PointMass graded-reward coverage, and recalibrating Hopper's
fault-sensitivity bounds after fixing its contact normal. The failed records are preserved;
no numerical limits were widened to make the tests pass. The
[deferred-failure record](../assets/dmc-mjwarp/deferred-fidelity.json) separates remaining issues from
the corrected contact causes and identifies the historical evidence.

Broader regression runs also encountered existing failures in BallInCup gravity variation,
Quadruped restored-image history, navigation declarations, and the distribution license inventory.
The two DMC GPU failures reproduce on pristine `main`; unrelated implementations and fixtures were
left unchanged. The [validation summary](../assets/dmc-mjwarp/validation-summary.json) records each
run separately and links to the archived logs, including failed checks.

### Measured performance

The matched sweep completed all **160 measurements**: 16 tasks × five batch sizes × native/Warp,
with 80 complete pairs and no execution errors. It used one RTX A6000 on a shared AMD EPYC 7443 host,
eight CPUs per reservation and one Torch thread per process. Both backends returned CUDA tensors.
Every case ran in a fresh process with seed 0, zero normalized actions, the task's default rendering
settings, five warmups and ten synchronized steady-state samples. Code hashes, versions, individual
latency samples, allocation measurements and the two reservations' metadata are retained in the
[benchmark summary](../assets/dmc-mjwarp/benchmark-summary-16.json),
[complete compressed report](../assets/dmc-mjwarp/benchmark-16.json.gz) and
[raw segments and analysis tools](../assets/dmc-mjwarp/benchmark-segments.zip).

The table gives the geometric mean of **native median latency / Warp median latency** across all
16 tasks. Values above 1 favor Warp. Public steps include the final owned State and task outputs;
the frame column adds the final frame. State history is omitted.

| Batch | Raw physics | Public step with State/tasks | Public step with frame | Tasks with faster public stepping |
|---:|---:|---:|---:|---:|
| 1 | 0.020× | 0.137× | 0.442× | 0/16 |
| 16 | 0.241× | 0.592× | 0.747× | 0/16 |
| 64 | 0.857× | 0.732× | 0.925× | 1/16 |
| 256 | 3.101× | 2.111× | 1.591× | 16/16 |
| 1024 | 13.343× | 7.734× | 2.982× | 16/16 |

The per-environment curves show all five measured batch sizes; values above 1× favor Warp.

![Native-to-Warp latency ratios for all 16 DMC environments](../assets/dmc-mjwarp-speedup.png)

[Absolute native and Warp public-step latency curves](../assets/dmc-mjwarp-latency.png) show the
underlying timings in milliseconds. Regenerate both figures from the committed measurements with
`uv run --no-project --with matplotlib==3.11.2 python benchmarks/plot_dmc_mjwarp.py`.
The plotting script validates all 80 matched pairs against the saved summary and does not run physics.

Raw native physics runs sequential CPU lanes without worker IPC; raw Warp includes runtime guards
and interop around two captured control rounds. Public native stepping retains existing worker
routing, including serial Quadruped. These measurements do not set a new automatic routing cutoff.

Initialization and restoration remain expensive. At batch 1024, the median cost across the 16 tasks
was:

| Operation | Native | Warp |
|---|---:|---:|
| Constructor/model initialization | 25.34 s | 86.74 s |
| Reset | 74.41 ms | 936.15 ms |
| Restore | 39.70 ms | 914.65 ms |

Construction excludes process imports and initial CUDA context setup. Construction and first-step
latencies each have one timed sample; disk kernel caches survive between processes. The largest
sampled parent/worker RSS was 25.29 GiB for native and 29.33 GiB for Warp; sampled process/worker GPU
usage reached 4.03 and 4.25 GiB respectively. RSS/NVML readings are sampled usage, not transient
peaks, and can include retained allocations from earlier stages. Python-tracked allocation and
Torch allocator peaks are recorded separately; they do not cover all native/Warp allocations.

This single-seed workload measures performance, not numerical correctness or tail-latency
guarantees. Full numerical acceptance and further optimization remain follow-up work.

## Acrobot

### DMCAcrobotBalance

Acrobot is a chain of two links with one actuator at the elbow. The task is to keep the free end above
the base. The episode starts near the upright pose.

<!-- env: DMCAcrobotBalance -->

## Ball in cup

### DMCBallInCupCatch

A cup on a planar arm holds a ball on a string. The agent swings the ball up and catches it in the cup.

<!-- env: DMCBallInCupCatch -->

## Cartpole

### DMCCartpoleBalance

The pole starts near the upright pose. The agent moves the cart to keep it upright.

<!-- env: DMCCartpoleBalance -->

### DMCCartpoleSwingup

The pole starts pointing down. The agent swings it up and keeps it upright.

<!-- env: DMCCartpoleSwingup -->

## Cheetah

### DMCCheetahRun

A planar biped runs forward. The reward increases with the forward speed, up to 10 m/s.

<!-- env: DMCCheetahRun -->

## Finger

### DMCFingerSpin

A planar finger with two joints spins a hinged body as fast as possible.

<!-- env: DMCFingerSpin -->

### DMCFingerTurnEasy

The finger turns the body until its tip points at a target. The target is large.

<!-- env: DMCFingerTurnEasy -->

### DMCFingerTurnHard

This task is the same as DMCFingerTurnEasy with a smaller target.

<!-- env: DMCFingerTurnHard -->

## Hopper

### DMCHopperHop

A planar one-legged body hops forward. The reward needs an upright torso and forward speed.

<!-- env: DMCHopperHop -->

## Humanoid

### DMCHumanoidWalk

A humanoid with 21 joints walks forward at 1 m/s.

<!-- env: DMCHumanoidWalk -->

## Manipulator

### DMCManipulatorBringBall

A planar arm with a gripper picks up a ball and brings it to a target position.

<!-- env: DMCManipulatorBringBall -->

## Pendulum

### DMCPendulumSwingup

The pendulum starts at a random angle. The agent swings it up and holds it upright with a limited
torque.

<!-- env: DMCPendulumSwingup -->

## Point mass

### DMCPointMassEasy

A point mass moves on a plane. The agent drives it to a target at the centre.

<!-- env: DMCPointMassEasy -->

## Quadruped

### DMCQuadrupedWalk

A body with four legs walks forward.

<!-- env: DMCQuadrupedWalk -->

## Reacher

### DMCReacher

A planar arm with two links moves its fingertip to a random target.

`simulation_backend="auto"` chooses physics when you construct the environment:

| Output device | Number of environments | Physics |
|---|---:|---|
| CPU | Any | Native MuJoCo on CPU |
| CUDA | Fewer than 32 | Native MuJoCo on CPU |
| CUDA | 32 or more | JAX/MJX on CUDA |

Previously, every CUDA batch used MJX, including one environment. Small CUDA batches now run
physics on CPU and return State and observations on CUDA. Rendering still uses the requested device.

```python
with snv.DMCReacher(num_envs=1, device="cuda") as env:
    print(env.simulation_backend)  # "native"

with snv.DMCReacher(num_envs=1024, device="cuda") as env:
    print(env.simulation_backend)  # "mjx"
```

Set `simulation_backend="native"` or `"mjx"` to choose explicitly. MJX requires CUDA and
`stable-envs[dmcontrol-gpu]`. Set `mjx_min_envs` to change the automatic cutoff.

The default cutoff of 32 is provisional. Earlier A40 measurements compared native physics with
CPU outputs against MJX with CUDA outputs. The comparison with both implementations returning
CUDA tensors remains unmeasured, so the cutoff does not guarantee the fastest choice.

Native physics uses float64. MJX uses float32 and can produce different trajectories and rewards.
Select the same backend when comparing experiments.

<!-- env: DMCReacher -->

## Swimmer

### DMCSwimmerSwimmer6

A planar swimmer with six links swims to a target in a viscous fluid.

<!-- env: DMCSwimmerSwimmer6 -->

## Walker

### DMCWalkerWalk

A planar biped walks forward at 1 m/s and keeps its torso upright.

<!-- env: DMCWalkerWalk -->
