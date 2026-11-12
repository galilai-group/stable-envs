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

On CPU, one compiled Numba kernel draws the frames of each group. The `dmcontrol` extra installs Numba.
The kernel divides each frame into tiles of 8 × 8 pixels. Each tile tests its rays only against the
moving geometry that its bounds can reach. Shadow rays and mirror rays use the same test. The kernel
then shades the pixels in the same order of float32 operations as the Torch renderer.
`backend="torch"` selects the Torch renderers, and `backend="auto"` (the default) selects Numba when it
imports. The two backends give the same frames, except for rare pixels. Torch's CPU square root can
round differently from the IEEE square root that the kernels use.

On CPU, all environments of a class step in one `mujoco.batch` call, which runs the MuJoCo steps on a
pool of threads. `dmc_threads` sets the largest number of threads. A step with little MuJoCo work uses
fewer threads, because each thread that the call wakes adds a delay. The class times one MuJoCo step when
you construct it and gives each thread about 0.1 ms of work for each call, and at least two environments.
The number of threads does not change State or outputs.

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

## GPU physics and backend selection

On a CUDA device, all 17 classes step with MuJoCo Warp at every batch size. On CPU they step with
MuJoCo's `mujoco.batch` backend and the `dmc_threads` tuning setting. Install
`stable-envs[dmcontrol-gpu]` for the GPU backend; it uses MuJoCo/MuJoCo Warp 3.17 and dm-control 1.0.50.
`backend` still selects rendering.

`simulation_backend` selects physics when you construct the environment:

| Value | CPU device | CUDA device |
|---|---|---|
| `None` (default) | Native | MuJoCo Warp at every batch size |
| `"native"` | Native | Native physics on CPU; State and outputs on CUDA |
| `"mjwarp"` | Error | MuJoCo Warp |
| `"auto"` | Native | Native below `mjwarp_min_envs`, MuJoCo Warp from there |

`"auto"` is opt-in. `mjwarp_min_envs` defaults to the measured cutoff of the task in the table
below. Pass an integer to change it. `"mjwarp"` raises an error if CUDA or the GPU dependencies are
unavailable.

The cutoff of a task is the smallest measured batch from which MuJoCo Warp steps faster than native
physics at every larger measured batch. The measurement used public `env.step` calls with
`states=None, frames=None` on an NVIDIA L40S with 8 CPU cores, and includes the episode reset
(an episode is 500 public calls). Batches were 1, 8, 64, 256, 1024, 2048 and 4096.

| Class | `mjwarp_min_envs` |
|---|---:|
| `DMCCheetahRun` | 1 |
| `DMCFingerTurnHard`, `DMCPendulumSwingup`, `DMCPointMassEasy`, `DMCReacher` | 8 |
| `DMCCartpoleBalance`, `DMCCartpoleSwingup`, `DMCHumanoidWalk`, `DMCQuadrupedWalk` | 64 |
| `DMCBallInCupCatch`, `DMCHopperHop` | 256 |
| `DMCAcrobotBalance`, `DMCFingerSpin`, `DMCFingerTurnEasy`, `DMCManipulatorBringBall`, `DMCSwimmerSwimmer6`, `DMCWalkerWalk` | 1024 |

```python
with snv.DMCFingerTurnHard(num_envs=256, device="cuda", simulation_backend="mjwarp") as env:
    state = env.reset(seed=7)
    result = env.step(env.action_spec.sample(256))
    checkpoint = env.get_state()
    env.set_state(checkpoint)
```

The adapter owns the lifecycle, the two action repetitions, native reset sampling and checkpoints.
The shared `WarpBatch` backend (`stable_envs/_execution/mjbatch.py`) owns the Warp buffers and
steps each control round in one captured CUDA graph. Bound domain tasks compute observations,
rewards and info on the device. Cartpole and Finger variants share formulas.

Warp physics uses float32. The public State keeps its shapes and dtypes, including the float64
physical columns. `set_state(s); get_state()` gives back the checkpoint exactly, including
warmstart, activation, lifecycle, cumulative reward, complete RNG, episode targets and variation.
Snapshots own their memory. MuJoCo Warp does not repeat a contact solve bit for bit, so two
rollouts from one State can differ after contacts. Rendering uses the State and does not depend on
the stepping backend.

Each control round follows dm_control's `legacy_step`: `mj_step2`, then `n - 1` calls of
`mj_step`, then `mj_step1`. RK4 tasks use `n` calls of `mj_step`, then `mj_step1`. After a round,
the data hold the position and velocity stages of the new state and the acceleration-stage sensor
values of the last integration. Contacts, constraints and the solver are those of the MuJoCo Warp
3.17 step, with no changes.

Native environments own resets and RNG. Each environment keeps its complete RNG state, episode
targets, physics values and compact model columns. The adapter writes the model values of each
environment into the per-environment model fields of MuJoCo Warp, and writes only the fields that
changed. For Cheetah and Quadruped, one native reset owner per `dmc_threads` thread resets a share
of the environments in parallel. Other tasks use one reset owner. A pending reset uses one control
round with zero reward. When only some environments reset, the full batch advances and the reset
environments get their new State after the round.

The step does not read values from the device. Nonfinite controls or physics raise
`dm_control.rl.control.PhysicsError` one or two steps later. A contact or constraint buffer
overflow raises `BufferOverflowError` in the same way. `check_buffers()` on the simulator raises
the error immediately.

### Validation status

Run the focused CUDA checks with `python scripts/ci/run_dmc_mjwarp.py`. Missing CUDA or skipped
required cases make this runner fail. It covers all 17 classes, checkpoint ownership, resets,
targets, physics variation, rendering of captured States, stream ordering, the legacy step order
and the sensor timing.

The table compares MuJoCo Warp with native CPU physics over 300 public calls (600 control rounds)
from the same reset (`seed=5`), on 16 environments with slow sinusoidal actions. It gives the
median over environments of the absolute return difference, the mean over the rollout of the
largest `qpos` difference of the median environment, and the largest `qpos` difference.

| Class | Return, CPU (median) | Return difference | Mean `qpos` difference | Largest `qpos` difference |
|---|---:|---:|---:|---:|
| `DMCAcrobotBalance` | 0 | 0 | 6.5e-06 | 0.0056 |
| `DMCBallInCupCatch` | 0 | 0 | 7.8e-08 | 0.0011 |
| `DMCCartpoleBalance` | 177 | 0.00021 | 1.6e-05 | 0.0024 |
| `DMCCartpoleSwingup` | 0 | 0 | 7.2e-06 | 0.00055 |
| `DMCCheetahRun` | 3.84 | 1.7e-06 | 1.4e-07 | 1.4e-06 |
| `DMCFingerSpin` | 0 | 0 | 1.7e-06 | 6.2e-05 |
| `DMCFingerTurnEasy` | 0 | 0 | 1.3e-06 | 4.7e-05 |
| `DMCFingerTurnHard` | 0 | 0 | 1.3e-06 | 4.7e-05 |
| `DMCHopperHop` | 0 | 0 | 0.00052 | 1.9 |
| `DMCHumanoidWalk` | 0.66 | 4.9e-07 | 0.69 | 2.9 |
| `DMCManipulatorBringBall` | 2.8e-53 | 1.3e-58 | 0.0048 | 2 |
| `DMCPendulumSwingup` | 0 | 0 | 4.3e-07 | 3.3e-06 |
| `DMCPointMassEasy` | 3.8e-155 | 2.6e-158 | 8.3e-08 | 1.7e-06 |
| `DMCQuadrupedWalk` | 12.9 | 3.9e-05 | 3.4e-06 | 0.088 |
| `DMCReacher` | 4 | 0 | 7.7e-07 | 3.9e-06 |
| `DMCSwimmerSwimmer6` | 11.3 | 6e-06 | 3.5e-06 | 2e-05 |
| `DMCWalkerWalk` | 16.0 | 0.024 | 0.0098 | 2 |

In tasks with contacts the float32 trajectories drift apart and the drift grows over the rollout, but
the returns stay close. Humanoid falls under these actions on both backends. Many tasks give a zero
return under these actions, so their return difference shows little; the `qpos` columns cover them.

### Benchmarking

`python benchmarks/dmc_mjwarp.py --help` describes the configurable latency and memory benchmark.
It separates construction, reset, restore, raw physics, public stepping and rendering costs.
The default matrix covers all 17 environments with native and MJWarp backends at batches
1, 16, 64, 256 and 1024, with reset seeds 91000, 91010 and 91020 (510 records).
`--seed` selects one seed; `--seeds` replaces the seed list. `--batches` selects arbitrary positive batch sizes;
`--extended-batches` appends 2048, 4096 and 8192 without duplicating selected sizes.
Preview a light/heavy task matrix without initializing CUDA, constructing environments or
launching workers:

```bash
python benchmarks/dmc_mjwarp.py --dry-run --pretty \
  --envs DMCPendulumSwingup,DMCCheetahRun --batches 1024 --extended-batches
```

Remove `--dry-run` and supply `--output results/dmc-mjwarp.json` to measure that matrix.
The script requires `psutil` and `nvidia-ml-py` for measurement, but not for the preview.

For a focused stepping and memory baseline, add `--measurement-profile focused`. This retains
construction, first public step, steady public stepping without pixels, raw physics, and their
memory measurements. It skips the repeated reset/restore and rendering phases; the seeded
reset and checkpoint restores needed to initialize each rollout still run outside timing.
The default `full` profile includes every phase. Select the same profile for both revisions.

```bash
python benchmarks/dmc_mjwarp.py --measurement-profile focused --dry-run --pretty \
  --envs DMCPendulumSwingup,DMCCheetahRun,DMCHumanoidWalk --batches 1,256,1024
```

This matrix has 54 records. Comparing public-step and raw-physics timings helps track the
cost of task evaluation, snapshots, validation and synchronization together; their difference
does not isolate synchronization alone. Construction RSS and process-lifetime peak RSS help
track host memory, with the scopes described below.

The stepping workload follows the PR's random-action recipe. NumPy `default_rng(seed + 91717)`
draws uniform `[-1, 1)` float64 actions in `[lane, call, component]` order, then transposes them
to contiguous `[call, lane, component]` arrays. Both backends receive identical precomputed
CUDA inputs, and smaller batches share the corresponding lane prefixes. By default, each
stepping phase runs 20 warmup actions, restores its seeded native reset checkpoint, and replays
the sequence from index zero for 100 timed calls. `--warmup` and `--repeats` override these counts;
the sequence length is their maximum. Action generation, upload, control scaling for raw physics,
and checkpoint restoration stay outside the timed intervals. The headline metric is
`public_step_no_pixels`: complete public State/task outputs with frames and history disabled.
Rendering and raw physics remain separate measurements. Memory profiling makes one additional
call using action index zero after each timed rollout.

Each environment/batch/seed/backend runs in a fresh process. Backend order alternates across
seed indices. Results retain action and initial-checkpoint SHA-256 hashes; a pair with different
inputs is marked as an error. The resolved protocol, actual native thread count, Torch thread
count and CPU affinity are recorded. The default render size is 32, as in the original harness.

Construction records include wall time and memory samples. On Linux/macOS,
`memory_after.rss_peak_self_bytes` is the operating system's process-lifetime peak resident
host RAM through construction, including startup; later memory samples retain the high-water
mark through their phase. Each row also includes `worker_peak_rss_bytes` collected when its
isolated worker exits, even if it is killed before writing a result. Unsupported platforms
report `null`. These are absolute process high-water marks, not per-phase allocation deltas
or peak concurrent RAM summed across worker processes. On Linux, the exit-time value can
also include the largest reaped descendant. Existing RSS sums and NVML readings remain point
samples. `worker_exit` records the exit code and signal; a `SIGKILL` alone does not prove OOM.
Failed cells remain explicit error rows, and the sweep continues to the next cell.

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

On CUDA, Reacher steps with MuJoCo Warp by default, like the other tasks. See
[GPU physics and backend selection](#gpu-physics-and-backend-selection) for `simulation_backend`
and `mjwarp_min_envs`. Reacher's cutoff is 8 environments.

```python
with snv.DMCReacher(num_envs=1, device="cuda") as env:
    print(env.simulation_backend)  # "mjwarp"

with snv.DMCReacher(num_envs=1, device="cuda", simulation_backend="auto") as env:
    print(env.simulation_backend)  # "native"
```

`simulation_backend="mjx"` selects JAX/MJX on CUDA. It needs `stable-envs[dmcontrol-gpu]`. On an
NVIDIA L40S, MuJoCo Warp stepped Reacher faster than MJX at every batch from 1 to 4096 (for example
1,167,156 and 552,472 steps per second at 4096 environments, including resets). Reacher's MJWarp path uses the same adapter as the
other DMC tasks, including native reset sampling, per-environment targets, physics variation and full
checkpoints.

```python
with snv.DMCReacher(num_envs=256, device="cuda", simulation_backend="mjwarp") as env:
    state = env.reset(seed=7)
    result = env.step(env.action_spec.sample(256), states=None, frames=None)
```

Native physics uses float64. MJX and MJWarp use float32 and can produce different trajectories and rewards.
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
