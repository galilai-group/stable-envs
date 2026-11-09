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

All 17 classes accept `simulation_backend="mjwarp"` for direct batched physics on
CUDA. Install `stable-envs[dmcontrol-gpu]`; this backend uses MuJoCo/MuJoCo Warp 3.17
and dm-control 1.0.50. Explicit selection
raises an error if CUDA or the dependencies are unavailable. `backend` still selects rendering.
Native CPU stepping uses MuJoCo's `mujoco.batch` backend and the `dmc_threads` tuning setting.
Automatic routing is unchanged, including Reacher's existing native/MJX selection.

```python
with snv.DMCFingerTurnHard(num_envs=256, device="cuda", simulation_backend="mjwarp") as env:
    state = env.reset(seed=7)
    result = env.step(env.action_spec.sample(256))
    checkpoint = env.get_state()
    env.set_state(checkpoint)
```

The shared adapter owns lifecycle, the two action repetitions, native reset sampling and checkpoints.
A private runtime owns Warp buffers, CUDA stream ordering, graph capture and physics substeps. Bound
domain tasks compute observations, rewards and info on device; Cartpole and Finger variants share
formulas. Existing model factories, State schemas and physical-variation calculations are reused.

Warp physics uses float32; public State retains its existing shapes and dtypes, including float64
physical columns. `set_state(s); get_state()` preserves the supplied checkpoint exactly, including
warmstart, activation, lifecycle, cumulative reward, complete RNG, episode targets and variation.
Snapshots own their memory. This does not guarantee bitwise identical future GPU trajectories.
Rendering continues to consume State, independently of the stepping backend.

Source physics substeps, RK4 and acceleration-stage sensor timing are retained. Native environments
remain authoritative for resets and RNG. A pending reset consumes a control round with zero reward;
continuing lanes use a compact scratch runtime. A changed active-lane count requires a new scratch
allocation, so mixed resets can cost more than steady-state stepping.

For Warp 3.17 the runtime corrects plane–capsule friction frames and capsule–capsule geometry to
follow native MuJoCo's [primitive collisions](https://github.com/google-deepmind/mujoco/blob/3.17.0/src/engine/engine_collision_primitive.c).
The capsule correction uses float64 local arithmetic for closest points and intersecting-axis normals.
It follows MuJoCo 3.17's relative angular cutoff, including the native one- or two-contact
result for near-parallel capsules. Stored dynamics and integration remain float32. Focused tests compare contact geometry, impulses and touch sensors
against native MuJoCo. Acceleration-stage readings are cleared before stepping so touch forces do
not accumulate from checkpoint reconstruction.
Cheetah and Humanoid order joint-limit and contact constraints consistently before each solve, including
after checkpoint restoration and mixed resets. This removes replay variation caused by parallel
constraint allocation; the solver equations, contacts and iteration settings stay unchanged.
Humanoid also orders changed constraint rows before incremental Newton updates and sums each world's
gradient norm in a fixed order, including in mixed-reset scratch runtimes. Its DMC-local Newton loop
retains upstream warmstart, line search, incremental Hessian updates, convergence and stable-state
shortcuts. Constraint ordering alone does not make these solver reductions deterministic.
Manipulator accumulates elliptic contact curvature in constraint-row order, with one writer per
world and Hessian entry. This removes a source of floating-point atomic-sum variation while keeping
worlds parallel. Its DMC-local dense Newton loop reuses MuJoCo Warp 3.17's line search, convergence
and factorization kernels, with the original warmstart and iteration settings.

### Validation status

Run the focused CUDA checks with `python scripts/ci/run_dmc_mjwarp.py`; missing CUDA or skipped
required cases fail this runner. It covers all 17 classes, checkpoint ownership, resets, targets,
physics variation, rendering captured States, stream ordering and contact/solver regressions.

The recorded MuJoCo 3.17 evaluation passed all 35 focused CUDA checks and 51 checkpoint-replay
cases (17 environments × three seeds, at batch 3) over 100 public calls. Exact replay in those
cases does not establish numerical equivalence with native float64 physics. CPU/GPU errors
grow on contact-heavy trajectories, and random actions provide limited sparse-reward coverage.
The backend remains experimental; no new numerical acceptance threshold was applied.

[Validation reports and 100-call error curves](https://github.com/galilai-group/stable-envs/blob/782cba6f35ee316f22499c7d83402320e6012f79/docs/assets/dmc-mjwarp-317/README.md)
include tested revisions, shared baseline failures and hardware-specific skips.

### Benchmarking

`python benchmarks/dmc_mjwarp.py --help` describes the configurable latency and memory benchmark.
It separates construction, reset, restore, raw physics, public stepping and rendering costs.
The default matrix covers all 17 environments with native and MJWarp backends at batches
1, 16, 64, 256 and 1024. `--batches` selects arbitrary positive batch sizes;
`--extended-batches` appends 2048, 4096 and 8192 without duplicating selected sizes.
Preview a light/heavy task matrix without initializing CUDA, constructing environments or
launching workers:

```bash
python benchmarks/dmc_mjwarp.py --dry-run --pretty \
  --envs DMCPendulumSwingup,DMCCheetahRun --batches 1024 --extended-batches
```

Remove `--dry-run` and supply `--output results/dmc-mjwarp.json` to measure that matrix.
The script requires `psutil` and `nvidia-ml-py` for measurement, but not for the preview.

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

The [MuJoCo 3.17 timing curves](https://github.com/galilai-group/stable-envs/blob/782cba6f35ee316f22499c7d83402320e6012f79/docs/assets/dmc-mjwarp-317/README.md#timing)
compare sequential CPU, native CPU batching and RTX 3090 stepping at batches 1, 256 and 1024,
using 100 synchronized public calls per seed. Their captions specify hardware, included work
and aggregation. Those random-action rollout measurements use a different protocol from the
general benchmark script.

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

Set `simulation_backend="native"`, `"mjx"` or `"mjwarp"` to choose explicitly. Both GPU backends
require CUDA and `stable-envs[dmcontrol-gpu]`. Reacher's MJWarp path uses the same adapter as the
other DMC tasks, including native reset sampling, per-lane targets, physics variation and full
checkpoints. Set `mjx_min_envs` to change the automatic native/MJX cutoff; MJWarp is selected explicitly.

```python
with snv.DMCReacher(num_envs=256, device="cuda", simulation_backend="mjwarp") as env:
    state = env.reset(seed=7)
    result = env.step(env.action_spec.sample(256), states=None, frames=None)
```

The default cutoff of 32 is provisional. Earlier A40 measurements compared native physics with
CPU outputs against MJX with CUDA outputs. Those measurements used different output devices,
so the cutoff does not guarantee the fastest choice.

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
