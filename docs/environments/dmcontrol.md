# DM Control

The [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) ([Tassa et al., 2018](https://arxiv.org/abs/1801.00690)) is a set of continuous control tasks, from a pendulum swing-up to a walking humanoid, and stable-envs has 17 of them as classes. dm_control builds, resets and restores each task. On CPU, MuJoCo steps the suite's own models in float64, and on CUDA, MuJoCo Warp steps them in float32. stable-envs ray casters draw the [frames](../terms.md#frame), except for the body-tracking tasks on CUDA, which MJWarp draws.

```python
# needs the dmcontrol extra
import stable_envs as snv

with snv.DMCReacher(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    out = env.step(env.action_spec.sample(8))
    print(out["reward"].shape)     # torch.Size([8])
    print(env.render().shape)      # torch.Size([8, 84, 84, 3])
```

## Install

The classes need the `dmcontrol` extra, which also installs Numba for the CPU renderer. On CUDA, the default physics and the body-tracking renderer need MuJoCo Warp, and Reacher's MJX physics needs MJX. The `dmcontrol-gpu` extra installs both; see [what each family needs](../install.md#what-each-family-needs).

These classes default to `device="cuda"`; pass `device="cpu"` on a machine without a GPU.

## Task

Each action dimension drives one actuator of the model, in the order of the MuJoCo model. Actions lie in [-1, 1] and are rescaled to each actuator's control range. Each `step` applies the action for two control steps of the task and returns the sum of their rewards, each in [0, 1]. Only Ball in cup defines a success criterion: `success` is 1 when the ball is in the cup.

## Simulation

dm_control resets and restores each task. On CPU, every environment of a batch steps in one `mujoco.batch` call on a pool of threads. `dmc_threads` caps the number of threads, and a step with little MuJoCo work uses fewer: when you construct the environment, it times one MuJoCo step and gives each thread about 0.1 ms of work per call and at least two environments. The number of threads doesn't change the [state](../terms.md#state) or the outputs. Build the environment in the process that steps it: the `mujoco.batch` threads don't survive `fork()`.

### Physics backends

`simulation_backend` picks the physics when you construct the environment:

| Value | CPU device | CUDA device |
|---|---|---|
| `None` (default) | native MuJoCo | MuJoCo Warp at every batch size |
| `"native"` | native MuJoCo | native MuJoCo on CPU, with state and outputs on CUDA |
| `"mjwarp"` | error | MuJoCo Warp |
| `"auto"` | native MuJoCo | native MuJoCo below `mjwarp_min_envs`, MuJoCo Warp from there |
| `"mjx"` | error | JAX/MJX, `DMCReacher` only |

`"mjwarp"` and `"mjx"` raise an error without CUDA or without the GPU dependencies. With `"auto"`, `mjwarp_min_envs` defaults to the measured cutoff of each task: the smallest measured batch from which MuJoCo Warp stepped faster than native MuJoCo at every larger measured batch. The measurement timed public `step` calls with `states=None, frames=None`, including episode resets, at batches 1, 8, 64, 256, 1024, 2048 and 4096 on an NVIDIA L40S with 8 CPU cores. Pass an integer to change the cutoff.

| Class | `mjwarp_min_envs` |
|---|---:|
| `DMCCheetahRun` | 1 |
| `DMCFingerTurnHard`, `DMCPendulumSwingup`, `DMCPointMassEasy`, `DMCReacher` | 8 |
| `DMCCartpoleBalance`, `DMCCartpoleSwingup`, `DMCHumanoidWalk`, `DMCQuadrupedWalk` | 64 |
| `DMCBallInCupCatch`, `DMCHopperHop` | 256 |
| `DMCAcrobotBalance`, `DMCFingerSpin`, `DMCFingerTurnEasy`, `DMCManipulatorBringBall`, `DMCSwimmerSwimmer6`, `DMCWalkerWalk` | 1024 |

```python
# needs the dmcontrol-gpu extra; CUDA only
with snv.DMCReacher(num_envs=1, device="cuda") as env:
    print(env.simulation_backend)  # mjwarp

with snv.DMCReacher(num_envs=1, device="cuda", simulation_backend="auto") as env:
    print(env.simulation_backend)  # native
```

On MuJoCo Warp, dm_control still resets the environments and owns their random streams, and the state keeps its shapes and dtypes, including the float64 physics columns. `set_state` followed by `get_state` returns the same state exactly, but two runs from one state on the GPU aren't guaranteed to match bit for bit. `CheetahRun` and `QuadrupedWalk` reset their environments on one dm_control owner per `dmc_threads` thread, and the other tasks on one owner. The physics substeps, RK4 integration and sensor timing follow the source.

`simulation_backend="mjx"` keeps Reacher's JAX/MJX physics. On an L40S, MuJoCo Warp stepped Reacher faster than MJX at every batch from 1 to 4096, for example 1,167,156 against 552,472 steps per second at 4096 environments, including resets.

## Rendering

The renderers come in two groups:

- The fixed-camera tasks (Acrobot, Ball in cup, Cartpole, Finger, Manipulator, Pendulum, Point mass and Reacher) use the analytic ray caster `raycast3d`. The tests compare their frames with MuJoCo's frames pixel by pixel.
- The body-tracking tasks (Cheetah, Hopper, Humanoid, Quadruped, Swimmer and Walker) use `cpu3d` on CPU and MJWarp on CUDA. Their frames differ from MuJoCo's; see [Limitations](#limitations).

On CPU, one compiled Numba kernel draws the frames of each group, and the `dmcontrol` extra installs Numba. The kernel splits each frame into tiles of 8 × 8 pixels, and each tile tests its rays only against the moving geometry its bounds can reach, for shadow and mirror rays too. It then shades the pixels in the same order of float32 operations as the Torch renderer. `backend="auto"` (the default) picks Numba when it imports, and `backend="torch"` picks the Torch renderers. The two backends give the same frames except for rare pixels where Torch's CPU square root rounds differently from the IEEE square root the kernels use.

## Variation

Every class has `material.*`, `floor.color` and `light.*` [variation leaves](../tutorials/variation.md). Eleven tasks have `camera.fovy_scale`, and most have `gravity.*` and `agent.*` leaves, such as body densities or the cart mass. Cheetah, Hopper, Humanoid, Quadruped and Walker also vary `floor.friction`. `env.physics_variation_leaves` lists the leaves the simulator reads as physics.

## Classes

### Acrobot

#### DMCAcrobotBalance

Acrobot is a chain of two links with one actuator at the elbow. The task is to swing the free end up above the base and keep it there, with a sparse reward. The episode starts from random joint angles.

<!-- env: DMCAcrobotBalance -->

### Ball in cup

#### DMCBallInCupCatch

A cup on a planar arm holds a ball on a string. The agent swings the ball up and catches it in the cup.

<!-- env: DMCBallInCupCatch -->

### Cartpole

#### DMCCartpoleBalance

The pole starts near the upright pose. The agent moves the cart to keep it upright.

<!-- env: DMCCartpoleBalance -->

#### DMCCartpoleSwingup

The pole starts pointing down. The agent swings it up and keeps it upright.

<!-- env: DMCCartpoleSwingup -->

### Cheetah

#### DMCCheetahRun

A planar biped runs forward. The reward increases with the forward speed, up to 10 m/s.

<!-- env: DMCCheetahRun -->

### Finger

#### DMCFingerSpin

A planar finger with two joints spins a hinged body as fast as possible.

<!-- env: DMCFingerSpin -->

#### DMCFingerTurnEasy

The finger turns the body until its tip points at a target. The target is large.

<!-- env: DMCFingerTurnEasy -->

#### DMCFingerTurnHard

This task is the same as DMCFingerTurnEasy with a smaller target.

<!-- env: DMCFingerTurnHard -->

### Hopper

#### DMCHopperHop

A planar one-legged body hops forward. The reward needs an upright torso and forward speed.

<!-- env: DMCHopperHop -->

### Humanoid

#### DMCHumanoidWalk

A humanoid with 21 joints walks forward at 1 m/s.

<!-- env: DMCHumanoidWalk -->

### Manipulator

#### DMCManipulatorBringBall

A planar arm with a gripper picks up a ball and brings it to a target position.

<!-- env: DMCManipulatorBringBall -->

### Pendulum

#### DMCPendulumSwingup

The pendulum starts at a random angle. The agent swings it up and holds it upright with a limited torque.

<!-- env: DMCPendulumSwingup -->

### Point mass

#### DMCPointMassEasy

A point mass moves on a plane. The agent drives it to a target at the center.

<!-- env: DMCPointMassEasy -->

### Quadruped

#### DMCQuadrupedWalk

A body with four legs walks forward.

<!-- env: DMCQuadrupedWalk -->

### Reacher

#### DMCReacher

A planar arm with two links moves its fingertip to a random target. It also has an MJX backend; see [Physics backends](#physics-backends).

<!-- env: DMCReacher -->

### Swimmer

#### DMCSwimmerSwimmer6

A planar swimmer with six links swims to a target in a viscous fluid.

<!-- env: DMCSwimmerSwimmer6 -->

### Walker

#### DMCWalkerWalk

A planar biped walks forward at 1 m/s and keeps its torso upright.

<!-- env: DMCWalkerWalk -->

## Limitations

- The body-tracking frames differ from MuJoCo's by a [measured amount](../environments.md#comparison-with-the-upstream-library).
- The body-tracking classes draw `floor.color` from a per-texel palette texture, and `DMCWalkerWalk` `floor.rotation_y` tilts the floor per batch row.
- Native physics uses float64, and MuJoCo Warp and MJX use float32. They can produce different trajectories and rewards, and the gap grows on contact-heavy tasks. Use the same backend for every run you compare.
- On MuJoCo Warp 3.17, the CUDA runtime corrects plane-capsule and capsule-capsule contacts to follow native MuJoCo, and orders the constraint rows of Cheetah, Humanoid and Manipulator to remove sources of variation between replays. [Pull request #28](https://github.com/galilai-group/stable-envs/pull/28) has the error curves against native physics.
- The tests compare against the upstream pins in `pin-dmc`: `mujoco==3.17.0` and `dm-control==1.0.50`. The `VISUAL_FIDELITY` declarations of the body-tracking classes record the versions they were measured with.
