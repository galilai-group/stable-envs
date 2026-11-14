# OGBench

[OGBench](https://github.com/seohongpark/ogbench) ([Park et al., 2024](https://arxiv.org/abs/2410.20092)) is a benchmark for offline goal-conditioned reinforcement learning. stable-envs has 27 of its tasks as classes: 25 locomotion mazes, where a point mass, an ant or a humanoid moves through a maze to a goal cell, and 2 manipulation tasks, `OGBCube` and `OGBScene`, where a UR5e arm with a gripper moves objects on a table. MuJoCo steps every environment of a batch in one call on CPU and MuJoCo Warp steps them on CUDA. stable-envs draws the maze [frames](../terms.md#frame) with its own ray caster, and Cube and Scene with MuJoCo OpenGL or MJWarp.

```python
# needs the ogbench extra
import stable_envs as snv

with snv.PointMazeLargeNavigate(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    print(env.render().shape)      # torch.Size([8, 224, 224, 3])
```

## Install

The mazes need the `ogbench` extra, and their default physics on CUDA also needs the `mjwarp` extra. Cube and Scene need `mujoco==3.17.0` and the `ogbench-manipulation` extra, and on CUDA their default physics and `supersample` 2 or more also need `ogbench-manipulation-gpu`; see [What each family needs](../install.md#what-each-family-needs).

The maze classes default to `device="cuda"`; pass `device="cpu"` on a machine without a GPU. Cube and Scene default to `device="cpu"`.

## Task

OGBench sets `success` when the agent reaches the goal, and the step output reports that value as `success`. In a maze, the body must be close to the goal position. In Cube and Scene, each object must match its goal state.

Each maze class name has three parts:

- The body: `PointMaze`, `AntMaze` or `HumanoidMaze`.
- The maze size: `Medium`, `Large`, `Giant` or `Teleport`. The teleport maze has cells that move the agent to another cell.
- The dataset: `Navigate`, `Stitch` or `Explore`. The suffix selects the OGBench task and leaves the frames unchanged: classes that differ only in the suffix have the same clips. The exception is `PointMazeLargeNavigate`, which draws OGBench's state-observation scene, with a checker floor, white walls and the goal disc.

## Simulation

The maze [state](../terms.md#state) holds the joint positions (`qpos`) and velocities of the body.

### Maze physics

One call steps all the environments of a batch, and `simulation_backend` selects the physics:

| Value | Device | Physics | Agreement with OGBench |
|---|---|---|---|
| `"default"` (default) | CPU | `mujoco.batch`, float64 | equal, bit for bit: state, observations, task outputs, resets and RNG streams |
| `"default"` (default) | CUDA | MuJoCo Warp, float32 | equal resets, goals, tasks, teleports and RNG streams; approximate physics |
| `"auto"` | CUDA | `mujoco.batch` below `mjwarp_min_envs` environments, MuJoCo Warp from there | as the selected physics |
| `"native"` | CPU or CUDA | `mujoco.batch`, float64 | equal, bit for bit |
| `"mjwarp"` | CUDA | MuJoCo Warp, float32 | as `"default"` on CUDA |

The default `mjwarp_min_envs` was measured for each body on an L40S with 8 CPU cores: 4096 for PointMaze, 256 for AntMaze and 256 for HumanoidMaze. The outputs are on the device of the environment.

On CPU, `mujoco.batch` steps the physics on its own threads, and the step does its Torch operations on one thread. Idle Torch threads would otherwise keep the CPU cores busy while the `mujoco.batch` threads wait for them. The frames and the state do not change with the thread count.

```python
# CUDA only; needs the ogbench and mjwarp extras
import stable_envs as snv

with snv.AntMazeMediumNavigate(num_envs=64, device="cuda", simulation_backend="auto") as env:
    print(env.simulation_backend)  # native
```

MuJoCo Warp agrees with MuJoCo to float32 rounding on a step where no contact starts or stops. When contacts start or stop, the contact forces of the two solvers differ and the trajectories move apart. After 200 steps of smooth actions from the same state on 64 environments, the median distance between the `xy` positions of MuJoCo Warp and OGBench was:

| Body | Median `xy` distance after 200 steps |
|---|---|
| PointMaze | less than 1e-5 (largest: 0.13 after wall contacts) |
| AntMaze | 0.05 |
| HumanoidMaze | 0.23 (0.11 when only the start state is rounded to float32 in MuJoCo) |

A PointMaze agent that follows the OGBench subgoal oracle reaches the goal at the same step on both backends.

On MuJoCo Warp, PointMaze integrates with Euler, where OGBench uses RK4. The point's step moves the position by the action and sets the velocity to zero, so away from walls the two integrators give the same positions, and only the response to a wall contact differs: over 300 steps of smooth actions on 32 environments, the largest `xy` distance from OGBench was 0.13 (2e-5 with RK4). RK4 runs four forward passes in each substep, so the Euler step is about four times faster.

### Cube and Scene physics

`simulation_backend` picks the physics in the same way:

| Value | CPU device | CUDA device |
|---|---|---|
| `"default"` (default) | `mujoco.batch` | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| `"auto"` | `mujoco.batch` | `mujoco.batch` below `mjwarp_min_envs` (256) environments, MuJoCo Warp from there |
| `"native"` | `mujoco.batch` | `mujoco.batch` |
| `"mjwarp"` | error | MuJoCo Warp |

`mujoco.batch` steps every environment in one native call on up to 8 threads. The step does the operations of OGBench's step in the same order: the end-effector controller and its inverse kinematics, the 25 `mj_step` substeps, `mj_rnePostConstraint` and the task's `post_step`. The state and the outputs equal OGBench's bit for bit, and a CUDA environment on `mujoco.batch` returns its state on CUDA. The default model has the OGBench physics, with a cube size of 0.02.

MuJoCo Warp steps the substeps of every environment in one captured CUDA graph, in float32, and one Warp kernel computes the controller and its inverse kinematics in float64. The state has the same leaves as on CPU, and its physics leaves hold the float32 values. On the scripted OGBench Cube oracle, each episode ends with the same success at the same first-success step as OGBench (see `tests/gpu/test_ogbench_manipulation_warp.py`). Over a few hundred steps, contacts can make the cube positions differ from OGBench.

MuJoCo Warp doesn't test the links of the Robotiq gripper against each other for contact. The pads keep their contacts with each other and with all other geoms. The linkage has one degree of freedom and the pads touch before two links can, so these pairs make no contacts in OGBench either, and the step is faster without their mesh-mesh tests.

The MuJoCo Warp step doesn't synchronize with the host. When a Scene cube leaves the workspace, OGBench puts it at a new position drawn from the environment's generator. The host draws the next two of these positions for each environment before the step, and the device uses them. The host draws again for an environment after it reads that the device used them, and then prints OGBench's message. If an environment needs a third position before that read, the device uses the second one again.

### Cube and Scene resets

A reset is the OGBench reset: resets, restores and variations run OGBench's own code on one OGBench object per environment, and the physics backend then loads the result. On Linux, a batch of at least `worker_min_lanes` environments resets in up to `workers` [worker processes](../performance-tuning.md#worker-processes). OGBench samples the task from NumPy's global RNG, and each environment in the batch keeps its own copy of that RNG in state (`rng/task`). A seeded reset seeds the copy the way `np.random.seed(seed)` would with that environment's seed, and an unseeded reset continues it. OGBench samples the two settling actions of a reset from an unseeded action space. stable-envs samples them from the seeded `rng/action` generator instead. Restoring a state therefore also fixes the next reset.

## Rendering

The analytic ray caster `raycast3d` draws the maze, the body and the floor from the state on CPU or CUDA. On CPU, the render uses the Torch thread count (`torch.set_num_threads`). It draws batches of fewer than 256 frames on one thread when the scene has no tracking light, which is every class except `PointMazeLargeNavigate`.

On CPU, `PointMazeLargeNavigate` draws the tracking light on the static pixels with Numba kernels that do all the frames of a render in one pass. The square roots and the powers of the light stay Torch operations on the same tensors as the Torch path, thus the frames are equal to the frames of the Torch path, byte for byte. Without Numba, the Torch path draws the light. The `ogbench` extra installs Numba.

### Cube and Scene render backends

The frames have the look of the OGBench pixel tasks `visual-cube-single-v0` and `visual-scene-v0`: the `front_pixels` camera, a purple gripper, a see-through arm and no goal cube.

| Device and setting | `backend="auto"` draws with | Frames |
|---|---|---|
| CPU | the MuJoCo rasterizer in Numba, as `backend="numba"` | differ from MuJoCo OpenGL; a draw of the same state gives the same frame each time, in any batch |
| CUDA, `supersample=1` | NVIDIA OpenGL | the driver can change a pixel by 1 or 2 levels (of 255) between two draws of the same state, and so can the OpenGL reference |
| CUDA, `supersample` 2 or more | two-pass MJWarp, as `backend="mjwarp_fidelity"` | can differ from MuJoCo in lighting and transparency |

- `backend="numba"` is the shared [MuJoCo rasterizer](../contributing/renderers.md#mujoco-rasterizer). It follows MuJoCo's OpenGL renderer: the same meshes and primitive tessellations, per-vertex lighting, the 4096 shadow map of the scene light, the floor texture with its mipmaps, the skybox, the see-through arm and 4 samples per pixel. It draws the arm meshes with one vertex per grid cell of 2 mm. On CPU, the first render after an install compiles the Numba kernels in about 1 to 2 minutes, and Numba keeps the compiled code in a cache for later processes. On CUDA, `backend="numba"` raises `ValueError`. `render_semantic(rows)` gives the segmentation and the planar depth of compact render rows.
- Without Numba, the CPU default is software OpenGL (`backend="opengl"`). The CUDA default at `supersample=1` is NVIDIA OpenGL.
- `backend="opengl"` draws the frames of the source. On CPU, these frames are bit-exact. On CUDA, the NVIDIA driver can change a pixel by 1 or 2 levels (of 255) between two draws of the same state, and so can the OpenGL reference.
- CPU OpenGL rendering refuses a hardware GL context. For `backend="opengl"`, configure [Mesa software OpenGL](../install.md#what-each-family-needs) before you import MuJoCo.
- The default `supersample` is 1 on every device. An explicit `supersample` argument keeps its value, and so does the sampling setting stored in a render state you pass in.
- `backend="mjwarp"` needs CUDA.
- On Linux, a batch of states draws in OpenGL [worker processes](../performance-tuning.md#worker-processes), at most `ogb_gl_workers`. On CPU, the workers start when there are at least 2 of them and at least as many states as workers. On CUDA, there is one worker for every 8 states, and the workers start from 16 states. The frames are the same as from one process.
- `light.intensity`, `camera.angle_delta`, `cube.size` and Scene `lock_color` are per-world inputs to one MJWarp context per family. Changing them adds no contexts or compiled models.

## Variation

The maze [variation leaves](../tutorials/variation.md) set the colors of the body, the walls and the floor, the camera, the light and the body's `agent.scale`. Cube and Scene vary colors, the camera, the light, the cube size and the start and goal poses of the cube.

In Cube and Scene, the arm starts at the position OGBench samples. It starts at the [variation](../terms.md#variation) leaf `agent.ee_start_position` only when the reset options name this leaf, or when `set_variation` sets it to a non-default value.

## Classes

### PointMaze

#### PointMazeMediumNavigate

<!-- env: PointMazeMediumNavigate -->

#### PointMazeMediumStitch

<!-- env: PointMazeMediumStitch -->

#### PointMazeLargeNavigate

<!-- env: PointMazeLargeNavigate -->

#### PointMazeLargeStitch

<!-- env: PointMazeLargeStitch -->

#### PointMazeGiantNavigate

<!-- env: PointMazeGiantNavigate -->

#### PointMazeGiantStitch

<!-- env: PointMazeGiantStitch -->

#### PointMazeTeleportNavigate

<!-- env: PointMazeTeleportNavigate -->

#### PointMazeTeleportStitch

<!-- env: PointMazeTeleportStitch -->

### AntMaze

#### AntMazeMediumNavigate

<!-- env: AntMazeMediumNavigate -->

#### AntMazeMediumStitch

<!-- env: AntMazeMediumStitch -->

#### AntMazeMediumExplore

<!-- env: AntMazeMediumExplore -->

#### AntMazeLargeNavigate

<!-- env: AntMazeLargeNavigate -->

#### AntMazeLargeStitch

<!-- env: AntMazeLargeStitch -->

#### AntMazeLargeExplore

<!-- env: AntMazeLargeExplore -->

#### AntMazeGiantNavigate

<!-- env: AntMazeGiantNavigate -->

#### AntMazeGiantStitch

<!-- env: AntMazeGiantStitch -->

#### AntMazeTeleportNavigate

<!-- env: AntMazeTeleportNavigate -->

#### AntMazeTeleportStitch

<!-- env: AntMazeTeleportStitch -->

#### AntMazeTeleportExplore

<!-- env: AntMazeTeleportExplore -->

### HumanoidMaze

#### HumanoidMazeMediumNavigate

<!-- env: HumanoidMazeMediumNavigate -->

#### HumanoidMazeMediumStitch

<!-- env: HumanoidMazeMediumStitch -->

#### HumanoidMazeLargeNavigate

<!-- env: HumanoidMazeLargeNavigate -->

#### HumanoidMazeLargeStitch

<!-- env: HumanoidMazeLargeStitch -->

#### HumanoidMazeGiantNavigate

<!-- env: HumanoidMazeGiantNavigate -->

#### HumanoidMazeGiantStitch

<!-- env: HumanoidMazeGiantStitch -->

### Manipulation

#### OGBCube

The arm moves one cube to a goal pose. This is the OGBench `cube-single` task, and the constructor argument `reward_task_id` picks the evaluation task the reward uses.

<!-- env: OGBCube -->

#### OGBScene

The scene has a cube, a drawer, a window and two button locks. The agent changes the objects to match a goal configuration. A locked object stays in place until the agent presses its button.

<!-- env: OGBScene -->

## Limitations

- Cube and Scene are compared with frames from MuJoCo's OpenGL renderer on the model stable-envs compiles, not with a default OGBench install. The [fidelity record](../environments.md#comparison-with-the-upstream-library) of the NVIDIA OpenGL path compares two stable-envs OpenGL instances with the same compiled appearance.
- With `supersample` 2 or more, MJWarp draws the CUDA frames. It is measured against the same stable-envs OpenGL path, but `visual_fidelity` holds only the values of `backend="opengl"`.
- `HumanoidMazeGiantNavigate` on CUDA isn't reproducible across processes on at least one GPU.
- MuJoCo Warp orders contacts with atomic operations, so two runs from one state can differ by rounding. Use `simulation_backend="native"` when you need results equal to OGBench.
- The default CPU `numba` frames differ from the `backend="opengl"` frames that `visual_fidelity` records. Only `backend="opengl"` gives the frames of the source.
