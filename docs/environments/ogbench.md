# OGBench

These 27 classes port environments of [OGBench](https://github.com/seohongpark/ogbench)
([Park et al., 2024](https://arxiv.org/abs/2410.20092)). OGBench is a benchmark for offline
goal-conditioned reinforcement learning. MuJoCo steps the physics: the mazes step all environments of a
batch together on CPU or CUDA, and the manipulation tasks step on CPU, or with MJX on CUDA for large `OGBCube`
batches.

The family has two groups:

- 25 locomotion mazes. A point mass, an ant or a humanoid moves through a maze to a goal cell.
- 2 manipulation tasks, `OGBCube` and `OGBScene`. A UR5e arm with a gripper moves objects on a table.

**Success:** OGBench sets `success` when the agent reaches the goal. In a maze, the body must be close
to the goal position. In Cube and Scene, each object must match its goal state. The step output reports
the source value as `success`.

```python
import stable_envs as snv

env = snv.PointMazeLargeNavigate(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

## Mazes

Each maze class name has three parts:

- The body: `PointMaze`, `AntMaze` or `HumanoidMaze`.
- The maze size: `Medium`, `Large`, `Giant` or `Teleport`. The teleport maze has cells that move the
  agent to another cell.
- The dataset: `Navigate`, `Stitch` or `Explore`. The dataset suffix selects the OGBench task. It does
  not change the frames, so classes that differ only in the suffix have the same clips.

The State holds the joint positions (`qpos`) and velocities of the body. The renderer is the analytic
ray caster `raycast3d`. It draws the maze, the body and the floor from the State on CPU or CUDA.

### Maze physics

One call steps all the environments of a batch. The `simulation_backend` argument selects the physics:

| `simulation_backend` | Device | Physics | Agreement with OGBench |
| --- | --- | --- | --- |
| `"default"` | CPU | `mujoco.batch`, float64 | Equal, bit for bit: State, observations, task outputs, resets and RNG streams. |
| `"default"` | CUDA | MuJoCo Warp, float32 | Resets, goals, tasks, teleports and RNG streams are equal. The physics is approximate (see below). |
| `"auto"` | CUDA | `mujoco.batch` for batches smaller than `mjwarp_min_envs`, MuJoCo Warp for larger batches | As the selected physics. |
| `"native"` | CPU or CUDA | `mujoco.batch`, float64 | Equal, bit for bit. |
| `"mjwarp"` | CUDA | MuJoCo Warp, float32 | As `"default"` on CUDA. |

The default `mjwarp_min_envs` is a measured batch size for each body: 4096 for PointMaze, 256 for AntMaze and
256 for HumanoidMaze (L40S GPU, 8 CPU cores). The outputs are always on the device of the environment.

```python
env = snv.AntMazeMediumNavigate(num_envs=64, device="cuda", simulation_backend="auto")
```

MuJoCo Warp agrees with MuJoCo to float32 rounding on a step without a change of contacts. When contacts start
or stop, the contact forces of the two solvers are different, and the trajectories move apart. These are the
median distances between the `xy` positions of MuJoCo Warp and of OGBench after 200 steps of smooth actions
from the same State (64 environments):

| Body | Median `xy` distance after 200 steps |
| --- | --- |
| PointMaze | less than 1e-5 (largest: 0.13 after wall contacts) |
| AntMaze | 0.05 |
| HumanoidMaze | 0.23 (0.11 when only the start State is rounded to float32 in MuJoCo) |

A PointMaze agent that follows the OGBench subgoal oracle reaches the goal at the same step on both backends.

On MuJoCo Warp, PointMaze integrates with Euler, and OGBench integrates with RK4. The point's step moves the
position by the action and sets the velocity to zero, so away from walls the two integrators give the same
positions. Only the response to a wall contact is different: over 300 steps of smooth actions on 32
environments, the largest `xy` distance from OGBench is 0.13 (2e-5 with RK4). RK4 runs four forward passes in
each substep, so the Euler step is about four times faster.
MuJoCo Warp orders contacts with atomic operations, so two runs from one State can differ by rounding. Use
`simulation_backend="native"` when you need results that are equal to OGBench.

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

## Manipulation

Cube and Scene need `mujoco==3.17.0` and the `ogbench-manipulation` extra (`ogbench-manipulation-gpu` for
MuJoCo Warp). MuJoCo OpenGL draws the frames on CPU (software OpenGL) and on CUDA (NVIDIA OpenGL).

| `simulation_backend` | CPU env | CUDA env |
|---|---|---|
| `"default"` (default) | `mujoco.batch` | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| `"auto"` | `mujoco.batch` | `mujoco.batch` below `mjwarp_min_envs` (256) environments, MuJoCo Warp from 256 |
| `"native"` | `mujoco.batch` | `mujoco.batch` |
| `"mjwarp"` | not supported | MuJoCo Warp |

`mujoco.batch` steps every environment in one native call on up to 8 threads. The step does the operations of
OGBench's step in the same order: the end-effector controller and its inverse kinematics, the 25 `mj_step`
substeps, `mj_rnePostConstraint` and the `post_step` of the task. The State and the outputs are equal to
OGBench, bit for bit. A CUDA env on `mujoco.batch` returns its State on CUDA.

MuJoCo Warp steps the substeps of every environment in one captured CUDA graph, in `float32`. One Warp kernel
computes the controller and its inverse kinematics in `float64`. The State has the same leaves as on CPU. Its
physics leaves hold the `float32` values. MuJoCo Warp is approximately equal to OGBench: on the scripted OGBench
Cube oracle, each episode ends with the same success at the same first-success step (see the test
`tests/gpu/test_ogbench_manipulation_warp.py`). Over a few hundred steps, contacts can make the cube positions
differ from OGBench.

MuJoCo Warp does not test the links of the Robotiq gripper against each other for contact. The pads keep their
contacts with each other and with all other geoms. The linkage has one degree of freedom, and the pads touch
before two links can touch, so these pairs make no contacts in OGBench. Their bounding volumes overlap on every
substep, and each test is a mesh-mesh collision test, so the step is faster without them.

The resets, restores and Variations run OGBench's own code on one OGBench object per environment, and the
backend then loads the result. A small CUDA batch steps faster on `mujoco.batch` than on MuJoCo Warp; use
`simulation_backend="auto"` for this.

The frames have the look of the OGBench pixel tasks `visual-cube-single-v0` and `visual-scene-v0`: the
`front_pixels` camera, a purple gripper, a see-through arm, and no goal cube.

- On CPU, the default backend is `opengl_fast`. It is the same software OpenGL path with simplified arm
  meshes and a 2048 shadow map (MuJoCo uses 4096). The arm meshes keep one vertex per grid cell of 3 mm.
  The mean frame MSE against `backend="opengl"` is 1.2 (Cube) and 3.4 (Scene) on
  the 0-255 scale. The frames are about 3 times faster.
- `backend="opengl"` draws the frames of the source.
- CPU frames are bit-exact. A draw of the same state gives the same frame each time.
- CUDA frames use the same MuJoCo OpenGL path. The NVIDIA driver can change a pixel by 1/255 between
  two draws of the same state, and the same variation appears in the comparison renderer.
- With `supersample` 2 or more, MJWarp draws the CUDA frames. MJWarp frames can differ from MuJoCo in
  lighting and transparency.

A reset is the OGBench reset. OGBench samples the task from NumPy's global RNG. Each lane keeps its own
copy of that RNG in State (`rng/task`). A seeded reset seeds the copy as `np.random.seed(seed)` does with
the lane seed, and an unseeded reset continues it. OGBench samples the two settling actions of a reset from
an action space that has no seed. The lanes sample them from the seeded `rng/action` generator. Thus a
restore also sets the next reset.

The arm starts at the position that OGBench samples. It starts at `agent.ee_start_position` only when
the reset options name this leaf, or when `set_variation` sets the leaf to a value that is not the default.

The default model has the OGBench physics, with a cube size of 0.02. With `mujoco.batch` physics, the reset and the
physics are equal to OGBench, bit for bit.

### OGBCube

The arm moves one cube to a goal pose. This is the OGBench `cube-single` task. The constructor argument `reward_task_id` selects the
evaluation task of the reward.

<!-- env: OGBCube -->

### OGBScene

The scene has a cube, a drawer, a window and two button locks. The agent changes the objects to match a
goal configuration. A locked object does not move until the agent presses its button.

<!-- env: OGBScene -->
