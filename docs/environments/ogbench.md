# OGBench

These 27 classes port environments of [OGBench](https://github.com/seohongpark/ogbench)
([Park et al., 2024](https://arxiv.org/abs/2410.20092)). OGBench is a benchmark for offline
goal-conditioned reinforcement learning. MuJoCo steps the physics on CPU.

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

Cube and Scene need `mujoco==3.12.0` and the `ogbench-manipulation` extra. MuJoCo OpenGL draws
the frames on CPU (software OpenGL) and on CUDA (NVIDIA OpenGL). Physics runs on CPU with both renderers.

The frames have the look of the OGBench pixel tasks `visual-cube-single-v0` and `visual-scene-v0`: the
`front_pixels` camera, a purple gripper, a see-through arm, and no goal cube.

- CPU frames are bit-exact. A draw of the same state gives the same frame each time.
- CUDA frames are the source's frames, up to a variation in the NVIDIA OpenGL driver. The driver can
  change a pixel by 1/255 between two draws of the same state. The source shows the same variation.
  In one pass over 256 Scene states, 0 to 5 frames change, with at most 32 changed pixels in each frame.
  Cube frames change less often.
- With `supersample` 2 or more, MJWarp draws the CUDA frames. MJWarp frames can differ from MuJoCo in
  lighting and transparency.

### OGBCube

The arm moves one cube to a goal pose. This is the OGBench `cube-single` task. The constructor argument `reward_task_id` selects the
evaluation task of the reward.

<!-- env: OGBCube -->

### OGBScene

The scene has a cube, a drawer, a window and two button locks. The agent changes the objects to match a
goal configuration. A locked object does not move until the agent presses its button.

<!-- env: OGBScene -->
