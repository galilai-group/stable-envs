# Fetch

These 12 classes port the Fetch tasks of
[Gymnasium-Robotics](https://robotics.farama.org/envs/fetch/)
([Plappert et al., 2018](https://arxiv.org/abs/1802.09464)). A Fetch robot arm with seven joints and a
parallel gripper works on a table. stable-envs follows the stable-worldmodel wrappers of these tasks.

The action has four values in [-1, 1]:

- The first three values move the gripper in x, y and z.
- The fourth value opens or closes the gripper.

**Success:** the gripper (Reach) or the block (Push, Slide, PickAndPlace) is within 5 cm of the goal.
The step output reports it as `success`.

Each task comes in three classes:

- `Fetch<Task>` uses the sparse reward. The reward is 0 at the goal and -1 otherwise.
- `Fetch<Task>Dense` uses the dense reward, the negative distance to the goal.
- `Fetch<Task>Dict` has the same task and reward as `Fetch<Task>`. Its name matches the
  stable-worldmodel id `swm/Fetch<Task>Dict-v3`.

Fetch needs `mujoco<3.12`. Install it in its own environment with the `robotics` extra. On CUDA, MJWarp
draws the frames. See [System setup](../system-setup.md).

On CPU, `backend="auto"` (the default) and `backend="numba"` draw the frames with a compiled Numba
rasterizer at the requested size. The rasterizer follows MuJoCo's OpenGL renderer: the same meshes and
primitive tessellations, per-vertex lighting with the headlight, 4 samples per pixel, the skybox, and the
translucent mocap markers and arm blended after the opaque scene. The frames differ from the Gymnasium
Robotics frames only at some edges and by rounding. The mean squared error per frame is about 0.15 on
average and less than 0.4 (0-255 scale). The first render after an install compiles the rasterizer in
about 30 s. Numba keeps the compiled code in a cache for later processes. Without Numba,
`backend="auto"` uses MuJoCo OpenGL.

`render_semantic(rows)` gives the segmentation and the planar depth of compact render rows on CPU, with the
goal as the site `target0` of the Gymnasium Robotics model. The shared MuJoCo rasterizer
([MuJoCo rasterizer](../api/mujoco-raster.md)) computes them from the OpenGL render model.

`backend="opengl"` draws the frames with MuJoCo OpenGL, as Gymnasium Robotics does. On CPU this needs a
software OpenGL backend, and the frames are equal to the source frames. On Linux, a CPU batch of at least
`render_worker_min_rows` states draws in up to `render_workers` worker processes. Each worker has its own
OpenGL context. The frames do not change. See [Performance tuning](../performance-tuning.md).

```python
import stable_envs as snv

env = snv.FetchPush(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

## Physics backends

| `simulation_backend` | CPU env | CUDA env |
|---|---|---|
| `"default"` (default) | CPU batch | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| `"auto"` | CPU batch | CPU batch below `mjwarp_min_envs` (256) environments, MuJoCo Warp from 256 |
| `"native"` | CPU batch | CPU batch |
| `"mjwarp"` | not supported | MuJoCo Warp |
| `"source"` | Gymnasium Robotics `step` per environment | Gymnasium Robotics `step` per environment |

The CPU batch does the Gymnasium Robotics step for all environments in one call: the action scaling, the
mocap target and the gripper controls on arrays, then `mj_step` with 20 substeps for every environment, then
the reward and the 50-step time limit on arrays. MuJoCo runs in `mujoco.batch` when the installed MuJoCo has
it. Otherwise up to 8 threads call `mj_step`, and each thread takes the next environment from one shared queue
when it is done with its last one. A thread that is slower than the others then steps fewer environments.
Resets draw the object and goal positions from the PCG64 stream of each environment, as Gymnasium Robotics does. State and
step outputs are equal, bit for bit, to those of Gymnasium Robotics on the same MuJoCo version. A CUDA env on
the CPU batch returns its State on CUDA.

MuJoCo Warp steps every environment in `float32`. The 20 substeps of a step run in one captured CUDA graph,
and the step does not synchronize with the host. A step takes about 8 ms for any batch up to a few hundred
environments, so a smaller CUDA batch steps faster with `"auto"`. Two steps from the same State can differ
in the last bits, because MuJoCo Warp sums contact forces with atomic operations. The resets draw the same starts and goals as the CPU batch.
The physics model has two changes, because MuJoCo Warp does not support the source model as it is:

- Push, PickAndPlace and Reach turn off multi-contact and native collision detection, and keep the 1 mm
  contact margin of the robot geoms.
- Slide replaces the cylinder puck with a 24-sided prism of the same size and inertia, and sets the contact
  margins to 0. MuJoCo Warp makes one contact for each cylinder pair, and with one contact the puck sinks into
  the table.

A scripted controller (reach the goal, push the block, strike the puck, or grasp and carry the block) runs
for 200 steps on Gymnasium Robotics. Its actions are then replayed on MuJoCo Warp. Over 32 episodes for each
task, the replayed task outcome is the same as the source outcome in 32 Reach, 30 Push, 30 Slide and 32
PickAndPlace episodes. The largest robot joint difference is 0.0003 (Reach), 0.012 (Push) and 0.004
(PickAndPlace). The mean block position difference is 1.4 cm (Push) and 1.5 mm (PickAndPlace). The puck of
Slide slides freely, so small differences grow: its mean position difference is 10 cm. On the CPU source, an
action noise of 1e-6 alone gives a mean puck position difference of 3.3 cm.

## Reach

The agent moves the gripper to a goal position in the air. The table has no block.

### FetchReach

<!-- env: FetchReach -->

### FetchReachDense

<!-- env: FetchReachDense -->

### FetchReachDict

<!-- env: FetchReachDict -->

## Push

The agent pushes a block across the table to a goal position on the table. The gripper stays closed.

### FetchPush

<!-- env: FetchPush -->

### FetchPushDense

<!-- env: FetchPushDense -->

### FetchPushDict

<!-- env: FetchPushDict -->

## Slide

The agent hits a puck so that it slides to a goal on a long table with low friction. The goal is out
of the reach of the arm.

### FetchSlide

<!-- env: FetchSlide -->

### FetchSlideDense

<!-- env: FetchSlideDense -->

### FetchSlideDict

<!-- env: FetchSlideDict -->

## PickAndPlace

The agent grasps a block and moves it to a goal position. The goal can be on the table or in the air.

### FetchPickAndPlace

<!-- env: FetchPickAndPlace -->

### FetchPickAndPlaceDense

<!-- env: FetchPickAndPlaceDense -->

### FetchPickAndPlaceDict

<!-- env: FetchPickAndPlaceDict -->
