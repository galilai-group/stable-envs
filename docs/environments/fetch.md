# Fetch

A Fetch arm with seven joints and a parallel gripper works on a table: it moves the gripper to a goal, or pushes, slides or picks up an object. These are the Fetch tasks of [Gymnasium-Robotics](https://robotics.farama.org/envs/fetch/) ([Plappert et al., 2018](https://arxiv.org/abs/1802.09464)), in 12 classes that follow the stable-worldmodel wrappers. MuJoCo steps the batch in one call on CPU and MuJoCo Warp on CUDA. A Numba rasterizer, MuJoCo OpenGL or MJWarp draws the [frames](../terms.md#frame).

```python
# needs the robotics extra
import stable_envs as snv

with snv.FetchPush(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    print(env.render().shape)      # torch.Size([8, 480, 480, 3])
```

## Install

Fetch needs `mujoco<3.12`. Install the `robotics` extra in an environment of its own. CUDA also needs the `mjwarp` extra; see [what each family needs](../install.md#what-each-family-needs) for both setups.

These classes default to `device="cpu"`. Pass `device="cuda"` for the CUDA path.

## Task

The action has four values in [-1, 1]: the first three move the gripper in x, y and z, and the fourth opens or closes the gripper.

A task succeeds when the gripper (Reach) or the block (Push, Slide, PickAndPlace) is within 5 cm of the goal, and the step output reports it as `success`.

Each task comes in three classes:

- `Fetch<Task>` uses the sparse reward, which is 0 at the goal and -1 otherwise.
- `Fetch<Task>Dense` uses the dense reward, the negative distance to the goal.
- `Fetch<Task>Dict` has the same task and reward as `Fetch<Task>`, and its name matches the stable-worldmodel id `swm/Fetch<Task>Dict-v3`.

## Simulation

`simulation_backend` picks the physics when you construct the environment:

| Value | CPU device | CUDA device |
|---|---|---|
| `"default"` (default) | CPU batch | [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) |
| `"auto"` | CPU batch | CPU batch below `mjwarp_min_envs` (256) environments, MuJoCo Warp from there |
| `"native"` | CPU batch | CPU batch |
| `"mjwarp"` | error | MuJoCo Warp |
| `"source"` | Gymnasium-Robotics' `step`, one environment at a time | the same |

The CPU batch runs Gymnasium-Robotics' step for every environment in one call: the action scaling, the mocap target and the gripper controls on arrays, then 20 MuJoCo substeps for every environment in `mujoco.batch`, then the reward and the 50-step time limit on arrays. Without `mujoco.batch` in the installed MuJoCo, up to 8 threads call `mj_step`, and each thread takes the next environment from one shared queue when it completes its last one. A thread that is slower than the others then steps fewer environments. Resets draw the object and goal positions from each environment's PCG64 stream, as Gymnasium-Robotics does. The [state](../terms.md#state) and the step outputs equal Gymnasium-Robotics' bit for bit on the same MuJoCo version, and a CUDA environment on the CPU batch returns its state on CUDA. With `"source"`, a batch of at least `worker_min_lanes` environments steps in up to `workers` [worker processes](../performance-tuning.md#worker-processes) on Linux with at least four CPUs, and otherwise in one process on up to `fetch_threads` threads.

MuJoCo Warp steps every environment in float32, with the 20 substeps of a step in one captured CUDA graph and no host synchronization. A step takes about 8 ms for any batch up to a few hundred environments, so a smaller CUDA batch steps faster with `"auto"`. Its resets draw the same starts and goals as the CPU batch. MuJoCo Warp doesn't support the source model as it is, so the model has two changes:

- Push, PickAndPlace and Reach turn off multi-contact and native collision detection, and keep the 1 mm contact margin of the robot geoms.
- Slide replaces the cylinder puck with a 24-sided prism of the same size and inertia, and sets the contact margins to 0. MuJoCo Warp makes one contact for each cylinder pair, and with one contact the puck sinks into the table.

A scripted controller ran 200 steps on Gymnasium-Robotics, and its actions were replayed on MuJoCo Warp. Over 32 episodes per task, the replay reached the source task outcome in 32 Reach, 30 Push, 30 Slide and 32 PickAndPlace episodes.

## Rendering

On CPU, `backend="auto"` (the default) and `backend="numba"` draw the frames at the requested size with a compiled Numba rasterizer that follows MuJoCo's OpenGL renderer: the same meshes and primitive tessellations, per-vertex lighting with the headlight, 4 samples per pixel, the skybox, and the translucent mocap markers and arm blended after the opaque scene. Its frames differ from Gymnasium-Robotics' frames only at some edges and by rounding: the mean squared error per frame is about 0.15 on average and below 0.4 on the 0-255 scale. The first render after an install compiles the rasterizer in about 30 s, and Numba caches the result for later processes. Without Numba, `backend="auto"` uses MuJoCo OpenGL.

`backend="raster"` draws the frames with the shared [MuJoCo rasterizer](../contributing/renderers.md#mujoco-rasterizer) on CPU. It reads the scene from the OpenGL render model, with MuJoCo's plane grid and primitive tessellations. On CUDA it raises `ValueError`. `render_semantic(rows)` gives the segmentation and the planar depth of compact render rows, with the goal as the site `target0` of the Gymnasium-Robotics model.

`backend="opengl"` draws with MuJoCo OpenGL, as Gymnasium-Robotics does. On CPU it needs a software OpenGL backend and gives the source frames. On CUDA, MJWarp draws the frames by default, and `backend="opengl"` draws them with NVIDIA OpenGL instead.

On Linux, a CPU batch of at least `render_worker_min_rows` [states](../terms.md#state) drawn with `backend="opengl"` uses up to `render_workers` [worker processes](../performance-tuning.md#worker-processes), each with its own OpenGL context, and gives the same frames as one process.

## Variation

Every class has [variation leaves](../tutorials/variation.md) for the colors of the arm, the table, the floor, the sky and the goal, plus `camera.*`, `light.diffuse` and `rendering.transparent_arm`. The classes with an object also have `object.color` and `block.mass`, and `block.mass` changes the dynamics.

## Classes

### Reach

The agent moves the gripper to a goal position in the air. The table has no block.

#### FetchReach

<!-- env: FetchReach -->

#### FetchReachDense

<!-- env: FetchReachDense -->

#### FetchReachDict

<!-- env: FetchReachDict -->

### Push

The agent pushes a block across the table to a goal position on the table. The gripper stays closed.

#### FetchPush

<!-- env: FetchPush -->

#### FetchPushDense

<!-- env: FetchPushDense -->

#### FetchPushDict

<!-- env: FetchPushDict -->

### Slide

The goal is out of the arm's reach on a long table with low friction. The agent hits a puck so that it slides to the goal.

#### FetchSlide

<!-- env: FetchSlide -->

#### FetchSlideDense

<!-- env: FetchSlideDense -->

#### FetchSlideDict

<!-- env: FetchSlideDict -->

### PickAndPlace

The agent grasps a block and moves it to a goal position. The goal can be on the table or in the air.

#### FetchPickAndPlace

<!-- env: FetchPickAndPlace -->

#### FetchPickAndPlaceDense

<!-- env: FetchPickAndPlaceDense -->

#### FetchPickAndPlaceDict

<!-- env: FetchPickAndPlaceDict -->

## Limitations

- MJWarp has no alpha blending, so it leaves out the translucent mocap markers (the 1 cm cube and three crosshair bars). The Numba rasterizer, the MuJoCo rasterizer (`backend="raster"`) and OpenGL draw them at the gripper, as Gymnasium-Robotics does.
- On MuJoCo Warp, two steps from the same state can differ in the last bits, because MuJoCo Warp sums contact forces with atomic operations. The puck of Slide slides freely, so small differences grow: after 200 replayed steps its mean position differs from the source by 10 cm, while an action noise of 1e-6 on the CPU source alone moves it by 3.3 cm.
- The default CPU frames differ from the source frames at some edges and by rounding. Only `backend="opengl"` on CPU gives the frames of the source.
- `sky.color` at its default draws the registered gradient skybox, which the official wrapper's single-color `background.color` can't express.
