# Fetch

These 12 classes port the Fetch tasks of
[Gymnasium-Robotics](https://robotics.farama.org/envs/fetch/)
([Plappert et al., 2018](https://arxiv.org/abs/1802.09464)). A Fetch robot arm with seven joints and a
parallel gripper works on a table. MuJoCo steps the physics on CPU. stable-envs follows the
stable-worldmodel wrappers of these tasks.

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

Fetch needs `mujoco<3.12`. Install it in its own environment with the `robotics` extra. On CPU, MuJoCo
OpenGL draws the frames at the requested size, as Gymnasium Robotics does. On CUDA, MJWarp draws them. See [System setup](../system-setup.md).
On Linux, a CPU batch of at least `render_worker_min_rows` states (default 2 for Fetch) draws in up
to `render_workers` worker processes. Each worker has its own OpenGL context. The frames do not change. See
[Performance tuning](../performance-tuning.md).

```python
import stable_envs as snv

env = snv.FetchPush(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

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
