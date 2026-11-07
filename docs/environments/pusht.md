# PushT

PushT is a 2D manipulation task with contact. The agent is a disc. It pushes a T-shaped block into a
target pose. [Pymunk](https://www.pymunk.org/) simulates the contact physics. The task comes from
Diffusion Policy ([Chi et al., 2023](https://arxiv.org/abs/2303.04137)). stable-envs follows the
stable-worldmodel version of [gym-pusht](https://github.com/huggingface/gym-pusht).

The block must match the target **position** and **orientation**. The agent must plan a sequence of
pushes. A single move to a point does not solve the task.

**Success:** the episode succeeds when two conditions are true:

- The L2 norm of the agent and block position error is less than 20 pixels. The error vector has four
  values: agent x, agent y, block x and block y.
- The block orientation error is less than π/9 rad (20°). The check uses the rotational symmetry of the
  block shape.

This criterion comes from [DINO-WM](https://arxiv.org/abs/2411.04983).

```python
import stable_envs as snv

env = snv.PushT(num_envs=8)
state = env.reset(seed=0)
frames = env.render()          # (8, 224, 224, 3) uint8
```

PushT draws its frames with the source pygame code on CPU. A frame is the source frame for the same
State. `device="cuda"` puts the output tensors on the GPU. Physics stays on CPU.

On CPU with Numba installed, `render` draws batches of 8 or more States with a Numba rasterizer. It
uses the same primitive records as the GPU path, and its frames are byte-identical to the pygame frames.
`backend="source"` keeps pygame; `backend="port"` uses the Numba rasterizer for every batch.

With `device="cuda"` and Triton, PushT draws the frames on the GPU. The GPU frames are byte-identical
to the pygame frames. Frames with action markers, and the few frames that the GPU path cannot draw
exactly, use pygame.

On Linux, `render` draws a batch of at least `worker_min_lanes` States in up to `workers` processes.
Step frames come from the running scenes, so a step does not export a State only to draw it. Worker
processes do not change frames. See [Performance tuning](../performance-tuning.md).

The State is a full replay checkpoint. It holds the bodies, the goal, the action history and the
Variation that the replay starts from. See [State](../concepts/state.md#pusht-checkpoints).

## PushT

The action is a relative control for each axis. It is not an absolute pixel target. The action space is [-1, 1], but PushT also accepts actions outside it and moves the target further, as stable-worldmodel does.

<!-- env: PushT -->

## PushTDiscrete

The action has two discrete indices in [0, 16). They select a polar action.

<!-- env: PushTDiscrete -->

## Reset states and colour Variation

This image shows 24 lanes after `reset(seed=0)`, at 96 px:

![24 PushT reset states](../assets/media/PushT/grid.png)

This image shows the same 24 lanes after `set_variation` samples the colour leaves (`agent`,
`background`, `block`, `goal`) for each lane:

![Per-lane colour variation](../assets/media/PushT/variation_batch.png)

The code below makes the two images:

```python
import numpy as np
import stable_envs as snv

env = snv.PushT(num_envs=24, device="cpu", size=96)
env.reset(seed=0)
grid = env.render()
colors = [p for p in env.variation_names if p.endswith(".color")]
env.set_variation(env.sample_variation(paths=colors, rng=np.random.default_rng(0)))
env.reset(seed=0)
varied = env.render()
env.close()
```
