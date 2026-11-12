# PushT

PushT is a 2D manipulation task with contact. The agent is a disc. It pushes a T-shaped block into a
target pose. The source simulates the contact physics with [Pymunk](https://www.pymunk.org/). The task comes from
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

A PushT frame is the source's pygame frame for the same State. `device="cuda"` puts the output tensors
on the GPU.

## Physics backends

stable-envs steps PushT with a port of the Pymunk contact solver. One call steps all environments. The port
has three implementations of the same math:

| `simulation_backend` | Device of the physics | Implementation |
|---|---|---|
| `"numba"` | CPU | Numba, `float64`, in parallel over environments. The default on CPU. |
| `"triton"` | CUDA | One Triton kernel launch per step, `float32`, one GPU thread per environment. The default on CUDA. |
| `"jax"` | the env device | JAX `vmap`, `float64`. The default on CUDA when Triton is not installed. |
| `"auto"` | CPU or CUDA | On CUDA, `"numba"` below `triton_min_envs` environments and the CUDA default at or above it. On CPU, `"numba"`. |

With `"numba"` on CUDA, the physics runs on the CPU and the State and outputs are CUDA tensors. On an
L40S with 8 CPU cores, `"triton"` was faster than `"numba"` at every batch size, so `triton_min_envs`
defaults to 1. A JAX step on CUDA takes about 8 ms at every batch size, and a Triton step takes about
0.3 to 0.5 ms. The first Triton step in a process compiles the kernel (about 10 s, or 20 s for agent
shapes other than the disc) unless Triton has it in its cache.

The port is not bit-identical to Pymunk. It uses the same integration, PD control, damping and
sequential impulse solver, but its walls are half-planes. In the source, a block pushed hard into a wall
can pass through it. In the port, the block stays inside. The State keeps the solver cache
(`current/jax`), so a restored State steps the same as the original on the same backend. The Numba and
JAX backends agree to rounding. The Triton backend computes in `float32`. For the disc agent it agrees
with them to about 0.001 px per step.

Pymunk on the CPU resets the environments, with the source reset distribution. On Linux, a batch of at least
`worker_min_lanes` lanes resets in up to `workers` processes. The worker processes do not change the
State, the frames or the task outputs.

On CPU with Numba installed, `render` draws with a Numba rasterizer. It builds the same primitive
records as the GPU path, in parallel over States. It paints and resizes the background, goal and walls
once for each distinct goal and colour set in the batch, then redraws only the region that the bodies
cover in each frame. Its frames are byte-identical to the pygame frames. The first render in a process
loads the compiled kernels, and compiles them if Numba has no cache yet. Without Numba, or with
`backend="source"`, CPU frames are drawn with pygame.

With `device="cuda"` and Triton, PushT draws the frames on the GPU. The GPU frames are byte-identical
to the pygame frames. Frames with action markers, and the few frames that the GPU path cannot draw
exactly, use pygame.

On Linux, `render` draws a batch of at least `worker_min_lanes` States in up to `workers` processes.
Step frames come from the running scenes, so a step does not export a State only to draw it. Worker
processes do not change frames. See [Performance tuning](../performance-tuning.md).

The State is a full checkpoint. It holds the bodies, the contact-solver cache, the goal, the action
history and the reset recipe. See [State](../concepts/state.md#pusht-checkpoints).

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
