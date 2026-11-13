# Navigation

A point agent moves through a flat arena to a target, past walls, doors or motion-bias zones depending on the task. The five classes reimplement the 2D navigation tasks of [stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) in Torch, Numba and JAX, and the tests check that resets, [state](../terms.md#state) and step outputs equal those of a pinned stable-worldmodel.

```python
import stable_envs as snv

with snv.TwoRoom(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    print(env.render().shape)      # torch.Size([8, 224, 224, 3])
```

## Install

TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones run with the core install. SimplePointMaze needs the `navigation` extra for Matplotlib, and so does Piecewise's `rendering.render_bias_field` leaf. The extra also installs Numba for the CPU kernels, and TwoRoom on CUDA needs the `navigation-gpu` extra.

## Task

The agent moves to a target. TwoRoom, Piecewise and SimplePointMaze terminate the episode when it reaches the target, and TwoRoomTarget and PiecewiseZones return only the state, with no reward or episode flags. Each class in [Classes](#classes) gives its arena and its success condition.

## Simulation

TwoRoomTarget, Piecewise and PiecewiseZones take `simulation_backend`. `"auto"` (the default) steps CPU batches with the Numba kernel when Numba imports and uses Torch otherwise, and `"torch"` and `"numba"` select one implementation. Both give the same state and task outputs.

### TwoRoom backends

TwoRoom runs on CPU or CUDA and takes and returns Torch tensors. On CPU, one call of a Numba kernel moves every agent in the batch and computes the state, the observation, the reward and the termination flags. On CUDA, JAX steps the batch on the GPU. JAX comes with the core install, but CUDA also needs `stable-envs[navigation-gpu]`:

```python
# needs the navigation-gpu extra; CUDA only
import torch
import stable_envs as snv

with snv.TwoRoom(num_envs=1024, device="cuda") as env:
    env.reset(seed=0)
    actions = torch.zeros(1024, 2, device="cuda")
    result = env.step(actions, states=None, frames=None)
```

The first step compiles the physics kernel and, on supported CUDA installs, the shared Torch task outputs. Run a few warm-up steps before you measure throughput. A new batch shape can compile again. Torch and JAX exchange tensors through DLPack on the selected device, without copying them to CPU while stepping. If the JAX CUDA runtime is missing, the error names the extra to install.

`simulation_backend="auto"` uses the Numba kernel on CPU when Numba imports. Without Numba, a CPU batch steps in JAX, or in Torch below `jax_min_envs`. On CUDA, it uses Torch below `jax_min_envs` environments and JAX at or above that cutoff. The default cutoff is 1: JAX was faster in our measurements even for a single environment. If Torch is faster for small CUDA batches on your machine, raise the cutoff:

```python
# needs the navigation-gpu extra; CUDA only
with snv.TwoRoom(num_envs=4, device="cuda", jax_min_envs=8) as env:
    print(env.simulation_backend)  # torch
```

The choice is fixed at construction and uses these defaults rather than a benchmark of your machine. Set `simulation_backend="torch"`, `"jax"` or `"numba"` to choose yourself at any batch size. Torch and JAX use the requested device, but Numba runs on CPU only. `env.simulation_backend` reports the selected implementation, and all three give the same state and task outputs.

## Rendering

TwoRoom, TwoRoomTarget, Piecewise and PiecewiseZones draw each dot as a normalized Gaussian over a base image. On CUDA a Triton kernel draws the dots, and on CPU a Numba kernel draws them with one frame per thread. The `navigation` extra installs Numba. Both kernels give the same bytes as the Torch path, and `backend="torch"` selects the Torch path on CPU.

SimplePointMaze draws a `Scene2D` that reproduces its Matplotlib figure in each [frame](../terms.md#frame). Only the `info.goal` image comes from Matplotlib.

## Variation

Every class has `agent.color`, `agent.radius`, `agent.speed` and `background.color` [variation leaves](../tutorials/variation.md). The other leaves set the layout: the wall and doors of TwoRoom, the zone biases of Piecewise and the walls of SimplePointMaze. `env.dynamics_variation_leaves` lists the leaves that change the dynamics.

## Classes

### TwoRoom

A wall with doors splits the arena into two rooms, and the agent moves through a door to a target in the other room. The target is invisible in the frame. The state keeps its position, and a restored state keeps the task.

**Success:** the episode terminates when the agent is less than 16 px from the target.

<!-- env: TwoRoom -->

### TwoRoomTarget

This class draws the TwoRoom arena with a visible target, and the target position is part of the renderer input.

<!-- env: TwoRoomTarget -->

### Piecewise

The arena is one open room with four zones, and each zone adds a motion bias to the agent. By default the zones and the target are invisible in the frame, and the `rendering.render_bias_field` and `rendering.render_target` leaves draw them.

**Success:** the episode terminates when the agent is less than 16 px from the target.

<!-- env: Piecewise -->

### PiecewiseZones

This class draws the Piecewise arena with a different color for each zone.

<!-- env: PiecewiseZones -->

### SimplePointMaze

A point moves in a 5 × 5 arena with rectangular walls toward a visible goal. The frame is 500 × 500 px, as in the source's Matplotlib figure.

**Success:** the episode terminates when the agent is inside the goal radius. The reward is 1 at the goal and -0.01 for every other step.

**Goal image:** `info.goal` is the Matplotlib figure of the goal that `reset` and `set_state` draw, and every step returns the same image tensor. Don't write into it. For `(B, T)` actions, `info.goal` is a view of this image along the time axis. When an environment's episode ends inside the sequence, `info.goal` is a copy with zeros after the end.

<!-- env: SimplePointMaze -->

## Limitations

Each class renders only at its fixed native size. Resize the output for a smaller frame.
