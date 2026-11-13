# Miniworld

`MiniWorldMovingBlockWorld` is a 3D room of moving colored blocks, the Dynamic Blockworld of Flow Equivariant World Models ([Lillemark et al., 2026](https://arxiv.org/abs/2601.01075)). It runs `MovingBlockWorld` from the [hlillemark/Miniworld](https://github.com/hlillemark/Miniworld) fork of [Miniworld](https://github.com/Farama-Foundation/Miniworld), pinned to the fork's `flowm_release` branch, which holds the code frozen for the paper. All environments step together in a batched port of the fork's step, in Numba on CPU and in Triton on CUDA, and pyglet OpenGL draws the [frames](../terms.md#frame).

```python
# needs the Miniworld fork and the miniworld extra
import stable_envs as snv

with snv.MiniWorldMovingBlockWorld(num_envs=8, size=128, device="cpu") as env:
    state = env.reset(seed=0)
    frames = env.render()
```

## Install

The class needs the `miniworld` extra, which also installs Numba for the CPU step, and the fork, which isn't on PyPI; see [the Miniworld install commands](../install.md#miniworld). On Linux it also needs an [EGL driver](../install.md#what-each-family-needs).

## Task

An agent walks in a room of colored blocks. The task has no reward and no success criterion: the reward is always 0 and an episode ends only at the step limit.

The action is one of the fork's nine ids:

| Id | Action |
|---|---|
| 0, 1 | turn left, turn right |
| 2, 3 | move forward, move back |
| 4 | do nothing |
| 5, 6 | pick up, drop |
| 7, 8 | toggle, done |

## Simulation

Every block moves by a fixed velocity each step and bounces off the walls. A preset sets the room and the motion:

| Preset | Configuration |
|---|---|
| `flowm` (default) | the paper's dynamic dataset: plain colored boxes in a 16 × 16 grid room, each moving one cell per step in a cardinal direction, through the agent and each other; 90° turns, unit forward steps, 500 steps |
| `flowm-textured` | the paper's textured split: `flowm` with random wall, floor and block textures, and balls among the blocks |
| `upstream` | `MovingBlockWorld()` as the fork registers it: a 12 × 12 room, continuous motion, 250 steps |

Keyword arguments override single options of a preset, for example `num_blocks`, `blocks_static`, `room_size` or `max_episode_steps`. `num_blocks` takes a count or a `(min, max)` range. With a range, each environment draws its count uniformly at every reset, as the paper's dataset script draws 6 to 10 blocks per video, and both FloWM presets use `(6, 10)`. The [state](../terms.md#state) pads the block fields to `max` blocks, and `blocks.present` marks the blocks an environment has.

### Physics backends

The `simulation_backend` argument selects the step:

| Value | Step | Device |
|---|---|---|
| `None` (default) | `"triton"` on CUDA, `"numba"` on CPU | |
| `"numba"` | a Numba kernel, from 64 environments on several CPU threads | CPU; on a CUDA device it returns the state on CUDA |
| `"triton"` | a Triton kernel, one environment per GPU thread, in a CUDA graph | CUDA |
| `"torch"` | Torch operations on the whole batch | any |
| `"auto"` | `"numba"` for CUDA batches of fewer than `triton_min_envs` environments, otherwise as `None` | |

`env.simulation_backend` reports the selected backend. On an NVIDIA L40S, the Triton step is as fast as the Numba step from one environment, so `triton_min_envs` is 1 by default.

The Numba step follows the fork's `MiniWorldEnv.step` and `MovingBlockWorld.step` operation by operation in float64, and from the same state it gives the fork's values bit for bit. The Triton and Torch steps use the same rules. On CUDA, the cosine and sine round differently and the Triton step compares squared distances, so continuous motion (`preset="upstream"`) agrees to 1e-9 per step in the tests, while the grid motion of the FloWM presets gives equal values.

Resets sample the fork's reset distribution: block count, colors, textures, kinds, sizes, positions, headings and velocities. Draw `k` of an episode is a hash of the environment's seed, its episode number and `k`, so every backend draws the same numbers. `reset(seed)` gives environment `i` the seed `seed + i`, and `reset()` starts the next episode of each environment. The state holds the seed and the episode number (`rng.seed`, `rng.episode`), so a restored state continues with the same episodes.

Actions on the host are checked, and an id outside [0, 8] raises `ValueError`. Actions on CUDA aren't read back, and there an id out of range does nothing, as in the fork.

## Rendering

Frames are the agent view that Miniworld draws with OpenGL through pyglet. On Linux the family uses pyglet's headless EGL mode, which needs an EGL driver. Mesa's llvmpipe is one that works without a GPU.

## Variation

This class has no [variation](../terms.md#variation) leaves. The presets and keyword arguments above set the room instead.

## Classes

### MiniWorldMovingBlockWorld

<!-- env: MiniWorldMovingBlockWorld -->

## Limitations

Resets don't use the fork's PCG64 generator, and the paper's dataset script draws the block count with Python's `random`. The episodes here follow the same distribution but not the same sequence.
