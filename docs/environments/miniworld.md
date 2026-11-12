# Miniworld

`MiniWorldMovingBlockWorld` runs the 3D Dynamic Blockworld of Flow Equivariant World Models
([Lillemark et al., 2026](https://arxiv.org/abs/2601.01075)). It is `MovingBlockWorld` from the
[hlillemark/Miniworld](https://github.com/hlillemark/Miniworld) fork of
[Miniworld](https://github.com/Farama-Foundation/Miniworld), pinned to the fork's `flowm_release` branch,
the code frozen for the paper. All environments step together in one batched port of the fork's step (see
[Physics backends](#physics-backends)).

An agent walks in a room of coloured blocks. Every block moves by a fixed velocity each step and bounces
off the walls. The task has no reward and no success criterion: the reward is always 0, and an episode
ends only at the step limit.

| Preset | Configuration |
|---|---|
| `flowm` (default) | the paper's dynamic dataset: plain coloured boxes in a 16 x 16 grid room, each moving one cell per step in a cardinal direction, through the agent and each other; 90 degree turns, unit forward steps, 500 steps |
| `flowm-textured` | the paper's textured split: `flowm` with random wall, floor and block textures, and balls among the blocks |
| `upstream` | `MovingBlockWorld()` as the fork registers it: a 12 x 12 room, continuous motion, 250 steps |

Keyword arguments override single options of a preset, for example `num_blocks`, `blocks_static`,
`room_size` or `max_episode_steps`. `num_blocks` takes a count or a `(min, max)` range. With a range, each
environment draws its count uniformly at every reset, as the paper's dataset script draws 6 to 10 blocks per
video; both FloWM presets use `(6, 10)`. State pads the block fields to `max` blocks, and `blocks.present`
marks the blocks of an environment.

The action is one of the fork's nine ids:

| Id | Action |
|---|---|
| 0, 1 | turn left, turn right |
| 2, 3 | move forward, move back |
| 4 | do nothing |
| 5, 6 | pick up, drop |
| 7, 8 | toggle, done |

Frames are the agent view that Miniworld draws with OpenGL through pyglet. On Linux the family uses
pyglet's headless EGL mode, which needs an EGL driver. Mesa's llvmpipe works without a GPU.

```python
import stable_envs as snv

env = snv.MiniWorldMovingBlockWorld(num_envs=8, size=128)
state = env.reset(seed=0)
frames = env.render()
```

The class needs the `miniworld` extra. The fork is not on PyPI. See [Install](../install.md).

## Physics backends

The `simulation_backend` argument selects the step:

| Value | Step | Device |
|---|---|---|
| `"numba"` | Numba kernel, one environment per CPU thread | CPU; on CUDA it returns State on CUDA |
| `"triton"` | Triton kernel, one environment per GPU thread, in a CUDA graph | CUDA |
| `"torch"` | Torch operations on all environments | any |
| `None` (default) | `"triton"` on CUDA, `"numba"` on CPU | |
| `"auto"` | `"numba"` for CUDA batches of fewer than `triton_min_envs` environments, else as `None` | |

`env.simulation_backend` reports the selected backend. On an NVIDIA L40S, the Triton step is as fast as
the Numba step from one environment, so `triton_min_envs` is 1 by default. Set a larger value on a machine
where the Numba step is faster for small CUDA batches.

The Numba step follows the fork's `MiniWorldEnv.step` and `MovingBlockWorld.step` operation by operation
in float64. From the same State, it gives the fork's values, bit for bit, and a test compares 200 steps of a
scripted policy. The Triton and Torch steps use the same rules. On CUDA, the cosine and sine round
differently, and the Triton step compares squared distances, so continuous motion (`preset="upstream"`)
agrees to 1e-9 per step in the tests. Grid motion (the FloWM presets) gives equal values.

Resets sample the fork's reset distribution: block count, colours, textures, kinds, sizes, positions,
headings and velocities. They do not use the fork's PCG64 generator. Draw `k` of an episode is a hash of
the environment's seed, its episode number and `k`, so all backends draw the same numbers and the
environments reset together. `reset(seed)` gives environment `i` the seed `seed + i`; `reset()` starts the
next episode of each environment. State holds the seed and the episode number (`rng.seed`,
`rng.episode`), so a restored State continues with the same episodes.

Actions on the host are checked: an id outside [0, 8] raises `ValueError`. Actions on CUDA are not read
back; there, such an id does nothing, as in the fork.

## MiniWorldMovingBlockWorld

<!-- env: MiniWorldMovingBlockWorld -->
