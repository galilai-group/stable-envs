# Miniworld

`MiniWorldMovingBlockWorld` runs the 3D Dynamic Blockworld of Flow Equivariant World Models
([Lillemark et al., 2026](https://arxiv.org/abs/2601.01075)). It is `MovingBlockWorld` from the
[hlillemark/Miniworld](https://github.com/hlillemark/Miniworld) fork of
[Miniworld](https://github.com/Farama-Foundation/Miniworld), pinned to the fork's `flowm_release` branch,
the code frozen for the paper. Each lane is one source env on CPU.

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
lane draws its count uniformly at every reset, as the paper's dataset script draws 6 to 10 blocks per video;
both FloWM presets use `(6, 10)`. State pads the block fields to `max` blocks, and `blocks.present` marks a
lane's blocks. The script draws the count with Python's `random`, so the counts follow the same
distribution but not the same sequence.

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

## MiniWorldMovingBlockWorld

<!-- env: MiniWorldMovingBlockWorld -->
