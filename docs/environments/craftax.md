# Craftax

In [Craftax](https://github.com/MichaelTMatthews/Craftax) ([Matthews et al., 2024](https://arxiv.org/abs/2402.16801)), an open-ended 2D survival game in JAX, the agent gathers resources, crafts tools, fights creatures and explores. The four classes cover Craftax and its Crafter-style Classic version, each with pixel or symbolic observations.

```python
# needs the craftax extra
import stable_envs as snv

with snv.CraftaxClassicPixels(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    frames = env.render()
```

## Install

The classes need the `craftax` extra, and `device="cuda"` needs the `craftax-gpu` extra, which installs the CUDA build of JAX.

Without the `craftax-gpu` extra, `device="auto"` selects the CPU, also on a host with CUDA.

`import stable_envs` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` if it is not set, so that JAX does not reserve most of the GPU memory. JAX reads this variable when it starts its GPU backend, so an earlier `import jax` has no effect on it. If JAX code ran on the GPU before `import stable_envs`, set the variable before you start Python.

## Task

Craftax-Classic reimplements [Crafter](https://github.com/danijar/crafter) ([Hafner, 2021](https://arxiv.org/abs/2109.06780)), and Craftax adds more levels, items and creatures. The task has no success criterion. The reward counts achievements, for example "collect wood" or "make a stone pickaxe", and the episode ends when the agent dies or reaches the game's step limit.

The `Pixels` and `Symbolic` classes use the same game, but a `Symbolic` class returns the game's symbolic vector as `observation`.

## Simulation

The whole batch is one stacked JAX pytree that one jitted `vmap` steps. [Batch row](../terms.md#batch-row) `i` follows the stable-worldmodel `CraftaxWrapper` with seed `seed + i`.

On the CPU, one XLA program uses only a few cores. So the batch rows are split evenly over XLA host devices, one device for each CPU that the process can use (its CPU affinity, limited by a cgroup CPU quota). Each device steps its rows in the same program. The results are equal bit for bit to the results on one device. When the rows do not divide evenly, the last device also steps copies of the first row, and their results are discarded.

The number of host devices is a JAX setting for the whole process, and JAX reads it when it starts its CPU backend. Craftax sets it when you make the first Craftax environment, before Craftax imports its game code. Thus:

- If you set the number (`XLA_FLAGS=--xla_force_host_platform_device_count=N`, `JAX_NUM_CPU_DEVICES=N` or `jax.config.update("jax_num_cpu_devices", N)`), Craftax keeps it and uses up to `N` devices.
- If JAX code ran before you made the first Craftax environment, JAX has one CPU device. Then the CPU environments step on one device, and Craftax logs a warning. To prevent this, make the first Craftax environment before other JAX code runs, or set the number before you start Python.
- Other JAX code in the process also sees the extra CPU devices. Code that does not place its arrays on them still runs on the first CPU device.

Actions from the host are checked before the step. Actions in a CUDA tensor are checked on the device, so that the host does not wait for the step. A step with an action outside the range leaves every environment unchanged, and the next `step`, `reset` or `set_state` raises `ValueError`.

The [state](../terms.md#state) holds each leaf of the upstream `EnvState` under `engine` and the wrapper's PRNG key under `key`, one row per environment. Leaves that are uint32 PRNG words are stored bit for bit as int32.

## Rendering

Both the `Pixels` and the `Symbolic` classes draw the same [frames](../terms.md#frame) from their state. A frame is Craftax's pixel observation scaled to uint8, as stable-worldmodel renders it.

The renderer is a port of the game's pixel renderer. The game paints each block type and each item type over the whole frame in turn. The port reads the block texture and the item texture of each pixel with one lookup each, and draws the player, creatures, light, night, sleep and inventory as the game does. The frames are equal to the game's frames bit for bit, on CPU and on CUDA.

When `step` returns a frame, the same jitted call steps the game and draws the frame.

## Variation

These classes have no [variation](../terms.md#variation) leaves.

## Classes

### CraftaxClassicPixels

<!-- env: CraftaxClassicPixels -->

### CraftaxClassicSymbolic

<!-- env: CraftaxClassicSymbolic -->

### CraftaxPixels

<!-- env: CraftaxPixels -->

### CraftaxSymbolic

<!-- env: CraftaxSymbolic -->

## Limitations

Craftax draws at its own frame size, and the constructor refuses other values of `size`.
