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

When you build a Craftax environment, stable-envs sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` if it is not set and JAX has not started, so that JAX does not reserve most of the GPU memory. Importing stable-envs does not set it. JAX reads this variable when it starts its GPU backend. If JAX code ran on the GPU before you built the first Craftax environment, set the variable before you start Python.

## Task

Craftax-Classic reimplements [Crafter](https://github.com/danijar/crafter) ([Hafner, 2021](https://arxiv.org/abs/2109.06780)), and Craftax adds more levels, items and creatures. The task has no success criterion. The reward counts achievements, for example "collect wood" or "make a stone pickaxe", and the episode ends when the agent dies or reaches the game's step limit.

The `Pixels` and `Symbolic` classes use the same game, but a `Symbolic` class returns the game's symbolic vector as `observation`.

## Simulation

The whole batch is one stacked JAX pytree that one jitted `vmap` steps. [Batch row](../terms.md#batch-row) `i` follows the stable-worldmodel `CraftaxWrapper` with seed `seed + i`. When the environment is built, stable-envs turns off JAX's GPU memory preallocation, so that Torch keeps its GPU memory. See [JAX GPU memory](../environment-variables.md#jax-gpu-memory).

On the CPU, one XLA program uses only a few cores. So the batch rows are split evenly over XLA host devices, one device for each CPU that the process can use (its CPU affinity, limited by a cgroup CPU quota). Each device steps its rows in the same program. The results are equal bit for bit to the results on one device. When the rows do not divide evenly, the last device also steps copies of the first row, and their results are discarded.

A device gets at least 4 rows (`MIN_ENVS_PER_HOST_DEVICE` in `stable_envs/envs/craftax/kernel.py`), and a smaller batch uses fewer devices or one device. One XLA:CPU program that runs alone spreads its small operations over the threads of XLA's thread pool. For Craftax, this is slower than programs that run at the same time on different devices, but each split also adds a cost to each call, for each device. The value was measured on 17 November 2026 on an AMD EPYC 9554 with 8 cores, for the four Craftax classes, with `env.step` of one action per call and a State per step, at batches 4, 16, 64 and 128 with 1, 2, 4, 8 and 32 rows for each device. At a batch of 16, 4 rows for each device was the fastest for each class (for example, Craftax-Classic-Symbolic: 1.95 ms for each step, against 2.52 ms with 2 rows and 2.93 ms on one device). Nothing is timed when you build an environment.

On CUDA, XLA reads the condition of each loop iteration on the host. The step and the reset unroll the short loops of the game, so that the GPU runs them without these round trips. The unrolled program gives the same results.

The number of host devices is a JAX setting for the whole process, and JAX reads it when it starts its CPU backend. Craftax sets it when you make the first Craftax environment on the CPU, before Craftax imports its game code. A Craftax environment on CUDA does not change it. Thus:

- If you set the number (`XLA_FLAGS=--xla_force_host_platform_device_count=N`, `JAX_NUM_CPU_DEVICES=N` or `jax.config.update("jax_num_cpu_devices", N)`), Craftax keeps it and uses up to `N` devices.
- If JAX code ran before you made the first CPU Craftax environment, JAX has one CPU device. Then the CPU environments step on one device, and Craftax logs a warning. To prevent this, make the first CPU Craftax environment before other JAX code runs, or set the number before you start Python. The same occurs with a JAX version that has no `jax_num_cpu_devices` setting.
- After Craftax sets the number, other JAX code in the process also sees the extra CPU devices in `jax.devices("cpu")`. Code that does not place its arrays on them still runs on the first CPU device.
- `device="auto"` starts JAX to find its GPU backend only when a JAX GPU plugin is installed. If JAX then has no GPU, the environment steps on the CPU with one device.

A sequence of actions, `step` with actions of shape `(B, T, 1)`, runs in one jitted call that steps the game `T` times with `lax.scan`. The results are equal to `T` single steps. The call is padded to a length that is a power of two, so a new `T` compiles again only when it passes a power of two. Every selection of `states` and `frames` uses this call: the last state and the last frame come from the end of the scan, and other selections take their steps from the states or the frames of every step.

Actions from the host are checked before the step, all the actions of a sequence before its first step. Actions in a CUDA tensor are checked on the device, so that the host does not wait for the step. A step or a sequence with an action outside the range leaves every environment unchanged, and the next `step`, `reset`, `set_state` or `close` raises `ValueError`. The rejected step or sequence itself returns `valid` set to true for each environment, zero rewards and observations, no ends, and the unchanged state.

The [state](../terms.md#state) has three leaves, one for each dtype: `craftax/bool`, `craftax/float32` and `craftax/int32` (`craftax_classic/...` for the Classic classes), with one row per environment. Each field is a slice of one of them. The fields are each leaf of the upstream `EnvState` under `engine`, such as `engine/player_health`, the variation values of the environment under `physics`, such as `physics/zombie/health`, and the wrapper's PRNG key, `key`. Fields that are uint32 PRNG words are stored bit for bit as int32. Read a field with `snv.get_field(state, "engine/player_health")` and write a field with `snv.set_field`.

## Rendering

Both the `Pixels` and the `Symbolic` classes draw the same [frames](../terms.md#frame) from their state. A frame is Craftax's pixel observation scaled to uint8, as stable-worldmodel renders it.

The renderer is a port of the game's pixel renderer. The game paints each block type and each item type over the whole frame in turn. The port reads the block texture and the item texture of each pixel with one lookup each, and draws the player, creatures, light, night, sleep and inventory as the game does. The frames are equal to the game's frames bit for bit, on CPU and on CUDA.

When `step` returns a frame, the same jitted call steps the game and draws the frame.

## Variation

The [variation](../terms.md#variation) leaves set fields of the upstream `EnvParams` for each environment. All of them change the game, not the frames. An environment takes its values at `reset`, and its state keeps them under `physics`, so `set_state` restores them with the rest of the game. A change with `set_variation` takes effect at the next `reset`. `reset(options={"variation_values": {...}})` sets values for one episode, and `reset(options={"variation": [...]})` samples them for each environment.

| Leaf | `EnvParams` field | Classic default | Craftax default | Range |
| --- | --- | --- | --- | --- |
| `world.day_length` | `day_length` | 300 | 300 | 50 to 1000 |
| `world.always_diamond` | `always_diamond` | 1 | 0 | 0 or 1 |
| `mobs.despawn_distance` | `mob_despawn_distance` | 14 | 14 | 6 to 30 |
| `zombie.health` | `zombie_health` | 5 | | 1 to 9 |
| `cow.health` | `cow_health` | 3 | | 1 to 9 |
| `skeleton.health` | `skeleton_health` | 3 | | 1 to 9 |
| `spawn.cow` | `spawn_cow_chance` | 0.1 | | 0 to 1 |
| `spawn.zombie_base` | `spawn_zombie_base_chance` | 0.02 | | 0 to 1 |
| `spawn.zombie_night` | `spawn_zombie_night_chance` | 0.1 | | 0 to 1 |
| `spawn.skeleton` | `spawn_skeleton_chance` | 0.05 | | 0 to 1 |
| `player.max_attribute` | `max_attribute` | | 5 | 1 to 10 |

When every environment has the default values, the step uses the upstream defaults as constants, as upstream does, and the states are the same as upstream's bit for bit. Other values go to upstream's `step` as arguments.

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
