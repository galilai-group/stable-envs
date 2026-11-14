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

Actions from the host are checked before the step. Actions in a CUDA tensor are checked on the device, so that the host does not wait for the step. A step with an action outside the range leaves every environment unchanged, and the next `step`, `reset` or `set_state` raises `ValueError`.

The [state](../terms.md#state) holds each leaf of the upstream `EnvState` under `engine` and the wrapper's PRNG key under `key`, one row per environment. Leaves that are uint32 PRNG words are stored bit for bit as int32.

## Rendering

Both the `Pixels` and the `Symbolic` classes draw the same [frames](../terms.md#frame) from their state with the game's pixel renderer. A frame is Craftax's pixel observation scaled to uint8, as stable-worldmodel renders it.

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
