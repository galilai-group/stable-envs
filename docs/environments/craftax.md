# Craftax

These four classes wrap [Craftax](https://github.com/MichaelTMatthews/Craftax)
([Matthews et al., 2024](https://arxiv.org/abs/2402.16801)). Craftax is an open-ended 2D survival game
in JAX. The agent gathers resources, crafts tools, fights creatures and explores. Craftax-Classic
reimplements [Crafter](https://github.com/danijar/crafter) ([Hafner, 2021](https://arxiv.org/abs/2109.06780)).
Craftax adds more levels, items and creatures.

All lanes are one stacked JAX pytree. One jitted `vmap` steps them. Lane `i` follows the
stable-worldmodel `CraftaxWrapper` with seed `seed + i`.

The task has no success criterion. The reward counts achievements, for example "collect wood" or "make
a stone pickaxe". The episode ends when the agent dies or reaches the step limit of the game.

The `Pixels` and `Symbolic` classes use the same game. A `Symbolic` class returns the symbolic vector
of the game as `observation`. Both draw the same frames from their State with the pixel renderer of the
game. Craftax draws its own frame size. The constructor refuses other values of `size`.

```python
import stable_envs as snv

env = snv.CraftaxClassicPixels(num_envs=8)
state = env.reset(seed=0)
frames = env.render()
```

The classes need the `craftax` extra. `device="cuda"` needs the `craftax-gpu` extra, which installs
the CUDA build of JAX.

## Craftax-Classic

### CraftaxClassicPixels

<!-- env: CraftaxClassicPixels -->

### CraftaxClassicSymbolic

<!-- env: CraftaxClassicSymbolic -->

## Craftax

### CraftaxPixels

<!-- env: CraftaxPixels -->

### CraftaxSymbolic

<!-- env: CraftaxSymbolic -->
