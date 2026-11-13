# Randomize appearance and physics

Every batch row can have its own [variation](../terms.md#variation): one render call can draw the same state in many looks, and one reset can give every batch row different physics. This tutorial inspects a variation space, renders one state in many sampled looks, installs a variation, and randomizes physics per batch row through reset options, following [`examples/domain_randomization.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/domain_randomization.py).

## Prerequisites

You need the core install (`pip install stable-envs`) and a CPU. Each code block continues from the one before it, so the page runs top to bottom as one script. On Linux, CartPole draws these batches of frames in worker processes, and each worker imports your script again. In a script, put the code in a function that runs under `if __name__ == "__main__":`, or run the page in a notebook. The PushT block near the end needs the `pusht` extra (`pip install "stable-envs[pusht]"`), and you can skip it.

## Inspect the space

Every environment has a `variation_space` tree of leaves, addressed by dotted paths such as `"pole.color"`, and its tensor form is one flat row per batch row:

```python
import numpy as np
import torch
import stable_envs as snv

env = snv.CartPole(num_envs=16, device="cpu")
print(env.variation_names[:3])  # ['axle.color', 'background.color', 'cart.color']
print(env.variation_dim)        # 22, the row width
print(sorted(env.dynamics_variation_leaves)[:3])  # ['physics.force_mag', 'physics.gravity', 'physics.length']
```

`variation_names` has one name per leaf, so an RGB color is one name and three columns. `dynamics_variation_leaves` names the leaves that change physics, so every other leaf only changes the picture. Each class declares only the factors its [upstream library](../terms.md#upstream-library) has:

| Kind | Examples |
|---|---|
| appearance only | `background.color`, `agent.color` |
| physics only | `physics.gravity`, joint damping, `agent.speed` |
| reset or task | start-pose distribution, `task.min_steps` |
| appearance and physics | `block.scale`, body shape |

## Render one state in many looks

`sample_variation(n, paths=...)` draws `n` rows that randomize the leaves you list and keep the rest at their defaults. An unknown path raises an error. `render(state, variation=rows)` gives batch row `i` the look in row `i`:

```python
state = env.reset(seed=0)
shared = state[torch.zeros(16, dtype=torch.long)]  # batch row 0, 16 times
looks = [name for name in env.variation_names if name not in env.dynamics_variation_leaves]
rows = env.sample_variation(16, paths=looks, rng=np.random.default_rng(0))
frames = env.render(shared, variation=torch.as_tensor(rows))

def distinct(frames):
    return len({frame.tobytes() for frame in frames.cpu().numpy()})

print(frames.shape)                                    # torch.Size([16, 400, 600, 3])
print(distinct(env.render(shared)), distinct(frames))  # 1 16
```

All 16 rows draw the same state, so any difference between their [frames](../terms.md#frame) comes from the variation: the default gives one distinct frame, and the sampled rows give 16. `python examples/domain_randomization.py --num-envs 256` times one batched call against a loop of one-row calls on your machine.

`render(..., variation=)` takes nothing (the installed variation), one row or a dict of leaf values shared by every batch row, or a `(B, variation_dim)` tensor with one row per batch row. Some renderers apply one appearance per call rather than per batch row, so count distinct frames if that matters for your experiment. `sample_variation(..., rng=)` is deterministic for a given generator, and `reset(seed=...)` reseeds `env.rng`, so pass your own `rng` to keep variation draws independent of resets.

## Install a variation

`set_variation` installs rows on the environment, and they stay until you change them:

```python
env.set_variation({"pole.color": [220, 40, 40]})  # a dict changes only the leaves it names
print(env.get_variation().shape)                  # torch.Size([16, 22])
print(env.default_variation().shape)              # (22,)
```

`get_variation()` returns the installed `(num_envs, variation_dim)` rows, and `default_variation()` returns the upstream defaults.

## Randomize physics at reset

`render(..., variation=)` only draws, so it can't change physics. Install physics leaves with `set_variation`, or set them for one episode through reset options. `reset(seed, options={"variation": [...]})` samples the named leaves for each batch row, and the state records the values CartPole simulates with:

```python
state = env.reset(seed=0, options={"variation": ["physics.gravity", "physics.length"]})
print(state["physics", "gravity"].flatten()[:4])  # gravities drawn in [4.9, 14.7]
values = {"physics.gravity": torch.arange(1.0, 17.0)}
state = env.reset(seed=1, options={"variation_values": values})
print(state["physics", "gravity"].flatten()[:4])  # tensor([1., 2., 3., 4.], dtype=torch.float64)
state = env.reset(seed=2)
print(state["physics", "gravity"].flatten()[:4])  # tensor([9.8000, 9.8000, 9.8000, 9.8000], dtype=torch.float64)
env.close()
```

Most classes follow stable-worldmodel's reset rule:

- Reset samples the leaves named in `options["variation"]` for each batch row.
- Reset sets the leaves in `options["variation_values"]`, with one value for the batch or one per batch row.
- Every other leaf goes back to the values `set_variation` last installed, physics leaves included, so values from reset options last for that episode only.

With a seed, the draws of each batch row are reproducible: a reset leaf such as a start pose in batch row `i` uses seed `seed + i`, and a sampled variation leaf uses `np.random.default_rng((seed, i))`. The Craftax classes, Miniworld and `MSHumanLocomotion` take only `seed`.

## Classes that keep variation in state

`PushT`, `PyFlytRocketLanding`, `OGBCube` and `OGBScene` wrap the stable-worldmodel implementation, which keeps the variation inside the state. `set_variation` changes their defaults, which apply at the next reset, so sample their leaves through reset options instead:

```python
# needs the pusht extra
with snv.PushT(num_envs=4, device="cpu", size=96) as pusht:
    state = pusht.reset(seed=0, options={"variation": ["agent.color", "block.color"]})
    print(pusht.render(state).shape)   # torch.Size([4, 96, 96, 3])
    print(pusht.get_variation().shape)  # torch.Size([4, 29])
```

Some leaves also apply to the running episode:

- PushT: the leaves in `PushT.immediate_variation_leaves`, which are the colors, `goal.position`, `goal.angle`, `goal.scale` and `rendering.render_goal`. Body shapes, `agent.scale`, `block.scale` and start poses wait for the next reset.
- `PyFlytRocketLanding`: the color leaves. Start leaves (drop height, lateral offset, tilt) wait for the next reset.
- `OGBCube` and `OGBScene`: cube size and color, floor color, camera angle, light intensity and lock color. Placements wait for the next reset.

Each family page lists its classes' variation leaves, and [Specs and spaces](../api/specs.md#spaces) covers the space classes.

## Checklist

- Change physics leaves with `set_variation` or `reset(options=...)`. `render` never steps the simulation.
- Values from reset options last one episode. The next reset without options goes back to the variation `set_variation` installed.
- Pass a `(B, variation_dim)` tensor to `render(..., variation=)` to give each batch row its own look. A dict gives every batch row the same values.
- Pass your own `rng` to `sample_variation`. `reset(seed=...)` reseeds `env.rng`, so draws from `env.rng` repeat after each seeded reset.
- For `PushT`, `PyFlytRocketLanding`, `OGBCube` and `OGBScene`, sample leaves through reset options. Their `set_variation` only changes the defaults for the next reset.
- Store the variation rows next to the states you render. A frame depends on both.

Next: [Write your own environment](custom-env.md) builds a small environment and registers it.
