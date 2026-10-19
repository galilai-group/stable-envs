![](assets/brand/mark.svg){ width="76" }

# stable-envs

**A batched environment library for research: explicit state, dynamics, and variation, with state-driven rendering across CPU and GPU.**

Store the full State. Render frames from it when you need them.

Authors: **Randall Balestriero**, **Sami BuGhanem**.

```python
import stable_envs as snv

with snv.CartPole(num_envs=64) as env:
    state = env.reset(seed=0)
    pixels = env.render(state)                 # Frame
    out = env.step(env.action_spec.sample(64, env.rng))
    state = out["state"]
```

Class names are the canonical API. `snv.make("CartPole", …)` takes the same names as strings. The full
walkthrough is the [Quickstart](quickstart.md).

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)**: install, `reset` / `step` / `render`, specs
- **[Install](install.md)**: extras, pins, nvdiffrast
- **[System setup](system-setup.md)**: driver, OpenGL and `stable-envs doctor`
- **[Environments](environments.md)**: `list_envs`, `info`, and one page per family with clips and specs
- **[State, Action, Variation and Frame](concepts/state.md)**: what each value holds
- **[Stepping](concepts/dynamics.md)**: single actions, sequences, States and frames
- **[Variations](concepts/variation-spaces.md)**: persistent environment configuration
- **[Lifecycle](concepts/lifecycle.md)**: `close()`, context manager, short batches
- **[Sibling projects](concepts/worldmodel.md)**: where this package ends
- **[Limitations](concepts/limitations.md)**: devices, extras, known limits
- **[Benchmarks](benchmarks.md)**: method and measured tables
- **[Command line](api/cli.md)**: `stable-envs doctor`
- **[Adding a component](adding/index.md)**: environment, task and renderer guides
- **[Changelog](changelog.md)**: changes in each release

</div>

## How it works

A frame is `f(state, variation)`. Where native dynamics exist, `next_state = g(state, action, variation)`.
Families render in different ways:

- 2-D families build a `Scene2D` and rasterize it. See [The Scene2D IR](concepts/scene-ir.md) and
  [2D raster rendering](api/raster2d.md).
- Tile-atlas families gather tiles from a precomputed atlas.
- OGBench mazes ray-cast analytic primitives.
- MuJoCo families use MuJoCo's OpenGL renderer or MJWarp.

## Building these docs

```bash
pip install "stable-envs[docs]"
mkdocs build --strict     # or: mkdocs serve
```

mkdocstrings reads docstrings statically. The build does not need torch or a GPU.
