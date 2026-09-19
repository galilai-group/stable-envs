![](assets/brand/mark.svg){ width="76" }

# stable-envs

**Famous research environments, rendered on the GPU as a pure function of a compact state, batched over thousands of environments in one launch.**

Store states, not frames. Choose the camera, the resolution and the appearance **afterwards**. PushT is 5 floats per step against a 96×96×3 frame: 20 bytes against 27,648, and 100% of the interior pixels are bit-identical to the official `gym_pusht` renderer.

Authors: **Randall Balestriero**, **Sami BuGhanem**.

```python
import stable_envs as snv

with snv.PushT(num_envs=64, device="cpu") as env:
    state = env.reset(seed=0)
    pixels = env.render(state)                 # (64, 96, 96, 3) uint8
    out = env.step(env.action_spec.sample(64, env.rng))
```

Concrete classes are canonical. `snv.make("swm/PushT-v1", …)` is the string API for configs and interop. The full walkthrough is the [Quickstart](quickstart.md).

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — install, `reset` / `step` / `render`, specs, logging
- **[Environments](environments.md)** — `list_envs`, `spec`, SWM aliases, optional extras
- **[State and specs](concepts/state.md)** — TensorDict keys, pack/unpack, StepResult
- **[Stateful vs functional](concepts/dynamics.md)** — `step` writes; `transition` / `unroll` do not
- **[Lifecycle](concepts/lifecycle.md)** — `close()`, context manager, short batches
- **[Registry](concepts/registry.md)** — classes canonical, strings secondary
- **[stable-worldmodel](concepts/worldmodel.md)** — ownership boundary; frozen interop
- **[Limitations](concepts/limitations.md)** — devices, extras, accepted Step-8 limits
- **[Throughput](benchmarks.md)** — methodology, then the measured tables
- **[Adding an environment](adding-an-env.md)** — the worked PushT port

</div>

The storage win is universal (compact state vs raw pixels). The *speed* win is per-family and is **not** universal — against nvJPEG on the GPU it is a wash. [Throughput](benchmarks.md) says which families beat which read path, including the ones that do not.

## Keep your training loop. Delete your image dataset.

Rendering is a drop-in: `render` is a pure function of state. Stepping is optional and is **not** the same claim — a contact solver cannot be bit-exact the way a rasterizer can. [Why physics is separate](concepts/physics.md).

=== "Before"

    ```python
    ds = ImageFolder("frames/", transform=to_tensor)
    loader = DataLoader(ds, batch_size=1024, num_workers=8)
    for frames, target in loader:
        frames = frames.cuda(non_blocking=True)
        loss = mse(net(frames), target)
    ```

=== "After"

    ```python
    with snv.PushT(num_envs=1024, device="cuda") as env:
        ds = TensorDataset(states)
        loader = DataLoader(ds, batch_size=1024)
        for (s,) in loader:
            frames = env.render(env.unpack(s))
            loss = mse(net(frames), target)
    ```

Side-by-side on identical recorded states:

<video controls muted loop playsinline width="100%" poster="assets/media/PushT/grid.png">
  <source src="assets/media/PushT/sidebyside.mp4" type="video/mp4">
</video>

More in the [Gallery](gallery.md). Fidelity claims, including the visual tier, are in [Fidelity tiers](concepts/fidelity-tiers.md).

## How it works

An env that can produce a `Scene2D` from a state has *proved* the frame is a function of the state. Tile-atlas families gather from a precomputed atlas; OGBench mazes ray-cast analytic primitives. Those override `render` and the suite applies the checks that still apply.

- [The Scene2D IR](concepts/scene-ir.md)
- [The fused rasterizer](concepts/rasterizer.md)
- [Variation spaces](concepts/variation-spaces.md)

Research notes, including families that were examined and rejected, stay in [research](research/). A no-go is a result, not a gap to hide: [when does this pay off](research/when_does_this_pay_off.md).

## Building these docs

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs build --strict     # or: mkdocs serve
```

mkdocstrings reads source docstrings statically — the site build does not require torch or a GPU.
