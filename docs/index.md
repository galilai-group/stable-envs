![](assets/brand/mark.svg){ width="76" }

# stable-envs

**A batched environment library for research: explicit state, dynamics, and variation, with fast state-driven rendering across CPU and GPU.**

Store compact render state, not frames. Choose the camera, the resolution and the configuration **afterwards**. On PushT the compact render state is 5 floats per step against a 96×96×3 frame: 20 bytes against 27,648, and 100% of the interior pixels are bit-identical to the official `gym_pusht` renderer.

Authors: **Randall Balestriero**, **Sami BuGhanem**.

```python
import stable_envs as snv

with snv.PushT(num_envs=64) as env:
    state = env.reset(seed=0)
    pixels = env.render(state)                 # Frame
    out = env.step(env.action_spec.sample(64, env.rng))
    state = out["state"]
```

Concrete classes are canonical. `snv.make("PushT", …)` is the string API for those class names. The full walkthrough is the [Quickstart](quickstart.md).

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — install, `reset` / `step` / `render`, specs, logging
- **[Install](install.md)** — extras, pins, nvdiffrast
- **[Environments](environments.md)** — `list_envs`, `info`, optional extras
- **[State, Action, Variation, Frame](concepts/state.md)** — the five nouns; compact render state; StepResult
- **[Stateful vs functional](concepts/dynamics.md)** — `step` writes; `transition` / `unroll` do not
- **[Variations](concepts/variation-spaces.md)** — persistent environment configuration
- **[Lifecycle](concepts/lifecycle.md)** — `close()`, context manager, short batches
- **[Sibling projects](concepts/worldmodel.md)** — where this package ends
- **[Limitations](concepts/limitations.md)** — devices, extras, accepted limits
- **[Throughput](benchmarks.md)** — methodology, then the measured tables
- **[Gallery](gallery.md)** — official renderer and ours, same states
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
            frames = env.render(env.unpack_render_state(s))
            loss = mse(net(frames), target)
    ```

Side-by-side on identical recorded states:

<video controls muted loop playsinline width="100%" poster="assets/media/PushT/grid.png">
  <source src="assets/media/PushT/sidebyside.mp4" type="video/mp4">
</video>

More in the [Gallery](gallery.md). Fidelity claims, including the visual tier, are in [Fidelity tiers](concepts/fidelity-tiers.md).

## How it works

A frame is `f(state, variation)`. Where native dynamics exist, `next_state = g(state, action, variation)`. Variation may live on the env rather than on every signature. Some families rasterize a `Scene2D`; tile-atlas families gather from a precomputed atlas; OGBench mazes ray-cast analytic primitives. Those override `render` and the suite applies the checks that still apply. Render fidelity and contact-solver fidelity stay separate claims.

- [The Scene2D IR](concepts/scene-ir.md)
- [The fused rasterizer](concepts/rasterizer.md)
- [Variation spaces](concepts/variation-spaces.md)

Research notes, including families that were examined and rejected, stay in the [research archive](research/index.md). A no-go is a result, not a gap to hide: [when does this pay off](research/when_does_this_pay_off.md).

## Building these docs

```bash
pip install "stable-envs[docs]"
mkdocs build --strict     # or: mkdocs serve
```

mkdocstrings reads source docstrings statically — the site build does not require torch or a GPU.
