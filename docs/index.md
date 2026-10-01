![](assets/brand/mark.svg){ width="76" }

# stable-envs

**A batched environment library for research: explicit state, dynamics, and variation, with fast state-driven rendering across CPU and GPU.**

Keep exhaustive State for replay and branching; generate frames when needed. Compact RenderState is a private renderer projection, separate from simulator State and downstream learner observations.

Authors: **Randall Balestriero**, **Sami BuGhanem**.

```python
import stable_envs as snv

with snv.CartPole(num_envs=64) as env:
    state = env.reset(seed=0)
    pixels = env.render(state)                 # Frame
    out = env.step(env.action_spec.sample(64, env.rng))
    state = out["state"]
```

Concrete classes are canonical. `snv.make("CartPole", …)` is the string API for those class names. The full walkthrough is the [Quickstart](quickstart.md).

<div class="grid cards" markdown>

- **[Quickstart](quickstart.md)** — install, `reset` / `step` / `render`, specs, logging
- **[Install](install.md)** — extras, pins, nvdiffrast
- **[Environments](environments.md)** — `list_envs`, `info`, optional extras
- **[State, Action, Variation, Frame](concepts/state.md)** — the five nouns; compact render state; StepResult
- **[Stepping and capture](concepts/dynamics.md)** — single actions, sequences, States and frames
- **[Variations](concepts/variation-spaces.md)** — persistent environment configuration
- **[Lifecycle](concepts/lifecycle.md)** — `close()`, context manager, short batches
- **[Sibling projects](concepts/worldmodel.md)** — where this package ends
- **[Limitations](concepts/limitations.md)** — devices, extras, accepted limits
- **[Throughput](benchmarks.md)** — methodology, then the measured tables
- **[Gallery](gallery.md)** — official renderer and ours, same states
- **[Adding a component](adding/index.md)** — environment, task and renderer responsibilities

</div>

Storage and throughput benefits depend on the family and the retained State. [Throughput](benchmarks.md) describes the measured workloads and their limits.

## State and pixels have different responsibilities

`render(full_state)` produces pixels without advancing simulation. Stable-worldmodel
selects and packages learner-visible frames, proprioception and goals. Exhaustive
State may contain privileged information; it is not the learner observation.

Renderer fidelity and simulator restoration need independent evidence. See
[State](concepts/state.md), [the gallery](gallery.md), and
[the contributor guides](adding/index.md).

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
