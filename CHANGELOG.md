# Changelog

## 0.1.0

First public release of `stable-envs`: a batched environment library with explicit state,
dynamics and variation, and state-driven rendering. Store compact states; choose the camera and
the configuration afterwards.

### What you get

- 75 registered environments in seven families: classic control, Craftax, DM Control, Fetch,
  navigation, OGBench and PushT. PyFlyt Rocket Landing is an eighth family when its extra is
  installed.
- Class-first API: `snv.CartPole(num_envs=..., device=...)`, plus `snv.make`, `snv.info`,
  `snv.list_envs` and `snv.list_families`.
- Exhaustive TensorDict `State` with `get_state` / `set_state`, and `render(state)` that draws a
  frame without advancing the simulation.
- One stateful `step(actions, states=slice(None), frames=slice(-1, None))`: `[B, A]` actions for
  one transition, `[B, T, A]` for a sequence. Sequence lanes stop at termination or truncation
  and report a validity mask; episodes are not reset silently. `frames=None` skips rendering.
- Variation and style spaces with `sample_variation` / `set_variation` and
  `sample_style` / `set_style`.
- `close()` and context-manager lifecycle on every environment.
- `stable_envs.testing`: the measurement helpers the test suite uses, for checking an env or
  renderer against its reference.
- Optional family extras. A missing extra costs that family only; `snv.import_errors()` names it.
- MkDocs documentation, examples, and the `benchmarks/` harness. Throughput numbers come with the
  [benchmark methodology](docs/benchmarks.md).

Renderer-only ports of other families (MiniGrid, Sokoban, Meta-World, HumanoidBench and others)
live in the repository's `contrib/` directory. They are not part of the released package.

### Limitations

- PushT physics and rendering run on CPU through the source Pygame renderer. `device="cuda"`
  places outputs on the GPU but does not render there.
- Fetch, OGBCube and OGBScene render on CPU through software OpenGL and on GPU through MJWarp.
- Fetch `background.color` is partial: the floor is mapped, the skybox is not.
- `HumanoidMazeGiantNavigate` on CUDA is not process-reproducible on at least one measured card.
- There is no runtime interop contract with other stable-* packages.

See [limitations](docs/concepts/limitations.md).
