# Changelog

## Unreleased

### Added

- Simulation for every DM Control cell (the six that were renderer-only now run dm_control's own
  tasks on CPU lanes) and for every OGBench locomotion maze cell on CPU MuJoCo. 76 environments
  register with every extra installed; only `CraftaxClassic` and `CraftaxFull` are renderer-only.
- Reset rule: every `reset` installs the defaults, then the leaves `options["variation"]` samples
  and `options["variation_values"]` sets, per lane (lane `i` seeded with `seed + i`), as
  stable-worldmodel's `reset_variation_space` does. `Env` declares `reset_leaves`,
  `reset_value_paths` and `reset_sample_paths`.
- Physics Variation leaves (gravity, densities, masses, friction, classic-control parameters) travel
  in State under `("physics", <name>)`; `set_state` installs them for that episode and refuses a
  State without them.
- Variation leaves gated against stable-worldmodel for classic control, navigation, PushT, Rocket,
  Fetch (including `rendering.transparent_arm`, flat `sky.color` and every reset leaf), OGBench
  mazes (floor colour, pixel encoding, light intensity) and Cube/Scene (every leaf per lane).
- DM Control trackcom `floor.color` and `DMCWalkerWalk` `floor.rotation_y`, per lane.
- `stable_envs.spaces`: `Dict(sampling_order=...)` and `Dict.ordered_names()`, `Box(exact=True)`,
  and `set_init_value(..., reset_value=False)`.
- CI: one `uv.lock` with `pin-*` dependency groups, `nox` sessions per profile
  (`uvx nox -s "test(<profile>)"`), a diff-to-profile plan from `ci/families.toml`, a single `ci-ok`
  check and a nightly run of every profile. Family profiles that compare against stable-worldmodel
  install it at the pinned commit and run those gates live. Pull requests skip the multi-minute
  `nightly` tests, and each family profile runs as duration-balanced shards (`ci/durations.json`).
- Two source-comparison layers per env: `ORIGINAL_GATES` against the library it ports, run with
  stable-worldmodel unimportable, and `SWM_GATES` against live stable-worldmodel, marked `swm`.
- Registry-wide rules for every registered class: the generic contract
  (`tests/unit/test_registry_conformance.py`), family ownership, a source comparison and a variation
  map (`tests/unit/test_registry_coverage.py`). Waivers live in `tests/exemptions.toml`, which has a
  code owner.

### Changed

- **Breaking:** the PushT and Rocket Variation spaces use the public `stable_envs.spaces` classes.
  Scalar `Box` leaves not declared `exact=True` hold their defaults as 0-d arrays cast to the leaf
  dtype, where the private copies kept Python scalars; `Box(..., exact=True)` keeps values as given.
- Classic-control State carries the physics Variation fields.
- `set_variation` sets the defaults as well as the current value, so a plain reset returns to what
  it last installed. Cube/Scene apply their leaves to the running episode; PushT applies its
  appearance leaves immediately.
- The stable-worldmodel reference is pinned at `21446f1ede6d5284e981bd7b47f432b994e6d812`; PushT
  and Cheetah follow its changes.
- Cube and Scene render every camera, light, cube-size and lock-colour configuration in one MJWarp
  context per family instead of one per configuration.
- Trackcom visual-fidelity declarations are re-measured at mujoco 3.12.0 for all six cells; the
  reference profile `pin-dmc` is `mujoco==3.12.0` and `dm-control==1.0.45`.
- The Humanoid walk cell is `DMCHumanoidWalk` everywhere.

### Fixed

- Cube and Scene ran out of GPU memory at 24 distinct camera, light or size configurations on an RTX 3090.
- Fetch `background.color` drove only the floor; it now also fills the sky flat away from the
  default.

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
- `HumanoidMazeGiantNavigate` on CUDA is not process-reproducible on at least one measured card.
- There is no runtime interop contract with other stable-* packages.

See [limitations](docs/concepts/limitations.md).
