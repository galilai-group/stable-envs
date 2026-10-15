# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Craftax: `CraftaxClassicPixels`, `CraftaxClassicSymbolic`, `CraftaxPixels` and `CraftaxSymbolic`
  run the upstream JAX simulator with all lanes in one jitted `vmap` (`craftax` / `craftax-gpu` extras).
- `snv.tuning`: chunk sizes, Triton block sizes, worker counts and memory budgets are named knobs with
  `snv.tuning.set`, `snv.tuning.override`, `STABLE_ENVS_<KNOB>` variables and a per-environment
  `tuning={...}` argument. Some defaults are derived from the GPU's free memory or the CPU's L3 size.
  Knobs never change frames or State. See docs/performance-tuning.md.
- `stable-envs tune <Env>` and `snv.tune(env)` measure the fastest knob values for one render path on
  this machine and cache them; `--show`, `--clear` and `--all` manage the cache.

### Changed

- `stable-envs doctor` reports how many render paths are tuned on this machine, and checks torch's
  compiled GPU architectures with binary-compatibility rules.
- `cpu3d.render(chunk_size=None)` and `raster2d.render_triton(block=None)` read the tuning knobs.
- `MJW_CHUNK` is no longer exported by the DMC and OGBench packages; use the `dmc_min_worlds` and
  `ogb_mjw_worlds` knobs.

### Fixed

### Removed

## [0.1.0] - 2026-10-14

Initial release.

### Added

- Batched environments in seven families: classic control, DM Control, Fetch, navigation,
  OGBench, PushT and PyFlyt Rocket Landing. 70 environments register when every extra is
  installed. A missing extra disables only its family; `snv.import_errors()` names it.
- Class-first API: `snv.CartPole(num_envs=..., device=...)`, plus `snv.make`, `snv.info`,
  `snv.list_envs` and `snv.list_families`.
- Batched stepping on CPU or GPU: one `step` call takes `[B, A]` actions for one transition or
  `[B, T, A]` for a sequence, and returns a validity mask instead of resetting finished lanes.
- Batched rendering: `render(state)` draws frames from a State without advancing the simulation.
- Save and restore: `get_state` / `set_state` with a complete TensorDict `State`.
- Variation spaces: `sample_variation` / `set_variation` and per-lane reset options, compatible
  with stable-worldmodel.
- Source-comparison tests: each environment is checked against the library it ports and against
  stable-worldmodel, for State, simulation, tasks and pixels.
- `stable-envs doctor` checks the driver, Warp and MuJoCo OpenGL setup per env family.
- `stable_envs.testing` measurement helpers, the `benchmarks/` harness and MkDocs documentation.

See [Limitations](https://galilai-group.github.io/stable-envs/concepts/limitations/) for known gaps.

[Unreleased]: https://github.com/galilai-group/stable-envs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/galilai-group/stable-envs/releases/tag/v0.1.0
