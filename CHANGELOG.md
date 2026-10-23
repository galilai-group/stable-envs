# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Craftax: `CraftaxClassicPixels`, `CraftaxClassicSymbolic`, `CraftaxPixels` and `CraftaxSymbolic`
  run the upstream JAX simulator with all lanes in one jitted `vmap` (`craftax` / `craftax-gpu` extras).
  With Craftax, the extras provide 74 classes. Fetch and `OGBCube`/`OGBScene` pin different MuJoCo versions,
  so they need separate Python environments.
- `snv.tuning`: chunk sizes, Triton block sizes, worker counts and memory budgets are named knobs with
  `snv.tuning.set`, `snv.tuning.override` and `STABLE_ENVS_<KNOB>` variables. Some defaults are derived from the GPU's free memory or the CPU's L3 size.
  Knobs never change frames or State. See docs/performance-tuning.md.
- `stable-envs tune <Env>` and `snv.tune(env)` measure the fastest knob values for one render path on
  this machine and cache them; `--show`, `--clear` and `--all` manage the cache.
- A large render on an untuned path logs one `INFO` hint to run `stable-envs tune`
  (to hide it, set `STABLE_ENVS_TUNING_HINT=0`). `STABLE_ENVS_AUTOTUNE=1` tunes such a path on first use.
- docs/environment-variables.md lists every `STABLE_ENVS_*` variable.

### Changed

- One process adapter (`stable_envs._execution.process.ProcessAdapter`) runs the worker processes of
  DM Control, classic Acrobot, Fetch, OGBench Maze and OGBench Cube/Scene. It replaces five copies of
  the worker loop. Outputs do not change.
- With worker processes, `set_state` checks every lane in every worker before any worker restores.
  A State that is not valid in one lane raises and changes no lane in DM Control, classic Acrobot and
  Fetch too.
- PushT runs in worker processes on Linux for batches of at least `worker_min_lanes` lanes (tuning
  path `pusht-sim`).
- `visual_fidelity` records the per-frame MSE against the source as `mse_mean`, `mse_p95` and `mse_max`
  on the 0-255 scale. Only `measured_at` is required.
- The tests compare frames with a recorded source MSE for each class (`tests/fixtures/source_mse.json`)
  and with golden frames of our own renderers (`tests/fixtures/golden/`).
- `stable-envs doctor` reports how many render paths are tuned on this machine, lists the
  `STABLE_ENVS_*` variables that are set and flags unknown names, and checks torch's compiled GPU
  architectures with binary-compatibility rules.
- A `STABLE_ENVS_*` variable with a value that does not parse raises `ValueError` naming the variable.
- `cpu3d.render(chunk_size=None)` and `raster2d.render_triton(block=None)` read the tuning knobs.
- `PushT.step` and `PushTDiscrete.step` do not export a State when the caller requests no State. Physics-only
  stepping is approximately 2.9 times faster.
- `PushT` and `PushTDiscrete` keep the State fields that change only at reset, restore or `set_variation` as
  cached batch columns, record the action history in one growing array, and draw step frames from the running
  scenes. Resets copy the engine without a second scene build. Frames, State and task outputs do not change.
- PushT worker processes draw step frames and copy only the State leaves that a step changes. The parent
  keeps the other leaves and copies them for the next State while the workers step. Process adapters
  declare this with `process_static_leaves`, `process_step_state`, `render_live` and `frame_shape`.
- With `device="cuda"` and Triton, `PushT` and `PushTDiscrete` draw `render` and step frames on the GPU.
  A Triton kernel builds the shape records from the State tensors, and a second kernel draws them and
  resizes the canvas. The frames are byte-identical to the pygame frames. Frames with action markers,
  shapes that are not supported, or a coordinate within 1e-9 of a rounding midpoint use pygame.
- `PushT.render` and `PushTDiscrete.render` draw a batch of at least `worker_min_lanes` States in up to `workers`
  spawned processes on Linux (render path `pusht-render`). The frames do not change.
- `PyFlytRocketLanding` reads its live colours directly for the default render style and copies each State
  leaf to the host once during validation. Rendering of supplied pose rows is approximately 70 times faster.
  Frames and State do not change.

### Fixed

- `PyFlytRocketLanding` on CUDA draws the fins, legs and fuel tank at the current body pose, as PyFlyt
  does, and does not draw the ground when the camera is below it. The mean per-frame MSE against PyFlyt
  is 4.2 on 480 px frames.
- `PyFlytRocketLanding.render(state)` on CUDA picks ground texels as TinyRenderer does, so frames that look
  down at the far ground match PyFlyt when a ground corner is close to the camera plane.
- `PyFlytRocketLanding` changes the flame colour at ignition, as the PyFlyt pixel env does. The State
  `render/rgba` flame row and the CPU and CUDA frames show the red ignition flame. The source makes this
  change only when it has a render mode. Supplied pose rows carry no ignition, so they render with the flame
  off. States saved after ignition with an earlier version do not restore.
- `PointMazeLargeNavigate` renders batches of 2048 and more on CUDA. Its spotlight pass casts wall shadow
  rays in one launch row per frame and lights a bounded number of rows at a time.

### Removed

- `MJW_CHUNK` in the DMC and OGBench packages. The `dmc_min_worlds` and `ogb_mjw_worlds` tuning knobs
  set the worlds per MJWarp launch.
- `Env.fidelity_tier`, `stable_envs.env.FIDELITY_TIERS` and the `fidelity_tier` key of `snv.info()`.
- `Env.parity_note` and `stable_envs.envs.dmcontrol.PARITY_NOTE`. `visual_fidelity` and `visual_note` describe
  how frames differ from the source.

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
