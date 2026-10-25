# Changelog

All notable changes to this project are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Audited official `DIVERSITY` execution context for G08/G44/G45/G46/G48,
  retaining source factor defaults without driver rendering/appearance hooks.
  Adds G44's rigid recording, independent 120-frame oracle and cache parity gates;
  context provenance/mismatch checks protect source comparisons. The other four
  remain deferred for orthographic cameras and G46's FONT labels.

- Complete first-nominal object-permanence audit: 94 certified scenes and 56
  explicit deferrals across all 150 task groups; 47 additional member scripts
  remain unregistered. Adds 61 reusable rigid recordings with independent
  120-frame depth/mask oracles, fresh-process geometry checks and two-resolution
  cache parity. Publishes measured source hashes/reasons alongside static hints;
  G103 cache-seed rebuilds and G132 source-normal variation remain deferred.

- Early object-permanence audit G01–G10: nominal G02/G03 ramp balls and G05–G07
  rigid vehicles, with independent 120-frame ray oracles and full-frame cache
  parity. Adds audited legacy construction-prefix and named-argument source
  preparation. Transparency, curves and orthographic cameras remain uncertified,
  with labeled deferred GIF diagnostics; the later context audit resolves G08's
  missing driver global while retaining its orthographic rejection.

- Three reusable rigid batches: nominal G93–G97/G99/G100 support/fall, opening-container
  and scripted-collision scenes, with independent 120-frame versioned oracles
  and all-frame depth/mask/cache gates. Skips G90 transparency and G91/G92
  orthographic cameras, and defers G98 after independent source-normal failures,
  without extending the contract; adds live camera rejection
  tests. Source comparison previews retain GIFs only after temporary PNG cleanup.

- Two further rigid import batches: nominal G85/G87/G88 cup scenes and G89 height
  gate, with independent 120-frame ray oracles/GIF workflow and full-resolution
  cache parity checks. Records the G90 active-glass stopping boundary with a live
  rejection test; runtime code, opaque contract and earlier references stay unchanged.

- Audited nominal G82/G84/G86 rigid recordings with independent 120-frame
  pixel-ray oracles. Static visibility seeds preserve the versioned depth and
  semantic-ID rules, with full drawing retained for comparison, input invalidation
  and explicit resource release. Adds upload-inclusive cache measurements.

- Opt-in opaque pixel-ray visibility contract with stable semantic-ID ties,
  independent versioned source oracles and CUDA replay for G63/G66/G80/G81.
  Adds audited G65/G83 nominal clips and all-frame depth/mask gates. Legacy
  references, thresholds, defaults and BVH deferrals remain unchanged.

- Source visibility diagnostics distinguish exact coplanar ties, BVH boundary
  deviations and camera ray conventions. Optional float64 projection retains
  vertex precision through refined coverage; legacy defaults and frozen gates
  stay unchanged. Source-camera calibration is explicitly non-certifying.

- Second object-permanence batch: nominal G27 support removal and G55 rising
  screen pass independent full-frame depth/mask gates with the unchanged renderer.
  Adds bounded lossless reference chunks and a reusable rendering validator;
  G63/G66/G80/G81 remain explicitly deferred after source-fidelity failures.

- First object-permanence recording expansion: nominal G37 curved track, G38
  pendulum and G64 windowed flip board, using the unchanged shared renderer.
  Adds conservative source capability queues, independent all-frame Blender
  references with lossless depth chunks, and cached/uncached depth/mask gates.
  Existing G34/G62 frozen files and runtime interfaces are unchanged.

- Object-permanence source/shared GIF and upload-inclusive benchmark tools, isolated
  batch planning/export with source geometry checks, and a 120-frame G34 v2 oracle
  preserving all original samples. Shared/refined/raycast gates cover all frames;
  historical native/scaled raster gaps remain explicit expected failures.

- Rigid object-permanence recordings: shared content-addressed mesh assets,
  memory-mapped per-frame poses and an offline Blender exporter for nominal
  G34/G62 scenes, with an all-source inventory and live serialization checks.
  Semantic asset directories and shared licenses replace the derived index.
  A shared full-frame CUDA renderer handles sampled multi-object motion, verified
  against the original G34 oracle and a new independent all-frame G62 oracle.
  Refined coverage caches static visibility with exact depth/face-ID ties and
  uploads only dynamic-node poses; uncached drawing remains a comparison option.
  Existing G34 replay, MuJoCo interfaces and frozen geometry fixtures are unchanged.

- Engine-free `object_permanence.G34Replay`: restorable trajectory cursors and
  nvdiffrast CUDA rendering, without MuJoCo or Blender runtime calls. Reuses
  triangle rendering with rectangular RGB/depth/ID output, static plates and
  precise cropped viewports; the existing G34 geometry oracle is unchanged.
  Viewport helpers support multiple enclosing spheres, mesh-derived bounds and
  reusable CUDA camera matrices; tighter G34 windows reduce coverage memory,
  and static outputs are cached only when requested.
  `nvdiff.BatchRenderer.render_many()` shares coverage and interpolation across
  RGB/depth/IDs. Optional float64 coverage recovers missed thin triangles;
  G34 uses it for low-memory static plates and retains scaled dynamic coverage.

- `MuJoCoOcclusionBall` (`object-permanence` extra): the source-authored G34 nominal
  animation under `object_permanence`, with restorable MuJoCo State, native and
  batched raycast CPU/CUDA rendering, independent Blender trajectory/depth/mask fixtures
  and unchanged geometry gates. Raycast caches static plates and preserves the
  faceted sphere silhouette; its shading differs from native RGB. CUDA fuses
  float64 intersection/refinement/composition, uploading only the seven-float
  render projection from CPU State. MuJoCo simulation stays on CPU.

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

- `stable-envs doctor` reports how many render paths are tuned on this machine, lists the
  `STABLE_ENVS_*` variables that are set and flags unknown names, and checks torch's compiled GPU
  architectures with binary-compatibility rules.
- A `STABLE_ENVS_*` variable with a value that does not parse raises `ValueError` naming the variable.
- `cpu3d.render(chunk_size=None)` and `raster2d.render_triton(block=None)` read the tuning knobs.

### Fixed

### Removed

- `MJW_CHUNK` in the DMC and OGBench packages. The `dmc_min_worlds` and `ogb_mjw_worlds` tuning knobs
  set the worlds per MJWarp launch.

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
