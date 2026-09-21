# Changelog

## 0.1.0

First public release of `stable-envs`: a batched environment library with explicit state,
dynamics, and variation, plus fast state-driven rendering. Store compact states; choose the
camera and the configuration afterwards.

### What you get

- Class-first public API: `snv.PushT(num_envs=..., device=...)`, plus `snv.make` / `snv.info` /
  `snv.list_envs` / `snv.list_families` for configs.
- TensorDict `State` with `pack_render_state` / `unpack_render_state` to the compact render
  row every renderer and certification corpus uses.
- Stateful `step` and functional `transition` / `unroll` (known actions only). There is no
  `env.rollout(policy=...)`.
- `close()` / context-manager lifecycle on every environment.
- Public developer library: `stable_envs.testing` (measurement, numpy.testing-style).
  Not advertised at `snv.*`.
- Optional family extras. A missing extra costs that family; `snv.import_errors()` names it.
- Explicit renderer resource ownership (`Closeable`, `BoundedCache`) for MJWarp / nvdiffrast /
  rendernet, including family-cache eviction and LRU CPU working-set caches.
- MkDocs, example CLIs, and the `benchmarks/` harness. Throughput numbers are tied to
  [benchmark methodology](docs/benchmarks.md).

### Limitations

- Fetch `background.color` is partial (floor mapped; skybox residual measured).
- `HumanoidMazeGiantNavigate` on CUDA is not process-reproducible on some measured nodes.
- There is no runtime interop contract with other stable-* packages.
- OGBCube, OGBScene and PyFlytRocketLanding refuse CPU rather than silently falling back.
- Fetch has no CPU backend.

See [limitations](docs/concepts/limitations.md).
