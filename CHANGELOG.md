# Changelog

## 0.1.0

First public release of `stable-envs`: famous research environments as GPU-batched, state-driven
renderers. Store compact states, choose the camera and the appearance afterwards.

### What you get

- Class-first public API: `snv.PushT(num_envs=..., device=...)`, plus `snv.make` / `snv.info` /
  `snv.list_envs` / `snv.list_families` for configs and interop.
- TensorDict `State` with `pack_render_state` / `unpack_render_state` to the compact render
  row every renderer and certification corpus uses.
- Stateful `step` and functional `transition` / `unroll` (known actions only). There is no
  `env.rollout(policy=...)`.
- `close()` / context-manager lifecycle on every environment.
- Public developer libraries: `stable_envs.testing` (measurement, numpy.testing-style) and
  `stable_envs.interop` (stable-worldmodel adapters). Neither is advertised at `snv.*`.
- Optional family extras. A missing extra costs that family; `snv.import_errors()` names it.
- Explicit renderer resource ownership (`Closeable`, `BoundedCache`) for MJWarp / nvdiffrast /
  rendernet, including family-cache eviction and LRU CPU working-set caches.
- MkDocs, example CLIs, and the `benchmarks/` harness. Throughput numbers are tied to
  [benchmark methodology](docs/benchmarks.md).

### Limitations

- Fetch `background.color` is partial (floor mapped; skybox residual measured).
- `HumanoidMazeGiantNavigate` on CUDA is not process-reproducible on some measured nodes.
- Native `swm.World(env=snv.PushT(...))` is unsupported by the installed `stable-worldmodel`;
  registered-gym and `swap_renderer` paths remain.
- OGBCube, OGBScene and PyFlytRocketLanding refuse CPU rather than silently falling back.
- Fetch has no CPU backend.

See [limitations](docs/concepts/limitations.md).
