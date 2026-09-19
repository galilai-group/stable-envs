# Changelog

## 0.1.0 — v0 draft

First public-shaped cut of `stable-envs` after the Step 8 functional-completion milestone
(`21cb306`) and the Step 9 refactor.

### Added

- Class-first public API: `snv.PushT(num_envs=..., device=...)`, plus `snv.make` / `snv.spec` /
  `snv.list_envs` for configs and interop.
- TensorDict `State` with `pack` / `unpack` to the unchanged Step-8 packed row.
- Stateful `step` and functional `transition` / `unroll` (known actions only).
- `close()` / context-manager lifecycle on every environment.
- Public `stable_envs.testing` measurement library and `stable_envs.interop` SWM facade.
- Explicit renderer resource ownership (`Closeable`, `BoundedCache`) for MJWarp / nvdiffrast /
  rendernet.
- Packaging extras, GitHub Actions, MkDocs, and OSS example / benchmark CLIs.

### Changed

- Environment families and render backends were reorganized without changing certified pixels.
- `batch=` remains accepted as a deprecated spelling of `num_envs=`.
- `StableEnv`, `REGISTRY`, and `register` still resolve once-per-process with a warning.

### Limitations carried from Step 8

- Fetch `background.color` stays partial (no re-freeze).
- `HumanoidmazeGiantNavigate` on CUDA is not process-reproducible on some Oscar nodes
  (`DOCUMENTED_LIMITATION`).
- Native `swm.World(env=snv.PushT(...))` is unsupported by the installed `stable-worldmodel`;
  registered-gym and `swap_renderer` paths remain.

Compatibility shims (`stable_envs.core`, `stable_envs.compat`, `envs/swm_nav.py`,
`envs/ogb_manip.py`, `envs/craftax_full.py`) stay importable in this draft so historical tests and
research scripts keep working. They are classified, not advertised.
