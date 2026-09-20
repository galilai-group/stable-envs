# v0.1.0 release surface (P0 contract)

Binding for the polish pass that follows `v0-draft` @ `973ea70`.
Audit: `STABLE_ENVS_V0_POLISH_AUDIT.md`. This pass is **repository polish**, not
renderer work. Do not change certified pixels. Do not push `main`. Do not tag
a release from these branches.

Authors remain **Randall Balestriero, Sami BuGhanem**. Commits: `sami-bg
<samibghanem@gmail.com>`, no `Co-authored-by` trailers (rewrite with
`git commit-tree` if Cursor injects one).

---

## Public import surface (`dir(snv)` / `__all__`)

```text
snv.Env, snv.State, snv.StepResult
snv.Dynamics, snv.Spec, snv.CompositeSpec
snv.make, snv.spec, snv.list_envs, snv.list_families, snv.import_errors
snv.set, snv.get_config, snv.SWM_IDS, snv.__version__
concrete env classes via snv.PushT / snv.make("PushT")
```

`import stable_envs as snv` still auto-discovers families. A missing optional
dependency costs that family alone (`snv.import_errors()`).

## Developer and internal modules

| module | status | why |
|---|---|---|
| `stable_envs.interop` | **public developer API** | SWM facade; not advertised at `snv.*` |
| `stable_envs.testing` | **public developer API** | measurement library (`numpy.testing`-style); stays |
| `stable_envs.testing._rollout` | **private** | certification/GPU helper only; not a user API |
| `stable_envs._data` | **private** | corpus helpers for tests; storage is not a stable-envs responsibility |
| `stable_envs.render` | **internal** | not a user import |
| `stable_envs.spaces` | **internal** | used by interop / envs |
| `stable_envs._style` | **internal** | |

`env.unroll(state, actions)` stays. `env.sample(policy=...)` and `env.rollout(policy)` are not APIs.

## Removed before first tagged 0.1.0

No unpublished-architecture compatibility in the first public cut:

- `stable_envs/core/`
- `stable_envs/compat/`
- `stable_envs/envs/swm_nav.py`
- `stable_envs/envs/simple_point_maze.py`
- `stable_envs/envs/ogb_manip.py`
- `stable_envs/envs/craftax_full.py` (teach `discover()` to load `_craftax_full`)
- `snv.StableEnv` / `snv.REGISTRY` / `snv.register` `__getattr__` aliases
- `batch=` as a spelling of `num_envs=` (never shipped in a tagged release)
- `__init__` `_LEGACY_MODULES` / `_DEPRECATED` machinery once callers are canonical

Scratch and milestone tags keep the old import paths reproducible. Maintained
tests, examples, and docs must use canonical modules **before** the shims go.

## Do not touch

- certified frames / variation maps / `21cb306` oracle JSON
- Fetch `background.color` skybox (re-freeze)
- EGL never-release rule
- native `swm.World(env=...)` (upstream)
- GiantNavigate CUDA process-repro (hardware)
- deleting `scratch/` or `examples/results/` until E’s classification is
  reviewed (classify only in this wave)

## Workstream file ownership (no overlap)

| id | branch | owns | does not own |
|---|---|---|---|
| A+B | `v0/api-surface` | `stable_envs/{__init__,registry,env}.py` (alias/`batch=` only), `core/`, `compat/`, env alias modules, tests/examples that still import shims, `discover()` `_craftax_full` | docs nav, ruff config, scratch moves |
| D+G | `v0/docs-product` | `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `docs/` (except `docs/research/` archaeology moves), `mkdocs.yaml` | deleting shims, ruff |
| E | `v0/archaeology` | classification write-up under `docs/research/` or `examples/results/step9/`; **no deletes** | product code |
| F | `v0/ci-packaging` | `.github/`, `scratch/step9/packaging-ci/` → `tests/packaging/` + `scripts/ci/`, pyproject scripts that only those files reference | unit test renames that A is editing; ruff |
| C | `v0/format` **after** the others merge | `pyproject.toml` ruff, `.pre-commit-config.yaml`, format of `stable_envs/ tests/ benchmarks/ scripts/ examples/*.py` | semantics, docs prose, archaeology |

C does not start until A+B, D+G, and F have been merged or explicitly parked.
Formatting the whole tree while others edit is a guaranteed conflict.

## Open items disposition

See [step9_open_items.md](step9_open_items.md). Items 1–4 stay documented
limitations. The `craftax_full.py` discovery shim (former item 5) is closed:
`discover()` loads `_craftax_full` explicitly.
