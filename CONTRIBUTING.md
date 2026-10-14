# Contributing

## Set up

Test environments are built from the single `uv.lock`, one per profile, with
[nox](https://nox.thea.codes) and [uv](https://docs.astral.sh/uv/):

```bash
uvx nox -l                              # list sessions
uvx nox -s "test(core)"                 # core tests, no family extras
uvx nox -s "test(dmc)" -- -k reacher    # one family under its pinned versions
uvx nox -s affected                     # the profiles CI selects for this branch
SE_PYTHON=3.10 uvx nox -s "test(core)"  # core profile on another interpreter
uvx nox -s lint                         # pre-commit over the tree
uvx nox -s typecheck                    # mypy on the adapter interface
uvx nox -s docs                         # strict documentation build
```

`test(<profile>)` runs `uv sync --frozen` with the family's extras and its `pin-<profile>` group into
`.venvs/<profile>-py3.12`. It then runs the family's unit, integration and CPU regression tests.
The profiles are `core`, `classic`, `navigation`, `pusht`, `dmc`, `fetch`, `ogbench`, `rocket` and
`craftax`. [`ci/families.toml`](ci/families.toml) lists the paths and tests each profile owns and what
it installs.

To work in one profile directly:

```bash
uv sync --python 3.12 --extra dmcontrol --group pin-dmc --group test
uv run pytest ...
```

`pip install -e ".[dev]"` also works. Extra conflicts, the Fetch MuJoCo pin and the nvdiffrast build
are on the [Install](docs/install.md) page. The ports in `contrib/` have their own requirements and
tests; see [contrib/README.md](contrib/README.md).

## Lint and format

```bash
ruff format .
ruff check .
uvx pre-commit run --all-files
```

CI runs the pre-commit hooks on the files a pull request changes.

## Tests

```bash
uvx nox -s "test(<profile>)"                   # what CI runs for one profile
pytest -m unit                                 # CPU-only unit tests
python -m tests._runner.supervisor --smoke     # one env per family
pytest tests/gpu -v                            # GPU conformance suite, on a CUDA machine
```

A family that is not installed is skipped, with the cause, not failed. CI sets
`SE_REQUIRE_FAMILIES` and `SE_REQUIRE_CAPABILITIES`, which turn that skip into an error.

Markers for tests that need something extra:

| marker | the test needs |
|---|---|
| `gl` | dm_control's OpenGL renderer |
| `egl` | an EGL library |
| `swm` | stable-worldmodel, imported without `importorskip` |
| `family("<name>")` | another family's environments |

The GPU suite requires CUDA and skips without it. Run it through `tests._runner.supervisor` when it
initializes native renderers. The supervisor isolates crashes and names the test that crashed.

The full test layout and what each check rules out: [Testing](docs/testing.md).

## What a new environment needs to pass CI

Every registered class is picked up automatically, in its family's CI profile:

- **Generic contract** (`tests/unit/test_registry_conformance.py`). Reset, step and render match the
  specs. The same seed gives the same rollout. A lane does not see its neighbours' actions.
  `get_state`/`set_state` restores bit-exactly, and the snapshot owns its memory. A variation round-trips
  through another env. A lane renders the same alone as in a batch.
- **Ownership** (`tests/unit/test_registry_coverage.py`). The module lies under exactly one family's
  `paths` in `ci/families.toml`.
- **Source comparisons** (`tests/helpers/reference/cases.py`).
  - `ORIGINAL_GATES` names a CPU test that compares the env with the library it ports, or with a corpus
    recorded from it. That test must not import stable-worldmodel and must not be `nightly`.
  - Where stable-worldmodel registers the env, `SWM_GATES` names a test marked `swm` that compares it
    with live stable-worldmodel.
- **Variation map.** If the stable-worldmodel source has a variation space, add a `VariationMap` in
  `tests/helpers/reference/variation_maps/`.

A test file must run in a profile that installs what it imports. Assign the file to a family in
`ci/families.toml`, or mark a single test with `pytest.mark.family("<name>")`.

The stable-worldmodel layer is built to be removed in one change. To remove it, delete:

- `SWM_GATES` in `tests/helpers/reference/cases.py`;
- the tests marked `swm` and the gate helpers only they use;
- the `swm` capability and the `[swm]` block in `ci/families.toml`;
- `_install_swm` in `noxfile.py`.

`ORIGINAL_GATES` then stands alone.

## Exemptions

If a check cannot apply (for example, a renderer-only class has no `step`), waive it in
`tests/exemptions.toml`:

```toml
[[exemption]]
env = "Fetch*Dense"          # class name or glob
family = "fetch"
checks = ["original", "swm"]
reason = "why the check cannot apply"
```

Known checks are `specs`, `seeded`, `lanes`, `state`, `variation`, `batch_render`, `owner`,
`original`, `swm` and `variation_map`. A waived check still runs and must keep failing. Once it
passes, the suite fails until you remove the entry. The file has a code owner.

## Continuous integration

`.github/workflows/ci.yml` runs on pull requests, pushes to `main`, and nightly.
`scripts/ci/plan.py` maps the diff to profiles through `ci/families.toml`:

- A docs-only change runs the docs build.
- A shared path (core modules, conftests, `pyproject.toml`, `uv.lock`, CI files) selects every profile.
- Otherwise CI runs the touched families plus `core`.

The label `ci:full` runs every profile on a pull request. The only required check is `ci-ok`.

Tests marked `nightly` run on pushes to `main`, the nightly schedule, manual runs and `ci:full`.
Each family profile is split into `shards` jobs (`ci/families.toml`) by `tests/_runner/shard.py`,
using per-test times from `ci/durations.json`. Locally:

```bash
SE_NIGHTLY=1 uvx nox -s "test(dmc)"   # include nightly tests
SE_SHARD=2/4 uvx nox -s "test(dmc)"   # run one shard
```

Refresh `ci/durations.json` from the `durations-*.json` files a CI run uploads when shards become
unbalanced.

A red run on `main` or the nightly opens or updates the issue "ci: main is red". Revert the change
that broke it unless a fix is merged within about two hours.

## Pull request rules

- Do not relax a parity threshold or edit a frozen fixture to make a test pass.
- Do not release an EGL context or close a `mujoco.Renderer`. It corrupts rendering in other live
  contexts. `tests/_runner/resource_guard.py` fails the test that does it.
- Do not create contexts, renderers or envs per state, per style value or per call.
- Read the upstream source, then confirm your reading by comparing against the official renderer.
  See [Fidelity methodology](docs/concepts/fidelity-methodology.md).
- Do not commit local artifacts (scratch output, screenshots, one-off JSON). Fixtures that a committed
  script regenerates live under `tests/fixtures/`.

## Adding a component

See the [environment, task and renderer guides](docs/adding/index.md).
