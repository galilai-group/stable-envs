# Contributing

## Your first contribution

### Get access

The repository is private. A maintainer adds you as a collaborator on GitHub. Accept the
invitation, then clone the repository.

```bash
git clone git@github.com:galilai-group/stable-envs.git
cd stable-envs
```

### Make a branch

Branch off `v1-dev`. Name the branch `<your-name>/<short-topic>`, for example `alice/pusht-docs`.

```bash
git switch v1-dev
git pull
git switch -c alice/pusht-docs
```

### Set up

Install [uv](https://docs.astral.sh/uv/). Then install the development tools and the git hooks:

```bash
uv sync --python 3.12 --group dev
uv run pre-commit install
```

The hooks run ruff and a few file checks on each commit.

### Run the tests

Run the core tests, then the tests of the family you changed:

```bash
uvx nox -s "test(core)"
uvx nox -s "test(<profile>)"
```

The profiles are `core`, `classic`, `navigation`, `pusht`, `dmc`, `fetch`, `ogbench`, `rocket`, `craftax` and
`miniworld`. [`ci/families.toml`](ci/families.toml) lists the files each profile owns. `uvx nox -s affected`
runs the test profiles that your branch touches. To only print them, run
`python scripts/ci/plan.py --base origin/main --profiles-only`. If you changed the docs, also run
`uvx nox -s docs`.

GPU tests need a CUDA machine. Run them with `pytest tests/gpu -v` if you changed GPU code.
`stable-envs doctor` checks a machine's setup.

### Open a pull request

1. Push your branch and open a draft pull request into `v1-dev`.
2. Fill in the pull request template.
3. When you finish the work and the tests pass locally, click "Ready for review".

CI runs only on pull requests that are ready for review. Draft pull requests do not run CI. CI
has a limited number of minutes, so keep the pull request in draft while you work on it.

### Review

A code owner reviews every pull request. The reviewer may ask for changes. Push new commits to the
same branch to answer them. Do not force-push after review starts. The pull request is merged
when a reviewer approves it and the `ci-ok` check passes.

Add a line to `CHANGELOG.md` under `Unreleased` for any change a user would notice.

### Ask questions

Comment on your pull request, open an issue, or ask a maintainer directly.

## Test environments

[nox](https://nox.thea.codes) and [uv](https://docs.astral.sh/uv/) build the test environments from the
single `uv.lock`, one per profile:

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
The profiles are `core`, `classic`, `navigation`, `pusht`, `dmc`, `fetch`, `ogbench`, `rocket`, `craftax` and
`miniworld`. [`ci/families.toml`](ci/families.toml) lists the paths and tests each profile owns and what it
installs.

To work in one profile directly:

```bash
uv sync --python 3.12 --extra dmcontrol --group pin-dmc --group test
uv run pytest ...
```

`pip install -e ".[dev]"` also works. The [Install](docs/install.md) page covers extra conflicts, the
Fetch MuJoCo pin and the nvdiffrast build.

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

If a family is not installed, its tests skip and report the cause. CI sets
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

[Testing](docs/testing.md) describes the full test layout and what each check rules out.

## Pull request rules

- Do not relax a parity threshold or edit a frozen fixture to make a test pass.
- Do not release an EGL context or close a `mujoco.Renderer`. It corrupts rendering in other live
  contexts. `tests/_runner/resource_guard.py` fails the test that does it.
- Do not create contexts, renderers or envs per state, per style value or per call.
- Read the upstream source, then check your reading by comparing against the official renderer.
  See [Fidelity methodology](docs/concepts/fidelity-methodology.md).
- Do not commit local artifacts (scratch output, screenshots, one-off JSON). Fixtures that a committed
  script regenerates live under `tests/fixtures/`.

## Adding a component

See the [environment, task and renderer guides](docs/adding/index.md).

## What a new environment needs to pass CI

CI collects every registered class automatically, in its family's CI profile:

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

## Maintainers

The sections below cover CI internals and rules that most contributions do not need.

### Continuous integration

`.github/workflows/ci.yml` runs on ready (non-draft) pull requests into `main` and `v1-dev`, on pushes to `main`, and weekly. The `ci`, `docs` and `extras-smoke` jobs run on these events only when the repository variable `STABLE_ENVS_CI` is `true`. A manual run from the Actions tab always runs. Pull requests test core on Python 3.12 only.
`scripts/ci/plan.py` maps the diff to profiles through `ci/families.toml`:

- A docs-only change runs the docs build.
- A shared path (core modules, conftests, `pyproject.toml`, `uv.lock`, CI files) selects every profile.
- Otherwise CI runs the touched families plus `core`.

The label `ci:full` runs every profile on a pull request. The only required check is `ci-ok`.

Tests marked `nightly` run on pushes to `main`, the weekly schedule, manual runs and `ci:full`.
`tests/_runner/shard.py` splits each family profile into `shards` jobs (`ci/families.toml`),
using per-test times from `ci/durations.json`. Use these variables locally:

```bash
SE_NIGHTLY=1 uvx nox -s "test(dmc)"   # include nightly tests
SE_SHARD=2/4 uvx nox -s "test(dmc)"   # run one shard
```

Refresh `ci/durations.json` from the `durations-*.json` files a CI run uploads when shards become
unbalanced.

A red run on `main` or the nightly opens or updates the issue "ci: main is red". Revert the change
that broke it unless a fix is merged within about two hours.

### Exemptions

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

### Removing the stable-worldmodel layer

You can remove the stable-worldmodel layer in one change. To do this, delete these items:

- `SWM_GATES` in `tests/helpers/reference/cases.py`
- the tests marked `swm` and the gate helpers only they use
- the `swm` capability and the `[swm]` block in `ci/families.toml`
- `_install_swm` in `noxfile.py`

`ORIGINAL_GATES` then stands alone.
