# Contributing

Thanks for helping. This page takes you from a fresh clone to a merged pull request, and [Adding an environment](https://galilai-group.github.io/stable-envs/contributing/environment/) covers adding a simulator, a task or a renderer.

## Your first contribution

### Get the code

Fork [galilai-group/stable-envs](https://github.com/galilai-group/stable-envs) on GitHub, then clone your fork:

```bash
git clone git@github.com:<your-user>/stable-envs.git
cd stable-envs
git remote add upstream https://github.com/galilai-group/stable-envs.git
```

### Make a branch

Branch off `main`. Name the branch `<your-name>/<short-topic>`, for example `alice/pusht-docs`.

```bash
git switch main
git pull upstream main
git switch -c alice/pusht-docs
```

### Set up

Install [uv](https://docs.astral.sh/uv/), then install the development tools and the git hooks:

```bash
uv sync --python 3.12 --group dev
uv run pre-commit install
```

The hooks run ruff and a few file checks on each commit. `pip install -e ".[dev]"` also works.

- uv takes the Miniworld fork from `[tool.uv.sources]` in `pyproject.toml`, so `uv sync --extra miniworld` needs no extra step. With pip, install the fork first, as [Install](https://galilai-group.github.io/stable-envs/install/#miniworld) shows.
- The `pin-<profile>` dependency groups are the reference profiles: the exact versions parity is certified under, on Python 3.12 only. `tests/unit/test_reference_pins.py` keeps them equal to `tests/helpers/reference/profiles.py`, and uv refuses to combine two of them.
- `[tool.uv] conflicts` also keeps the `robotics` extra and `pin-fetch` (MuJoCo 3.11) apart from the extras and groups that need MuJoCo 3.17, so sync Fetch work into its own environment.

To work in one profile directly:

```bash
uv sync --python 3.12 --extra dmcontrol --group pin-dmc --group test
uv run pytest ...
```

### Run the tests

Run the core tests, then the tests of the family you changed:

```bash
uvx nox -s "test(core)"
uvx nox -s "test(<profile>)"
uvx nox -s affected                     # the profiles your branch touches
uvx nox -s lint                         # pre-commit over the tree
uvx nox -s typecheck                    # mypy on the adapter interface
```

`test(<profile>)` runs `uv sync --frozen` with the family's extras and its `pin-<profile>` group into `.venvs/<profile>-py3.12`, then runs the family's unit, integration and CPU regression tests. [`ci/families.toml`](ci/families.toml) lists the paths and tests each profile owns and what it installs, and `python scripts/ci/plan.py --base origin/main --profiles-only` prints the profiles your branch selects without running them.

GPU tests need a CUDA machine, so if you changed GPU code, run `python -m tests._runner.supervisor -- tests/gpu -v` there. The supervisor names the test that crashed when a native renderer ends the interpreter. `stable-envs doctor` checks a machine's setup. [Tests](https://galilai-group.github.io/stable-envs/contributing/testing/) describes the layout, markers, profiles and what each check rules out.

### Build the docs

If you changed the docs, run the same strict build that CI runs:

```bash
uvx nox -s docs
```

To preview the site while you edit, serve it locally and open the address it prints:

```bash
uv run --group docs mkdocs serve --config-file mkdocs.yaml
```

mkdocstrings reads the docstrings statically, so the build doesn't need a GPU.

Write each Markdown paragraph and list item on one line; don't hard-wrap.

### Open a pull request

1. Push your branch to your fork and open a pull request into `main`.
2. Fill in the pull request template.
3. If the work isn't finished, open the pull request as a draft and click "Ready for review" once the tests pass locally.

CI runs only on pull requests that are ready for review, because CI minutes are limited and drafts change often. Draft pull requests don't run CI.

### Review

A code owner reviews every pull request and may ask for changes. Push new commits to the same branch to answer them, but don't force-push after review starts, because that hides what changed since the last review. The pull request is merged when a reviewer approves it and the `ci-ok` check passes.

If a user would notice your change, add a line to `CHANGELOG.md` under an `## [Unreleased]` heading at the top, and create the heading if it isn't there yet.

### Ask questions

Comment on your pull request or [open an issue](https://github.com/galilai-group/stable-envs/issues/new/choose).

## Pull request rules

- Don't relax a parity threshold or edit a frozen fixture to make a test pass.
- Don't release an EGL context or close a `mujoco.Renderer`, because that corrupts rendering in other live contexts. `tests/_runner/resource_guard.py` fails the test that does it.
- Don't create contexts, renderers or environments per state, per style value or per call.
- Read the upstream source, then check your reading against the upstream renderer, as [Matching the upstream library](https://galilai-group.github.io/stable-envs/contributing/upstream/#measure-constants-against-the-upstream-library) describes.
- Don't commit local artifacts like scratch output, screenshots or one-off JSON. Fixtures that a committed script regenerates live under `tests/fixtures/`.
- Run `ruff format .` and `ruff check .`, or `uvx pre-commit run --all-files`. CI runs the pre-commit hooks on the files a pull request changes.

## What a new environment needs to pass CI

CI collects every registered class in its family's profile, so you don't list a new class anywhere. It runs the generic contract in `tests/unit/test_registry_conformance.py` and the ownership, source comparison and variation map checks in `tests/unit/test_registry_coverage.py`. [Registry rules](https://galilai-group.github.io/stable-envs/contributing/testing/#registry-rules) lists each check and the steps to add a gate. A test file must run in a profile that installs what it imports, so assign the file to a family in `ci/families.toml` or mark a single test with `pytest.mark.family("<name>")`.

## Maintainers

<details>
<summary>CI internals, repository variables and exemptions. Most contributions don't need these.</summary>

### Continuous integration

`.github/workflows/ci.yml` runs on ready (non-draft) pull requests into `main`, on pushes to `main`, in the merge queue and weekly. A manual run from the Actions tab always runs. Pull requests test core on Python 3.12 only, and other runs test every supported Python.

`scripts/ci/plan.py` maps the diff to profiles through `ci/families.toml`:

- A docs-only change runs the docs build.
- A shared path (core modules, conftests, `pyproject.toml`, `uv.lock`, CI files) selects every profile.
- Otherwise CI runs the touched families plus `core`.

The label `ci:full` runs every profile on a pull request. The only required check is `ci-ok`. [Profiles](https://galilai-group.github.io/stable-envs/contributing/testing/#profiles) covers nightly tests and shards. Refresh `ci/durations.json` from the `durations-*.json` files a CI run uploads when the shards become unbalanced.

A red run on `main` or the weekly schedule opens or updates the issue "ci: main is red". Revert the change that broke it unless a fix is merged within about two hours.

`.github/workflows/regression.yml` is manual only. Its GPU target needs a self-hosted runner with the `stable-envs-gpu` label, an NVIDIA GPU, EGL, a CUDA build of torch and nvdiffrast.

### Pull requests from forks

The jobs in `ci.yml`, the `docs.yml` build and `extras-smoke.yml` run on every pull request that is not a draft, also from forks. Fork workflows get a read-only token and no secrets. A maintainer approves the first run for a first-time contributor (Settings, Actions, General). `ci-ok` and the docs `build` are the required checks on `main`.

### Repository variables

Set it under Settings, Secrets and variables, Actions, Variables.

| Variable | When it's `true` |
|---|---|
| `DEPLOY_DOCS` | a push to `main` deploys the built site to GitHub Pages. Enable Pages with GitHub Actions as the source first. |

### Exemptions

If a check can't apply (for example, a renderer-only class has no `step`), waive it in `tests/exemptions.toml`, as [Registry rules](https://galilai-group.github.io/stable-envs/contributing/testing/#registry-rules) describes:

```toml
[[exemption]]
env = "Fetch*Dense"          # class name or glob
family = "fetch"
checks = ["original", "swm"]
reason = "why the check cannot apply"
```

</details>
