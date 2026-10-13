# Contributing

## Install

Test environments come from the one `uv.lock`, one per profile, through [nox](https://nox.thea.codes)
and [uv](https://docs.astral.sh/uv/):

```bash
uvx nox -l                              # every session
uvx nox -s "test(core)"                 # core tests, floating dependencies, no family extras
uvx nox -s "test(dmc)" -- -k reacher    # one family under its exact reference pins
uvx nox -s affected                     # the profiles CI selects for this branch
SE_PYTHON=3.10 uvx nox -s "test(core)"  # the core profile on another interpreter
```

`test(<profile>)` runs `uv sync --frozen --extra <family extras> --group pin-<profile> --group test`
into `.venvs/<profile>-py3.12` and then the family's unit, integration and CPU regression tests.
Profiles are `core`, `classic`, `navigation`, `pusht`, `dmc`, `fetch`, `ogbench`, `rocket` and
`craftax`; [`ci/families.toml`](ci/families.toml) says which paths and tests each owns and what it
needs installed. To work in one directly, `uv sync --python 3.12 --extra dmcontrol --group pin-dmc
--group test` and `uv run pytest ...`.

A plain `pip install -e ".[dev]"` also works. Extra incompatibilities, the Fetch MuJoCo pin,
and the nvdiffrast source build live on the [installation page](docs/install.md). The unreleased
renderer-only ports in `contrib/` have their own requirements and tests; see
[contrib/README.md](contrib/README.md).

## Format

```bash
ruff format .
ruff check .
```

## Tests

Nothing in this repo is measured on a login node, and nothing that measures is measured on a busy one.

```bash
uvx nox -s "test(<profile>)"                   # what CI runs for one profile
pytest -m unit                                 # CPU-only family/unit contracts, any interpreter
sbatch scripts/run_smoke_tests.sbatch          # one env per family, ~3.4 min
sbatch scripts/run_gpu_tests.sbatch            # full conformance suite
pytest tests/ -v -rs                           # the same, if you already hold a GPU
```

A family this interpreter lacks (or holds at a version its code refuses) does not fail its tests:
they are reported as skipped with the cause, and the run ends with a count of skips by cause. CI
profile jobs set `SE_REQUIRE_FAMILIES` / `SE_REQUIRE_CAPABILITIES`, which turn that absence into an
error. Mark a test that needs dm_control's OpenGL renderer `@pytest.mark.gl`, one that needs an EGL
library `@pytest.mark.egl`, one that imports stable-worldmodel without `importorskip`
`@pytest.mark.swm`, and a test that uses another family's environments
`@pytest.mark.family("<name>")`.

The full suite requires CUDA by design and skips loudly rather than going green without it. Run it
through `tests._runner.supervisor` (which is what the sbatch scripts do): a segfault or
`SIGABRT` out of MuJoCo, EGL or the driver ends the interpreter, and a bare pytest then reports
nothing at all about which test caused it.

**CPU baselines are not reproducible under contention** and GPU numbers are. The same
official-renderer measurement gave 422 frames/s on an idle node and 3 frames/s with three other
jobs resident, while the GPU column reproduces across nodes to 0.4%. Take a CPU baseline on a
node you have to yourself, quote the ratio as an order of magnitude rather than to three digits,
and say which it was.

What each check rules out: [Testing](docs/testing.md).

## Adding a component

Choose the [environment, task, or renderer guide](docs/adding/index.md).
Simulation calls the authoritative upstream backend; full State is a replay checkpoint.
Compact RenderState and learner observation packaging have different responsibilities.
Each guide separates local development tests from final promotion evidence.

Construct with `num_envs=`. Concrete classes register by unique class name.
Do not promote a registration from renderer parity or a sibling's passing task alone.

### What a new environment needs to pass CI

Every registered class is picked up automatically, in its family's CI profile:

- **Generic contract** (`tests/unit/test_registry_conformance.py`): reset, step and render match
  `state_spec`, `action_spec` and `frame_spec`; the same seed gives the same rollout; a lane does not
  see its neighbours' actions; `get_state`/`set_state` restores bit-exactly and the snapshot owns its
  memory; a variation installed in one env round-trips through another; a lane renders the same alone
  as in a batch.
- **Ownership** (`tests/unit/test_registry_coverage.py`): its module lies under exactly one family's
  `paths` in `ci/families.toml`, and that family's env modules import wherever its `requires` are met.
- **Source comparisons**, in two layers (`tests/helpers/reference/cases.py`). An `ORIGINAL_GATES` entry
  names a CPU test, owned by the same family, comparing it with the library it ports (dm_control,
  gymnasium-robotics, ogbench, ...) or a corpus recorded from it; that test imports no
  stable-worldmodel (it runs with the import blocked). Where stable-worldmodel
  registers the env, a `SWM_GATES` entry names a test marked `swm` comparing it with live
  stable-worldmodel.
- **Variation map**: when its stable-worldmodel source has a variation space, a `VariationMap` in
  `tests/helpers/reference/variation_maps/`.

Every test that needs stable-worldmodel carries `pytest.mark.swm`, and families with such tests
declare the `swm` capability; their profile then installs it at `[swm].commit` and, in CI, a test that
skips for want of it fails. A test file must run in a profile that installs what it imports: give the
file to a family in `ci/families.toml`, or a single test to one with `pytest.mark.family("<name>")`.
When stable-worldmodel drops its envs, the SWM layer goes in one change: delete `SWM_GATES`, the `swm`
tests and their gate helpers, the `swm` capability and `[swm]` block in `ci/families.toml`, and
`_install_swm` in `noxfile.py`; `ORIGINAL_GATES` then stands alone. A check that
truly cannot apply (a renderer-only cell has no `step`) is waived only in `tests/exemptions.toml`, with
the env, its family, the check and the reason; that file has a code owner. A waived check keeps
running, and the suite fails once it passes, so the entry is removed in the change that fixes it.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request, every push to `main` and nightly.
`scripts/ci/plan.py` maps the diff to profiles through `ci/families.toml`: documentation alone
runs the docs build; a shared path (core modules, conftests, `pyproject.toml`, `uv.lock`, CI
files) selects every profile; otherwise the touched families plus `core`. The label `ci:full`
forces every profile on a pull request. Pushes to `main`, the nightly run and manual runs test
every profile. The only required check is `ci-ok`.

A red run on `main` or the nightly opens or updates the issue "ci: main is red". Revert the
offending change first unless a fix is merged within about two hours; the issue names the run.

## Pull requests

- Do not relax a parity criterion or modify a frozen fixture to make a test pass. A proposed
  change to a fidelity claim requires independent evidence and review.
- Do not release an EGL context or close a `mujoco.Renderer`. Releasing one corrupts rendering in
  every other live context in the process, silently, and the damage surfaces later as pixel
  differences in an unrelated env. `tests/_runner/resource_guard.py` fails the test that does it.
- Do not create contexts, renderers or envs per state, per style value or per call. Context
  creation that scales with the batch means a per-environment value is being turned into a
  per-environment *scene*.
- Read the authoritative source, then validate its interpretation against the official renderer. The method,
  and the defects that produced it, are in
  [Fidelity methodology](docs/concepts/fidelity-methodology.md).

## Artifacts

Do **not** commit ad-hoc local artifacts (scratch dumps, screenshots, one-off JSON from a laptop
run). Canonical JSON fixtures that a committed script regenerates live under
`tests/fixtures/`. Large binary corpora stay out of `main`.
