# Testing

Simulation, task and rendering claims are tested separately. A rendering check does not prove that
State is complete, and a State check does not prove that pixels are right. A skipped required check is
not a pass.

## Layout

| directory | content | marker |
|---|---|---|
| `tests/unit/` | fast CPU tests: contracts, registry rules, renderer checks, repository and docs checks | `unit` |
| `tests/integration/` | several components together, including CPU source comparisons | `integration` |
| `tests/regression/` | frozen corpora, digests, declarations and provenance | `regression` |
| `tests/contracts/` | shared source contracts, run per profile with `--contract-profile` | |
| `tests/gpu/` | conformance against the official renderers; needs CUDA | `gpu` |
| `tests/packaging/` | package metadata, extras, docs config and workflow checks | |
| `tests/helpers/` | shared helpers, including `reference/` source bridges and gate tables | |
| `tests/fixtures/` | frozen data used by the tests | |
| `tests/_runner/` | supervisor, resource guard, sharding, pytest plugins | |

Directory markers are applied automatically, so a file without `pytestmark` is still selected by its
directory. Other markers:

| marker | meaning |
|---|---|
| `reference` | needs the official reference env |
| `slow` | long-running throughput benchmark |
| `nightly` | heavy source comparison; not run on ordinary pull requests |
| `smoke` | one env per family |
| `selftest` | crashes the interpreter on purpose; runs only with `SE_SELFTEST_CRASH=1` |
| `gl`, `egl` | needs dm_control's OpenGL renderer, or an EGL library |
| `swm` | imports stable-worldmodel without `importorskip` |
| `family("<name>")` | needs another family's environments |

## Profiles

Each family has a test profile in `ci/families.toml`. A profile names its extras, its `pin-<profile>`
dependency group, the source paths and test files it owns, and what it requires.

```bash
uvx nox -s "test(core)"                 # core profile
uvx nox -s "test(dmc)" -- -k reacher    # one family, with pytest arguments
uvx nox -s affected                     # profiles selected by the diff against origin/main
SE_PYTHON=3.10 uvx nox -s "test(core)"  # another interpreter
```

`test(<profile>)` syncs `.venvs/<profile>-py3.12` from `uv.lock` and runs `tests/unit`,
`tests/integration` and `tests/regression` with `-m "family_<profile> and not gpu and not slow"`.
Profiles with the `swm` capability also install stable-worldmodel at the commit in `[swm]` of
`ci/families.toml`.

A family that is not installed is skipped, with the cause. In CI, `SE_REQUIRE_FAMILIES` and
`SE_REQUIRE_CAPABILITIES` turn such a skip into a failure.

Without nox:

```bash
pytest -m unit                          # CPU-only unit tests
pytest -m "regression and not gpu"      # frozen corpora and digests
pytest -m "regression and gpu"          # GPU regression, needs CUDA
```

## Pull requests, nightly and shards

- Pull requests run the profiles the diff selects (`scripts/ci/plan.py`), without `nightly` tests.
- Pushes to `main`, the weekly schedule, manual runs and the `ci:full` label run every profile and
  include `nightly` tests.
- Each profile is split into `shards` jobs (`ci/families.toml`, per tier). `tests/_runner/shard.py`
  balances them with the per-test seconds in `ci/durations.json`. Tests of one env stay in one shard.

```bash
SE_NIGHTLY=1 uvx nox -s "test(dmc)"     # add nightly tests
SE_SHARD=2/4 uvx nox -s "test(dmc)"     # run shard 2 of 4
```

`--record-durations PATH` merges a run's per-test times into PATH. Use it, or the `durations-*.json`
files a CI run uploads, to refresh `ci/durations.json`.

GPU parity is not part of CI. Run `.github/workflows/regression.yml` manually, or run the GPU suite on
a CUDA machine.

## Registry rules

Every registered class is checked in its family's profile without being listed anywhere.

`tests/unit/test_registry_conformance.py`:

| check | rules out |
|---|---|
| reset, step and render match `state_spec`, `action_spec`, `frame_spec` | specs that describe something other than the outputs |
| the same seed gives the same rollout | hidden nondeterminism |
| a lane does not see its neighbours' actions | cross-lane leaks in batched stepping |
| `get_state`/`set_state` restores bit-exactly; the snapshot owns its memory | incomplete State, or a snapshot that changes later |
| a variation installed in one env round-trips through another | Variation that does not survive storage |
| a lane renders the same alone as in a batch | batch-dependent frames |

`tests/unit/test_registry_coverage.py`:

| check | requires |
|---|---|
| owner | the class's module is under exactly one family's `paths` |
| original | an `ORIGINAL_GATES` entry in `tests/helpers/reference/cases.py` |
| swm | an `SWM_GATES` entry, where stable-worldmodel registers the env |
| variation map | a `VariationMap` in `tests/helpers/reference/variation_maps/`, where the stable-worldmodel source has a variation space |
| imports | every test file runs in a profile that installs what it imports |

Source comparisons come in two layers:

- `ORIGINAL_GATES` name a CPU test that compares the env with the library it ports (dm_control,
  gymnasium-robotics, OGBench, …) or with a corpus recorded from it. These tests do not import
  stable-worldmodel, run with that import blocked, and are not `nightly`.
- `SWM_GATES` name a test marked `swm` that compares the env with live stable-worldmodel.

## Exemptions

`tests/exemptions.toml` is the only place a check is waived. Each entry names the env (a class name or
glob), the family, the checks and the reason. The check names are `specs`, `seeded`, `lanes`, `state`,
`variation`, `batch_render`, `owner`, `original`, `swm` and `variation_map`.

A waived check still runs and must keep failing. When it passes, the suite fails until the entry is
removed. `.github/CODEOWNERS` assigns the file a code owner.

## Adding a gate for a new environment

1. Put the env module under one family's `paths` in `ci/families.toml`.
2. Write a CPU test that compares the env with the upstream library, or with a corpus recorded from
   it. Give the test file to the same family (`tests` in `ci/families.toml`). Do not import
   stable-worldmodel in it.
3. Add the env and that test's node id to `ORIGINAL_GATES` in `tests/helpers/reference/cases.py`.
4. If stable-worldmodel registers the env, add a test marked `swm` and an `SWM_GATES` entry.
5. If the stable-worldmodel source has a variation space, add a `VariationMap`.
6. If a check cannot apply, add an exemption with the reason.
7. Run `uvx nox -s "test(<family>)"`.

## Source contracts

`tests/contracts/` holds shared source-contract tests for representative registrations. Each profile
compares State, simulation, task and source pixels against a pinned stable-worldmodel checkout. Pins
are in `tests/helpers/reference/profiles.py`.

```bash
python -m pytest tests/contracts --contract-profile native \
  --contract-source /path/to/pinned/stable-worldmodel \
  --contract-output /path/to/new-run
```

- `--contract-device cuda` runs the CUDA output path.
- `--contract-output` must be a new directory. It receives the plan, hashes, per-case progress and a
  `SHA256SUMS` seal.
- The required case set is computed before running. Missing, skipped, duplicated or deselected cases
  fail the profile. `-k` is for development only.
- Run one profile per interpreter, without xdist. After a crash, rerun the whole profile into a new
  directory.

The generators for frozen corpora are in `scripts/testing/`; see its README.

## GPU conformance

`tests/gpu/test_conformance.py` runs on every registered env on a CUDA machine. It skips without
CUDA.

```bash
pytest tests/gpu -v
python -m tests._runner.supervisor -- tests/gpu -v   # crash-isolated
```

| test | rules out |
|---|---|
| `test_render_parity_with_official_renderer` | wrong colour, shape or position against the official renderer (parity tier) |
| `test_visual_tier_matches_its_declared_fidelity` | a visual-tier class that no longer meets its declared thresholds |
| `test_fidelity_tier_is_fully_declared` | a class that neither parity nor visual checks apply to |
| `test_relaxed_parity_floors_are_all_explained` | a lowered threshold without a `parity_note` |
| `test_render_state_roundtrip` | a render state that is clipped, normalized or reordered on the way back |
| `test_every_state_dim_reaches_the_render` | a render-state dimension that does not affect the frame |
| `test_batch_invariance` | a frame that depends on its batch |
| `test_per_env_variation_is_per_env` | a per-lane Variation that collapses to one value |
| `test_style_is_state` | a frame that depends on something not stored with the State |
| `test_default_style_is_the_official_appearance` | a default appearance that differs from the official one |
| `test_color_knobs_apply_the_requested_color` | a colour knob wired to the wrong primitive or channel |
| `test_determinism` | dependence on allocation order, cache state or a stale bake |
| `test_backend_agreement` | a fused Triton kernel that differs from the torch reference |
| `test_variations_all_do_something` | a Variation leaf that the renderer ignores |
| `test_a_declared_task_object_is_drawn_iff_the_reference_draws_it` | a task object drawn or hidden differently from the reference |
| `test_gpu_step_tracks_the_reference_simulator` | a batched GPU simulator that drifts from the reference beyond its bound |
| `test_state_is_sufficient` | a state that does not determine the official frame |

Parity thresholds, defined at the top of the file:

| constant | value | meaning |
|---|---|---|
| `INTERIOR_EXACT_MIN` | 1.0 | interior pixels (flat 3×3 neighbourhood in the reference) must match exactly |
| `ALL_EXACT_MIN` | 0.90 | fraction of all pixels that must match; edges may differ |
| `MEAN_ABS_MAX` | 4.0 / 255 | mean absolute difference |

A class can lower `parity_interior_min` or `parity_all_exact_min` only with a `parity_note` that names
the cause. Visual-tier classes are checked against their own `visual_fidelity`. See
[Fidelity tiers](concepts/fidelity-tiers.md). If a new env cannot reach the interior floor, find the
wrong constant; see [Identify authority](adding/renderer.md#identify-authority).

## Running native renderers safely

A crash in MuJoCo, EGL or the CUDA driver ends the interpreter, and pytest then reports nothing about
the test that caused it. Run such suites through the supervisor:

```bash
python -m tests._runner.supervisor -- tests/ -v   # full suite, crash-isolated
python -m tests._runner.supervisor --smoke        # one env per family
```

- The supervisor records the running test, restarts after a crash, and reports crashed tests as
  failures.
- `tests/_runner/resource_guard.py` counts EGL contexts per test. Releasing a context fails as
  `EGL_CONTEXT_RELEASED`. Creating contexts in proportion to the batch fails as
  `EGL_CONTEXTS_SCALE_WITH_BATCH`.
- Each test has a 900 s timeout when pytest-timeout is installed.

`tests/integration/test_safety_net.py` tests these mechanisms. `tests/gpu/test_crash_isolation.py`
tests them on a GPU and runs only with `SE_SELFTEST_CRASH=1`.

## Using the checks outside pytest

The checks are plain functions that return dicts:

```python
from stable_envs.testing import conformance as C

C.check_render_state_roundtrip(env)  # {'max_abs_err': 0.0}
C.check_state_dims_matter(env)       # {'dead_dims': []}
C.check_backend_agreement(env)       # {'exact_frac': 1.0, 'max_abs': 0, ...}
C.pixel_agreement(ours, official)    # all-pixel and interior-pixel agreement
```

API details: [Testing helpers](api/testing.md).
