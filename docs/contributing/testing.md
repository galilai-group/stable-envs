# Tests

The stable-envs test suite checks simulation, task and rendering separately: matching pixels don't prove that [state](../terms.md#state) is complete, and a state check doesn't prove the pixels are right. To test a change, run the [profile](#profiles) of each family it touches, starting with core:

```bash
uvx nox -s "test(core)"
```

CI turns a skipped required check into a failure, since a skip isn't a pass. The source contracts against stable-worldmodel are on [Matching the upstream library](upstream.md#source-contracts).

## Layout

| Directory | Content | Marker |
|---|---|---|
| `tests/unit/` | fast CPU tests: contracts, registry rules, renderer checks, repository and docs checks | `unit` |
| `tests/integration/` | several components together, including CPU comparisons with the upstream libraries | `integration` |
| `tests/regression/` | frozen corpora, digests, declarations and provenance | `regression` |
| `tests/contracts/` | shared source contracts, run per profile with `--contract-profile` | |
| `tests/gpu/` | conformance against the upstream renderers (needs CUDA) | `gpu` |
| `tests/packaging/` | package metadata, extras, docs config and workflow checks | |
| `tests/helpers/` | shared helpers, including `reference/` bridges to the upstream libraries and gate tables | |
| `tests/fixtures/` | frozen data used by the tests | |
| `tests/_runner/` | supervisor, resource guard, sharding, pytest plugins | |

Directory markers apply automatically. The other markers are:

| Marker | The test |
|---|---|
| `reference` | needs the upstream library's environment installed |
| `slow` | is a long throughput benchmark |
| `nightly` | is a heavy comparison with the upstream library that ordinary pull requests skip |
| `smoke` | checks one environment per family |
| `selftest` | crashes the interpreter on purpose, and runs only with `SE_SELFTEST_CRASH=1` |
| `gl`, `egl` | needs dm_control's OpenGL renderer or an EGL library |
| `swm` | imports stable-worldmodel without `importorskip` |
| `family("<name>")` | needs another family's environments |

## Profiles

Each family has a test profile in `ci/families.toml` that names its extras, its `pin-<profile>` dependency group if it has one, the source paths and test files it owns, and what it requires. The profiles are `core`, `classic`, `navigation`, `pusht`, `dmc`, `fetch`, `ogbench`, `mshuman`, `rocket`, `roadster`, `craftax` and `miniworld`.

```bash
uvx nox -s "test(dmc)" -- -k reacher    # one family, with pytest arguments
uvx nox -s affected                     # profiles selected by the diff against origin/main
SE_PYTHON=3.10 uvx nox -s "test(core)"  # another interpreter
SE_NIGHTLY=1 uvx nox -s "test(dmc)"     # add nightly tests
SE_SHARD=2/4 uvx nox -s "test(dmc)"     # run shard 2 of 4
pytest -m "regression and not gpu"      # without nox
```

`test(<profile>)` syncs `.venvs/<profile>-py3.12` from `uv.lock` and runs `tests/unit`, `tests/integration` and `tests/regression` with `-m "family_<profile> and not gpu and not slow"`. Profiles with the `swm` capability also install stable-worldmodel at the commit in `[swm]` of `ci/families.toml`. The suite skips a family that isn't installed and reports the cause. In CI, `SE_REQUIRE_FAMILIES` and `SE_REQUIRE_CAPABILITIES` turn such a skip into a failure.

Pull requests run the profiles the diff selects (`scripts/ci/plan.py`), without `nightly` tests. Pushes to `main`, the weekly schedule, manual runs and the `ci:full` label run every profile with the `nightly` tests. Each profile splits into `shards` jobs, and `tests/_runner/shard.py` balances them with the per-test seconds in `ci/durations.json`, keeping the tests of one environment in one shard so its per-process caches are built once. `--record-durations PATH` merges a run's per-test times into PATH; use it, or the `durations-*.json` files a CI run uploads, to refresh `ci/durations.json`. GPU parity runs outside CI: run `.github/workflows/regression.yml` manually, or the GPU suite on a CUDA machine.

## Registry rules

The family's profile checks every registered class, with no list to add a new class to. `tests/unit/test_registry_conformance.py` checks each [batch row](../terms.md#batch-row) and the batch as a whole:

| Check | Rules out |
|---|---|
| reset, step and render match `state_spec`, `action_spec`, `frame_spec` | specs that describe something other than the outputs |
| the same seed gives the same rollout | hidden nondeterminism |
| a batch row never sees its neighbors' actions | leaks between batch rows in batched stepping |
| `get_state`/`set_state` restores bit-exactly, and the snapshot owns its memory | incomplete state, or a snapshot that changes later |
| a [variation](../terms.md#variation) installed in one environment round-trips through another | a variation that doesn't survive storage |
| a batch row renders the same alone as in a batch | batch-dependent [frames](../terms.md#frame) |

`tests/unit/test_registry_coverage.py` checks:

| Check | Requires |
|---|---|
| owner | the class's module is under exactly one family's `paths` |
| original | an `ORIGINAL_GATES` entry in `tests/helpers/reference/cases.py` |
| `swm` | an `SWM_GATES` entry, where stable-worldmodel registers the environment |
| variation map | a `VariationMap` in `tests/helpers/reference/variation_maps/`, where the stable-worldmodel environment has a variation space |
| imports | every test file runs in a profile that installs what it imports |

`ORIGINAL_GATES` name a CPU test that compares the environment with its [upstream library](../terms.md#upstream-library) (dm_control, gymnasium-robotics, OGBench and so on) or with a corpus recorded from it. These tests run with the stable-worldmodel import blocked and aren't `nightly`. `SWM_GATES` name a test marked `swm` that compares the environment with live stable-worldmodel.

A `VariationMap` says how each variation leaf of a stable-worldmodel environment relates to the stable-envs class. Matching by name would get cases like PushT's absolute scales against stable-worldmodel's multipliers wrong. Each stable-worldmodel leaf lands in exactly one group: mapped (identical, renamed, scaled, affine, color or enum), unsupported, dead upstream, not applicable, or stable-worldmodel only. The maps run as tests and stay out of the package's API.

`tests/exemptions.toml` is the only place that can waive a check. Each entry names the environment (a class name or glob), the family, the checks (`specs`, `seeded`, `lanes`, `state`, `variation`, `batch_render`, `owner`, `original`, `swm` and `variation_map`) and the reason. A waived check still runs and must keep failing. Once it passes, the suite fails until you remove the entry, which keeps a waiver from outliving its reason. `.github/CODEOWNERS` assigns the file a code owner.

To add a gate for a new environment:

1. Put the environment module under one family's `paths` in `ci/families.toml`.
2. Write a CPU test that compares it with its upstream library or a corpus recorded from it, give the test file to the same family (`tests` in `ci/families.toml`), and keep stable-worldmodel imports out of it.
3. Add the environment and that test's node id to `ORIGINAL_GATES`.
4. If stable-worldmodel registers the environment, add a test marked `swm` and an `SWM_GATES` entry, and if its environment has a variation space, a `VariationMap`.
5. If a check can't apply, add an exemption with the reason, then run `uvx nox -s "test(<family>)"`.

## GPU conformance

`tests/gpu/test_conformance.py` runs on every registered environment on a CUDA machine and skips without CUDA. The navigation classes and `PyFlytRocketLanding` get their upstream environments from stable-worldmodel. For a family with the `swm` capability, install a CUDA build of torch and stable-worldmodel at the commit in `[swm]` of `ci/families.toml` with the family's `swm_with` distributions. Without it, these checks fail with `NamespaceNotFound: swm`. `test(<profile>)` installs stable-worldmodel with `noxfile._install_swm`, but on Linux its environment has the CPU build of torch.

```bash
python -m tests._runner.supervisor -- tests/gpu -v   # crash-isolated
```

`python scripts/ci/run_dmc_mjwarp.py` runs the CUDA checks of DM Control on MuJoCo Warp: all 17 classes, checkpoint ownership, resets, targets, physics variation, rendering of saved states, stream ordering, the legacy step order and the sensor timing. It fails without CUDA and when any required case skips.

| Test | Rules out |
|---|---|
| `test_source_mse_matches_the_record` | a frame MSE against the upstream renderer above its record, or a record that's out of date |
| `test_cuda_frames_match_the_golden_frames` | any change to the frames of a class on CUDA |
| `test_render_state_roundtrip` | a render row that the round trip clips, normalizes or reorders |
| `test_every_state_dim_reaches_the_render` | a render-row dimension that doesn't affect the frame |
| `test_batch_invariance` | a frame that depends on its batch |
| `test_per_env_variation_is_per_env` | a per-environment variation that collapses to one value |
| `test_style_is_state` | a frame that depends on something not stored with the state |
| `test_default_style_is_the_official_appearance` | a default appearance that differs from the upstream library's |
| `test_color_knobs_apply_the_requested_color` | a color knob wired to the wrong primitive or channel |
| `test_determinism` | dependence on allocation order, cache state or a stale bake |
| `test_backend_agreement` | a fused Triton kernel that differs from the torch reference |
| `test_variations_all_do_something` | a variation leaf that the renderer ignores |
| `test_a_declared_task_object_is_drawn_iff_the_reference_draws_it` | a task object drawn or hidden differently from the upstream renderer |
| `test_gpu_step_tracks_the_reference_simulator` | a batched GPU simulator that drifts from the upstream simulator beyond its bound |
| `test_state_is_sufficient` | a state that doesn't determine the upstream renderer's frame |

### Frame regression checks

Two fixtures record frames, and each check compares a new measurement with its record instead of a fixed limit.

`tests/fixtures/source_mse.json` holds one record per class with an upstream renderer: the per-frame MSE against the upstream library on the 0-255 scale, for 64 render rows from a fixed seed, as `mse_mean`, `mse_p95` and `mse_max`, with `seed`, `n`, `size`, `reference_versions`, `gpu` and `torch` describing the setup. `test_source_mse_matches_the_record` measures the same rows again, and a value fails above 1.10 × record + 0.01 or below 0.5 × record − 0.01. A class that matches exactly records 0 and still passes with an MSE up to 0.01. A value far below its record fails until you record it again. A class that declares `visual_fidelity` must agree with its record within 5% or 0.01, whichever is larger (`tests/regression/test_declarations.py`).

```bash
python scripts/testing/record_source_mse.py                    # new file, on the GPU profile
python scripts/testing/record_source_mse.py --update PushT     # replace one record
```

`tests/fixtures/golden/<device>/<Class>.npz` holds 8 render rows and the golden frames for `cuda` and `cpu`, and `manifest.json` gives a sha256 per file with the package versions, machine and GPU per class. `test_cuda_frames_match_the_golden_frames` and `test_cpu_frames_match_the_golden_frames` require the same frames exactly and skip on a machine or package version that differs from the manifest. The `cpu` golden frames cover classic control and navigation only. The MuJoCo families draw part of the frame with OSMesa on CPU, which makes their CPU frames depend on the OSMesa build. PushT has none, since `sample_render_rows` can't pack its state. To accept an intended render change, capture new frames into a new directory, compare them, and move the directory into place:

```bash
python scripts/testing/capture_golden.py /tmp/golden-cuda --device cuda --source-commit "$(git rev-parse HEAD)"
```

If a new environment can't match the upstream library, look for the wrong constant first: see [Find the upstream renderer](upstream.md#find-the-upstream-renderer) and [measuring constants](upstream.md#measure-constants-against-the-upstream-library).

## Running native renderers safely

A crash in MuJoCo, EGL or the CUDA driver ends the interpreter, and pytest then reports nothing about the test that caused it. Run such suites through the supervisor. CPU-only unit tests can run with plain pytest.

```bash
python -m tests._runner.supervisor -- tests/ -v   # full suite, crash-isolated
python -m tests._runner.supervisor --smoke        # one environment per family
python -m tests._runner.node_health               # host check before measuring
```

`tests._runner.supervisor` runs pytest in restartable subprocesses. It records the running test, restarts after a crash without the tests that already finished or crashed, and reports crashed tests as failures in one combined report. It gives up after a bounded number of restarts. `--stall-timeout` (900 s by default) kills a segment in which no test starts or finishes, and `--json PATH` writes the combined result to a file. `tests/_runner/safety_plugin.py` provides its pytest hooks: resource accounting, progress events, the host check and smoke selection, which picks one representative per family and leaves out checks that start many subprocesses. Set `SE_SMOKE_ENVS=A,B` to pick the environments yourself.

`tests/_runner/resource_guard.py` counts EGL and MuJoCo OpenGL contexts per test, since those contexts live for the whole process. Releasing a context fails as `EGL_CONTEXT_RELEASED`, and creating contexts in proportion to the batch fails as `EGL_CONTEXTS_SCALE_WITH_BATCH`. `SE_MAX_CONTEXTS_PER_TEST` and `SE_MAX_LIVE_CONTEXTS` change the per-test and session limits, and each test has a 900 s timeout when pytest-timeout is installed.

`tests._runner.node_health` asks the cluster scheduler (Slurm's `sinfo`), if there is one, whether this node is drained or down, and renders the same states twice through an upstream renderer. If the renders differ, the session stops with exit code 78: parity numbers from a host that can't repeat its own upstream frames are unreliable. The result can be `ok` while a check was unavailable or skipped. Read the details. `SE_SKIP_NODE_GATE=1` skips the check, and nox and CI set it.

`tests._runner.render_worker` renders seeded inputs for every registered environment in a fresh interpreter and writes frames and digests to `--out-npz` and `--out-json`. It records per-environment errors and keeps going. Inspect the report, not the exit status. `tests/integration/test_safety_net.py` tests these mechanisms, and `tests/gpu/test_crash_isolation.py` tests them on a GPU with `SE_SELFTEST_CRASH=1`.

## Testing helpers

`stable_envs.testing.conformance` holds the checks the test suite runs against every environment, as plain functions that measure render agreement, determinism and state round trips and return dicts of numbers. You decide what passes: call them from pytest, a report or a benchmark and apply your own thresholds. The [GPU conformance](#gpu-conformance) tests use them. `stable_envs.testing` is a public measurement library outside the `snv.*` namespace; import it by its full name.

```python
import stable_envs as snv
from stable_envs.testing import conformance

with snv.TwoRoom(num_envs=4, device="cpu") as env:
    print(conformance.check_batch_invariance(env))   # {'max_abs_diff': 0}
    print(conformance.check_determinism(env))        # {'identical': True}
    print(conformance.check_state_dims_matter(env))  # {'dead_dims': []}

    state = env.reset(seed=0)
    a = env.render(state).numpy()
    print(conformance.pixel_agreement(a, a)["exact_frac"])  # 1.0
```

Most checks work on the private renderer projection of state, the compact rows that `env.sample_states` returns, rather than on the full state. For the public `get_state` and `set_state` contract, use the trajectory checks the test suite runs. The reference environments that `reference_env` hands out live for the whole process and are shared. Leave them open.

### Frames and metrics

Reference environments, render rows, frame digests and the metrics that compare two batches of frames.

::: stable_envs.testing.conformance
    options:
      show_root_heading: false
      heading_level: 4
      members: [reference_env, sample_render_rows, frames_digest, check_reference_renderer_is_stable, pixel_agreement, ssim, frame_mse, mse_summary, check_source_mse]

### State and determinism

Restores of state and of the private render rows, state dimensions that never reach the frame, determinism, and stepping against the upstream simulator.

::: stable_envs.testing.conformance
    options:
      show_root_heading: false
      heading_level: 4
      members: [check_render_state_roundtrip, check_state_roundtrip, check_state_dims_matter, check_set_state_copies, check_determinism, check_step_matches_reference]

### Batches and backends

Frames that must not depend on the batch, its order or size, the chunking, the input form or the backend, and rendering that leaves the stored render rows and style unchanged.

::: stable_envs.testing.conformance
    options:
      show_root_heading: false
      heading_level: 4
      members: [check_batch_invariance, check_batch_size_invariance, check_chunked_render_matches, check_order_independence, check_input_form_invariance, check_render_is_pure, check_backend_agreement]

### Variation and appearance

Variation leaves that change pixels, per-environment variation, the default appearance, exact colors and drawn task objects.

::: stable_envs.testing.conformance
    options:
      show_root_heading: false
      heading_level: 4
      members: [check_variations_do_something, check_per_env_variation, check_style_is_state, check_default_style_is_official, color_exactness_applicability, partition_colour_exactness, check_colors_apply_exactly, check_task_object_visibility]

### Resolution

Native-size frames compared with box-downsampled larger renders, and the fixed corpus and state bands that comparison uses.

::: stable_envs.testing.conformance
    options:
      show_root_heading: false
      heading_level: 4
      members: [downsample_protocol, resolution_consistency_applicability, check_resolution_consistency]
