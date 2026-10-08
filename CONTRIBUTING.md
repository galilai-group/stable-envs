# Contributing

## Install

```bash
pip install -e ".[dev]"
```

That is numpy, torch, gymnasium, tensordict, loguru, plus pytest and ruff. Add family extras as
needed (`"[dev,pusht]"`, `"[dev,dmcontrol]"`). Extra incompatibilities, the Fetch MuJoCo pin,
Overcooked's private interpreter, and the nvdiffrast source build live on the
[installation page](docs/install.md).

## Format

```bash
ruff format .
ruff check .
```

## Tests

Nothing in this repo is measured on a login node, and nothing that measures is measured on a busy one.

```bash
pytest -m unit                                 # CPU-only family/unit contracts
sbatch scripts/run_smoke_tests.sbatch          # one env per family, ~3.4 min
sbatch scripts/run_gpu_tests.sbatch            # full conformance suite
pytest tests/ -v -rs                           # the same, if you already hold a GPU
```

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
