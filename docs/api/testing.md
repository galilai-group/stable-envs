# `stable_envs.testing`

Conformance checks as plain functions returning numbers, so they can be called from a report or a
benchmark and not only from pytest.

::: stable_envs.testing.conformance

## Running the suite safely

The suite is a GPU program, and GPU programs die in ways Python cannot catch: a segfault or a
`SIGABRT` out of MuJoCo, EGL or the CUDA driver ends the interpreter, reports nothing about the test
that caused it, and silently costs every test that had not run yet. The four modules below exist so
that such a failure costs **one test**, is **named**, and is **attributed** — and so that a broken
machine is diagnosed as a broken machine rather than as a renderer bug.

```
python -m stable_envs.testing.supervisor -- tests/ -v      # full suite, crash-isolated
python -m stable_envs.testing.supervisor --smoke           # one env per family
python -m stable_envs.testing.node_health                  # is this host fit to measure on?
sbatch scripts/run_gpu_tests.sbatch                        # the same, on a GPU node
sbatch scripts/run_smoke_tests.sbatch
```

::: stable_envs.testing.supervisor

::: stable_envs.testing.resource_guard

::: stable_envs.testing.node_health

::: stable_envs.testing.render_worker

The pytest side of this — the per-test resource accounting, the crash breadcrumbs, the node gate and
the smoke selection — lives in `stable_envs/testing/safety_plugin.py`, loaded from the root
`conftest.py`. It is not rendered here because it imports `pytest` at module scope and the docs build
should not need the test dependencies to describe the library; read the module docstring, which states
which of the four incidents each mechanism answers.
