# `stable_envs.testing`

Conformance checks as plain functions returning numbers, so they can be called from a report or a
benchmark and not only from pytest.

::: stable_envs.testing.conformance

## Running the suite safely

Use the supervisor for suites that initialize native renderers. It records the active test,
restarts after a crash, and includes crashed tests in the combined failure report. CPU-only unit
tests can run directly through pytest. See [the testing guide](../testing.md) for source profiles
and acceptance requirements.

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

`stable_envs/testing/safety_plugin.py` supplies the pytest hooks for resource accounting, flushed
progress events, node checks, and smoke selection. The supervisor consumes those events; node checks
can be unavailable or skipped, so inspect their details rather than treating `ok` alone as proof
that the host is healthy.
