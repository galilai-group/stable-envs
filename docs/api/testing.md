# `stable_envs.testing`

Conformance checks as plain functions that return numbers. Call them from pytest, a report or a
benchmark.

::: stable_envs.testing.conformance

## Test runner

Use the supervisor for suites that initialize native renderers. It records the running test,
restarts after a crash, and reports crashed tests as failures. CPU-only unit tests can run with plain
pytest. See [Testing](../testing.md).

```bash
python -m tests._runner.supervisor -- tests/ -v      # full suite, crash-isolated
python -m tests._runner.supervisor --smoke           # one env per family
python -m tests._runner.node_health                  # host check before measuring
uvx nox -s "test(<profile>)"                         # one CI profile
```

::: tests._runner.supervisor

::: tests._runner.resource_guard

::: tests._runner.node_health

::: tests._runner.render_worker

`tests/_runner/safety_plugin.py` provides the pytest hooks for resource accounting, progress events,
host checks and smoke selection. Host checks can be skipped. Read their details; `ok` alone does not
prove the host is healthy.
