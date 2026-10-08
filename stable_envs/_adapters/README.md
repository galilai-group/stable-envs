# Family adapters

Family simulator adapters and worker-process code. Shared transport (the
`SimulatorAdapter` protocol, process-selection policy, request lifecycle and
shared memory) lives in `stable_envs/_execution/`.

| Location | Ownership |
|---|---|
| `classic/` | Gymnasium classic control and the Acrobot process route |
| `fetch/` | Fetch simulation, observations and execution selection |
| `dmc/` | DMC domains; `factory.py` selects the same implementation locally and in workers |
| `ogbench/` | Maze and manipulation, including staged world restoration |
| `pusht/`, `rocket/` | Source task behavior, complete replay and family-specific spaces |
| `craftax.py` | Upstream JAX simulation; rendering belongs to `envs/craftax/` |

For a concrete path, read `fetch/execution.py`, `fetch/adapter.py`, then
`fetch/process.py`. Family code owns State validation, RNG, reset, task
behavior and inactive lanes. OGBench prepares replacement worlds before
committing restoration; do not replace that protocol with a generic `set_state`
broadcast. Keep source space types local: their precision, sampling order and
default/value semantics differ from the public spaces.

`tests/unit/test_fetch_process_adapter.py` illustrates snapshot ownership,
continuation, inactive lanes and worker failure checks. Each process route also
needs its family/source and device gates; serial agreement alone is insufficient.
