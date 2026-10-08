# Simulator adapters

`SimulatorAdapter` in `__init__.py` describes reset, one transition per lane,
get/set State and close. Concrete adapters inherit this protocol; its abstract
methods require implementations. Process adapters can inherit implementations
from their local adapter or transport mixin. `Env` owns action sequences,
capture slices and rendering.

| Location | Ownership |
|---|---|
| `classic/` | Gymnasium classic control and the Acrobot process route |
| `fetch/` | Fetch simulation, observations and execution selection |
| `dmc/` | DMC domains; `factory.py` selects the same implementation locally and in workers |
| `ogbench/` | Maze and manipulation, including staged world restoration |
| `pusht/`, `rocket/` | Source task behavior, complete replay and family-specific spaces |
| `craftax.py` | Upstream JAX simulation; rendering belongs to `envs/craftax/` |
| `execution.py` | Shared process-selection policy |
| `process_requests.py` | Request lifecycle and worker-failure cleanup |
| `shared_memory.py`, `state_columns.py` | Typed shared outputs and owned State assembly |

For a concrete path, read `fetch/execution.py`, `fetch/adapter.py`, then
`fetch/process.py`. Workers call the local simulator implementation; they do
not define another physics model. Selecting CUDA output does not move CPU
physics to the GPU.

## Interfaces and ownership

Use `SimulatorAdapter` when only lifecycle operations are needed. Keep family
configuration explicitly typed: for example, Fetch's `object_mass` returns an
owned copy. `StepPipeline` describes optional submit/receive/abort operations.
The pipeline receives independent outputs before workers reuse shared storage.

Family code owns State validation, RNG, reset, task behavior and inactive lanes.
Shared transport owns request delivery and failure cleanup. OGBench prepares
replacement worlds before committing restoration; do not replace that protocol
with a generic `set_state` broadcast.

Keep source space types local: their precision, sampling order and default/value
semantics differ from the public spaces. Drawing helpers live with their owning
environments, not with process transport.

## Checks

`tests/unit/test_fetch_process_adapter.py` illustrates snapshot ownership,
continuation, inactive lanes and worker failure checks. Each process route also
needs its family/source and device gates; serial agreement alone is insufficient.

Run the focused static checks from the repository root with an interpreter that
has Torch and TensorDict installed:

```sh
mypy --config-file tests/typing/mypy.ini --python-executable /path/to/venv/bin/python
```

These checks cover the shared interfaces and family factory boundaries, not all
simulation internals. Keep the public action and capture APIs unchanged and
compare horizon-1 and horizon-8 throughput when execution mechanics change.
