# Shared process execution

`protocol.py` defines `SimulatorAdapter`: reset, one transition per lane,
get/set State and close. Concrete adapters inherit this protocol; its abstract
methods require implementations. Process adapters can inherit implementations
from their local adapter or transport mixin. `Env` owns action sequences,
capture slices and rendering.

| Module | Ownership |
|---|---|
| `protocol.py` | `SimulatorAdapter` and `StepPipeline` interfaces |
| `policy.py` | Shared process-selection policy |
| `requests.py` | Request lifecycle and worker-failure cleanup |
| `shared_memory.py`, `state_columns.py` | Typed shared outputs and owned State assembly |

## Interfaces and ownership

Use `SimulatorAdapter` when only lifecycle operations are needed. `StepPipeline`
describes optional submit/receive/abort operations. The pipeline receives
independent outputs before workers reuse shared storage.

Family code owns State validation, RNG, reset, task behavior and inactive lanes.
Shared transport owns request delivery and failure cleanup. Workers call the
local simulator implementation; they do not define another physics model.
Selecting CUDA output does not move CPU physics to the GPU.

## Checks

Run the focused static checks from the repository root with an interpreter that
has Torch and TensorDict installed:

```sh
mypy --config-file tests/typing/mypy.ini --python-executable /path/to/venv/bin/python
```

These checks cover the shared interfaces and family factory boundaries, not all
simulation internals.
