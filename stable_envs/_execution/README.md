# Shared process execution

`protocol.py` defines `SimulatorAdapter`: reset, one transition per lane,
get/set State and close. Concrete adapters inherit this protocol. Its abstract
methods require implementations. `Env` owns action sequences, capture slices
and rendering.

| Module | Ownership |
|---|---|
| `protocol.py` | `SimulatorAdapter` and `StepPipeline` interfaces, and the default two-phase restore |
| `process.py` | `ProcessAdapter` and the worker loop that runs any local adapter in spawned processes |
| `policy.py` | Shared process-selection policy |
| `requests.py` | Lane slices, worker start-up, request lifecycle and worker-failure cleanup |
| `workers.py` | Worker CPU pinning and ready/failure replies |
| `shared_memory.py`, `state_columns.py` | Typed shared outputs and owned State assembly |

## Interfaces and ownership

Use `SimulatorAdapter` when only lifecycle operations are needed. `StepPipeline`
describes optional submit/receive/abort operations. The pipeline receives
independent outputs before workers reuse shared storage.

Family code owns State validation, RNG, reset, task behavior and inactive lanes.
Shared transport owns request delivery and failure cleanup. Workers call the
local simulator implementation. They do not define another physics model.
Selecting CUDA output does not move CPU physics to the GPU.

## Process adapters

`ProcessAdapter` runs the lanes of one local adapter in worker processes. A
family subclass supplies:

- a picklable factory. `factory(count, seed)` builds the local adapter of one
  worker. Lane `i` of the batch gets `seed + i`.
- `_inspect(prototype)`, which reads family metadata from a one-lane prototype.
- parent-side checks of actions (`_host_step`), State (`validate`) and family
  options. A request that fails these checks reaches no worker.

The local adapter must accept `step(actions, *, active=None, capture_state=True)`.
With `capture_state=False`, it does not export State. The worker then copies
only the task outputs.

A local adapter lists its family methods in `process_methods`, for example
`{"configure_physics": "per_lane"}`. `ProcessAdapter.call(name, *args)` runs a
listed method on every worker. For `per_lane`, each worker receives the rows of
the first argument that it owns. For `broadcast`, every worker receives the same
arguments. The parent joins the return values in lane order. A listed method
without a parent wrapper is available as an attribute of the process adapter.

## Restore

`set_state` has two phases:

1. Every worker calls `prepare_restore` on its rows. The default implementation
   calls `validate` and keeps the State. Adapters that build new simulators
   during restore stage them here.
2. If every worker accepts, every worker calls `commit_restore`. If any worker
   rejects, every worker calls `discard_restore`, and the parent raises the
   first error. No lane changes.

State leaves that do not fit the shared buffers go to the workers through the
pipe. PushT uses this for its action history, which has no fixed length.

## Checks

Run the focused static checks from the repository root with an interpreter that
has Torch and TensorDict installed:

```sh
mypy --config-file tests/typing/mypy.ini --python-executable /path/to/venv/bin/python
```

These checks cover the shared interfaces and family factory boundaries, not all
simulation internals.
