# Shared process execution

`protocol.py` defines `SimulatorAdapter`: reset, one transition per batch row, get/set State and close. Concrete adapters inherit this protocol. Its abstract methods require implementations. `Env` owns action sequences, capture slices and rendering.

| Module | Ownership |
|---|---|
| `protocol.py` | `SimulatorAdapter` and `StepPipeline` interfaces, and the default two-phase restore |
| `process.py` | `ProcessAdapter` and the worker loop that runs any local adapter in spawned processes |
| `policy.py` | Shared process-selection policy |
| `requests.py` | Lane slices, worker start-up, request lifecycle and worker-failure cleanup |
| `workers.py` | Worker CPU pinning and ready/failure replies |
| `shared_memory.py`, `state_columns.py` | Typed shared outputs and owned State assembly |
| `mjbatch.py` | Batched MuJoCo physics for B environments (`mujoco.batch` on CPU, MuJoCo Warp on CUDA), backend routing and the parallel source-reset pool |
| `mjwarp_fused.py` | One-launch MuJoCo Warp tree passes (subtree center of mass, composite inertia, RNE internal forces) that `WarpBatch` graphs use |

## Interfaces and ownership

Use `SimulatorAdapter` when only lifecycle operations are needed. `StepPipeline` describes optional submit/receive/abort operations. The pipeline receives independent outputs before workers reuse shared storage. An env returns its pipeline from the private `_step_pipeline()` hook, and the shared step loop in `stable_envs/_core/stepping.py` drives it. The contributor guide `docs/contributing/environment.md` explains how the two fit together.

Family code owns State validation, RNG, reset, task behavior and inactive batch rows. Shared transport owns request delivery and failure cleanup. Workers call the local simulator implementation. They do not define another physics model. Selecting CUDA output does not move CPU physics to the GPU.

## Process adapters

`ProcessAdapter` runs the batch rows of one local adapter in worker processes. A family gives it a picklable factory. `factory(count, seed)` builds the local adapter of one worker, and batch row `i` gets `seed + i`.

The local adapter declares what the parent needs:

- `process_attributes` names attributes that the parent copies from a one-lane prototype, for example `action_dim` or `fields`. The prototype stays open in the parent as `_metadata`.
- `process_methods` names family methods that run on every worker, for example `{"configure": "per_lane"}`. `ProcessAdapter.call(name, *args)` runs one. For `per_lane`, each worker receives the rows of the first argument that it owns. For `broadcast`, every worker receives the same arguments. The parent joins the return values in lane order. A listed method without a parent wrapper is also available as an attribute of the process adapter.
- `check_step(actions, active)` checks a step and returns the host arrays. The parent calls it with the process adapter as `self`, so it can read only `num_envs` and `process_attributes`. A step that fails it reaches no worker.
- `probe_actions()` returns actions for one lane. The parent steps the prototype once with them to size the shared task buffers.
- `step(actions, *, active=None, capture_state=True)`. With `capture_state=False`, the adapter does not export State, and the worker copies only the task outputs.
- `process_static_leaves` (optional) names the State leaves that only reset, restore and process methods change. After a step with State, the worker writes only `process_step_state()`, a `{key tuple: array}` of the other leaves in the shared buffers. The parent copies the static leaves from a cache. A thread copies the cache for the next snapshot while the workers step.
- `render_live()` and `frame_shape` (optional) let the workers draw the frame of their running batch rows into a shared buffer. `live_frames` is then true, and `submit_step(..., capture_frame=True)` returns the frame as `"frame"`. The step then needs no State to draw the frame.
- `frame_fields()` (optional) returns a `{name: array}` with one row per batch row. When `frame_mode` is `FRAME_FIELDS` before `ProcessAdapter.__init__` runs, the workers write it to shared buffers after a step that needs a frame. `take_frame()` then gives a copy of the fields to the family's `_fields_frame`, which draws the frame in the parent while the workers run the next step. Rocket uses this to draw CUDA frames.

`policy.process_count(..., lanes_per_process=n)` gives a batch larger than `n` rows enough workers to hold it. Rocket uses this for the PyBullet client limit.

A family subclass adds parent-side checks that need the whole batch: State checks (`validate`), reset arguments and the arguments of family methods.

## Restore

`set_state` has two phases:

1. Every worker calls `prepare_restore` on its rows. The default implementation calls `validate` and keeps the State. Adapters that build new simulators during restore stage them here.
2. If every worker accepts, every worker calls `commit_restore`. If any worker rejects, every worker calls `discard_restore`, and the parent raises the first error. No batch row changes.

State leaves that do not fit the shared buffers go to the workers through the pipe. PushT uses this for its action history, which has no fixed length.

## Checks

Run the focused static checks from the repository root with an interpreter that has Torch and TensorDict installed:

```sh
mypy --config-file tests/typing/mypy.ini --python-executable /path/to/venv/bin/python
```

These checks cover the shared interfaces and family factory boundaries, not all simulation internals.
