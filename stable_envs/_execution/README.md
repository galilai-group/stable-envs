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
family gives it a picklable factory. `factory(count, seed)` builds the local
adapter of one worker. Lane `i` of the batch gets `seed + i`.

The local adapter declares what the parent needs:

- `process_attributes` names attributes that the parent copies from a
  one-lane prototype, for example `action_dim` or `fields`. The
  prototype stays open in the parent as `_metadata`.
- `process_methods` names family methods that run on every worker, for
  example `{"configure": "per_lane"}`. `ProcessAdapter.call(name, *args)`
  runs one. For `per_lane`, each worker receives the rows of the first argument
  that it owns. For `broadcast`, every worker receives the same arguments. The
  parent joins the return values in lane order. A listed method without a
  parent wrapper is also available as an attribute of the process adapter.
- `check_step(actions, active)` checks a step and returns the host arrays. The
  parent calls it with the process adapter as `self`, so it can read only
  `num_envs` and `process_attributes`. A step that fails it reaches no worker.
- `probe_actions()` returns actions for one lane. The parent steps the
  prototype once with them to size the shared task buffers.
- `step(actions, *, active=None, capture_state=True)`. With
  `capture_state=False`, the adapter does not export State, and the worker
  copies only the task outputs.
- `process_static_leaves` (optional) names the State leaves that only reset,
  restore and process methods change. After a step with State, the worker
  writes only `process_step_state()`, a `{key tuple: array}` of the other
  leaves in the shared buffers. A thread copies the static leaves from the
  shared buffers for the next snapshot while the workers step. Resets,
  restores and process methods wait for this copy before they start.
- `process_sequences = True` (optional) lets one worker request run a sequence
  of actions. `step_sequence` sends up to `sequence_steps` actions per lane in
  one request. Each worker steps its lanes once per action, and a lane whose
  step sets `terminated`, `truncated` or `done` takes no more actions. The
  workers write the task outputs of every action to shared buffers. An adapter
  declares this only when a step with no active lane changes no simulator and
  `step(actions, active=None)` equals a step with every lane active. The
  optional `step_arrays(actions, active)` returns the task outputs of `step` as
  `{key tuple: array}` without State, and the optional `check_steps(actions)`
  checks all actions of a sequence as `check_step` checks each one. Without
  them, the worker converts the outputs of `step`, and the parent calls
  `check_step` once per action. The step dispatcher runs the same loop in the
  calling process for an env whose `_sequence_simulator()` returns its local
  adapter.
- `render_live()` and `frame_shape` (optional) let the workers draw the frame
  of their running lanes into a shared buffer. `live_frames` is then true, and
  `submit_step(..., capture_frame=True)` returns the frame as `"frame"`. The
  step then needs no State to draw the frame.

`policy.process_count(..., lanes_per_process=n)` gives a batch larger than `n`
lanes enough workers to hold it. Rocket uses this for the PyBullet client limit.

A family subclass adds parent-side checks that need the whole batch: State
checks (`validate`), reset arguments and the arguments of family methods.

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
