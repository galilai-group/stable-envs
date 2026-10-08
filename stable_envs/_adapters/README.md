# Simulator adapters

Start with `SimulatorAdapter` in `__init__.py`. It describes the common methods
for an owned batch of simulators: reset, step, get/set State and close. It is a
Python `Protocol`, so implementations do not need to inherit from it.

An adapter advances **one transition per lane**. The public `Env.step()` handles
action sequences, State/frame selection and coordination with the renderer.
Changing the renderer or output device does not by itself move physics to a GPU.

| Files | Responsibility |
|---|---|
| `*_execution.py` | Choose local or process execution for a family. |
| `fetch_pickandplace.py`, `dmcontrol*.py`, `ogbench*.py` | Wrap upstream simulators and translate complete State. Start with files without a `_process` suffix. |
| `*_process.py` | Own persistent workers, partition lanes and copy shared outputs. |
| `process_requests.py` | Send requests, wait for results and clean up failed workers. |
| `state_columns.py` | Assemble State tensors from simulator fields. |
| `pusht_*`, `rocket_*` | Restore task and simulator state, including replay where direct restoration is insufficient. |
| `craftax.py`, `craftax_render.py` | Delegate the registered Craftax configurations to upstream JAX. |

For a concrete path, read `fetch_execution.py`, then `fetch_pickandplace.py`, then
`fetch_process.py`. The process implementation uses the local implementation in
its workers; it does not implement a different physics model.

## Types

Use `adapter: SimulatorAdapter` when the caller only needs the common methods.
Use a concrete family type when it needs extra fields or methods, such as Fetch
object masses or model access. Factory type variables are bounded by the common
protocol and retain the selected family's type. Fetch's process adapter inherits
from its local adapter, so its factory returns `FetchAdapter`.

`reset()` and `get_state()` return owned `TensorDict` snapshots. `set_state()`
restores those snapshots. A normal `step()` returns a `TensorDict` with State and
the task outputs defined by the family. Some implementations also support
skipping State capture, inactive lanes or asynchronous `submit_step()` /
`receive_step()`; those options are not universal protocol requirements.

These hints document interfaces and help static checking. They neither enforce
runtime validation nor add worker dispatch or synchronization.

## Editing

Preserve hidden simulator, task, lifecycle and RNG fields when changing State
handling. Keep shared-buffer ownership and worker-failure cleanup explicit.
Test local/process agreement, restoration after different histories, output
ownership and termination. See `tests/unit/test_fetch_process_adapter.py` for
an existing example. Source comparisons and device gates remain separate tests.

The next layout pass should group these files by family. Keep shared request
transport separate; do not introduce a common adapter base class merely to
force different simulators into the same implementation.
