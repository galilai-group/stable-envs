# Stepping

```python
step(actions, *, states=slice(None), frames=slice(-1, None))
```

`step` advances the live simulator. `states` and `frames` are keyword-only arguments.

- Actions are `[B,A]` for one action per lane or `[B,T,A]` for a sequence. `T` must be positive.
- Discrete actions use `A=1`. An integer `[B]` tensor is also accepted for a single action.
- `step` keeps the action dtype.

`states` and `frames` are Python slices over the post-action outputs. They are independent. `None`
disables that history. The default returns all States and renders only the final one. The outputs
do not include the initial State.

```python
import torch
import stable_envs as snv

with snv.CartPole(num_envs=8, device="cpu") as env:
    env.reset(seed=17)
    one = env.step(torch.zeros(8, 1, dtype=torch.int64))
    frame = one["frames"][:, 0]          # [B,H,W,3]
    chunk = env.step(torch.zeros(8, 4, 1, dtype=torch.int64))
    final_state = chunk["state"]         # full State, batch [B]
    states = chunk["states"]             # full States, batch [B,T]
    final_frame = chunk["frames"][:, 0]
```

| request | `states` | `frames` |
| --- | --- | --- |
| all States, final frame (default) | `slice(None)` | `slice(-1,None)` |
| final State and frame only | `None` | `slice(-1,None)` |
| all States, no rendering | `slice(None)` | `None` |
| final State only, no rendering | `None` | `None` |
| all States and frames | `slice(None)` | `slice(None)` |
| States at 0,4,8… and frames at 3,7,11… | `slice(None,None,4)` | `slice(3,None,4)` |

## Slice rules

- Slices follow Python rules for clipping, negative indices and negative strides.
- A reverse slice changes the output order. The simulation order stays the same.
- Integer selectors are not supported.
- Invalid selectors, zero strides and bad action shapes fail before the simulator moves.

## Output shapes

- A single action counts as a sequence of length one.
- Selected States have batch `[B,K]`. Selected frames have shape `[B,K,H,W,3]`, also when `K=1`.
- An empty slice returns `K=0` and renders nothing. `None` omits the history.
- The result always contains the final `state`, with batch `[B]`.
- There is no singular `frame` key. Use `frames[:,0]` for the default final frame.

Task fields such as reward and termination keep their family names and dtypes. For a sequence they
gain a time axis. `valid` always has shape `[B,T]`. `state_valid` and `frame_valid` give the validity
of the selected States and frames, in output order.

## Termination inside a sequence

Each lane stops after its first termination or truncation. The lane does not use its remaining actions. The
terminal transition is valid. Later task outputs are zero. Selected States and frames repeat the final
endpoint with validity false. The executor does not reset lanes. Single actions keep the upstream
behaviour after termination.

## Memory and ownership

- Snapshots and frames own their storage. Later calls do not change them.
- A family can return one task output tensor at each step, for example `info.goal` of
  `SimplePointMaze`. The family does not change this tensor. Do not write into it.
- `step` renders only the selected frames.
- Retained histories cost memory in proportion to the number of selected outputs.
- Save the endpoint from `result["state"]`. Calling `get_state()` right after `step()` exports it again.

Only submit action chunks whose actions are already known. Selecting fewer frames does not skip
physics steps. Symbolic environments without RGB output need `frames=None`.

There is no `capture`, `transition`, `unroll` or `step_sequence` method. To branch, use another env,
`set_state` and `step`.

## Process workers

On Linux with at least four CPU cores, some families run physics in worker processes. These families
are Fetch, classic Acrobot, the OGBench registrations, PushT and `PyFlytRocketLanding`. Each process has
one simulation thread and stays within the allowed CPU affinity. DM Control steps every lane in one
`mujoco.batch` call on `dmc_threads` threads (8 by default), at most one per lane and one per CPU in the
affinity. Physics stays on CPU even when rendering uses CUDA.

PyBullet allows 1024 clients in one process, and a Rocket restore holds two clients per lane. Thus one
process holds at most 512 Rocket lanes. A larger Rocket batch uses enough processes to hold it, on every
platform and for every value of the tuning knobs.

`set_state` checks the State in every worker before any worker restores it. If one lane is not valid,
`set_state` raises the error and no lane changes.

Two tuning knobs control the pool:

- `worker_min_lanes` is the smallest batch that uses workers. The default is 64.
- `workers` is the largest number of processes. The default is 8.

See [Performance tuning](../performance-tuning.md).

When a worker pool is active, the next known action runs while the parent renders the previous
snapshot. At most one transition is in flight. Every submitted transition finishes before `step`
returns. This does not change the outputs.

Scripts that use this path must follow the usual rules for spawn-based process pools:

- Put env construction and the main loop inside a function guarded by `if __name__ == "__main__":`.
- Call `close()` or use the context manager.
- If rendering fails with a transition in flight, the workers stop. Construct a new environment.

Contributors add a family to the shared scheduler with the private `_step_pipeline()` hook. Its
executor:

- submits one masked transition.
- returns owned task outputs and any requested State.
- aborts pending work on failure.

Rendering stays on the calling thread, except for PushT step frames. PushT workers draw the frames of
their running lanes.
