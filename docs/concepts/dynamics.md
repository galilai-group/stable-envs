# Stepping and capture

`step(actions, capture=None)` advances the live simulator. Supply one action per
lane as `[B, A]`, or a sequence as `[B, T, A]`. `B` is `num_envs`, `T` is a positive
sequence length, and `A` is the per-environment action dimension. Discrete actions
use `A=1`; existing integer `[B]` inputs remain accepted as single steps. The
single-step and sequence forms preserve the input action dtype.

```python
import torch
import stable_envs as snv

with snv.CartPole(num_envs=8, device="cpu") as env:
    saved = env.reset(seed=17)
    one = env.step(torch.zeros(8, 1, dtype=torch.int64))
    frame = one["frame"]                 # [B, height, width, 3], uint8
    env.set_state(saved)
    chunk = env.step(torch.zeros(8, 4, 1, dtype=torch.int64))
    states = chunk["states"]            # complete States, batch [B, T]
    final_state = chunk["state"]        # complete State, batch [B]
    final_frame = chunk["frame"]
```

| capture | One action | Sequence |
| --- | --- | --- |
| `None` | Final State and frame | State history, final State and final frame |
| `"states"` | State, no rendering | State history and final State, no rendering |
| `"final_frame"` | Final State and frame | Final State and frame, no intermediate history |
| `"all_frames"` | Final State, frame and one-element frame history | Final State, every frame and final frame |

Every result is a TensorDict with batch `[B]` and a `state` entry. Applicable
reward, termination, truncation, observation and other task outputs retain their
family-specific fields. For sequences those outputs have an additional time
axis: for example reward `[B,T]`. `states` is a nested TensorDict with batch
`[B,T]`; `frames` is `[B,T,height,width,3]`. A sequence of length one still retains
its time axis. Each captured State or image is **after** its corresponding action;
the initial State is not included.

Sequence lanes stop after their first reported termination or truncation. `valid[B,T]`
marks transitions actually executed, including the terminal transition. Remaining
actions in that lane are not consumed. The sequence executor adds no resets;
each control step retains the environment's established lifecycle behavior. Padded task outputs
are zero; State/frame history holds the lane's final endpoint. Use `valid` when
computing statistics. Final State/frame always describes that endpoint. Single
steps retain each upstream environment's established lifecycle behavior.

Snapshots own their storage and remain unchanged by later calls. Internal buffers
and worker scheduling are implementation details. Retaining all States or images
costs memory proportional to `B*T`; submit bounded action chunks for long runs.
Only use chunks whose actions are already known. A policy depending on the next
image must wait for that image before supplying the next action.

`capture="states"` is the explicit physics-only path. Symbolic environments
without RGB output require it. Other capture modes require a working renderer
backend. There is no `transition`, `unroll`, or `step_sequence` method, and no
public `Dynamics` protocol. To branch, construct an independent environment,
restore a complete checkpoint with `set_state`, and call `step`.

For contributors, implement `_step(actions)` as a single-transition kernel.
Families with terminal states also implement `_step(actions, active=mask)` to
skip inactive lanes without stepping, restoring, or resetting them. The shared
public `step` owns action-shape dispatch, histories and capture. Accepted delegated
backends continue to call their authoritative upstream task/controller directly.

Renderer acceptance and simulator acceptance remain independent. This API does
not promote incomplete simulator registrations or change their acceptance gates.

Large Fetch batches use a persistent, bounded CPU worker pool selected internally.
Batches below 64 remain serial; larger batches use at most four workers, capped
by process CPU affinity where available. This is separate from renderer placement:
Fetch physics remains the authoritative CPU simulator even with CUDA frames.
Other families currently retain their existing execution strategies.
