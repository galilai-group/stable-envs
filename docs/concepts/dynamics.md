# Stepping and selecting outputs

`step(actions, states=slice(None), frames=slice(-1, None))` advances the live
simulator. Actions are `[B,A]` for one action per lane or `[B,T,A]` for a sequence.
Discrete actions use `A=1`; integer `[B]` remains accepted for a single action.
`T` must be positive. Input action dtype is preserved.

`states` and `frames` independently select post-action outputs using Python
slices; `None` disables that history. The default retains all States and renders
only the final endpoint. The initial State is not included.

```python
import torch
import stable_envs as snv

with snv.CartPole(num_envs=8, device="cpu") as env:
    env.reset(seed=17)
    one = env.step(torch.zeros(8, 1, dtype=torch.int64))
    frame = one["frames"][:, 0]          # [B,H,W,3]
    chunk = env.step(torch.zeros(8, 4, 1, dtype=torch.int64))
    final_state = chunk["state"]        # complete State, batch [B]
    states = chunk["states"]            # complete States, batch [B,T]
    final_frame = chunk["frames"][:, 0]
```

| Request | `states` | `frames` |
| --- | --- | --- |
| All States, final frame (default) | `slice(None)` | `slice(-1,None)` |
| Only final State and frame | `None` | `slice(-1,None)` |
| All States, no rendering | `slice(None)` | `None` |
| Only final State, no rendering | `None` | `None` |
| All States and frames | `slice(None)` | `slice(None)` |
| States at 0,4,8…; frames at 3,7,11… | `slice(None,None,4)` | `slice(3,None,4)` |

Slices follow Python clipping, negative-index and negative-stride rules. For
example, `slice(-1,None)` selects the last output; `slice(None,-1)` excludes it.
A reverse slice changes output order, not simulation order. Integer selectors
are intentionally not supported. Invalid selectors, zero strides and invalid
action shapes fail before simulation advances.

A single action behaves as length one for selection. Selected States always
have batch `[B,K]`; selected frames always have shape `[B,K,H,W,3]`, including
`K=1`. Empty slices return empty histories with `K=0` and perform no requested
rendering; `None` omits that history entirely. The full final `state`, batch
`[B]`, is always returned regardless of selections. There is no singular
`frame` alias; use `frames[:,0]` for the default final-frame selection.

Task fields such as reward/termination preserve their family-specific names and
dtypes. They retain their existing single-action shape, and acquire a time axis
for action sequences. `valid` always has shape `[B,T]` (including `T=1`).
`state_valid` and `frame_valid` select the matching validity columns in output
order, making reverse/sparse selections unambiguous.

Within a sequence, each lane stops after its first reported termination or
truncation. Remaining actions in that lane are not consumed. That terminal
transition is valid; later task outputs are zero, and selected States/frames
hold the final endpoint with validity false. The sequence executor adds no
resets; each control step retains its established upstream lifecycle behavior.
Ordinary single actions retain the upstream post-terminal behavior.

Snapshots and frames own their storage and remain unchanged by later calls.
Only selected frames are rendered. Fetch and OGBench provide task-only advance
hooks, and DMC process workers can also step without capturing a snapshot.
Classic paths can still construct snapshots internally, even when the process
path omits their transport. A final State is always needed. Retained
histories cost memory proportional to the selected output count. Use the returned
`result["state"]` when saving the endpoint; calling `get_state()` immediately
after `step()` would export that endpoint again.

Only submit action chunks whose actions are already known. A policy needing the
next image must wait for it before supplying the next action. Selecting fewer
frames does not skip physics steps. Symbolic environments without RGB support
require `frames=None`: they have no image shape, even for an empty frame slice.

There is no `capture`, `transition`, `unroll`, `step_sequence`, or public
`Dynamics` protocol. Branch via an independent environment, `set_state`, and
`step`. Contributors implement the private `_step` kernel; terminating families
also support an active-lane mask, without stepping and restoring inactive lanes.

Large Fetch, supported DMC, classic Acrobot and the three accepted OGBench
registrations on Linux use persistent simulation processes when at least
four CPU cores are available: batches of at least 64 use up to eight processes,
with one simulation thread per process, respecting the allocated CPU affinity.
Small batches, other platforms, and daemon workers retain the local execution
path. Physics remains on CPU even with CUDA rendering. The enabled DMC tasks
are Acrobot, Pendulum, Cartpole, Reacher, Cheetah, Hopper, Walker, Ball-in-Cup,
Finger, Manipulator and Humanoid. The OGBench routes cover `swm/OGBMaze-v0`
(`PointMazeLargeNavigate`), `swm/OGBCube-v0` (`OGBCube`) and `swm/OGBScene-v0`
(`OGBScene`). Classic Acrobot is `swm/AcrobotControl-v1` (`Acrobot`). These are
27 accepted registrations with process routing. The three navigation tasks keep
native Torch batching. Quadruped and the other four classic tasks retain local
execution; process overhead is not assumed worthwhile for every simulator.

When one of these process pools is selected, the dispatcher starts the next known
action while the parent renders the previous owned snapshot. There is at most
one transition in flight. The same path accepts a single action or a sequence;
one action has no later transition to overlap within that call. Selecting only
the final frame also leaves no intermediate rendering to overlap. Use
`frames=slice(None)` or a sparse slice when intermediate images are wanted.
This scheduling changes neither selected outputs nor rendering fidelity.

All submitted transitions are received before `step` returns; no future actions
are inferred or launched across calls. A policy that needs each next image must
still supply its actions one call at a time. CPU stepping and rendering can
compete for host CPU resources, so overlap is measured rather than assumed to
achieve the maximum of their separate rates.

Scripts using the process path must put environment construction and their main
loop inside a function guarded by `if __name__ == "__main__":`, as with other
Python spawn-based process pools. Call `close()` or use an environment context
manager. If rendering fails with a transition in flight, its simulator owners
are stopped; construct a fresh environment instead of continuing an ambiguous
partial sequence.

Contributors can opt a family into the shared scheduler by providing the private
`_step_pipeline()` hook. Its executor submits one masked transition, returns
owned task outputs and any requested full State on receive, and aborts pending
work on failure. The renderer stays on the calling thread. Fetch, the eleven
DMC tasks above, the three OGBench tasks and classic Acrobot have independent
default-routing acceptance gates; the hook alone does not certify concurrency
for other families. Routing acceptance is also distinct from an upstream speedup.

## Reading online performance results

| Measurement | What is timed | Rate denominator |
| --- | --- | --- |
| Task stepping without State capture | Task/physics and the reported IPC/output transfer; no requested rendering | Valid lane transitions |
| Full State export | `get_state`, including transport to the selected output device | Exported lane States |
| Render only | Rendering an already prepared State batch, with transfers identified by the benchmark | Rendered images |
| Online step and render | One complete public `step` call, including required final State, requested histories and frames | Valid transitions for steps/s; emitted images for FPS |

For a nonterminal batch B with T known actions, one call executes B×T
transitions. Rendering only its final endpoint emits B images; rendering every
endpoint emits B×T images. Those two FPS numbers use different image counts and
cannot be compared as if they were the same workload. The `valid`, `state_valid`
and `frame_valid` masks identify terminal padding. If a benchmark counts emitted
held terminal images, it must say so; those images are not new transitions.

Separate component rates are diagnostic, not additive timing partitions. Online
overlap shares CPU, memory and transfer resources, and its speedup must be measured.
Source comparisons should use the same device, image size, matched States,
action sequence, output selection and CPU allocation. When source rendering
supports a GPU, compare against that GPU path. If it has no GPU renderer, label
the CPU-source comparison explicitly. An improvement against a previous library
implementation or deferred scheduling is not an upstream speedup.

Timing tables report mean and sample standard deviation across repeated measured
calls. Image MSE statistics instead summarize the checked matched-state frames,
with pixel values normalized to [0,1]. The current registration-specific evidence
and unfinished comparisons are in the [stepping backlog](../research/stepping_benchmark_backlog.csv).
