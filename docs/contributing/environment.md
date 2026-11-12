# Adding an environment

A new environment is a registration with three layers. The backend holds the simulator, the actions and the [state](../terms.md#state) schema. The task holds the reward, goal, reset distribution, success rule and time limit. The renderer draws the [frames](../terms.md#frame) with its default configuration. There's no `Task`, `Renderer` or `Observation` class to subclass: find the layer you're changing below and follow its section. To try a small environment in your own code first, follow [Write your own environment](../tutorials/custom-env.md). For branches, tests and pull requests, read [`CONTRIBUTING.md`](https://github.com/galilai-group/stable-envs/blob/main/CONTRIBUTING.md).

| You're changing | Section |
|---|---|
| a simulator, dynamics, action mechanism or state schema | [Backend](#backend) |
| a reward, goal, reset distribution, success rule or time limit on an existing backend | [Task](#task) |
| pixels, a camera or a rendering backend, without changing the simulation | [Renderer](#renderer) |
| a comparison with the upstream library, for any layer | [Matching the upstream library](upstream.md) |
| a fused GPU kernel that must match a reference exactly | [Bit-exact Triton kernels](bit-exact-triton.md) |

Every layer comes back to one check: a saved state restores into an independent instance with a different history, and rendering it changes nothing.

```python
import numpy as np
import torch
import stable_envs as snv

with snv.CartPole(num_envs=2, device="cpu") as source, \
     snv.CartPole(num_envs=2, device="cpu") as restored:
    source.reset(seed=17)
    action = np.array([[0], [1]], dtype=np.int64)
    source.step(action)
    saved = source.get_state()
    held = saved.clone()

    restored.reset(seed=42)
    restored.step(1 - action)  # unrelated history
    assert restored.set_state(saved) is None
    assert (restored.get_state() == saved).all()
    for actions in (action, 1 - action, action):
        assert (source.step(actions) == restored.step(actions)).all()
    assert (saved == held).all()

    before = restored.get_state()
    saved["x"].add_(1)  # snapshots don't share storage with the simulator
    assert (restored.get_state() == before).all()
```

## What you say it supports

Each registration states what it supports: the devices, backends, batch sizes, task configurations and fidelity to its [upstream library](../terms.md#upstream-library) that its docs and metadata promise. Every item needs a test or a recorded comparison behind it. Anything you haven't tested stays off the list. A sibling's result counts only when the code path and configuration are identical, and you still run each public class.

## Where code goes

| Code | Location |
|---|---|
| shared public types: `Env`, state, specs, spaces, registration, configuration | package root |
| family-specific code | `envs/<family>/` |
| a family's simulator adapter and state conversion | `envs/<family>/simulation/` |
| reusable rendering engines | `render/` |
| shared worker-process code | `_execution/` |
| shared internals | `_core/` |
| reference comparisons and corpus utilities | `testing/` |
| experiment drivers | `scripts/` |

A helper used by one family lives with that family, even if it's graphics code. A leading underscore marks an internal module, and a new subpackage only makes sense when several files share an owner.

## Who owns which value

stable-envs owns the full state, reset, step, actions, reward, success, done, RNG, [variation](../terms.md#variation) and rendering. For the dynamics it calls the upstream simulator instead of writing its own. stable-worldmodel picks what a learner sees, which keeps observation classes and state-filtering layers out of this package. Keep these values apart:

| Value | What it holds | Restores the simulator? |
|---|---|---|
| State | a TensorDict of every simulator, task, RNG and lifecycle field the future depends on | yes |
| RenderState | the private renderer input, often a compact float32 row | no, and `set_state` must refuse it |
| Frame | uint8 pixels, `(B, H, W, 3)` | no |
| learner observation | what stable-worldmodel picks from frames, proprioception and goals | not necessarily |

## Backend

The backend owns the simulation and every field needed to resume it, and it calls the upstream engine and controller. A simulating `snv.Env` provides:

| Operation | Contract |
|---|---|
| `reset(seed=None, options=...)` | return an independent full state, keeping the upstream reset and seeding behavior |
| `get_state()`, `state` | an independent TensorDict snapshot with documented field shapes and dtypes |
| `set_state(state)` | validate the whole input before changing anything, restore every field, return `None` |
| `step(actions)` | advance the upstream simulator and return state, reward, success, termination and truncation |
| `state_spec`, `action_spec` | match the actual fields and actions, and document constraints the spec can't express |
| `close()` | release the resources this instance owns, as [Resources](#resources) describes |

You implement the private `_step(actions)` kernel and masked stepping for finished [batch rows](../terms.md#batch-row). The shared public `step` handles action sequences and capture. Keep snapshots and restores out of an ordinary `step()`: they can discard controller or warm-start history and change the trajectory. Renderer-only classes must refuse simulator operations.

The private protocol `stable_envs._execution.protocol.SimulatorAdapter` has `num_envs`, reset, get, set, step and close, and no rendering method. For examples, see `ClassicControlAdapter` in `stable_envs/envs/classic_control/simulation/local.py`, which `ClassicControlSemantics` in `stable_envs/envs/classic_control/semantics.py` uses, and the Fetch adapter in `stable_envs/envs/fetch/simulation/adapter.py`. Family mixins combine backends with renderers. Register public classes with `@register` from `stable_envs.registry`, which uses the class name as the lookup key. Set the family, domain and default configuration explicitly, and let `snv.import_errors()` report missing optional dependencies.

### Designing state

Find every field that affects the future: integration state, control targets and actuator state, applied forces and contact caches, task bookkeeping and reset anchors, and RNG state. Each engine needs its own schema: trace the new family's physics, controller and task before reusing any of CartPole's or Fetch's fields. Record the restore order: which engine call recomputes derived arrays, which fields must be set after it, and which setters have side effects.

Validate before changing any batch row: missing or extra keys, batch shape, dtype, finite values, ranges and semantic constraints. A bad field must leave every earlier row as it was, since the caller has no way to tell which rows changed. A static model or configuration can stay outside state only if restoring requires the same configuration and the docs say so.

Reset seeding follows the upstream library. Some upstream libraries reseed with fresh entropy on `reset(seed=None)` instead of continuing a single generator. State and the reset anchors must capture everything a replay you promise depends on.

### Running in worker processes

To run a backend in worker processes, subclass `stable_envs._execution.process.ProcessAdapter`. The local adapter must accept `step(actions, *, active=None, capture_state=True)`, and its `validate(state)` must leave every batch row unchanged. `stable_envs/_execution/README.md` gives the full contract.

The environment hands its pool to the shared step loop through the private `_step_pipeline()` hook, which returns a `StepPipeline` from `stable_envs._execution.protocol`, or `None` to step in-process. The pipeline submits one masked transition with `submit_step(actions, active=..., capture_state=...)`, returns owned task outputs and any requested state from `receive_step()` before the workers reuse their shared buffers, and releases the workers with `abort_step()` when something fails mid-step. At most one transition is in flight: the next known action runs while the parent renders the previous snapshot, and every submitted transition finishes before `step` returns. A pipeline that sets `host_outputs` keeps the outputs and step bookkeeping on the CPU when the environment is on CUDA, then moves them to the device at the end of `step`.

Rendering stays on the calling thread unless the local adapter defines `render_live()` and `frame_shape`. Then the workers draw the frames of their running batch rows into a shared buffer, and a step that only needs frames builds no state. PyFlyt Rocket Landing works this way.

## Task

A task supplies reset and goal rules, reward, success and lifecycle through the same `Env.reset`, `step`, `get_state` and `set_state`. Goal, task id, reset anchors and counters go in state when future transitions or resets need them, and task variation must reach the upstream code that uses it. Also:

- Keep the goal in state separate from the rendered goal marker and from learner goal images.
- Document the default task and every task id you expose.
- Keep constructor configuration, persistent variation and one-reset overrides apart.
- Raise on unsupported options instead of ignoring them.

Each registration needs its own checks, even when it shares code. The twelve Fetch classes, for example, share one simulator adapter per backend, `FetchBatchAdapter` by default on CPU, with task-specific configuration and a sparse or dense reward, and the `CELLS` table and `_make_classes` in `stable_envs/envs/fetch/envs.py` define them. The same seed and action give the same state in each, and only the reward differs:

```python
# needs the robotics extra
from contextlib import ExitStack

with ExitStack() as stack:
    names = ("FetchPickAndPlace", "FetchPickAndPlaceDense", "FetchPickAndPlaceDict")
    envs = [stack.enter_context(snv.make(n, num_envs=1, device="cpu")) for n in names]
    states = [env.reset(seed=17) for env in envs]
    action = np.array([[0.1, -0.2, 0.05, -1.0]], dtype=np.float32)
    outputs = [env.step(action, frames=None) for env in envs]  # no rendering, no OpenGL
    for out in outputs[1:]:
        assert (out["state"] == outputs[0]["state"]).all()
    assert torch.equal(outputs[0]["reward"], outputs[2]["reward"])
    assert outputs[0]["reward"].item() in (-1.0, 0.0)
    assert outputs[1]["reward"].item() <= 0.0
```

The Dict class doesn't build a dict observation here. The family's own tests check the goal-distance threshold and the exact reward formulas.

### Time limits

A registration's own `TimeLimit` belongs to its lifecycle state. A training or collection cap that a caller adds belongs to that caller's wrapper. Trace the construction path to see which one you have. If the registration owns the limit, keep the elapsed steps in state so a restore resumes the count. Compare termination and truncation separately: a horizon that ends the episode is truncation, not failure.

## Renderer

A renderer draws the same frame from the full state as from its private compact input, without touching the state or the live simulator. A change to collision shapes or physics is a backend change, even if the image changes too.

```python
with snv.CartPole(num_envs=2, device="cpu") as env:
    saved = env.reset(seed=17)
    held, live = saved.clone(), env.get_state()
    rows = env._pack_render_state(saved)
    full = env.render(saved)
    assert torch.equal(full, env._render(rows))
    assert (saved == held).all() and (env.get_state() == live).all()
    for incomplete in (rows, env._unpack_render_state(rows)):
        try:
            env.set_state(incomplete)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("a compact render row restored the simulator")
```

Keep this test for a new renderer, and also compare with the upstream renderer: two equal frames can both be wrong.

`snv.Env.render(state=None, ...)` is the public entry point. A family supplies `_pack_render_state` and the private decode helpers, `_render` or a `scene()` that an existing renderer consumes, and `state_names` and `state_dim` (also `render_state_names` and `render_state_dim`), the layout of the renderer input. A frame is uint8 `(B, H, W, 3)` on the declared device. Support the size, variation, batch and `out=` contracts where you can, and raise a clear error for unsupported backends. Reuse the upstream renderer, which [Matching the upstream library](upstream.md#find-the-upstream-renderer) says how to pin and read, or an existing path from [Renderers](renderers.md): CartPole uses `gfxdraw`, MuJoCo families use MuJoCo or MJWarp, and `Scene2D` is optional.

### The compact RenderState is private

Renderers use `_pack_render_state`, `_unpack_render_state`, `_get_render_state`, `_set_render_state` and `_render`. The compact rows are often float32 with frozen layouts, and they leave out velocities, lifecycle and RNG fields. Decoding a row must not invent those fields. Instead, `_unpack_render_state` may return a partial named projection, and `set_state()` must refuse both the rows and such partial dictionaries. Matching pixels say nothing about whether the state is complete: a renderer can ignore exactly the fields a restore would lose.

### Resources

An instance releases only the resources it owns, a second `close()` does nothing, and later calls raise. `close()` leaves shared process-lifetime rendering contexts and caches alone, since other live environments draw with them. Releasing an EGL context or closing a `mujoco.Renderer` corrupts their rendering, and `tests/_runner/resource_guard.py` fails the test that does it. Create contexts and buffers a bounded number of times, never per state, per style value or per call.

## Checks

Run these local checks for every layer:

- Field validation rejects bad input before any change, and snapshots are independent of each other and of the simulator.
- Reset matches the upstream library on fixed seeds and a fixed variation, and one-step and multi-step continuation match, then restoration after a different history with contacts and controller state.
- Action dtypes, bounds, and clipping or rejection follow the upstream library.
- Reward and success thresholds are tested at the boundary and at the nearest representable values on each side, in the upstream dtype, and termination and truncation are compared separately.
- Dense, sparse and alias classes share physics and differ only in reward, and every public class calls the real simulator on every step.
- The full state and the RenderState give equal frames, rendering is deterministic, and it changes neither the live state, the snapshot, RNG, goals nor later transitions. To catch hidden mutation, render, then step, and compare with a control that didn't render.
- Frames stay the same across batch orders and chunk sizes, where the renderer promises this, and visual-only variation changes pixels and not physics.

```bash
uvx nox -s "test(core)"
uv run pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py
uv run pytest -q tests/unit/test_native_classic_semantics.py tests/unit/test_raster2d_torch_backend.py
uvx nox -s docs
uvx ruff@0.15.0 check . && uvx ruff@0.15.0 format --check .
git diff --check
```

Then run your family's profile with `uvx nox -s "test(<profile>)"`, since skips for missing optional dependencies prove nothing. CI runs the [registry rules](testing.md#registry-rules) on every class, and suites that start native renderers run through the [supervisor](testing.md#running-native-renderers-safely).

Before review, run the final acceptance comparison with the upstream library that [Matching the upstream library](upstream.md#final-acceptance) describes.

## When a registration is done

A new registration is ready for review when everything you say it supports is implemented, every required check passes, and each supported item points to the test or recorded comparison behind it. Unsupported paths raise a clear error instead of running, and a skipped required check counts as a failure. Frozen fixtures and thresholds stay as they are when a check fails, and a sibling class's result doesn't count for your registration unless the code path and configuration are identical. In the pull request, name the layer and every affected registration, report fast tests and final acceptance separately, and keep the driver script, upstream versions and hashes, dependency and platform versions, raw logs and the result of each check for review.

Mistakes reviewers see most: saving only `qpos`, dropping reset RNG or anchors, restoring fields in the wrong order, a setter that changes hidden history, casting float64 upstream arithmetic to float32, silently clipping actions, confusing the visual goal with the reward goal, changing `<` to `<=`, dropping velocity terms from a dense reward, using a trainer's horizon as the simulator's, accepting siblings by name, changing physics while rendering a snapshot, making frames depend on batch order, releasing shared contexts, and saying the physics is right because the pixels match.
