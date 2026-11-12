# Terms

Each word below has one fixed meaning in these docs, and other pages link here at its first use.

## Environment

An `snv.Env` instance, such as `snv.TwoRoom(num_envs=2)`. It runs `num_envs` copies of one task side by side on one device, and every call works on the whole batch.

## Batch row

One of those copies. Row `i` of every state tensor, action and frame belongs to the same copy, and `len(env)` equals `env.num_envs`. A batch row can end its episode while the others keep going.

## Family

A group of classes that share a module and an install extra, such as `dmcontrol` or `pusht`. `snv.list_families()` lists the families this interpreter can import.

## State

Everything an environment needs to continue exactly where it left off: positions and velocities, task counters, random number generator state and the step count. It's a [`tensordict.TensorDict`](https://pytorch.org/tensordict/) with batch size `(num_envs,)`, and `snv.State` is an alias for that class. It also holds values hidden from the agent, such as TwoRoom's target.

## Action

What you send to `step` for one batch row: `(B, A)` for one step, or `(B, T, A)` for a sequence of `T` steps. Its dtype, bounds and scaling come from the upstream library.

## Variation

The configuration that steps don't change: colors, sizes, physical constants and how reset draws start poses. It stays until `set_variation` or reset options change it. Every class has a `variation_space`, and its defaults reproduce the upstream picture and physics, except where a family page says otherwise.

## Frame

One rendered RGB image, a `uint8` tensor of shape `(H, W, 3)`. A frame depends only on the state and the variation, and you can draw it again from a saved state at any time.

## Step result

The TensorDict that `step` returns, aliased as `snv.StepResult`: the final `state`, the task outputs such as `reward`, and the selected `states` and `frames`.

## Spec

`env.state_spec`, `action_spec`, `frame_spec` or `variation_spec`. Each describes one batch row's shape, dtype and bounds without the batch axis. A spec only describes values. The environment still checks the rules a spec can't express.

## Upstream library

The original implementation a class reimplements or wraps, such as Gymnasium for `CartPole`. `snv.info(name)["reference_id"]` names it, for example `"CartPole-v1"`, and the tests compare frames and dynamics against it.

## Observation

What a model sees, for example a frame plus the agent's position. You pick it from frames and state yourself, or let stable-worldmodel pick it. Step-result fields named `observation` mirror the upstream library for comparison, not as a curated model input.

## Shapes

Shapes on these pages use `B` for `num_envs`, `T` for the steps in an action sequence, `A` for the action shape, `S` and `F` for the number of states and frames a step returns, and `(H, W)` for the frame size.
