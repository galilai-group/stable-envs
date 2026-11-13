# MS-Human-700

[MS-Human-700](https://github.com/LNSGroup/MS-Human-700) ([Zuo et al.](https://lnsgroup.cc/research/MS-Human/)) is a whole-body musculoskeletal human: 100 muscle actuators drive its 36 degrees of freedom. `MSHumanLocomotion` wraps its locomotion model. Native MuJoCo steps the batch on CPU and [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) steps it on CUDA, and MuJoCo OpenGL draws the [frames](../terms.md#frame).

```python
# needs the mshuman extra
import stable_envs as snv

with snv.MSHumanLocomotion(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    out = env.step(env.action_spec.sample(8))
    print(env.render().shape)      # torch.Size([8, 128, 128, 3])
```

## Install

Install the backend with the `mshuman` extra, and see [What each family needs](../install.md#what-each-family-needs) for the GPU setup. The model is Apache-2.0 and isn't on PyPI. The first use downloads a pinned commit into the cache. Set [`STABLE_ENVS_MSHUMAN_HOME`](../environment-variables.md) to an existing checkout to skip the download.

## Task

This is a control sandbox with no task reward. The action has 100 values in `[0, 1]`, one activation per muscle, and the step output has no reward, success or done values. `reset` starts every environment from the model's `init` keyframe.

## Simulation

The [state](../terms.md#state) holds `qpos` (36 values), `qvel` (36), the muscle activations `act` (100), `time` and the solver's `warmstart` (36), all in float32.

`simulation_backend` picks the physics when you construct the environment:

| Value | CPU device | CUDA device |
|---|---|---|
| `"default"` (default) | `mujoco.batch` | MuJoCo Warp |
| `"auto"` | `mujoco.batch` | `mujoco.batch` below `mjwarp_min_envs` (256) environments, MuJoCo Warp from there |
| `"native"` | `mujoco.batch` | `mujoco.batch` |
| `"mjwarp"` | error | MuJoCo Warp |

`mujoco.batch` steps every environment in one native call, on up to 8 threads, at most one per CPU in the process affinity and one per environment. It runs MuJoCo in float64 and rounds the integration state to float32 after every step, so a state restores bit for bit, and the steps equal a per-environment `mj_step` loop with the same rounding. A CUDA environment on `mujoco.batch` returns its state on CUDA. MuJoCo Warp steps every environment in float32 in one captured CUDA graph.

MS-Human-700 ships only this MJCF model and no task code, so the reference dynamics are CPU MuJoCo in float64 stepping the pinned model. Over the first 12 steps (24 ms) from the `init` keyframe under uniformly random activations, across 3,000 trajectories, the CPU path's worst gaps from direct CPU MuJoCo were below `1.5e-5` in `qpos`, `3e-3` in `qvel`, `1.2e-7` in activation and `2e-9` in time. MuJoCo Warp in float32 isn't bit-identical to that reference.

Build the environment in the process that steps it, for example in a DataLoader's `worker_init_fn`: an environment built before `fork()` hangs when the child steps it.

## Rendering

Frames come from host-side MuJoCo OpenGL, one state at a time, which draws the muscle tendons that MuJoCo Warp's renderer leaves out. The renderer reads only `qpos` from each state, and a CUDA environment copies its states to the host and its frames back.

## Variation

This class has no [variation](../terms.md#variation) leaves.

## Classes

### MSHumanLocomotion

<!-- env: MSHumanLocomotion -->

## Limitations

- Rollouts longer than the 12 steps above aren't checked against direct CPU MuJoCo.
- On CUDA, frames are still drawn one state at a time on the host with MuJoCo's OpenGL renderer, then copied to the GPU.
