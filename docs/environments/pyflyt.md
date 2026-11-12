# PyFlyt

A rocket falls from a height with a random tilt and offset, and the agent uses the engine and the fins to land it upright on a pad. `PyFlytRocketLanding` wraps this rocket landing task of [PyFlyt](https://github.com/jjshoots/PyFlyt) ([Tai et al., 2023](https://arxiv.org/abs/2304.01305)). PyFlyt and PyBullet step the physics on CPU, and TinyRenderer or nvdiffrast draws the [frames](../terms.md#frame).

```python
# needs the rocket extra
import stable_envs as snv

with snv.PyFlytRocketLanding(num_envs=8, device="cpu") as env:
    state = env.reset(seed=0)
    frames = env.render()
```

## Install

The class needs the `rocket` extra, which pins `PyFlyt==0.29.0` and `pybullet==3.2.7`. CUDA rendering needs Triton and nvdiffrast, which isn't on PyPI; [build it from source](../install.md#nvdiffrast).

`snv.import_errors()` reports a failed import of the family under `rocket_landing`, `rocket_render` and `rocket_semantics`.

## Task

The action has seven values:

| Index | Control | Bounds |
|---|---|---|
| 0, 1, 2 | finlet deflection in x and y, and finlet roll | [-1, 1] |
| 3 | engine ignition | [0, 1] |
| 4 | engine throttle | [0, 1] |
| 5, 6 | engine gimbal in x and y | [-1, 1] |

PyFlyt sets `env_complete` when the rocket lands on the pad at a low speed, and the step output reports it as `success`.

## Simulation

PyFlyt and PyBullet step the physics on CPU. With worker processes on CPU, the workers draw the step frames of the environments they run with TinyRenderer, without exporting a [state](../terms.md#state) to draw them. On CUDA, the workers send the frame fields and nvdiffrast draws them in the main process.

A step reads from PyBullet only the state leaves that a step can change: the rocket pose, velocity and fuel tank mass, the cached link poses, and the task, RNG and history fields. The other leaves come from the last reset, configuration or restore, and the state equals a full read of the engine. A large leaf that is all zero, such as `current/wrapper/current_goal` without a render mode, isn't stored or copied for each environment. On CUDA, the outputs of the worker processes move to the device as they arrive.

Each PyBullet client holds about 27 MB. A reset or a restore replaces the client of one environment at a time: it makes the new client and then closes the old one, so a worker process holds at most one client more than its environments. A reset checks the arguments of every environment before it changes one. If PyBullet fails during a reset, the environments before the failed one can already have a new episode. A restore keeps the state of each environment it replaced, and if a later environment fails, it replays these states, so no environment changes.

## Rendering

On CPU, TinyRenderer, PyFlyt's own renderer, draws the frames. On CUDA, nvdiffrast draws them.

On Linux, `render` draws a batch of 2 or more states in render worker processes while the main process checks the state. `render_workers` sets the number of workers, and `render_worker_min_rows` sets the smallest such batch (default 2). These frames are byte-identical to frames drawn in the main process.

## Variation

The `environment.start_*` [variation leaves](../tutorials/variation.md) set the start height, horizontal offset and tilt, and change the dynamics. The other leaves are colors: of the rocket parts, of the pad, and `environment.sky_color`, which PyFlyt declares but never draws.

## Classes

### PyFlytRocketLanding

<!-- env: PyFlytRocketLanding -->

## Limitations

The nvdiffrast frames differ from the TinyRenderer frames by a [measured amount](../environments.md#comparison-with-the-upstream-library).
